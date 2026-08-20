import os
import re
import fitz
import chromadb
import ollama
import numpy as np
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

PDF_PATH = "epilepsy.pdf"

# IMPORTANT:
# New collection name because we changed the embedding model
# and chunking strategy.
COLLECTION_NAME = "epilepsy_rag_v4_bge"

CHROMA_PATH = "./chroma_db"

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

RERANKER_MODEL = "BAAI/bge-reranker-base"

LLM_MODEL = "llama3.2:latest"

# BGE v1.5 query instruction
QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

# ------------------------------------------------------------
# Chunking
# ------------------------------------------------------------

CHUNK_SIZE = 1200

# Number of previous sentences reused in next chunk
CHUNK_OVERLAP_SENTENCES = 2

# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------

# Chroma retrieves a larger candidate pool first
RETRIEVAL_K = 12

# Final number of chunks sent to LLM
FINAL_K = 4

# ------------------------------------------------------------
# Relevance thresholds
# ------------------------------------------------------------

# This is NOT the only decision anymore.
# It is only an initial dense retrieval filter.
DENSE_MINIMUM = 0.30

# Reranker score is sigmoid-normalized to 0-1.
RERANKER_THRESHOLD = 0.35

# ------------------------------------------------------------
# Lexical matching
# ------------------------------------------------------------

LEXICAL_BOOST = 0.10

# ------------------------------------------------------------
# Database
# ------------------------------------------------------------

# MUST be True the first time you run V4.
# After database is created successfully, you can change to False.
REBUILD_DATABASE = True


# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================

def normalize_text(text):
    """
    Clean PDF extraction without destroying sentence structure.
    """

    # Fix words split by line-break hyphenation.
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Convert line breaks to spaces.
    text = re.sub(r"\n+", " ", text)

    # Remove excessive spaces.
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation.
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


def split_into_sentences(text):
    """
    Lightweight sentence splitter.

    We avoid aggressive NLP dependencies and preserve
    the scientific PDF text as much as possible.
    """

    text = text.strip()

    if not text:
        return []

    # Protect common abbreviations.
    replacements = {
        "e.g.": "e<prd>g<prd>",
        "i.e.": "i<prd>e<prd>",
        "vs.": "vs<prd>",
        "Fig.": "Fig<prd>",
        "Dr.": "Dr<prd>",
        "et al.": "et al<prd>",
    }

    protected = text

    for old, new in replacements.items():
        protected = protected.replace(old, new)

    # Split after sentence-ending punctuation.
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        protected
    )

    restored = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        for old, new in replacements.items():
            sentence = sentence.replace(
                new,
                old
            )

        restored.append(sentence)

    return restored


# ============================================================
# 3. LOAD PDF
# ============================================================

def load_pdf(pdf_path):

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    pdf = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(pdf):

        raw_text = page.get_text("text")

        text = normalize_text(raw_text)

        if text.strip():

            pages.append({
                "page": page_number + 1,
                "text": text
            })

    pdf.close()

    print(
        f"Loaded {len(pages)} pages from PDF."
    )

    return pages


# ============================================================
# 4. BETTER CHUNKING
# ============================================================

def create_chunks(pages):

    chunks = []

    chunk_id = 0

    for page in pages:

        page_number = page["page"]
        text = page["text"]

        sentences = split_into_sentences(text)

        if not sentences:
            continue

        current_sentences = []
        current_length = 0

        for sentence in sentences:

            sentence_length = len(sentence)

            # If adding the sentence makes the chunk too large,
            # save the current chunk first.
            if (
                current_sentences
                and
                current_length + sentence_length > CHUNK_SIZE
            ):

                chunk_text = " ".join(
                    current_sentences
                ).strip()

                if chunk_text:

                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": chunk_text,
                        "page": page_number
                    })

                    chunk_id += 1

                # Keep the last few sentences as overlap.
                current_sentences = (
                    current_sentences[
                        -CHUNK_OVERLAP_SENTENCES:
                    ]
                )

                current_length = sum(
                    len(s)
                    for s in current_sentences
                )

            current_sentences.append(sentence)

            current_length += sentence_length

        # Add final chunk.
        if current_sentences:

            chunk_text = " ".join(
                current_sentences
            ).strip()

            if chunk_text:

                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "page": page_number
                })

                chunk_id += 1

    print(
        f"Created {len(chunks)} sentence-aware chunks."
    )

    return chunks


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print(
        "\nLoading embedding model..."
    )

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    return model


# ============================================================
# 6. LOAD RERANKER
# ============================================================

def load_reranker():

    print(
        "\nLoading reranker..."
    )

    print(
        f"Model: {RERANKER_MODEL}"
    )

    reranker = CrossEncoder(
        RERANKER_MODEL,
        activation_fn=torch.nn.Sigmoid()
    )

    print(
        "Reranker loaded."
    )

    return reranker


# ============================================================
# 7. CREATE / LOAD CHROMADB
# ============================================================

def create_vector_database(
    chunks,
    embedding_model,
    collection_name=COLLECTION_NAME,
    chroma_path=CHROMA_PATH,
    rebuild_database=REBUILD_DATABASE
):

    client = chromadb.PersistentClient(
        path=chroma_path
    )

    # Rebuild database when changing embedding/chunking.
    if rebuild_database:

        try:

            client.delete_collection(
                name=collection_name
            )

            print(
                "\nOld V4 collection deleted."
            )

        except Exception:

            print(
                "\nNo previous V4 collection found."
            )

    collection = client.get_or_create_collection(

        name=collection_name,

        metadata={
            "description":
                "PDF RAG V4",

            "hnsw:space":
                "cosine"
        }
    )

    if collection.count() == 0:

        print(
            "\nCreating embeddings..."
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = (
            embedding_model.encode(

                texts,

                show_progress_bar=True,

                normalize_embeddings=True
            )
        )

        collection.add(

            ids=[
                chunk["id"]
                for chunk in chunks
            ],

            documents=texts,

            embeddings=embeddings.tolist(),

            metadatas=[
                {
                    "page": chunk["page"]
                }

                for chunk in chunks
            ]
        )

        print(
            f"Added {len(chunks)} chunks to ChromaDB."
        )

    else:

        print(
            "\nChromaDB already contains "
            f"{collection.count()} chunks."
        )

    return collection


# ============================================================
# 8. LEXICAL SCORE
# ============================================================

def tokenize(text):

    words = re.findall(
        r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b",
        text.lower()
    )

    # Remove very common words.
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "how",
        "does",
        "do",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "with",
        "from",
        "about",
        "this",
        "that",
        "than"
    }

    return set(
        word
        for word in words
        if word not in stopwords
        and len(word) > 2
    )


def lexical_score(question, document):

    question_tokens = tokenize(
        question
    )

    document_tokens = tokenize(
        document
    )

    if not question_tokens:
        return 0.0

    overlap = (
        question_tokens
        &
        document_tokens
    )

    return len(overlap) / len(
        question_tokens
    )


# ============================================================
# 9. RETRIEVE FROM CHROMADB
# ============================================================

def retrieve_chunks(
    question,
    collection,
    embedding_model,
    top_k=RETRIEVAL_K
):

    # --------------------------------------------------------
    # BGE query embedding
    # --------------------------------------------------------

    query_for_embedding = (
        QUERY_INSTRUCTION
        +
        question
    )

    question_embedding = (
        embedding_model.encode(

            [query_for_embedding],

            normalize_embeddings=True
        )
    )

    # --------------------------------------------------------
    # Chroma retrieval
    # --------------------------------------------------------

    results = collection.query(

        query_embeddings=
            question_embedding.tolist(),

        n_results=top_k,

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_chunks = []

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    for document, metadata, distance in zip(

        documents,
        metadatas,
        distances
    ):

        # Chroma cosine distance:
        # similarity = 1 - distance
        dense_similarity = (
            1 - distance
        )

        lexical = lexical_score(
            question,
            document
        )

        retrieved_chunks.append({

            "text": document,

            "page": metadata["page"],

            "dense_similarity":
                float(dense_similarity),

            "lexical_score":
                float(lexical)
        })

    # --------------------------------------------------------
    # Sort by dense similarity
    # --------------------------------------------------------

    retrieved_chunks.sort(

        key=lambda x:
            x["dense_similarity"],

        reverse=True
    )

    return retrieved_chunks


# ============================================================
# 10. RERANK WITH CROSS ENCODER
# ============================================================

def rerank_chunks(
    question,
    retrieved_chunks,
    reranker
):

    if not retrieved_chunks:
        return []

    # Only rerank candidates that are not extremely weak.
    candidates = [
        chunk
        for chunk in retrieved_chunks
        if chunk["dense_similarity"]
        >= DENSE_MINIMUM
    ]

    # If all candidates are weak,
    # keep the strongest few anyway for diagnosis.
    if not candidates:

        candidates = retrieved_chunks[:FINAL_K]

    pairs = [
        (
            question,
            chunk["text"]
        )

        for chunk in candidates
    ]

    reranker_scores = reranker.predict(
        pairs
    )

    for chunk, score in zip(
        candidates,
        reranker_scores
    ):

        chunk["reranker_score"] = float(
            score
        )

        # Small lexical boost.
        chunk["combined_score"] = (
            0.75 * chunk["reranker_score"]
            +
            0.15 * chunk["dense_similarity"]
            +
            LEXICAL_BOOST
            * chunk["lexical_score"]
        )

    candidates.sort(

        key=lambda x:
            x["combined_score"],

        reverse=True
    )

    return candidates


# ============================================================
# 11. PRINT RETRIEVAL RESULTS
# ============================================================

def display_retrieval_results(
    retrieved_chunks
):

    print(
        "\n" + "-" * 70
    )

    print(
        "RETRIEVAL + RERANKING RESULTS"
    )

    print(
        "-" * 70
    )

    for i, chunk in enumerate(

        retrieved_chunks,

        start=1
    ):

        print(
            f"{i}. "
            f"Page {chunk['page']} | "
            f"Dense: "
            f"{chunk['dense_similarity']:.3f} | "
            f"Lexical: "
            f"{chunk['lexical_score']:.3f} | "
            f"Reranker: "
            f"{chunk.get('reranker_score', 0):.3f} | "
            f"Combined: "
            f"{chunk.get('combined_score', 0):.3f}"
        )


# ============================================================
# 12. ANSWERABILITY CHECK
# ============================================================

def check_answerability(
    question,
    context
):

    prompt = f"""
You are an answerability classifier for a PDF question-answering system.

Your ONLY job is to decide whether the provided PDF context contains
ENOUGH EXPLICIT INFORMATION to answer the user's question.

IMPORTANT:

- Use ONLY the provided context.
- Do NOT use general knowledge.
- Do NOT infer missing facts.
- Do NOT assume information that is not explicitly stated.
- If the question asks for a number, percentage, rate, ratio,
  prevalence, incidence, age, date, odds, or other quantitative value,
  the required value must be explicitly supported by the context.
- If the context contains only related information but not the answer,
  return NO.
- If the context directly contains the answer or the information
  needed to answer it, return YES.

Return ONLY one word:

YES

or

NO

PDF CONTEXT:
----------------
{context}
----------------

QUESTION:
{question}

ANSWERABILITY:
"""

    try:

        response = ollama.chat(

            model=LLM_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result = response[
            "message"
        ][
            "content"
        ].strip().upper()

        if "YES" in result:
            return True

        return False

    except Exception as e:

        print(
            "\nAnswerability check error:"
        )

        print(e)

        # If LLM check fails, fall back to retrieval.
        return None


# ============================================================
# 13. BUILD CONTEXT
# ============================================================

def build_context(
    chunks
):

    context_parts = []

    for i, chunk in enumerate(

        chunks,

        start=1
    ):

        context_parts.append(

            f"""
SOURCE {i}

PDF PAGE:
{chunk['page']}

DENSE SIMILARITY:
{chunk['dense_similarity']:.3f}

RERANKER SCORE:
{chunk.get('reranker_score', 0):.3f}

TEXT:
{chunk['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# 14. DETECT NUMERIC QUESTIONS
# ============================================================

def is_numeric_question(question):

    numeric_keywords = [

        "how many",

        "how much",

        "percentage",

        "percent",

        "rate",

        "prevalence",

        "incidence",

        "number",

        "ratio",

        "odds",

        "risk",

        "probability",

        "frequency",

        "proportion",

        "age",

        "years",

        "per 100",

        "per 1000",

        "per 100,000",

        "fold",

        "million",

        "thousand"
    ]

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in numeric_keywords
    )


# ============================================================
# 15. EXTRACT NUMBERS
# ============================================================

def extract_numbers(text):

    patterns = [

        # Percentages
        r"\b\d+(?:\.\d+)?\s*%",

        # Ranges such as 30%-50%
        r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*%",

        # Decimal numbers
        r"\b\d+\.\d+\b",

        # Numbers with commas
        r"\b\d{1,3}(?:,\d{3})+\b",

        # Integers
        r"\b\d+\b"
    ]

    matches = []

    for pattern in patterns:

        matches.extend(
            re.findall(
                pattern,
                text
            )
        )

    return matches


# ============================================================
# 16. FIND NUMERIC SENTENCES IN CONTEXT
# ============================================================

def find_numeric_evidence(
    question,
    context
):

    sentences = split_into_sentences(
        normalize_text(context)
    )

    question_tokens = tokenize(
        question
    )

    candidates = []

    for sentence in sentences:

        numbers = extract_numbers(
            sentence
        )

        if not numbers:
            continue

        sentence_tokens = tokenize(
            sentence
        )

        overlap = len(
            question_tokens
            &
            sentence_tokens
        )

        candidates.append(
            (
                overlap,
                sentence
            )
        )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        sentence
        for _, sentence in candidates[:3]
    ]


# ============================================================
# 17. GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context
):

    numeric_question = (
        is_numeric_question(
            question
        )
    )

    numeric_rule = ""

    if numeric_question:

        numeric_rule = """
NUMERIC ACCURACY RULE:

This question requires quantitative information.

If the PDF contains numbers, percentages, rates,
ranges, dates, ages, frequencies, or other numerical values
needed to answer the question:

1. You MUST include those values.
2. Copy the numerical values accurately from the PDF context.
3. Do NOT replace exact values with vague wording.
4. Do NOT round or approximate unless the PDF itself does so.
5. Do NOT invent any number.
6. If multiple values are requested, answer ALL of them.
"""

    prompt = f"""
You are a strict PDF question-answering assistant.

Your job is to answer the user's question using ONLY
the provided PDF context.

STRICT RULES:

1. Use ONLY information explicitly supported by the PDF context.

2. Do NOT use your general knowledge.

3. Do NOT guess.

4. Do NOT fill missing information.

5. Do NOT invent facts.

6. Do NOT invent numbers.

7. Answer exactly what the user asked.

8. If the context does not contain enough information,
   respond exactly:

I couldn't find an answer to this question in the PDF.

9. If the question asks for multiple pieces of information,
   provide all supported parts.

10. If the PDF provides exact numbers, preserve them.

11. If the PDF provides percentages, preserve the percentage.

12. If the PDF provides ranges, preserve the range.

13. If the PDF provides "approximately", "about", "nearly",
    or similar wording, preserve that qualification.

14. Do not omit important quantitative details.

15. Do not add unsupported explanations.

16. Do not confuse PDF reference numbers such as [23]
    with actual numerical answers.

17. The final answer must be concise.

{numeric_rule}

PDF CONTEXT
==================================================

{context}

==================================================

USER QUESTION
==================================================

{question}

==================================================

FINAL ANSWER
==================================================
"""

    try:

        response = ollama.chat(

            model=LLM_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response[
            "message"
        ][
            "content"
        ].strip()

    except Exception as e:

        print(
            "\nOllama Error:"
        )

        print(e)

        return None


# ============================================================
# 18. NUMERIC COMPLETENESS REPAIR
# ============================================================

def repair_numeric_answer(
    question,
    draft_answer,
    context
):

    prompt = f"""
You are a strict answer-verification assistant.

The user asked a question about a PDF.

You are given:

1. The question.
2. A draft answer.
3. The PDF context.

Your job is to produce a corrected final answer.

RULES:

- Use ONLY the PDF context.
- Do NOT use outside knowledge.
- Do NOT invent information.
- Keep the answer concise.
- If the draft omitted important numerical values
  that are explicitly present in the context, add them.
- If the question asks for two or more values, make sure
  all supported requested values are included.
- Preserve the exact numerical values from the PDF.
- Preserve "approximately", "about", "nearly", etc.
- Do not introduce numerical values that are not supported
  by the context.
- Do not discuss this verification process.

QUESTION:
{question}

DRAFT ANSWER:
{draft_answer}

PDF CONTEXT:
{context}

CORRECTED FINAL ANSWER:
"""

    try:

        response = ollama.chat(

            model=LLM_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response[
            "message"
        ][
            "content"
        ].strip()

    except Exception as e:

        print(
            "\nNumeric repair error:"
        )

        print(e)

        return draft_answer


# ============================================================
# 19. CHECK NUMERIC COMPLETENESS
# ============================================================

def answer_has_numbers(answer):

    if not answer:
        return False

    numbers = extract_numbers(
        answer
    )

    return len(numbers) > 0


# ============================================================
# 20. FALLBACK NUMERIC EVIDENCE
# ============================================================

def numeric_fallback(
    question,
    context
):

    evidence = find_numeric_evidence(
        question,
        context
    )

    if not evidence:
        return None

    # Return the strongest evidence sentence.
    return (
        "The PDF states: "
        + evidence[0]
    )


# ============================================================
# 21. FINAL ANSWER PIPELINE
# ============================================================

def answer_question(
    question,
    reranked_chunks
):

    if not reranked_chunks:

        return (
            "I couldn't find an answer "
            "to this question in the PDF."
        )

    # --------------------------------------------------------
    # Final chunks
    # --------------------------------------------------------

    final_chunks = []

    for chunk in reranked_chunks:

        # Require reasonable reranker score.
        if chunk.get(
            "reranker_score",
            0
        ) >= RERANKER_THRESHOLD:

            final_chunks.append(
                chunk
            )

    # If threshold is too strict,
    # keep the best result for LLM diagnosis.
    if not final_chunks:

        final_chunks = reranked_chunks[
            :FINAL_K
        ]

    final_chunks = final_chunks[
        :FINAL_K
    ]

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        final_chunks
    )

    # --------------------------------------------------------
    # Answerability check
    # --------------------------------------------------------

    print(
        "\nChecking answerability..."
    )

    answerable = check_answerability(
        question,
        context
    )

    if answerable is False:

        print(
            "Answerability: NO"
        )

        return (
            "I couldn't find an answer "
            "to this question in the PDF."
        )

    elif answerable is True:

        print(
            "Answerability: YES"
        )

    else:

        print(
            "Answerability check unavailable."
        )

        print(
            "Continuing using retrieval scores."
        )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    print(
        "\nGenerating answer..."
    )

    answer = generate_answer(
        question,
        context
    )

    if answer is None:

        return None

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    if is_numeric_question(
        question
    ):

        if not answer_has_numbers(
            answer
        ):

            print(
                "\nWARNING:"
            )

            print(
                "Numeric question detected, "
                "but draft answer contains "
                "no numerical values."
            )

            print(
                "Running numeric repair..."
            )

            repaired_answer = (
                repair_numeric_answer(
                    question,
                    answer,
                    context
                )
            )

            answer = repaired_answer

        # ----------------------------------------------------
        # Final fallback
        # ----------------------------------------------------

        if not answer_has_numbers(
            answer
        ):

            print(
                "\nNumeric repair did not "
                "produce a number."
            )

            fallback = numeric_fallback(
                question,
                context
            )

            if fallback:

                print(
                    "Using PDF numeric evidence "
                    "fallback."
                )

                answer = fallback

    return answer


# ============================================================
# 22. DISPLAY FINAL SOURCES
# ============================================================

def display_sources(
    chunks
):

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL RETRIEVED SOURCES"
    )

    print(
        "=" * 70
    )

    for i, chunk in enumerate(

        chunks,

        start=1
    ):

        print(

            f"\nSource {i} | "
            f"PDF Page {chunk['page']}"
        )

        print(
            f"Dense similarity: "
            f"{chunk['dense_similarity']:.3f}"
        )

        print(
            f"Lexical score: "
            f"{chunk['lexical_score']:.3f}"
        )

        print(
            f"Reranker score: "
            f"{chunk.get('reranker_score', 0):.3f}"
        )

        print(
            f"Combined score: "
            f"{chunk.get('combined_score', 0):.3f}"
        )

        print(
            "\nText:"
        )

        print(
            chunk["text"][:500]
            + "..."
        )


# ============================================================
# 23. MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "PDF RAG SYSTEM V4"
    )

    print(
        "=" * 70
    )

    print(
        "\nEmbedding model:"
    )

    print(
        EMBEDDING_MODEL
    )

    print(
        "\nReranker:"
    )

    print(
        RERANKER_MODEL
    )

    print(
        "\nLLM:"
    )

    print(
        LLM_MODEL
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    pages = load_pdf(
        PDF_PATH
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    chunks = create_chunks(
        pages
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    reranker = load_reranker()

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    collection = (
        create_vector_database(

            chunks,

            embedding_model,

            collection_name=COLLECTION_NAME,

            chroma_path=CHROMA_PATH,

            rebuild_database=REBUILD_DATABASE
        )
    )

    # --------------------------------------------------------
    # READY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG SYSTEM IS READY!"
    )

    print(
        "=" * 70
    )

    print(
        "\nAsk questions about the PDF."
    )

    print(
        "Type 'exit' to stop."
    )

    # ========================================================
    # CHAT LOOP
    # ========================================================

    while True:

        print(
            "\n" + "-" * 70
        )

        question = input(
            "Your question: "
        ).strip()

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if question.lower() == "exit":

            print(
                "\nGoodbye!"
            )

            break

        # ----------------------------------------------------
        # EMPTY QUESTION
        # ----------------------------------------------------

        if not question:

            continue

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        retrieved_chunks = (
            retrieve_chunks(

                question,

                collection,

                embedding_model,

                RETRIEVAL_K
            )
        )

        # ----------------------------------------------------
        # RERANKING
        # ----------------------------------------------------

        reranked_chunks = (
            rerank_chunks(

                question,

                retrieved_chunks,

                reranker
            )
        )

        # ----------------------------------------------------
        # DISPLAY SCORES
        # ----------------------------------------------------

        display_retrieval_results(
            reranked_chunks
        )

        # ----------------------------------------------------
        # FINAL ANSWER
        # ----------------------------------------------------

        answer = answer_question(

            question,

            reranked_chunks
        )

        if answer is None:

            continue

        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "ANSWER"
        )

        print(
            "=" * 70
        )

        print(
            answer
        )

        # ----------------------------------------------------
        # DISPLAY SOURCES
        # ----------------------------------------------------

        final_chunks = (
            reranked_chunks[
                :FINAL_K
            ]
        )

        display_sources(
            final_chunks
        )


# ============================================================
# 24. RUN
# ============================================================

if __name__ == "__main__":

    main()
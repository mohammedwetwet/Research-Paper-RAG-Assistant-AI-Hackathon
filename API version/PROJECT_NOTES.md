# Epilepsy RAG Project: Technical Notes & Design Decisions

This document details the architectural decisions, design rationale, data processing strategies, and operational considerations of the Epilepsy RAG system.

---

## 1. Data Ingestion & Chunking Rationale

### Strategy: Two-Stage Hybrid Chunking
Located in [`main.py`](file:///c:/Users/abdul/OneDrive/Desktop/API_RAG/main.py).

1. **Structural Splitting (`MarkdownHeaderTextSplitter`)**:
   - The knowledge base [`epilepsy_parsed.md`](file:///c:/Users/abdul/OneDrive/Desktop/API_RAG/epilepsy_parsed.md) is heavily structured with headings (`# Section`, `## SubSection`, `### SubSubSection`).
   - By first splitting along Markdown headers, each chunk inherits its exact hierarchical section path (e.g. `2.2 Etiological Classification and Risk Factors`) in `chunk.metadata`.
   - This metadata enables granular section filtering, source tracing, and section-level retrieval evaluation.

2. **Recursive Refinement (`RecursiveCharacterTextSplitter`)**:
   - `chunk_size = 800` characters, `chunk_overlap = 150` characters.
   - Separators: `["\n\n", "\n", ". ", " ", ""]` to prioritize semantic sentence boundaries.
   - Ensures chunks fit comfortably within the embedding model context window while maintaining sufficient semantic density without clipping mid-sentence.
   - **Result**: Exactly **412 chunks** generated across the entire clinical document.

---

## 2. Remote Embedding & Vector Store Design

### Embedding Model: `BAAI/bge-base-en-v1.5`
- **Dimensions**: 768-dimensional normalized dense vectors.
- **Remote Execution**: Processed entirely through Hugging Face's Inference API (`https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5`).
- **Batching**: Grouped into batches of 32 (`EMBEDDING_BATCH_SIZE = 32`) to minimize network overhead and respect rate limits.
- **Asymmetric Search Prefix**: In [`retrieval.py`](file:///c:/Users/abdul/OneDrive/Desktop/API_RAG/retrieval.py), search queries are prefixed with:
  ```text
  Represent this sentence for searching relevant passages: <query>
  ```
  This is recommended by BAAI to align short query representations with longer document passages in embedding space.

### Vector Database: Local Qdrant
Located in [`vector_store.py`](file:///c:/Users/abdul/OneDrive/Desktop/API_RAG/vector_store.py).
- **Storage**: Persisted locally in `qdrant_data/` without requiring external Docker containers or separate background server processes.
- **Metric**: Cosine similarity (`Distance.COSINE`).
- **Payload Schema**: Each point stores the vector ID, 768-dim float vector, text payload, and structural metadata dict (`Section`, `SubSection`, etc.).

---

## 3. LLM Generation & Grounding Strategy

Located in [`rag_chat.py`](file:///c:/Users/abdul/OneDrive/Desktop/API_RAG/rag_chat.py).

### Model: Groq `openai/gpt-oss-20b`
- **Low Latency**: High inference speed via Groq LPUs.
- **Context Grounding**: System prompt strictly constrains the assistant in high-efficiency English instructions:
  ```text
  You are an expert medical information assistant specializing in clinical epilepsy.
  Answer the user's question accurately, concisely, and strictly in English based on the provided retrieved context.

  Core Guidelines:
  1. Language: Always answer exclusively in English.
  2. Strict Grounding: Base your answer solely on the retrieved context. Do not extrapolate, assume, or incorporate unmentioned facts.
  3. Insufficient Context: If the retrieved context does not contain sufficient information to answer the question, clearly state: "The provided context does not contain sufficient information to answer this question."
  4. Citation Markers: Place source markers like [1], [2], or [3] directly after every factual statement to cite the relevant context passage.
  5. Efficiency & Conciseness: Deliver direct, well-structured, and focused responses. Avoid conversational filler.
  6. Medical Safety: Provide objective, scientific information only. Do not provide personalized medical advice.
  ```
- **Context Numbering**: Retrieved chunks are formatted into numbered blocks `[1] ...`, `[2] ...` so the LLM can reference specific chunks directly in its response.

### Rate-Limiting & Resilience
- Free/on-demand tiers enforce Tokens Per Minute (TPM) limits.
- Both [`rag_chat.py`](file:///c:/Users/abdul/OneDrive/Desktop/API_RAG/rag_chat.py) and [`evaluation.py`](file:///c:/Users/abdul/OneDrive/Desktop/API_RAG/evaluation.py) implement automatic **exponential backoff retry** for HTTP 429 errors.
- Custom `User-Agent: API-RAG/1.0` header is included to prevent Cloudflare/WAF blocks on standard python-urllib requests.

---

## 4. Evaluation Methodology

Located in [`evaluation.py`](file:///c:/Users/abdul/OneDrive/Desktop/API_RAG/evaluation.py).

### Metrics Tracked
1. **Hit@k (Hit@1, Hit@3, Hit@4)**:
   - Verifies whether the expected clinical section appears in the top-$k$ retrieved chunks.
   - **Observed Hit@1**: `80.0%`
   - **Observed Hit@3 & Hit@4**: `100.0%`
2. **Mean Reciprocal Rank (MRR)**:
   - Measures ranking quality ($\frac{1}{\text{rank}}$ of first relevant section).
   - **Observed MRR**: `0.900`
3. **LLM-as-a-Judge Faithfulness (1–5)**:
   - Evaluates whether factual claims in the answer can be directly verified from the provided context.
   - **Observed Score**: `5.0 / 5.0`
4. **LLM-as-a-Judge Relevance (1–5)**:
   - Evaluates whether the answer directly and comprehensively addresses the question.
   - **Observed Score**: `5.0 / 5.0`
5. **Citation Precision**:
   - Ensures all citations (`[1]`, `[2]`, etc.) refer to valid source indices ($1 \le index \le k$).
   - **Observed Precision**: `100.0%`
6. **Out-of-Domain Refusal**:
   - Verifies that when an out-of-scope question is asked (e.g. Type 2 diabetes), the model refuses to fabricate advice and explicitly states context insufficiency.
   - **Observed Status**: `PASSED (True)`

---

## 5. API Layer Architecture

Located in [`api.py`](file:///c:/Users/abdul/OneDrive/Desktop/API_RAG/api.py).

### Endpoints
- **`GET /health`**:
  Returns health status without exposing sensitive credentials:
  ```json
  {
    "ready": true,
    "vector_database_ready": true,
    "hugging_face_key_available": true,
    "groq_key_available": true
  }
  ```
- **`POST /ask`**:
  Input:
  ```json
  {
    "question": "What is the role of EEG in epilepsy diagnosis?",
    "limit": 4
  }
  ```
  Output:
  ```json
  {
    "answer": "The electroencephalogram (EEG) is a cornerstone...",
    "model": "openai/gpt-oss-20b",
    "sources": [
      {
        "id": 214,
        "score": 0.7357,
        "metadata": {"Section": "4 Clinical Evaluation", "SubSection": "4.1.2 Electrophysiological Assessment"}
      }
    ]
  }
  ```
- **`GET /docs`**: Interactive Swagger UI for live testing.

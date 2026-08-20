import os
import tempfile
import hashlib

import streamlit as st

import rag


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    .answer-box {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 1rem;
    }

    .source-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.20);
        margin-bottom: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "collection" not in st.session_state:
    st.session_state.collection = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "reranker" not in st.session_state:
    st.session_state.reranker = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# 4. LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    embedding_model = rag.load_embedding_model()

    reranker = rag.load_reranker()

    return embedding_model, reranker


# ============================================================
# 5. PDF HASH
# ============================================================

def get_file_hash(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# ============================================================
# 6. CREATE TEMPORARY PDF
# ============================================================

def save_uploaded_pdf(uploaded_file):

    suffix = ".pdf"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        return temp_file.name


# ============================================================
# 7. BUILD VECTOR DATABASE
# ============================================================

def process_pdf(
    uploaded_file,
    embedding_model
):

    temp_pdf_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded PDF
        # ----------------------------------------------------

        temp_pdf_path = save_uploaded_pdf(
            uploaded_file
        )

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        pages = rag.load_pdf(
            temp_pdf_path
        )

        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------

        chunks = rag.create_chunks(
            pages
        )

        # ----------------------------------------------------
        # Create unique collection
        # ----------------------------------------------------

        file_hash = get_file_hash(
            uploaded_file
        )

        collection_name = (
            f"streamlit_pdf_{file_hash}"
        )

        # ----------------------------------------------------
        # Create isolated ChromaDB
        # ----------------------------------------------------

        collection = (
            rag.create_vector_database(

                chunks,

                embedding_model,

                collection_name=
                    collection_name,

                chroma_path=
                    rag.CHROMA_PATH,

                rebuild_database=False
            )
        )

        return (
            collection,
            file_hash,
            len(pages),
            len(chunks)
        )

    finally:

        # ----------------------------------------------------
        # Remove temporary file
        # ----------------------------------------------------

        if (
            temp_pdf_path
            and
            os.path.exists(temp_pdf_path)
        ):

            os.remove(
                temp_pdf_path
            )


# ============================================================
# 8. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 PDF RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload a PDF and ask questions based only on its content.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 PDF")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    st.divider()

    st.subheader("⚙️ RAG Configuration")

    st.write(
        f"**Embedding:** "
        f"`{rag.EMBEDDING_MODEL}`"
    )

    st.write(
        f"**Reranker:** "
        f"`{rag.RERANKER_MODEL}`"
    )

    st.write(
        f"**LLM:** "
        f"`{rag.LLM_MODEL}`"
    )

    st.write(
        f"**Retrieval K:** "
        f"`{rag.RETRIEVAL_K}`"
    )

    st.write(
        f"**Final K:** "
        f"`{rag.FINAL_K}`"
    )

    st.divider()

    if st.session_state.pdf_loaded:

        st.success(
            f"Loaded: {st.session_state.pdf_name}"
        )

    else:

        st.info(
            "Upload a PDF to start."
        )


# ============================================================
# 10. PROCESS UPLOADED PDF
# ============================================================

if uploaded_file is not None:

    current_hash = get_file_hash(
        uploaded_file
    )

    # Only process when a new PDF is uploaded.
    if (
        current_hash
        !=
        st.session_state.pdf_hash
    ):

        st.session_state.pdf_loaded = False

        st.session_state.collection = None

        st.session_state.chat_history = []

        st.session_state.pdf_hash = current_hash

        with st.spinner(
            "Loading AI models..."
        ):

            (
                embedding_model,
                reranker
            ) = load_models()

        st.session_state.embedding_model = (
            embedding_model
        )

        st.session_state.reranker = (
            reranker
        )

        with st.spinner(
            "Processing PDF and creating vector database..."
        ):

            try:

                (
                    collection,
                    file_hash,
                    page_count,
                    chunk_count
                ) = process_pdf(

                    uploaded_file,

                    embedding_model
                )

                st.session_state.collection = (
                    collection
                )

                st.session_state.pdf_loaded = True

                st.session_state.pdf_name = (
                    uploaded_file.name
                )

                st.session_state.page_count = (
                    page_count
                )

                st.session_state.chunk_count = (
                    chunk_count
                )

                st.success(
                    "PDF processed successfully!"
                )

            except Exception as e:

                st.session_state.pdf_loaded = False

                st.error(
                    "Failed to process the PDF."
                )

                st.exception(e)


# ============================================================
# 11. PDF INFORMATION
# ============================================================

if st.session_state.pdf_loaded:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "PDF",
            st.session_state.pdf_name
        )

    with col2:

        st.metric(
            "Pages",
            st.session_state.page_count
        )

    with col3:

        st.metric(
            "Chunks",
            st.session_state.chunk_count
        )

else:

    st.info(
        "👈 Upload a PDF from the sidebar to begin."
    )


# ============================================================
# 12. QUESTION INPUT
# ============================================================

if st.session_state.pdf_loaded:

    st.divider()

    st.subheader(
        "💬 Ask a Question"
    )

    question = st.text_input(
        "Your question",
        placeholder=
            "Ask something about the uploaded PDF..."
    )

    ask_button = st.button(
        "🔍 Ask",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # 13. ASK QUESTION
    # ========================================================

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the PDF and generating an answer..."
            ):

                try:

                    # ------------------------------------------------
                    # Retrieval
                    # ------------------------------------------------

                    retrieved_chunks = (
                        rag.retrieve_chunks(

                            question,

                            st.session_state.collection,

                            st.session_state.embedding_model,

                            rag.RETRIEVAL_K
                        )
                    )

                    # ------------------------------------------------
                    # Reranking
                    # ------------------------------------------------

                    reranked_chunks = (
                        rag.rerank_chunks(

                            question,

                            retrieved_chunks,

                            st.session_state.reranker
                        )
                    )

                    # ------------------------------------------------
                    # Answer
                    # ------------------------------------------------

                    answer = (
                        rag.answer_question(

                            question,

                            reranked_chunks
                        )
                    )

                    # ------------------------------------------------
                    # Save history
                    # ------------------------------------------------

                    st.session_state.chat_history.append({

                        "question":
                            question,

                        "answer":
                            answer,

                        "sources":
                            reranked_chunks[
                                :rag.FINAL_K
                            ]
                    })

                except Exception as e:

                    st.error(
                        "An error occurred while answering the question."
                    )

                    st.exception(e)


# ============================================================
# 14. CHAT HISTORY
# ============================================================

if st.session_state.chat_history:

    st.divider()

    st.subheader(
        "💬 Conversation"
    )

    for item in reversed(
        st.session_state.chat_history
    ):

        # ----------------------------------------------------
        # Question
        # ----------------------------------------------------

        st.markdown(
            f"### 👤 Question"
        )

        st.write(
            item["question"]
        )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.markdown(
            "### 🤖 Answer"
        )

        st.markdown(
            f"""
            <div class="answer-box">
            {item["answer"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        with st.expander(
            "📚 Retrieved Sources"
        ):

            for i, chunk in enumerate(

                item["sources"],

                start=1
            ):

                st.markdown(
                    f"#### Source {i}"
                )

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                with col1:

                    st.write(
                        f"**Page:** "
                        f"{chunk['page']}"
                    )

                with col2:

                    st.write(
                        f"**Dense:** "
                        f"{chunk['dense_similarity']:.3f}"
                    )

                with col3:

                    st.write(
                        f"**Reranker:** "
                        f"{chunk.get('reranker_score', 0):.3f}"
                    )

                with col4:

                    st.write(
                        f"**Combined:** "
                        f"{chunk.get('combined_score', 0):.3f}"
                    )

                st.write(
                    chunk["text"]
                )

                st.divider()


# ============================================================
# 15. CLEAR CHAT
# ============================================================

if st.session_state.chat_history:

    if st.button(
        "🗑️ Clear Conversation"
    ):

        st.session_state.chat_history = []

        st.rerun()
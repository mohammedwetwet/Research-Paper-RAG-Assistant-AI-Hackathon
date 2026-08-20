"""
streamlit_app.py - Interactive Streamlit Web Interface
-------------------------------------------------------
This script provides an interactive web browser UI for the Epilepsy RAG system:
- Multi-turn conversation chat with message history.
- Live health indicators for Qdrant, Hugging Face, and Groq.
- Collapsible evidence drawers displaying chunk text, section headers, and Cosine similarity chips.
- Sidebar controls: retrieval depth slider (top-k), sample prompt buttons, and clear chat.
"""

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure project directory is on sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from main import EMBEDDING_MODEL, get_env_value
from rag_chat import GROQ_MODEL, generate_rag_answer
from vector_store import COLLECTION_NAME, QDRANT_PATH
from api import vector_database_is_ready


# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Epilepsy RAG Clinical Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS for Premium Aesthetics
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .hero-container {
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
    }
    
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.5rem;
        margin-top: 0.4rem;
    }
    
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    .badge-info {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    /* Source Card Styling */
    .source-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        background: rgba(30, 41, 59, 0.75);
    }
    
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    
    .source-tag {
        font-size: 0.75rem;
        font-weight: 600;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
    }
    
    .score-chip {
        font-size: 0.75rem;
        font-weight: 600;
        color: #a78bfa;
        background: rgba(167, 139, 250, 0.12);
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
    }
    
    .source-text {
        color: #cbd5e1;
        font-size: 0.85rem;
        line-height: 1.45;
        white-space: pre-wrap;
    }

    /* Disclaimer box */
    .disclaimer-box {
        padding: 0.6rem 0.9rem;
        background: rgba(245, 158, 11, 0.08);
        border-left: 3px solid #f59e0b;
        border-radius: 6px;
        color: #fbbf24;
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# System Status & Health Check
# -----------------------------------------------------------------------------
qdrant_ready = vector_database_is_ready()
hf_ready = bool(get_env_value("HF_TOKEN"))
groq_ready = bool(get_env_value("GROQ_API_KEY"))
all_ready = qdrant_ready and hf_ready and groq_ready


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")

    # Retrieval Slider
    k_limit = st.slider(
        "Context Chunks (Top-k)",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of most similar context passages to retrieve from Qdrant.",
    )

    st.markdown("---")
    st.markdown("### 🩺 System Diagnostics")

    col1, col2 = st.columns(2)
    with col1:
        if qdrant_ready:
            st.markdown('<span class="badge badge-success">✓ Qdrant DB</span>', unsafe_allow_html=True)
        else:
            st.error("Qdrant Missing")
    with col2:
        if groq_ready:
            st.markdown('<span class="badge badge-success">✓ Groq API</span>', unsafe_allow_html=True)
        else:
            st.error("Groq Key Missing")

    if hf_ready:
        st.markdown('<span class="badge badge-info">✓ BAAI / Hugging Face</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Quick Prompt Suggestions")
    
    sample_queries = [
        "What are the main causes of epilepsy?",
        "How common is epilepsy worldwide?",
        "How do ion-channel problems contribute to epilepsy?",
        "What is the role of EEG in epilepsy diagnosis?",
        "How is drug-resistant epilepsy managed?",
    ]

    for sample in sample_queries:
        if st.button(f"🔍 {sample}", use_container_width=True, key=f"btn_{sample}"):
            st.session_state.pending_prompt = sample

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"**Embeddings:** `{EMBEDDING_MODEL}` (768d)\n\n**LLM:** `{GROQ_MODEL}`")


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">🧠 Epilepsy Clinical RAG Assistant</h1>
        <p class="hero-subtitle">Interactive retrieval-augmented medical information system with strictly grounded evidence & source citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Clinical Disclaimer:</strong> This assistant generates factual responses grounded exclusively on the local epilepsy clinical document. It is intended for informational/research review and does not provide personalized medical diagnoses or treatment prescriptions.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Chat History Management
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your Epilepsy Clinical Assistant. I can answer questions regarding etiological categories, "
                "epidemiology, ion-channel pathomechanisms, EEG diagnostics, and treatment of drug-resistant epilepsy.\n\n"
                "How can I help you today?"
            ),
            "sources": [],
        }
    ]

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

        # Display collapsible source citations if present
        if msg.get("sources"):
            with st.expander(f"📚 View Retrieved Evidence ({len(msg['sources'])} Context Chunks)"):
                for idx, src in enumerate(msg["sources"], start=1):
                    sec_name = (
                        src["metadata"].get("SubSection")
                        or src["metadata"].get("Section")
                        or "General Section"
                    )
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-header">
                                <span class="source-tag">Source [{idx}] • {sec_name}</span>
                                <span class="score-chip">Cosine Score: {src['score']:.4f} ({src['score']*100:.1f}%)</span>
                            </div>
                            <div class="source-text">{src['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# -----------------------------------------------------------------------------
# Input Handling & Answer Generation
# -----------------------------------------------------------------------------
# Check if a sample button was clicked
prompt_input = st.chat_input("Ask a clinical or scientific question about epilepsy...")
active_prompt = None

if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif prompt_input:
    active_prompt = prompt_input

if active_prompt:
    if not all_ready:
        st.error("System dependencies are not ready. Please verify Qdrant database and API keys in .env.")
    else:
        # Append and render user message
        st.session_state.messages.append({"role": "user", "content": active_prompt, "sources": []})
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_prompt)

        # Generate assistant response
        with st.chat_message("assistant", avatar="🧑‍⚕️"):
            with st.spinner("🔍 Retrieving clinical evidence & synthesizing answer..."):
                try:
                    response = generate_rag_answer(active_prompt, limit=k_limit)
                    answer = response["answer"]
                    sources = response["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📚 View Retrieved Evidence ({len(sources)} Context Chunks)"):
                            for idx, src in enumerate(sources, start=1):
                                sec_name = (
                                    src["metadata"].get("SubSection")
                                    or src["metadata"].get("Section")
                                    or "General Section"
                                )
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-header">
                                            <span class="source-tag">Source [{idx}] • {sec_name}</span>
                                            <span class="score-chip">Cosine Score: {src['score']:.4f} ({src['score']*100:.1f}%)</span>
                                        </div>
                                        <div class="source-text">{src['text']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Save to conversation state
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except Exception as error:
                    st.error(f"Generation error: {error}")

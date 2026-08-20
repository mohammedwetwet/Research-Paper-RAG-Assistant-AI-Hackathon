"""
rag_chat.py - Grounded LLM Answering & Interactive Terminal Chat
-----------------------------------------------------------------
This module connects the retrieval layer to Groq's LLM (openai/gpt-oss-20b):
1. Retrieves top-k matching chunks from Qdrant using retrieval.py.
2. Formats retrieved chunks into numbered context blocks: [1] ..., [2] ...
3. Sends the prompt to Groq with strict medical-information system instructions:
   - Answer strictly from context.
   - Refuse/state insufficient context if outside domain.
   - Cite facts using source markers like [1], [2].
4. Handles rate-limit errors (HTTP 429) automatically via exponential backoff.
5. Provides both single-query execution and an interactive multi-turn REPL loop.
"""

import argparse
import json
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from main import get_env_value
from retrieval import retrieve


# -----------------------------------------------------------------------------
# Configuration & Prompt Design
# -----------------------------------------------------------------------------
# Groq Model ID and Chat Completions API endpoint
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# High-efficiency system instructions enforcing English output, grounding, medical safety, and citations
SYSTEM_PROMPT = """You are an expert medical information assistant specializing in clinical epilepsy.
Answer the user's question accurately, concisely, and strictly in English based on the provided retrieved context.

Core Guidelines:
1. Language: Always answer exclusively in English.
2. Strict Grounding: Base your answer solely on the retrieved context. Do not extrapolate, assume, or incorporate unmentioned facts.
3. Insufficient Context: If the retrieved context does not contain sufficient information to answer the question, clearly state: "The provided context does not contain sufficient information to answer this question."
4. Citation Markers: Place source markers like [1], [2], or [3] directly after every factual statement to cite the relevant context passage.
5. Efficiency & Conciseness: Deliver direct, well-structured, and focused responses (e.g., bullet points or concise paragraphs). Avoid unnecessary conversational filler, greetings, or pleasantries.
6. Medical Safety: Provide objective, scientific information only. Do not provide personalized medical advice or individual clinical diagnosis."""


# -----------------------------------------------------------------------------
# Context Assembly Helper
# -----------------------------------------------------------------------------
def build_context(results: list[dict[str, Any]]) -> str:
    """Format retrieved Qdrant chunks into numbered context blocks for the LLM.

    Args:
        results: List of retrieved chunk dictionaries from retrieval.py.

    Returns:
        A numbered string like:
        [1] First chunk text...
        [2] Second chunk text...
    """
    return "\n\n".join(
        f"[{index}] {result['text']}"
        for index, result in enumerate(results, start=1)
    )


# -----------------------------------------------------------------------------
# Core RAG Generation Function
# -----------------------------------------------------------------------------
def generate_rag_answer(question: str, limit: int = 4) -> dict[str, Any]:
    """Retrieve relevant chunks and prompt Groq to produce a grounded, cited answer.

    Args:
        question: The user's query string.
        limit: Number of context passages to retrieve (default: 4).

    Returns:
        Dictionary containing:
        - 'answer': The LLM-generated response string.
        - 'sources': List of retrieved chunk metadata dictionaries.
    """
    api_key = get_env_value("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Please add your Groq key to .env.")

    # Step 1: Retrieve closest document chunks from Qdrant
    results = retrieve(question, limit)

    # Step 2: Build the combined prompt payload
    user_message = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{build_context(results)}"
    )
    payload = json.dumps(
        {
            "model": GROQ_MODEL,
            "temperature": 0.2,             # Low temperature for factual, deterministic output
            "max_completion_tokens": 700,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        }
    ).encode("utf-8")

    request = Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "API-RAG/1.0",     # Custom User-Agent prevents WAF/Cloudflare blocks
        },
        method="POST",
    )

    # Step 3: Call Groq API with exponential backoff on HTTP 429 rate limits
    max_retries = 4
    data = None

    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
                break
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            # If rate limited (HTTP 429), wait and retry
            if error.code == 429 and attempt < max_retries - 1:
                wait_time = 2.0 * (2 ** attempt)
                match = re.search(r"try again in (\d+\.?\d*)s", details)
                if match:
                    wait_time = max(wait_time, float(match.group(1)) + 1.0)
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Groq API request failed ({error.code}): {details}") from error
        except URLError as error:
            if attempt < max_retries - 1:
                time.sleep(2.0)
                continue
            raise RuntimeError(f"Could not reach the Groq API: {error.reason}") from error

    if not data:
        raise RuntimeError("Failed to receive data from Groq API.")

    # Step 4: Extract the assistant content
    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as error:
        raise RuntimeError("The Groq API returned an unexpected response format.") from error

    return {"answer": answer, "sources": results}


# -----------------------------------------------------------------------------
# Interactive Terminal REPL Chat Loop
# -----------------------------------------------------------------------------
def interactive_chat_session(limit: int = 4) -> None:
    """Run a continuous interactive terminal chat session with the RAG system."""
    print("=" * 65)
    print("💬 Epilepsy RAG Interactive Chat Session")
    print("Type your question and press Enter. Type 'exit' or 'q' to quit.")
    print("=" * 65)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            print("\n🤖 Assistant is thinking...")
            response = generate_rag_answer(user_input, limit)
            print(f"\n{response['answer']}")
            print("\n📚 Sources:")
            for index, source in enumerate(response["sources"], start=1):
                sec = source["metadata"].get("SubSection") or source["metadata"].get("Section") or "General"
                print(f"  [{index}] Score: {source['score']:.4f} | Section: {sec}")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break
        except Exception as error:
            print(f"\n❌ Error: {error}")


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    """Handle CLI execution: single question or interactive chat mode."""
    # Ensure UTF-8 output formatting on Windows terminals
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask questions using the RAG pipeline.")
    parser.add_argument("question", nargs="?", default=None, help="Optional question to answer immediately")
    parser.add_argument("--limit", type=int, default=4, help="Number of context chunks (default: 4)")
    args = parser.parse_args()

    # If question passed via CLI arguments, answer once and exit
    if args.question:
        response = generate_rag_answer(args.question, args.limit)
        print(response["answer"])
        print("\nSources:")
        for index, source in enumerate(response["sources"], start=1):
            sec = source["metadata"].get("SubSection") or source["metadata"].get("Section") or "General"
            print(f"[{index}] score={source['score']:.4f} | section: {sec}")
    # Otherwise, launch the interactive REPL chat session
    else:
        interactive_chat_session(args.limit)


if __name__ == "__main__":
    main()

"""
main.py - Document Ingestion, Chunking & Remote Embeddings
-----------------------------------------------------------
This script handles the first phase of the RAG pipeline:
1. Reads the raw Markdown file (epilepsy_parsed.md).
2. Splits the document into 412 manageable text chunks using a two-stage strategy.
3. Sends each chunk to the remote Hugging Face API to generate 768-dimensional
   embeddings using the BAAI/bge-base-en-v1.5 model.
4. Saves all text chunks, metadata, and vectors to document_embeddings.json.
"""

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
# Define base paths dynamically so the script runs from any working directory.
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DOCUMENT_PATH = PROJECT_DIR / "epilepsy_parsed.md"
DEFAULT_EMBEDDINGS_PATH = PROJECT_DIR / "document_embeddings.json"

# Remote embedding model details (Runs via Hugging Face Serverless API)
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_API_URL = (
    "https://router.huggingface.co/hf-inference/models/" f"{EMBEDDING_MODEL}"
)
# Batch size: Send 32 chunks per HTTP request to optimize throughput without hitting timeouts.
EMBEDDING_BATCH_SIZE = 32

# Document structure headers to preserve as metadata tags (Section, SubSection, SubSubSection)
HEADERS_TO_SPLIT_ON = [
    ("#", "Section"),
    ("##", "SubSection"),
    ("###", "SubSubSection"),
]


# -----------------------------------------------------------------------------
# Step 1: Read the Source Document
# -----------------------------------------------------------------------------
def read_markdown_file(file_path: str | Path) -> str:
    """Read a UTF-8 Markdown file and return its full text content.

    Args:
        file_path: Path to the .md or .txt file.

    Returns:
        The complete text of the file as a string.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise ValueError("Unsupported file type. Use a Markdown or text file.")

    # utf-8-sig automatically strips any invisible Byte Order Mark (BOM) at the start
    return path.read_text(encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# Step 2: Split the Document into Chunks (Hybrid Strategy)
# -----------------------------------------------------------------------------
def create_chunks(file_path: str | Path):
    """Read a Markdown document and split it into embedding-ready chunks.

    Two-stage splitting process:
    1. MarkdownHeaderTextSplitter: Splits by headers (#, ##, ###) so each chunk knows
       its exact chapter/section hierarchy in metadata.
    2. RecursiveCharacterTextSplitter: Breaks large sections into ~800 character chunks
       with 150-character overlap to avoid cutting sentences mid-thought.

    Returns:
        List of LangChain Document objects containing chunk text and metadata.
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    text = read_markdown_file(file_path)

    # Stage 1: Structural split by headings
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    header_splits = markdown_splitter.split_text(text)

    # Stage 2: Recursive character split to enforce chunk size bounds
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return text_splitter.split_documents(header_splits)


# -----------------------------------------------------------------------------
# Step 3: Secure Environment Key Loader
# -----------------------------------------------------------------------------
def get_env_value(name: str) -> str:
    """Read a secret API key from the system environment or local .env file.

    Args:
        name: Name of the environment variable (e.g., 'HF_TOKEN', 'GROQ_API_KEY').

    Returns:
        The secret key value as a string, or an empty string if not found.
    """
    # Check OS environment first
    token = os.environ.get(name)
    if token:
        return token

    # Fallback to reading the local .env file
    env_file = PROJECT_DIR / ".env"
    if not env_file.is_file():
        return ""

    # Parse key=value pairs manually to avoid external dotenv dependencies
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key == name:
            return value.strip().strip('"').strip("'")

    return ""


def get_hf_token() -> str:
    """Read the Hugging Face API token from .env or environment."""
    return get_env_value("HF_TOKEN")


# -----------------------------------------------------------------------------
# Step 4: Generate Remote Vector Embeddings via Hugging Face API
# -----------------------------------------------------------------------------
def embed_texts(texts: list[str], api_token: str) -> list[list[float]]:
    """Generate normalized 768-dimensional BAAI embeddings through Hugging Face's API.

    Args:
        texts: List of text strings to embed.
        api_token: Hugging Face API bearer token.

    Returns:
        List of 768-dimensional float vectors (one vector per text).
    """
    if not api_token:
        raise ValueError("HF_TOKEN is not set. Please add your Hugging Face token to .env.")

    embeddings: list[list[float]] = []

    # Process in batches of EMBEDDING_BATCH_SIZE (32) to avoid request payload limits
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        
        # normalize: True produces unit-length vectors ideal for Cosine similarity
        # truncate: True ensures text exceeding 512 tokens is truncated safely
        payload = json.dumps(
            {"inputs": batch, "normalize": True, "truncate": True}
        ).encode("utf-8")
        
        request = Request(
            EMBEDDING_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding API request failed ({error.code}): {details}"
            ) from error
        except URLError as error:
            raise RuntimeError(f"Could not reach the embedding API: {error.reason}") from error

        # Verify response integrity
        if not isinstance(result, list) or len(result) != len(batch):
            raise RuntimeError("The embedding API returned an unexpected or partial response.")

        embeddings.extend(result)

    return embeddings


# -----------------------------------------------------------------------------
# Step 5: Save Processed Chunks & Vectors to JSON
# -----------------------------------------------------------------------------
def save_embedding_records(
    chunks: list[Any], embeddings: list[list[float]], output_path: str | Path
) -> None:
    """Save chunk text, metadata, and vectors into a JSON file for database ingestion.

    Args:
        chunks: List of chunk Document objects.
        embeddings: List of matching float vectors.
        output_path: Target JSON filepath.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("The number of chunks and embeddings must match.")

    # Combine chunk content, metadata, and vector into structured records
    records = [
        {
            "id": index,
            "text": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": embedding,
        }
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    Path(output_path).write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    """Perform the full ingestion workflow: read -> chunk -> embed -> save."""
    print("Reading and chunking document...")
    chunks = create_chunks(DEFAULT_DOCUMENT_PATH)
    print(f"Created {len(chunks)} text chunks.")

    print("Generating remote embeddings with BAAI/bge-base-en-v1.5...")
    api_token = get_hf_token()
    embeddings = embed_texts([chunk.page_content for chunk in chunks], api_token)

    print(f"Saving records to {DEFAULT_EMBEDDINGS_PATH.name}...")
    save_embedding_records(chunks, embeddings, DEFAULT_EMBEDDINGS_PATH)
    print(
        f"✓ Successfully created {len(embeddings)} embeddings ({len(embeddings[0])}-dim). "
        f"Saved to {DEFAULT_EMBEDDINGS_PATH.name}."
    )


if __name__ == "__main__":
    main()

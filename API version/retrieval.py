"""
retrieval.py - Semantic Retrieval Engine
-----------------------------------------
This module performs vector similarity search:
1. Takes a user question as input.
2. Prefixes the query with BAAI's recommended instruction prompt.
3. Converts the question into a 768-dimensional vector using Hugging Face's API.
4. Searches the local Qdrant collection using Cosine similarity.
5. Returns the top-k most relevant document chunks along with their metadata and scores.
"""

import argparse
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from main import EMBEDDING_MODEL, embed_texts, get_hf_token
from vector_store import COLLECTION_NAME, QDRANT_PATH


# -----------------------------------------------------------------------------
# Asymmetric Query Prefix
# -----------------------------------------------------------------------------
# BAAI recommends adding this instruction prefix to short queries when searching longer passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# -----------------------------------------------------------------------------
# Core Retrieval Function
# -----------------------------------------------------------------------------
def retrieve(query: str, limit: int = 4) -> list[dict[str, Any]]:
    """Embed a user question with BAAI and retrieve the top-k matching chunks from Qdrant.

    Args:
        query: The user's search question or query string.
        limit: Maximum number of closest chunks to retrieve (default: 4).

    Returns:
        A list of dictionaries, each containing:
        - 'id': Chunk integer ID
        - 'score': Cosine similarity score (e.g. 0.7950)
        - 'text': The actual chunk text passage
        - 'metadata': Section/chapter tags (e.g. {'SubSection': '3.2 Ion-Channel Dysfunction'})
    """
    if not query.strip():
        raise ValueError("The search query must not be empty.")
    if limit < 1:
        raise ValueError("The result limit must be at least 1.")
    if not Path(QDRANT_PATH).is_dir():
        raise FileNotFoundError("Vector database not found. Please run vector_store.py first.")

    # Step 1: Embed the user question using the exact same remote BAAI model
    query_text = f"{QUERY_INSTRUCTION}{query.strip()}"
    query_vector = embed_texts([query_text], get_hf_token())[0]

    # Step 2: Query the local Qdrant collection
    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        # query_points calculates cosine distances and returns the closest vectors
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
    finally:
        client.close()

    # Step 3: Format the returned points into clean dictionary objects
    return [
        {
            "id": point.id,
            "score": point.score,
            "text": point.payload["text"],
            "metadata": point.payload["metadata"],
        }
        for point in response.points
    ]


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    """Allow direct testing of the retrieval engine from the command line."""
    parser = argparse.ArgumentParser(
        description="Retrieve relevant chunks from the local Qdrant vector database."
    )
    parser.add_argument("query", help="Question to search for")
    parser.add_argument("--limit", type=int, default=4, help="Maximum results to return (default: 4)")
    args = parser.parse_args()

    results = retrieve(args.query, args.limit)
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Top-{len(results)} Chunks Retrieved for: '{args.query}'\n")
    
    for index, result in enumerate(results, start=1):
        sec = result['metadata'].get('SubSection') or result['metadata'].get('Section') or 'General'
        print(f"[{index}] Score: {result['score']:.4f} | Section: {sec}")
        print(f"    {result['text'][:200]}...\n")


if __name__ == "__main__":
    main()

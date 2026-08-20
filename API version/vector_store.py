"""
vector_store.py - Local Qdrant Vector Database Ingestion
---------------------------------------------------------
This script takes the precomputed embeddings and text chunks from
document_embeddings.json and stores them in a local, embedded Qdrant database.

Key points:
- Uses embedded Qdrant (stored directly on disk in the qdrant_data/ folder).
- No Docker or separate database service required.
- Uses Cosine similarity metric to match normalized BAAI query vectors.
"""

import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = PROJECT_DIR / "document_embeddings.json"
QDRANT_PATH = PROJECT_DIR / "qdrant_data"
COLLECTION_NAME = "epilepsy_chunks"


# -----------------------------------------------------------------------------
# Step 1: Load Precomputed Embeddings
# -----------------------------------------------------------------------------
def load_embedding_records(file_path: str | Path) -> list[dict[str, Any]]:
    """Load the chunk text, metadata, and 768-dim BAAI vectors from JSON.

    Args:
        file_path: Path to the document_embeddings.json file.

    Returns:
        List of dictionary records, each containing 'id', 'text', 'metadata', and 'embedding'.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Embedding file not found: {path}. Please run main.py first to generate embeddings."
        )

    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("The embedding file contains no records.")
    return records


# -----------------------------------------------------------------------------
# Step 2: Initialize & Ingest Points into Qdrant
# -----------------------------------------------------------------------------
def build_vector_database(records: list[dict[str, Any]]) -> int:
    """Create a persistent local Qdrant collection and insert all vector points.

    Args:
        records: List of chunk dictionaries containing vectors and payloads.

    Returns:
        The total count of indexed points in the collection.
    """
    vector_size = len(records[0]["embedding"])
    if vector_size == 0:
        raise ValueError("Embeddings must not be empty.")

    # Initialize local Qdrant client persisting to disk at QDRANT_PATH
    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        # Recreate collection cleanly if it already exists for deterministic reruns
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        # Cosine distance compares the angle between normalized vectors (1.0 = identical)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        # Wrap each chunk into a Qdrant PointStruct:
        # - id: Integer identifier (0..411)
        # - vector: 768-dimensional float list
        # - payload: Text and chapter metadata to return during search
        points = [
            PointStruct(
                id=record["id"],
                vector=record["embedding"],
                payload={"text": record["text"], "metadata": record["metadata"]},
            )
            for record in records
        ]

        # Upsert points into Qdrant and wait for indexing to finish
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        return client.count(collection_name=COLLECTION_NAME, exact=True).count
    finally:
        # Ensure client connection and lock files are cleanly closed
        client.close()


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    """Load JSON embeddings and build the persistent local Qdrant vector database."""
    print(f"Loading embeddings from {EMBEDDINGS_PATH.name}...")
    records = load_embedding_records(EMBEDDINGS_PATH)
    
    print(f"Ingesting {len(records)} points into local Qdrant at {QDRANT_PATH.name}/...")
    point_count = build_vector_database(records)
    print(
        f"✓ Successfully created Qdrant collection '{COLLECTION_NAME}' with {point_count} vectors "
        f"at {QDRANT_PATH.name}."
    )


if __name__ == "__main__":
    main()

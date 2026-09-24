"""A thin interface over Qdrant.

Nothing outside this file imports qdrant_client. Retrieval, evaluation and
the app all talk to VectorStore's four methods, so swapping Qdrant for
another store means rewriting this file and nothing else.

Two ways to connect, chosen by environment:
  QDRANT_URL set    -> a Qdrant server (the Docker container, locally)
  QDRANT_URL unset  -> Qdrant's embedded mode, reading from data/qdrant_local/
                       No server at all - this is what the deployed app uses,
                       since Streamlit Cloud can't reach a container on a laptop.

Each chunking strategy gets its own collection (fastapi_naive,
fastapi_structured) so the two indexes never mix.
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from ingest.chunk import Chunk

load_dotenv()

LOCAL_PATH = "data/qdrant_local"


def collection_name(strategy):
    return f"fastapi_{strategy}"


class VectorStore:
    def __init__(self, url=None, path=None):
        url = url or os.getenv("QDRANT_URL")
        self.client = QdrantClient(url=url) if url else QdrantClient(path=path or LOCAL_PATH)

    def recreate(self, collection, dimensions):
        """Drop and recreate a collection, so re-indexing starts clean."""
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
        )

    def upsert(self, collection, chunks, vectors, batch_size=256):
        """Store chunks with their vectors. The whole Chunk rides along as payload."""
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.client.upsert(
                collection,
                points=[
                    models.PointStruct(
                        # Qdrant ids must be ints or UUIDs; our 16-hex id fits in a 64-bit int.
                        id=int(chunk.id, 16),
                        vector=vector.tolist(),
                        payload=chunk.__dict__,
                    )
                    for chunk, vector in zip(batch, vectors[start : start + batch_size])
                ],
            )

    def search(self, collection, vector, k):
        """Return the k nearest chunks as [(Chunk, cosine_similarity)], best first."""
        hits = self.client.query_points(collection, query=vector.tolist(), limit=k, with_payload=True).points
        return [(Chunk(**hit.payload), hit.score) for hit in hits]

    def count(self, collection):
        return self.client.count(collection).count

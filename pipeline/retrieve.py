"""First-stage retrieval: nearest neighbours in embedding space.

Fast and broad. A bi-encoder embeds the question and each chunk separately,
so chunk vectors can be computed once, ahead of time, and search is a single
vector lookup. The cost of that speed: question and chunk never "see" each
other, so fine distinctions between near-miss chunks get lost. That's what
the reranker is for.
"""

from dataclasses import dataclass

from ingest.chunk import Chunk
from pipeline.embed import embed_query
from pipeline.vector_store import VectorStore, collection_name


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, or cross-encoder score after reranking


_store = None


def store():
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def retrieve(question, strategy, k):
    """Return the k chunks nearest to the question, best first."""
    results = store().search(collection_name(strategy), embed_query(question), k)
    return [Hit(chunk, score) for chunk, score in results]

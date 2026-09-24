"""Second-stage reranking with a cross-encoder.

A cross-encoder reads the question and one chunk TOGETHER, in a single pass,
and outputs a relevance score. Seeing both at once lets it judge whether the
chunk actually answers the question, not just whether it's about the same
topic - much more accurate than comparing two independently computed vectors.

It's also far too slow to run over 3,000 chunks per query, since nothing can
be precomputed. Hence the two-stage design: the bi-encoder cheaply narrows
thousands of chunks to ~20 candidates, and the cross-encoder carefully
reorders those 20.

The reranker can only reorder what first-stage retrieval found. It cannot
raise recall beyond the candidate pool; it changes which chunks make the
final cut and in what order.

MODEL: ms-marco-MiniLM-L-6-v2, trained on MS MARCO passage ranking, run via
fastembed (ONNX).
"""

from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from pipeline.retrieve import Hit

MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model():
    return TextCrossEncoder(MODEL_NAME)


def rerank(question, hits, top_n):
    """Score each candidate against the question and return the best top_n."""
    scores = list(_model().rerank(question, [h.chunk.text for h in hits]))
    reranked = sorted(
        (Hit(h.chunk, float(s)) for h, s in zip(hits, scores)),
        key=lambda h: h.score,
        reverse=True,
    )
    return reranked[:top_n]

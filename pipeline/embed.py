"""Turn text into vectors with a local embedding model.

An embedding maps text to a point in a 384-dimensional space such that texts
with similar meaning land near each other. "Near" is measured by cosine
similarity - the angle between two vectors, ignoring their length. Retrieval
is then just: embed the question, find the chunks whose vectors are closest.

MODEL: BAAI/bge-small-en-v1.5, run through fastembed (ONNX Runtime).
Local, so embedding the corpus and every eval question costs nothing, and
no PyTorch dependency, which keeps the deployed app within Streamlit Cloud's
memory limit.

BGE models are ASYMMETRIC: they were trained with queries carrying an
instruction prefix and passages without one. fastembed's query_embed() does
not add the prefix (verified: identical vectors to plain embed), so we do it
explicitly. Forgetting it degrades retrieval silently.
"""

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    # Loading the model takes a second or two; do it once per process.
    return TextEmbedding(MODEL_NAME)


def embed_passages(texts, batch_size=64):
    """Embed corpus chunks. Returns an (n, 384) float32 array."""
    return np.array(list(_model().embed(texts, batch_size=batch_size)), dtype=np.float32)


def embed_query(text):
    """Embed a question, with the instruction prefix BGE was trained on."""
    return next(_model().embed([QUERY_PREFIX + text])).astype(np.float32)

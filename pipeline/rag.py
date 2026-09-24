"""The pipeline's public interface: search() and answer().

These two functions are the only entry points the app and the evaluation
harness use. Evaluation treats the pipeline as a black box through exactly
the calls a real user's request goes through, so every measured number
describes the system as deployed.
"""

import time
from dataclasses import dataclass, field

from pipeline.rerank import rerank
from pipeline.retrieve import retrieve


@dataclass(frozen=True)
class Config:
    strategy: str = "structured"  # "naive" or "structured"
    use_rerank: bool = True
    k_final: int = 10  # length of the ranked list retrieval metrics are computed on
    k_context: int = 5  # chunks handed to the generator
    k_candidates: int = 20  # first-stage pool the reranker chooses from

    @property
    def name(self):
        return f"{self.strategy}+{'rerank' if self.use_rerank else 'vector'}"


@dataclass
class SearchResult:
    hits: list  # final ranked list, length k_final
    candidates: list  # first-stage pool before reranking (== hits without rerank)
    timings: dict = field(default_factory=dict)  # seconds per stage


def search(question, config):
    timings = {}

    started = time.perf_counter()
    pool_size = config.k_candidates if config.use_rerank else config.k_final
    candidates = retrieve(question, config.strategy, pool_size)
    timings["retrieve"] = time.perf_counter() - started

    if config.use_rerank:
        started = time.perf_counter()
        hits = rerank(question, candidates, config.k_final)
        timings["rerank"] = time.perf_counter() - started
    else:
        hits = candidates

    return SearchResult(hits=hits, candidates=candidates, timings=timings)


def answer(question, config):
    """Retrieve, then generate a cited answer from the top k_context chunks."""
    from pipeline.generate import generate  # imported lazily: needs an API key

    result = search(question, config)
    started = time.perf_counter()
    generated = generate(question, result.hits[: config.k_context])
    result.timings["generate"] = time.perf_counter() - started
    return result, generated

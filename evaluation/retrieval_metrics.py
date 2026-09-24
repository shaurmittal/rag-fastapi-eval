"""Retrieval metrics. Pure arithmetic against ground truth - no LLM involved,
so these numbers are deterministic, free, and reproducible.

RELEVANCE RULE
    A retrieved chunk is relevant if its source_file is one of the question's
    expected_sources. File-level matching is what lets one set of ground-truth
    labels score both chunking strategies: chunk ids differ between strategies,
    file paths do not. It is lenient - any chunk from a long relevant file
    counts - but equally lenient to every configuration being compared.

Because a relevant file usually yields several chunks, precision@k can reach
1.0 here. (With a single gold *chunk* per question, precision@5 would cap at
0.2, which is why chunk-level precision is rarely useful for RAG.)
"""


def relevant_flags(hits, expected_sources):
    """One boolean per retrieved chunk, in rank order."""
    expected = set(expected_sources)
    return [hit.chunk.source_file in expected for hit in hits]


def hit_rate(flags, k):
    """1.0 if anything relevant made the top k, else 0.0."""
    return float(any(flags[:k]))


def precision(flags, k):
    """Fraction of the top k that is relevant."""
    top = flags[:k]
    return sum(top) / len(top) if top else 0.0


def recall(hits, answer_facets, k):
    """Fraction of the answer's facets covered by the top k.

    A facet is one part of the answer, with every file that fully covers it
    listed as an alternative. Question #3 ("return 202 for a long-running
    job") has two facets: setting the status code, and running background
    work. Retrieving three files about status codes still covers only one.

    Counting facets rather than files matters once alternative sources exist:
    with a flat file list, discovering a second valid source for a question
    would enlarge the denominator and LOWER recall for a system that finds
    either one. For single-facet questions recall@k equals hit@k.
    """
    retrieved = {hit.chunk.source_file for hit in hits[:k]}
    covered = sum(1 for facet in answer_facets if retrieved & set(facet))
    return covered / len(answer_facets)


def reciprocal_rank(flags, k):
    """1 / rank of the first relevant chunk within the top k; 0 if none.

    Rank matters because models under-use information in the middle of a long
    context ("lost in the middle"), so a relevant chunk at rank 1 is worth more
    than the same chunk at rank 5 - a difference recall@5 cannot see.
    """
    for rank, is_relevant in enumerate(flags[:k], start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def score_retrieval(result, example):
    """All retrieval metrics for one question's SearchResult."""
    flags = relevant_flags(result.hits, example["expected_sources"])
    pool_flags = relevant_flags(result.candidates, example["expected_sources"])
    return {
        "hit@1": hit_rate(flags, 1),
        "hit@5": hit_rate(flags, 5),
        "recall@5": recall(result.hits, example["answer_facets"], 5),
        "recall@10": recall(result.hits, example["answer_facets"], 10),
        "precision@5": precision(flags, 5),
        "mrr@10": reciprocal_rank(flags, 10),
        # Whether the first stage found anything relevant at all. For reranked
        # configs this is the ceiling the reranker works under.
        "pool_hit": hit_rate(pool_flags, len(pool_flags)),
    }

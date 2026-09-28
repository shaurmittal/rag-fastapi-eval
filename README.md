# RAG over FastAPI — with a Rigorous Eval Harness

A question-answering system over the [FastAPI](https://github.com/fastapi/fastapi) codebase and
docs that answers with citations — built around an evaluation harness that measures retrieval and
generation separately, on 51 real questions mined from FastAPI's GitHub issues.

> **Status:** retrieval evaluation, generation evaluation and judge validation complete.

## Findings

1. **Measurement error outweighed every pipeline change.** A pooled review of the retrieved
   results found 28 correct answers the first-pass labels had missed. Completing the labels raised
   every configuration's hit@5 by about 0.10 — more than any chunking or reranking difference.
2. **No single pipeline factor is statistically supported at n=51.** Every paired 95% bootstrap
   interval includes zero. The comparisons are reported as hypotheses, not wins.
3. **Reranking appears to interact with chunking.** It helped structured chunks (hit@5 +0.059,
   3 wins / 0 losses / 48 ties; MRR +0.043) but not fixed-size windows (MRR −0.017). A plausible
   mechanism: the cross-encoder reads question and chunk together, and coherent, heading-prefixed
   sections give it more to judge than windows cut mid-sentence.
4. **Reranking costs ~15× retrieval latency** (0.97 s vs 0.06 s p50) with no demonstrable
   quality gain on this set.
5. **Retrieval is the ceiling on answer quality.** When an expected source reached the top 5,
   81% of answers were judged correct; when it didn't, 31% were. No generation-side metric
   separated the four configurations: every paired interval includes zero.
6. **Faithfulness measures "stated in the context", not "true".** About a third of extracted
   claims were judged unsupported. Reading them, most aren't inventions: they're claims the
   context *implies* (e.g. behaviour readable from a `routing.py` signature) but doesn't state,
   which the verifier is instructed to reject. That's the rule working as specified, not a
   miscalibrated judge: independent raters applying the same rule agreed with the judge on
   95% of sampled claims ([validation](#judge-validation)).

## Retrieval results

51 questions × 4 configurations. A retrieved chunk is relevant if its source file is one of the
question's expected sources ([why file-level](#relevance-and-ground-truth)).

| configuration | hit@1 | hit@5 | precision@5 | MRR@10 |
|---|---|---|---|---|
| naive + vector | 0.490 | 0.765 | 0.408 | 0.603 |
| naive + rerank | 0.471 | 0.765 | 0.365 | 0.586 |
| structured + vector | 0.471 | 0.706 | 0.443 | 0.577 |
| **structured + rerank** | **0.510** | **0.765** | **0.443** | **0.620** |

Paired comparisons, one factor at a time (Δ = first minus second, 95% paired bootstrap CI):

| comparison | metric | Δ | 95% CI | win / loss / tie |
|---|---|---|---|---|
| rerank vs vector, structured chunks | hit@5 | +0.059 | [+0.000, +0.137] | 3 / 0 / 48 |
| rerank vs vector, structured chunks | MRR@10 | +0.043 | [−0.032, +0.122] | 14 / 9 / 28 |
| rerank vs vector, naive chunks | MRR@10 | −0.017 | [−0.107, +0.070] | 12 / 11 / 28 |
| structured vs naive, with rerank | precision@5 | +0.078 | [−0.000, +0.161] | 17 / 11 / 23 |
| structured vs naive, no rerank | hit@5 | −0.059 | [−0.157, +0.039] | 2 / 5 / 44 |

Full tables, including docs/code slices and latency: [`results/retrieval.md`](results/retrieval.md).
The same run scored against first-pass labels: [`results/retrieval_first_pass_labels.md`](results/retrieval_first_pass_labels.md).

## Generation results

The same 51 questions × 4 configurations, answered by Claude Opus 5 from the top 5 chunks and
judged by Claude Sonnet 5 ([how](#generation-evaluation)).

| configuration | faithfulness | relevance | correctness | has citation | cited-source precision |
|---|---|---|---|---|---|
| naive + vector | 0.609 | 0.961 | 0.725 | 1.000 | 0.563 |
| naive + rerank | 0.628 | 0.922 | 0.647 | 1.000 | 0.526 |
| structured + vector | 0.677 | 0.882 | 0.686 | 1.000 | 0.541 |
| structured + rerank | 0.641 | 0.882 | 0.686 | 0.980 | 0.594 |

Correctness split by whether retrieval put an expected source in the top 5:

| configuration | hit@5 = 1 | hit@5 = 0 |
|---|---|---|
| naive + vector | 0.79 (n=39) | 0.50 (n=12) |
| naive + rerank | 0.77 (n=39) | 0.25 (n=12) |
| structured + vector | 0.83 (n=36) | 0.33 (n=15) |
| structured + rerank | 0.85 (n=39) | 0.17 (n=12) |

The closest call is chunking without reranking on faithfulness (structured − naive +0.068,
95% CI [−0.004, +0.140], 30 wins / 20 losses): suggestive, not supported. Relevance is near
ceiling for every configuration — Opus answers the question asked — so it doesn't discriminate
between them. Full tables with paired comparisons and the docs/code slices:
[`results/generation.md`](results/generation.md).

**Cost.** Generation and judging ran through the Message Batches API at half price: $9.00 for
all 204 answers ($0.024 generation + $0.020 judging per answer; median 9 claims judged each).
Batch requests have no per-request latency; a live smoke run measured 10–15 s p50 to generate
an answer.

## Architecture

```mermaid
flowchart LR
    Q[Question] --> E[Embed<br/>bge-small] --> V[(Qdrant<br/>one collection<br/>per strategy)]
    V -->|top 20| R[Cross-encoder<br/>rerank]
    R -->|top 5| G[Claude<br/>native citations]
    G --> A[Cited answer]
    V -. top 10 .-> M1[Retrieval metrics<br/>no LLM]
    R -. top 10 .-> M1
    A -.-> M2[Generation metrics<br/>LLM judge]
```

`pipeline.rag.search()` and `pipeline.rag.answer()` are the only entry points. The Streamlit app
and the evaluation harness both call them, so every number describes the path users actually hit.
`evaluation/` imports from `pipeline/`, never the reverse.

| Stage | Choice | Why |
|---|---|---|
| Corpus | `fastapi/fastapi` @ `v0.115.0` (`40e33e4`), English docs + `fastapi/` source | Ground truth is only valid against a frozen corpus |
| Chunking | fixed-size (baseline) vs structure-aware | The ablation's first factor |
| Embeddings | `bge-small-en-v1.5` via fastembed (ONNX) | Local, so eval sweeps cost nothing; no PyTorch in the deployed app |
| Vector store | Qdrant, behind a four-method interface | Docker server locally, in-memory from a portable export when deployed |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder | The ablation's second factor |
| Generation | `claude-opus-5` with native citations | Each chunk is a citable document; answers carry the exact passages used |

## The evaluation set

**Questions are real.** 613 closed FastAPI issues labelled `answered` + `question` were fetched
with their 3,915 comments and frozen into [`data/raw_issues.jsonl`](data/raw_issues.jsonl).
Candidates were ranked by heuristics — template noise stripped; the answer comment identified by
maintainer endorsement ("Thanks for the help here @user!") rather than position, since the first
reply is usually "+1" or a clarifying question — then reviewed by hand.

**Kept only if groundable.** A question survived only if its answer can be pointed at in a
specific file of the pinned corpus. About 20% of reviewed candidates survived: bug reports,
feature debates, environment problems and "fixed in 0.x, please upgrade" answers were dropped.

**Questions from GitHub, answers from the corpus.** Several threads predate the feature that now
answers them (`include_in_schema`, form models, return-type response models). The question is kept
verbatim; the expected answer is written against v0.115.0.

**Ground truth located independently.** Expected sources were found by searching the corpus for
the APIs each answer names — never with the system's own retriever, which would let the system
under test choose its own answers.

**The split is 45 docs-answerable / 6 code-answerable**, reflecting what users actually ask.
The code slice is reported but is too small to draw conclusions from.

See [`mine_eval_set/curated.py`](mine_eval_set/curated.py) for every decision.

## Relevance and ground truth

- **File-level relevance.** Chunk ids differ between chunking strategies; file paths don't. One
  set of labels scores both strategies. It's lenient (any chunk of a relevant file counts), but
  equally lenient to every configuration.
- **Answer facets.** Each question's ground truth is a list of facets, each listing every file
  that fully covers it; recall@k is the fraction of facets covered. With a flat list, discovering a
  second valid source would lower recall for a system that finds either one.
- **Pooled relevance review** (the TREC method). Every configuration's top 5 was pooled and each
  unlabelled file judged on its merits: 28 of 272 pairs qualified. Most were source files — at
  v0.115.0 FastAPI's code carries `Doc()` annotations that mirror the documentation.
  `--original-labels` reproduces first-pass scoring for comparison.

## Diagnosing a failure: the reranker "regression" that wasn't

On issue #1708 ("hide a parameter from the OpenAPI docs"), reranking dropped the labelled answer
from rank 1 to rank 7 — an apparent regression. The promoted chunks came from
`fastapi/param_functions.py`, whose `Query()` and `Header()` definitions document
`include_in_schema` directly. The reranker had found a correct answer the labels didn't list.
That one case prompted the pooled review — and the review turned out to be the largest single
effect in the evaluation.

## Generation evaluation

Implemented in [`evaluation/generation_metrics.py`](evaluation/generation_metrics.py):

| Metric | How |
|---|---|
| Faithfulness | The answer is decomposed into atomic claims; each is checked against the retrieved context, and the judge must quote supporting evidence before its verdict |
| Relevance | Does the answer address what was asked (binary) |
| Correctness | Is its main point consistent with the reference answer (binary) |
| Has citation | Whether the answer cites any retrieved passage (no LLM) |
| Cited-source precision | Share of citations pointing into an expected source (no LLM) |

Judge design choices against known LLM-as-judge failure modes: binary verdicts only (1–5 scales
cluster on 3–4), per-claim scoring (length can't inflate it), evidence before verdict, structured
outputs, and a versioned prompt. The judge is Claude Sonnet 5, a different model from the
generator, which reduces self-preference bias. It doesn't accept a temperature setting, so it
can't be pinned to greedy decoding; [`evaluation/judge_validation.py`](evaluation/judge_validation.py)
measures that directly with a test-retest check, and measures agreement with blind human labels
using Cohen's kappa (which scores an always-"yes" judge at 0, however high its raw agreement).

Two judge revisions were made after reading smoke-test items, before the full run, and applied to
every configuration: claim extraction originally pulled ~19 boilerplate claims per answer
("FastAPI is imported from fastapi"), which measured boilerplate coverage rather than
hallucination; and block-level citation coverage sat near 0.5 by construction (native citations
attach to quoted passages, and connecting prose is emitted as separate uncited blocks), so it was
replaced with `has_citation`.

## Judge validation

A fixed random sample of 60 claim verdicts and 30 correctness verdicts from the full run was
re-labelled blind — raters never saw the judge's verdict.

| comparison | claims: agreement / κ | correctness: agreement / κ |
|---|---|---|
| judge vs human (random subset, n = 12 / 8) | 0.75 / 0.25 | 1.00 / 1.00 |
| judge vs independent LLM rater (n = 60 / 30) | 0.95 / 0.89 | 0.97 / 0.91 |
| human vs LLM rater (n = 12 / 8) | 0.83 / 0.56 | 1.00 / 1.00 |
| judge vs itself, re-run (n = 60 / 30) | 0.95 / 0.89 | 1.00 / 1.00 |

- **Correctness verdicts are reliable**: every rater agreed with the judge on all but one item,
  and the judge reproduced every verdict on a re-run.
- **Claim verdicts are stable** (κ 0.89 test-retest) and match an independent rater, but the
  human subset disagreed on 3 of 12 claims — 2 where the judge accepted a claim the human
  rejected. Twelve items can't pin κ down (one flipped label moves it by ~0.1–0.2), so the
  human-agreement figure for claims is inconclusive rather than negative. One human label was
  corrected on re-review after unblinding (its supporting sentence is quoted near-verbatim in
  the context); the first-pass figures were 0.67 / κ 0.14.
- **The LLM rater is Claude Opus 5.5**: a different model from the judge, but the same vendor
  and lineage, so its agreement is weaker evidence than human agreement. Its labels and the
  human labels are both in [`data/judge_validation.jsonl`](data/judge_validation.jsonl);
  full output in [`results/judge_validation.md`](results/judge_validation.md).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync
docker compose up -d
uv run python -m ingest.clone_repo              # pinned corpus
uv run python -m ingest.build_chunks            # both chunking strategies
uv run python -m pipeline.index                 # embed + index (~10 min on CPU)
uv run python -m evaluation.run_ablations       # retrieval metrics, no API key needed

cp .env.example .env                            # add ANTHROPIC_API_KEY for generation
uv run python -m evaluation.run_ablations --generate --limit 5   # live smoke test
uv run python -m evaluation.batch_run                            # full run, Batch API (half price)
uv run streamlit run app.py
```

To rebuild the evaluation set: `mine_eval_set.scrape_github_issues` (needs `GITHUB_TOKEN`), then
`mine_eval_set.build_eval_set` and `mine_eval_set.curated`.

## Layout

```
data/raw_issues.jsonl        frozen snapshot of the mined issues (committed)
data/golden_eval_set.jsonl   the 51-question evaluation set (committed)
data/index/                  portable vector index used by the deployed app (committed)
ingest/                      corpus loading (include resolution, tab collapsing) + both chunkers
mine_eval_set/               issue scraper, candidate ranking, curated ground truth
pipeline/                    embed → vector store → retrieve → rerank → generate; rag.py is the API
evaluation/                  retrieval metrics, LLM judge, statistics, ablation runner, judge validation
results/                     committed result tables
app.py                       Streamlit demo
```

## Limitations

- **n = 51.** Differences of a few points are within noise; the paired bootstrap intervals say so.
- **File-level relevance** is lenient toward long files with many chunks.
- **The judge is an LLM, and human validation is small.** The judge is a different model from
  the generator but shares its vendor; the blind human check covers 20 items, and the larger
  second-rater check (90 items) is itself an LLM.
- **Relevance saturates** near 1.0 and doesn't discriminate between configurations.
- **The code-answerable slice (n=6)** is too small for conclusions.
- **One reranker** was tested; a code-aware reranker might behave differently on source files.

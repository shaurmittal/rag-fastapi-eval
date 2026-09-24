# RAG over FastAPI — with a Rigorous Eval Harness

Ask questions about the [FastAPI](https://github.com/fastapi/fastapi) framework and
get answers with citations — plus an evaluation harness that measures how often those
answers are actually correct, and how often they are made up.

> **Status:** in active development. The ablation results table will be published here
> once the evaluation runs are complete.

## Why this project

Most RAG demos stop at "it returned something plausible." The interesting question is
*how good is it, and why is one configuration better than another.* This repo is built
around that question:

- **Retrieval and generation are measured separately.** Retrieval quality is a hard
  ceiling on generation quality — if the answer isn't in the retrieved context, no
  prompt fixes it. Measuring the stages separately localizes failures instead of just
  scoring them.
- **The eval set is authentic.** Questions are mined from real closed FastAPI GitHub
  issues, where real users asked real questions and maintainers gave real answers
  grounded in real code and docs. That reflects a genuine query distribution rather
  than questions invented by the author.
- **Hallucination is measured, not asserted.** Generated answers are decomposed into
  atomic claims, each checked independently against the retrieved context.
- **The judge is itself validated.** The LLM-as-judge is scored against hand-labelled
  human judgments and reported with Cohen's kappa.

## Architecture

```
Ingestion → Chunking → Embedding → Vector Store → Retrieval → (Rerank) → Generation (cited)
                                                                              ↑
                                                                  Eval harness measures every stage
```

## Corpus (pinned)

Evaluation numbers are only meaningful against a frozen corpus, so the source is pinned:

| | |
|---|---|
| Repository | `fastapi/fastapi` |
| Tag | `v0.115.0` |
| Commit | `40e33e4` |
| Docs indexed | `docs/en/docs/` — English only (the repo ships 26 translations; indexing all of them would flood retrieval with near-duplicate chunks) |
| Source indexed | `fastapi/` |

## Stack

| Component | Choice | Notes |
|---|---|---|
| Vector store | Qdrant (Docker) | Behind a thin interface, so it is swappable |
| Embeddings | `bge-small-en-v1.5` | Local — no API cost per eval run, so the full sweep can be re-run freely |
| Generation + judge | Claude (`claude-opus-5`) | |
| Reranker | Cross-encoder (`ms-marco-MiniLM`) | Second-stage reordering |
| Eval | Custom metrics | Implemented directly rather than via RAGAS — see notes |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
# 1. Dependencies (Python 3.12 is pinned via .python-version)
uv sync

# 2. Start the vector store
docker compose up -d

# 3. Fetch the pinned corpus
git clone --depth 1 --branch 0.115.0 \
    https://github.com/fastapi/fastapi.git data/raw/fastapi

# 4. Configure credentials
cp .env.example .env    # then fill in GITHUB_TOKEN and ANTHROPIC_API_KEY
```

## Layout

```
data/raw/            # pinned corpus (gitignored — regenerable from the clone command above)
data/golden_eval_set.jsonl   # hand-curated ground truth (committed — not regenerable)
ingest/              # chunking strategies: naive fixed-size vs structure-aware
mine_eval_set/       # builds the golden set from GitHub issues
pipeline/            # embed → vector store → retrieve → rerank → generate
evaluation/          # retrieval metrics, generation metrics, ablation runner
app.py               # Streamlit demo
```

`evaluation/` imports from `pipeline/`, never the reverse — measurement must not be able
to influence what it measures.

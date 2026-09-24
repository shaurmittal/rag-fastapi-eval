"""Run every pipeline configuration over the golden set and compare them.

The design is a 2x2 factorial: {naive, structured} chunking x {vector only,
cross-encoder rerank}. One factor changes at a time, everything else is held
fixed, so each difference can be attributed to that factor.

Retrieval metrics need no API key. Generation metrics (faithfulness,
relevance, correctness) call Claude and run only with --generate.

OUTPUT (data/runs/<timestamp>/, gitignored)
    per_question.jsonl   every metric for every (config, question)
    summary.md           the comparison tables

RUN
    uv run python -m evaluation.run_ablations               # retrieval only
    uv run python -m evaluation.run_ablations --generate    # full evaluation
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from evaluation.retrieval_metrics import score_retrieval
from evaluation.stats import bootstrap_ci, mean, paired_record, percentile
from pipeline.rag import Config, search

GOLDEN_PATH = "data/golden_eval_set.jsonl"
RUNS_DIR = Path("data/runs")

CONFIGS = [
    Config(strategy="naive", use_rerank=False),  # baseline
    Config(strategy="naive", use_rerank=True),
    Config(strategy="structured", use_rerank=False),
    Config(strategy="structured", use_rerank=True),
]

RETRIEVAL_METRICS = ["hit@1", "hit@5", "recall@5", "recall@10", "precision@5", "mrr@10"]
GENERATION_METRICS = ["faithfulness", "relevance", "correctness", "has_citation", "cited_source_precision"]


def load_golden(original_labels=False):
    """Load the golden set; optionally strip the pooled-review additions.

    --original-labels reproduces scoring against the first-pass labels, so the
    effect of the pooled review can be measured on identical retrieval output.
    """
    examples = [json.loads(line) for line in open(GOLDEN_PATH)]
    if original_labels:
        for ex in examples:
            added = set(ex["pooled_additions"])
            ex["answer_facets"] = [[s for s in facet if s not in added] for facet in ex["answer_facets"]]
            ex["expected_sources"] = [s for s in ex["expected_sources"] if s not in added]
    return examples


def evaluate(config, example, generate_answers):
    row = {"config": config.name, "id": example["id"], "question_type": example["question_type"]}

    if generate_answers:
        from evaluation.generation_metrics import score_generation
        from pipeline.rag import answer

        result, generated = answer(example["question"], config)
        row.update(score_generation(example, result.hits[: config.k_context], generated))
        row["answer"] = generated.text
        row["answer_model"] = generated.model
        row["citations"] = [c.__dict__ for c in generated.citations]
        # The exact context the answer was generated from, for audit and human labelling.
        row["context"] = [{"source_file": h.chunk.source_file, "text": h.chunk.text}
                          for h in result.hits[: config.k_context]]
        row["generation_cost_usd"] = generated.cost_usd
        row["cost_usd"] = generated.cost_usd + row["judge_cost_usd"]
    else:
        result = search(example["question"], config)

    row.update(score_retrieval(result, example))
    row["retrieved_sources"] = [h.chunk.source_file for h in result.hits]
    row.update({f"latency_{stage}": secs for stage, secs in result.timings.items()})
    return row


def rescore(row, example):
    """Re-judge a stored answer with the current judge, without regenerating it.

    Retrieval metrics, latencies, and generation cost are kept from the
    original row; only the generation metrics and judge cost are recomputed.
    """
    from evaluation.generation_metrics import score_generation
    from ingest.chunk import Chunk
    from pipeline.generate import Answer, Citation
    from pipeline.retrieve import Hit

    hits = [Hit(Chunk(id="", text=c["text"], source_file=c["source_file"], section="",
                      chunk_type="", strategy=""), 0.0) for c in row["context"]]
    stored = Answer(text=row["answer"], citations=[Citation(**c) for c in row["citations"]],
                    stop_reason="refusal" if row.get("refused") else "end_turn")
    new = dict(row)
    new.update(score_generation(example, hits, stored))
    new["cost_usd"] = row["generation_cost_usd"] + new["judge_cost_usd"]
    return new


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


LABELS = "pooled (first-pass + pooled-review additions)"


def summarize(rows, metrics):
    by_config = {c.name: [r for r in rows if r["config"] == c.name] for c in CONFIGS}
    n = len(next(iter(by_config.values())))
    out = [f"# Ablation results\n\n{n} questions per configuration. Labels: {LABELS}.\n"]

    # 1. Headline table: mean of each metric per config.
    out.append("## Mean scores\n")
    out.append(table(
        ["config"] + metrics,
        [[name] + [f"{mean([r.get(m) for r in rs]):.3f}" for m in metrics] for name, rs in by_config.items()],
    ))

    # 2. The same, sliced by where the answer lives.
    for qtype in ("docs", "code"):
        subset = {name: [r for r in rs if r["question_type"] == qtype] for name, rs in by_config.items()}
        size = len(next(iter(subset.values())))
        out.append(f"\n## {qtype}-answerable questions only (n={size})\n")
        out.append(table(
            ["config"] + metrics,
            [[name] + [f"{mean([r.get(m) for r in rs]):.3f}" for m in metrics] for name, rs in subset.items()],
        ))

    # 3. Paired comparisons, one factor at a time.
    out.append("\n## Paired comparisons (95% bootstrap CI of the mean difference)\n")
    out.append("A difference is only supported by the data if its interval excludes 0.\n")
    pairs = [
        ("structured+vector", "naive+vector", "chunking, no rerank"),
        ("structured+rerank", "naive+rerank", "chunking, with rerank"),
        ("naive+rerank", "naive+vector", "reranking, naive chunks"),
        ("structured+rerank", "structured+vector", "reranking, structured chunks"),
    ]
    rows_out = []
    for a, b, label in pairs:
        for m in metrics:
            # Compare only questions where both configs have a defined score
            # (faithfulness is undefined for answers that make no claims).
            both = [(ra[m], rb[m]) for ra, rb in zip(by_config[a], by_config[b])
                    if ra.get(m) is not None and rb.get(m) is not None]
            if not both:
                continue
            xa, xb = [x for x, _ in both], [y for _, y in both]
            w, l, t = paired_record(xa, xb)
            lo, hi = bootstrap_ci(xa, xb)
            verdict = "better" if lo > 0 else "worse" if hi < 0 else "no clear difference"
            rows_out.append([label, m, f"{mean(xa) - mean(xb):+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", f"{w}/{l}/{t}", verdict])
    out.append(table(["factor", "metric", "Δ mean", "95% CI", "win/loss/tie", "verdict"], rows_out))

    # 4. Latency and cost.
    out.append("\n## Latency per query (seconds)\n")
    stages = sorted({k for r in rows for k in r if k.startswith("latency_")})
    lat_rows = []
    for name, rs in by_config.items():
        cells = []
        for s in stages:
            vals = [r[s] for r in rs if s in r]
            cells.append(f"p50 {percentile(vals, 50):.3f} / p95 {percentile(vals, 95):.3f}" if vals else "-")
        lat_rows.append([name] + cells)
    out.append(table(["config"] + [s.removeprefix("latency_") for s in stages], lat_rows))

    if any("cost_usd" in r for r in rows):
        out.append("\n## Cost\n")
        out.append(table(
            ["config", "total USD", "USD / question"],
            [[name, f"{sum(r['cost_usd'] for r in rs):.2f}", f"{mean([r['cost_usd'] for r in rs]):.4f}"]
             for name, rs in by_config.items()],
        ))
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="also generate answers and judge them (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    parser.add_argument("--original-labels", action="store_true", help="score against first-pass labels, without pooled additions")
    parser.add_argument("--rescore", metavar="RUN_DIR", help="re-judge the answers stored in a previous --generate run")
    args = parser.parse_args()

    golden = load_golden(args.original_labels)[: args.limit]
    global LABELS
    if args.original_labels:
        LABELS = "first-pass only"
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)

    rows = []
    started = time.perf_counter()
    if args.rescore:
        golden_by_id = {g["id"]: g for g in golden}
        old = [json.loads(line) for line in open(Path(args.rescore) / "per_question.jsonl")]
        with open(run_dir / "per_question.jsonl", "w") as f:
            for i, row in enumerate(old, 1):
                if row["id"] not in golden_by_id:
                    continue
                new = rescore(row, golden_by_id[row["id"]])
                rows.append(new)
                f.write(json.dumps(new) + "\n")
                f.flush()
                print(f"\rre-judged {i}/{len(old)}", end="", flush=True)
        print()
        args.generate = True
    with open(run_dir / "per_question.jsonl", "a") as f:
        for config in ([] if args.rescore else CONFIGS):
            for i, example in enumerate(golden, 1):
                row = evaluate(config, example, args.generate)
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"\r{config.name:<20} {i:>3}/{len(golden)}", end="", flush=True)
            print()

    metrics = RETRIEVAL_METRICS + (GENERATION_METRICS if args.generate else [])
    summary = summarize(rows, metrics)
    (run_dir / "summary.md").write_text(summary)
    print(f"\nFinished in {time.perf_counter() - started:.0f}s -> {run_dir}\n")
    print(summary)


if __name__ == "__main__":
    main()

"""Run the generation evaluation through the Message Batches API, at half price.

Same configurations, prompts and scoring as `run_ablations --generate` - the
request builders and the scoring function are shared - but every Claude call
goes through the Batch API: asynchronous, usually done within an hour, and
50% cheaper. Evaluation doesn't need answers in real time, so the trade is
free apart from two things:

  - Generation latency isn't measured here (batch requests have no
    meaningful per-request latency). Report it from a live run instead.
  - Server-side refusal fallbacks aren't available on batches; a refusal is
    recorded as a refused answer.

Three rounds, because each judging step depends on the previous one:
  1. generate every answer
  2. extract claims + judge relevance/correctness (both need only the answer)
  3. verify claims against the context (needs the extracted claims)

Every batch id is saved to the run directory the moment it's created, so an
interrupted run resumes with --resume and re-uses submitted batches instead
of paying for them again.

RUN
    uv run python -m evaluation.batch_run [--limit N]
    uv run python -m evaluation.batch_run --resume data/runs/<run>
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import anthropic
from anthropic.types.messages.batch_create_params import Request

from evaluation.generation_metrics import (AnswerJudgement, Claims, Verification, answer_prompt,
                                           batch_params, decompose_prompt, finalize, verify_prompt)
from evaluation.retrieval_metrics import score_retrieval
from evaluation.run_ablations import (CONFIGS, GENERATION_METRICS, RETRIEVAL_METRICS, RUNS_DIR,
                                      load_golden, summarize)
from ingest.chunk import Chunk
from pipeline.generate import answer_from_message, request_params
from pipeline.rag import search
from pipeline.retrieve import Hit

POLL_SECONDS = 30


class Run:
    """Persistent state for one batch run, stored as files in the run directory."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "batches.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.client = anthropic.Anthropic()

    def save_state(self):
        self.state_path.write_text(json.dumps(self.state, indent=1))

    def run_batch(self, stage, requests):
        """Submit (or resume) one batch and return {custom_id: message or None}."""
        if not requests:
            return {}
        if stage not in self.state:
            batch = self.client.messages.batches.create(
                requests=[Request(custom_id=cid, params=params) for cid, params in requests]
            )
            self.state[stage] = batch.id
            self.save_state()  # before waiting: an interrupted run must not resubmit
            print(f"{stage}: submitted {len(requests)} requests as {batch.id}")
        batch_id = self.state[stage]

        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            print(f"\r{stage}: {batch.processing_status}  succeeded {counts.succeeded}  "
                  f"errored {counts.errored}  processing {counts.processing}   ", end="", flush=True)
            if batch.processing_status == "ended":
                print()
                break
            time.sleep(POLL_SECONDS)

        # Results arrive in any order - key them by custom_id, never by position.
        results = {}
        for entry in self.client.messages.batches.results(batch_id):
            results[entry.custom_id] = entry.result.message if entry.result.type == "succeeded" else None
            if entry.result.type != "succeeded":
                print(f"  {entry.custom_id}: {entry.result.type}")
        return results


def parse_structured(message, schema):
    """Structured-output batch results arrive as a JSON text block."""
    if message is None or message.stop_reason == "refusal":
        return None
    text = next((b.text for b in message.content if b.type == "text"), None)
    return schema.model_validate_json(text) if text else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", metavar="RUN_DIR")
    args = parser.parse_args()

    golden = load_golden()[: args.limit]
    run = Run(args.resume or RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-batch")

    # Retrieval runs locally and is deterministic, so it's simply recomputed on resume.
    items = []
    for config in CONFIGS:
        for example in golden:
            result = search(example["question"], config)
            items.append({
                "key": f"q{len(items):04d}",
                "config": config,
                "example": example,
                "result": result,
                "hits": result.hits[: config.k_context],
            })
    print(f"retrieval done for {len(items)} (config, question) pairs")

    # Round 1: generate.
    generated = run.run_batch("generate", [
        (it["key"], request_params(it["example"]["question"], it["hits"])) for it in items
    ])
    for it in items:
        message = generated.get(it["key"])
        it["answer"] = answer_from_message(message, it["hits"], batch=True) if message else None

    # Round 2: extract claims, and judge relevance/correctness.
    answered = [it for it in items if it["answer"] and it["answer"].text]
    round2 = run.run_batch("judge-claims-and-answer", [
        request for it in answered for request in (
            (f"{it['key']}-d", batch_params(decompose_prompt(it["example"], it["answer"].text), Claims)),
            (f"{it['key']}-a", batch_params(answer_prompt(it["example"], it["answer"].text), AnswerJudgement)),
        )
    ])
    for it in answered:
        decomposed = parse_structured(round2.get(f"{it['key']}-d"), Claims)
        it["claims"] = decomposed.claims if decomposed else []
        it["judged"] = parse_structured(round2.get(f"{it['key']}-a"), AnswerJudgement)
        it["judge_cost"] = sum(
            _cost(round2.get(f"{it['key']}-{suffix}")) for suffix in ("d", "a")
        )

    # Round 3: verify claims against the context.
    round3 = run.run_batch("judge-verify", [
        (f"{it['key']}-v", batch_params(verify_prompt(it["hits"], it["claims"]), Verification))
        for it in answered if it["claims"]
    ])

    rows = []
    with open(run.dir / "per_question.jsonl", "w") as f:
        for it in items:
            config, example, result, answer = it["config"], it["example"], it["result"], it["answer"]
            row = {"config": config.name, "id": example["id"], "question_type": example["question_type"]}
            row.update(score_retrieval(result, example))
            row["retrieved_sources"] = [h.chunk.source_file for h in result.hits]
            row.update({f"latency_{stage}": secs for stage, secs in result.timings.items()})

            if answer is None:  # the generation request itself errored or expired
                row.update({m: None for m in GENERATION_METRICS}, generation_failed=True)
            else:
                verification = parse_structured(round3.get(f"{it['key']}-v"), Verification)
                judge_cost = it.get("judge_cost", 0.0) + _cost(round3.get(f"{it['key']}-v"))
                row.update(finalize(example, answer, verification.verdicts if verification else [],
                                    it.get("judged"), judge_cost))
                row.update({
                    "answer": answer.text,
                    "answer_model": answer.model,
                    "citations": [c.__dict__ for c in answer.citations],
                    "context": [{"source_file": h.chunk.source_file, "text": h.chunk.text} for h in it["hits"]],
                    "generation_cost_usd": answer.cost_usd,
                    "cost_usd": answer.cost_usd + judge_cost,
                })
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    summary = summarize(rows, RETRIEVAL_METRICS + GENERATION_METRICS)
    summary += ("\nGeneration and judging ran through the Message Batches API (50% price); "
                "generation latency is not measured in batch runs.\n")
    (run.dir / "summary.md").write_text(summary)
    total = sum(r.get("cost_usd", 0) for r in rows)
    print(f"\n{run.dir}  |  total cost ${total:.2f}\n")
    print(summary)


def _cost(message):
    from pipeline.generate import cost
    return cost(message.usage, message.model, batch=True) if message is not None else 0.0


if __name__ == "__main__":
    main()

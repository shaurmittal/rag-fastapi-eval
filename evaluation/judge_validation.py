"""Validate the LLM judge: does it agree with a human, and with itself?

A judge's score is only meaningful if the judge is right. This measures that
two ways, on a fixed random sample of items from a --generate run:

1. HUMAN AGREEMENT. A person labels the same items the judge scored, without
   seeing the judge's verdict (showing it would anchor the human and inflate
   agreement). Reported as raw agreement and Cohen's kappa.

2. TEST-RETEST STABILITY. The judge scores the same items a second time.
   Claude Opus 5 can't be pinned to temperature 0, so the judge isn't
   deterministic; this measures how much that matters in practice.

COHEN'S KAPPA
    kappa = (p_observed - p_chance) / (1 - p_chance)

    p_chance is the agreement two raters would reach by guessing with their
    own yes/no rates. Raw agreement flatters imbalanced data: if 85% of claims
    are supported, a judge that always says "supported" agrees 85% of the time
    while knowing nothing. Kappa scores that judge at 0.
    Rough reading: < 0.4 poor, 0.4-0.6 moderate, 0.6-0.8 substantial, > 0.8
    near-perfect.

ITEMS
    claim        "Is this claim supported by the context?"  (faithfulness)
    correctness  "Is this answer consistent with the reference?"

USAGE
    uv run python -m evaluation.judge_validation sample data/runs/<run>   # pick items
    uv run python -m evaluation.judge_validation label [--claims N --answers M]  # you label them
    uv run python -m evaluation.judge_validation retest                   # judge re-scores them
    uv run python -m evaluation.judge_validation report                   # agreement + kappa

Human labels are saved to data/judge_validation.jsonl, which is committed:
they're not regenerable. The same file carries a second rater's labels
(`llm_rater`, an independent blind pass by Claude Opus 5.5 over all items),
which `report` compares alongside the human and the re-run.
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

PATH = Path("data/judge_validation.jsonl")
N_CLAIMS = 60
N_ANSWERS = 30


def cohens_kappa(a, b):
    """Chance-corrected agreement between two lists of booleans."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    yes_a, yes_b = sum(a) / n, sum(b) / n
    chance = yes_a * yes_b + (1 - yes_a) * (1 - yes_b)
    if chance == 1:
        return float("nan")  # both raters constant: kappa undefined
    return (observed - chance) / (1 - chance)


def load():
    return [json.loads(line) for line in open(PATH)] if PATH.exists() else []


def save(items):
    with open(PATH, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def sample(run_dir):
    rows = [json.loads(line) for line in open(Path(run_dir) / "per_question.jsonl")]
    rows = [r for r in rows if r.get("answer")]
    rng = random.Random(0)

    claims = [
        {"kind": "claim", "config": r["config"], "id": r["id"], "claim": v["claim"],
         "context": r["context"], "judge": v["supported"]}
        for r in rows for v in r.get("claims", [])
    ]
    golden = {g["id"]: g for g in map(json.loads, open("data/golden_eval_set.jsonl"))}
    answers = [
        {"kind": "correctness", "config": r["config"], "id": r["id"], "question": golden[r["id"]]["question"],
         "answer": r["answer"], "reference": golden[r["id"]]["expected_answer"],
         "judge": bool(r["correctness"])}
        for r in rows if r.get("correctness") is not None
    ]
    items = rng.sample(claims, min(N_CLAIMS, len(claims))) + rng.sample(answers, min(N_ANSWERS, len(answers)))
    save(items)
    print(f"sampled {sum(i['kind'] == 'claim' for i in items)} claims and "
          f"{sum(i['kind'] == 'correctness' for i in items)} answers -> {PATH}")


def ask(prompt):
    while True:
        reply = input(f"{prompt} [y/n/q] ").strip().lower()
        if reply in ("y", "n", "q"):
            return reply


def label(n_claims=None, n_answers=None):
    """Label items blind. The sample is already in random order, so limiting
    to the first n of each kind labels a random subset."""
    items = load()
    limit = {"claim": n_claims, "correctness": n_answers}
    subset = [i for kind in limit for i in [i for i in items if i["kind"] == kind][: limit[kind]]]
    todo = [i for i in subset if "human" not in i]
    print(f"{len(todo)} of {len(subset)} items left. The judge's verdict is hidden on purpose.\n")
    for n, item in enumerate(todo, 1):
        print("=" * 80, f"\n[{n}/{len(todo)}] {item['kind']}  ({item['id']})\n")
        if item["kind"] == "claim":
            for doc in item["context"]:
                print(f"--- {doc['source_file']}\n{textwrap.indent(doc['text'], '    ')}\n")
            print(f"CLAIM: {item['claim']}\n")
            reply = ask("Is this claim supported by the context above?")
        else:
            print(f"QUESTION:\n{textwrap.shorten(item['question'], 700)}\n")
            print(f"ANSWER:\n{item['answer']}\n")
            print(f"REFERENCE:\n{item['reference']}\n")
            reply = ask("Is the answer consistent with the reference's main point?")
        if reply == "q":
            break
        item["human"] = reply == "y"
        save(items)  # after every label, so quitting loses nothing
    print(f"\nlabelled {sum('human' in i for i in items)} of {len(items)}")


def retest():
    from evaluation.generation_metrics import (ANSWER_PROMPT, VERIFY_PROMPT, AnswerJudgement,
                                               Verification, judge)
    # The retest re-uses the exact prompts the judge was scored with.
    items = load()
    for n, item in enumerate(items, 1):
        if "judge_retest" in item:
            continue
        if item["kind"] == "claim":
            context = "\n\n".join(f'<document source="{d["source_file"]}">\n{d["text"]}\n</document>'
                                  for d in item["context"])
            out, _ = judge(VERIFY_PROMPT.format(context=context, claims=f"- {item['claim']}"), Verification)
            item["judge_retest"] = bool(out.verdicts[0].supported) if out and out.verdicts else None
        else:
            out, _ = judge(ANSWER_PROMPT.format(question=item["question"], answer=item["answer"],
                                                reference=item["reference"]), AnswerJudgement)
            item["judge_retest"] = bool(out.consistent_with_reference) if out else None
        save(items)
        print(f"\rretested {n}/{len(items)}", end="", flush=True)
    print()


def compare(items, rater, reference="judge"):
    """Agreement between two label fields, over the items that have both."""
    both = [i for i in items if i.get(rater) is not None and i.get(reference) is not None]
    if not both:
        return
    r = [i[rater] for i in both]
    j = [i[reference] for i in both]
    agree = sum(x == y for x, y in zip(r, j)) / len(r)
    print(f"{reference} vs {rater:<10} n={len(r):<3} agreement={agree:.2f}  kappa={cohens_kappa(r, j):.2f}"
          f"  ({rater} yes {sum(r)}/{len(r)}, {reference} yes {sum(j)}/{len(j)};"
          f" {reference} stricter on {sum(x and not y for x, y in zip(r, j))},"
          f" more lenient on {sum(y and not x for x, y in zip(r, j))})")


def report():
    # Raters: "human" is the project author, labelling blind; "llm_rater" is a
    # second, independent LLM pass (Claude Opus 5.5), also blind to the judge.
    items = load()
    for kind in ("claim", "correctness"):
        subset = [i for i in items if i["kind"] == kind]
        print(f"\n## {kind} (n={len(subset)} sampled)")
        compare(subset, "human")
        compare(subset, "llm_rater")
        compare(subset, "llm_rater", reference="human")

        compare(subset, "judge_retest")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sample").add_argument("run_dir")
    label_parser = sub.add_parser("label")
    label_parser.add_argument("--claims", type=int, help="label only the first N sampled claims")
    label_parser.add_argument("--answers", type=int, help="label only the first N sampled answers")
    sub.add_parser("retest")
    sub.add_parser("report")
    args = parser.parse_args()
    {"sample": lambda: sample(args.run_dir), "label": lambda: label(args.claims, args.answers),
     "retest": retest, "report": report}[args.command]()


if __name__ == "__main__":
    main()

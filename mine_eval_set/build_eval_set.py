"""Stage 1 of curation: turn raw issues into a ranked list of candidates.

Reads data/raw_issues.jsonl (the frozen cache). Touches no network, so it can
be re-run as often as the rules change.

For each issue it:
  1. Cleans the question - strips the issue template's checklists and
     environment sections, which every issue shares and which would make every
     question look alike to an embedding.
  2. Picks the most likely answer comment. The first comment is usually NOT
     the answer (reconnaissance showed "+1", "any updates?", "same issue").
     Signals, strongest first:
       a. A maintainer thanked someone by name ("Thanks for the help here
          @euri10!") - that person's comment is the endorsed answer.
       b. A substantive maintainer comment (not a closing boilerplate).
       c. The most-reacted comment that contains code.
  3. Scores the pair on groundability signals and drops obvious rejects.

The output is a shortlist for HUMAN review, not the golden set. Deciding
whether an answer is grounded in the pinned corpus - and which file grounds
it - is a judgment this script deliberately does not attempt.

OUTPUT
    data/eval_candidates.jsonl

RUN
    uv run python -m mine_eval_set.build_eval_set
"""

import json
import re

RAW_PATH = "data/raw_issues.jsonl"
OUT_PATH = "data/eval_candidates.jsonl"

MAINTAINER = {"OWNER", "MEMBER", "COLLABORATOR"}

# Closing remarks that look like maintainer answers but carry no content.
BOILERPLATE = re.compile(
    r"^(thanks for the help here|thanks for reporting back|assuming the original need"
    r"|sorry, i can't help you if|it seems this is a duplicate|i'll close this issue"
    r"|thanks for the discussion|this issue was moved to a discussion)",
    re.IGNORECASE,
)
THANKED = re.compile(r"thanks for the help here @([\w-]+)", re.IGNORECASE)

# Template sections that describe the reporter's setup, not their question.
DROP_SECTIONS = re.compile(
    r"^#{2,4}\s*(first check|commit to help|operating system|operating system details"
    r"|fastapi version|python version|pydantic version|additional context|environment)\b",
    re.IGNORECASE,
)
CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\]")


def clean_question(title, body):
    # Template instructions live in HTML comments: "<!-- Replace the code below... -->"
    body = re.sub(r"<!--.*?-->", "", (body or "").replace("\r\n", "\n"), flags=re.DOTALL)
    kept, dropping = [], False
    for line in body.splitlines():
        if re.match(r"^#{2,4}\s", line):
            dropping = bool(DROP_SECTIONS.match(line))
            if not dropping:
                continue  # keep the content, drop the "### Description" label
        if dropping or CHECKBOX.match(line):
            continue
        kept.append(line)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return f"{title}\n\n{body}".strip()


def is_bot(comment):
    return comment["user"]["type"] == "Bot" or comment["user"]["login"].endswith("[bot]")


def substantive(comment):
    text = (comment["body"] or "").strip()
    return len(text) > 80 and not BOILERPLATE.match(text) and not is_bot(comment)


def pick_answer(comments):
    """Return (answer_comment, how_it_was_chosen) or (None, None)."""
    for c in comments:
        if c["author_association"] in MAINTAINER:
            match = THANKED.search(c["body"] or "")
            if match:
                by_them = [x for x in comments if x["user"]["login"] == match[1] and substantive(x)]
                if by_them:
                    return max(by_them, key=lambda x: len(x["body"])), f"endorsed:@{match[1]}"

    maintainer = [c for c in comments if c["author_association"] in MAINTAINER and substantive(c)]
    if maintainer:
        return max(maintainer, key=lambda x: len(x["body"])), "maintainer"

    with_code = [c for c in comments if substantive(c) and "`" in c["body"]]
    if with_code:
        return max(with_code, key=lambda x: x["reactions"]["total_count"]), "reacted"

    return None, None


def score(question, answer):
    """Heuristic groundability score. Higher = more worth a human's time."""
    s = 0
    s += 2 if "```" in answer else 0  # a code example
    s += min(answer.count("`") // 2, 4)  # named API surface: `Depends`, `include_in_schema`
    s += 1 if "fastapi.tiangolo.com" in answer else 0  # points at the docs
    s -= 2 if len(question) > 3000 else 0  # long pasted user code
    s -= 2 if re.search(r"traceback \(most recent call last\)", question, re.I) else 0
    return s


def main():
    issues = [json.loads(line) for line in open(RAW_PATH)]
    candidates = []
    for issue in issues:
        labels = {label["name"] for label in issue["labels"]}
        if labels & {"bug", "feature", "enhancement", "duplicate", "invalid"}:
            continue
        answer, how = pick_answer(issue["comments_data"])
        if not answer:
            continue
        question = clean_question(issue["title"], issue["body"])
        candidates.append(
            {
                "issue": issue["number"],
                "url": issue["html_url"],
                "question": question,
                "answer": answer["body"].replace("\r\n", "\n").strip(),
                "answer_by": answer["user"]["login"],
                "answer_chosen_by": how,
                "reactions": issue["reactions"]["total_count"],
                "score": score(question, answer["body"]),
            }
        )

    candidates.sort(key=lambda c: (-c["score"], -c["reactions"]))
    with open(OUT_PATH, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    print(f"{len(issues)} issues -> {len(candidates)} candidates with an identifiable answer")
    for how in ("endorsed", "maintainer", "reacted"):
        print(f"  answer chosen by {how:<10} {sum(c['answer_chosen_by'].startswith(how) for c in candidates)}")
    print(f"  score >= 4: {sum(c['score'] >= 4 for c in candidates)}")


if __name__ == "__main__":
    main()

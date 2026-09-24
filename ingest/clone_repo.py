"""Fetch the pinned FastAPI corpus into data/raw/fastapi.

The eval set is only valid against a frozen corpus, so this clones one tag,
shallowly, and refuses to reuse a checkout of any other version.

RUN
    uv run python -m ingest.clone_repo
"""

import subprocess

from ingest.corpus import CORPUS_COMMIT, CORPUS_ROOT, CORPUS_VERSION

REPO_URL = "https://github.com/fastapi/fastapi.git"


def main():
    if CORPUS_ROOT.exists():
        head = subprocess.run(
            ["git", "-C", str(CORPUS_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if not CORPUS_COMMIT.startswith(head[: len(CORPUS_COMMIT)]):
            raise SystemExit(f"{CORPUS_ROOT} is at {head}, expected {CORPUS_COMMIT}. Delete it and re-run.")
        print(f"Corpus already present at {CORPUS_VERSION} ({head}).")
        return

    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", CORPUS_VERSION, REPO_URL, str(CORPUS_ROOT)],
        check=True,
    )
    print(f"Cloned {REPO_URL} @ {CORPUS_VERSION}.")


if __name__ == "__main__":
    main()

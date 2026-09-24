"""Fetch answered FastAPI issues and their comment threads into a raw cache.

This script fetches. It does not judge. Every filtering and curation decision
happens later, in build_eval_set.py, reading the file this one writes. That
separation means filter rules can be rewritten many times without re-hitting
the network, and two runs of the filter on the same cache always agree.

OUTPUT
    data/raw_issues.jsonl - one JSON object per line: the raw issue from the
    search API, plus a "comments_data" key holding the raw comment list.

RUN
    uv run python -m mine_eval_set.scrape_github_issues

Two endpoints are involved:

    GET /search/issues                       30 requests per MINUTE, max 1000 results
    GET /repos/{repo}/issues/{n}/comments    5000 requests per HOUR

Comments need their own call because search returns only a comment COUNT,
never the text, and the text is where the answers are.
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# The repo moved from tiangolo/ to the fastapi/ org. Git follows the redirect,
# but the Search API's repo: qualifier does not, so the new name is required.
REPO = "fastapi/fastapi"
QUERY = f"repo:{REPO} is:issue is:closed label:answered label:question"
OUTPUT_PATH = "data/raw_issues.jsonl"
PER_PAGE = 100

# Set to a small number (e.g. 5) for a quick test run; None fetches everything.
LIMIT = None

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def get_json(url, params=None, max_retries=5):
    """Make one GET request and return the parsed JSON, retrying on failure.

    This is the only function that touches the network, so it is the only one
    that has to cope with the network misbehaving: rate limits (403/429) and
    transient errors like the 401 blip seen during reconnaissance.
    """
    for attempt in range(max_retries):
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()

        if response.status_code in (403, 429):
            # Rate limited. GitHub sometimes says how long to wait; the header
            # arrives as text, and is absent entirely on some 403s.
            wait = response.headers.get("Retry-After")
            time.sleep(int(wait) if wait is not None else 10)
        else:
            # Exponential backoff: 2, 4, 8, 16 seconds.
            time.sleep(2 ** (attempt + 1))

    raise RuntimeError(
        f"GET {url} failed after {max_retries} attempts "
        f"(last status {response.status_code})"
    )


def fetch_comments(issue_number):
    """Return the raw list of comments on one issue.

    This endpoint returns a list directly, unlike search, which wraps its
    results in a dict under "items".
    """
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    return get_json(url)


def search_issues(query):
    """Return every issue matching `query`, walking through the result pages."""
    issues = []
    page = 1
    while True:
        data = get_json(
            "https://api.github.com/search/issues",
            params={
                "q": query,
                "per_page": PER_PAGE,
                "page": page,
                # Highest-reaction first, so a partial run gets the issues the
                # community found most useful.
                "sort": "reactions",
                "order": "desc",
            },
        )
        items = data["items"]
        issues.extend(items)

        # A short page means there is nothing after it.
        if len(items) < PER_PAGE:
            break
        page += 1

        # Search allows only 30 requests per minute.
        time.sleep(2)

    return issues


def load_already_fetched(path):
    """Return the set of issue numbers already present in the output file.

    This is what makes the scraper resumable: stop it with Ctrl-C, run it
    again, and it picks up where it left off.
    """
    numbers = set()
    if not os.path.exists(path):
        return numbers

    with open(path) as f:
        for line in f:
            if line.strip():
                numbers.add(json.loads(line)["number"])
    return numbers


def main():
    print(f"Searching: {QUERY}")
    issues = search_issues(QUERY)[:LIMIT]
    print(f"Found {len(issues)} candidate issues.")

    already = load_already_fetched(OUTPUT_PATH)
    print(f"Already cached: {len(already)}. Fetching the rest.\n")

    fetched = 0
    # Append mode: "w" would truncate the file and destroy the cache that
    # load_already_fetched just read.
    with open(OUTPUT_PATH, "a") as out:
        for position, issue in enumerate(issues, start=1):
            if issue["number"] in already:
                continue

            issue["comments_data"] = fetch_comments(issue["number"])
            out.write(json.dumps(issue) + "\n")
            # Flush after every line so an interrupted run loses nothing.
            out.flush()
            fetched += 1

            print(
                f"  [{position:>3}/{len(issues)}] #{issue['number']:<6} "
                f"{len(issue['comments_data']):>3} comments  {issue['title'][:50]}"
            )
            time.sleep(0.5)

    print(f"\nDone. Fetched {fetched}, skipped {len(already)} already cached.")


if __name__ == "__main__":
    main()

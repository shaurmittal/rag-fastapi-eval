"""Load the pinned FastAPI corpus into (path, text) documents.

Two preprocessing steps happen here, before any chunking, because both
strategies need them equally:

1. Resolve include directives. The markdown doesn't contain its own code
   examples; it contains lines like

       {!../../../docs_src/query_params/tutorial001.py!}

   which the docs site replaces with file contents at build time. Chunking the
   raw markdown would leave every code example as a meaningless path, and
   questions whose answer IS the code example could never be retrieved.

2. Collapse tabbed variants. Most examples appear in up to six near-identical
   versions (Python 3.8+ / 3.9+ / 3.10+, with and without Annotated). Keeping
   all of them floods retrieval with near-duplicates - the same failure mode as
   indexing all 26 translations. We keep the first tab of each group, which is
   the variant FastAPI recommends.
"""

import re
from pathlib import Path

CORPUS_ROOT = Path("data/raw/fastapi")
CORPUS_VERSION = "0.115.0"
CORPUS_COMMIT = "40e33e4"

# English docs only - the repo ships 26 translations under docs/<lang>/.
DOCS_DIR = CORPUS_ROOT / "docs" / "en" / "docs"
SOURCE_DIR = CORPUS_ROOT / "fastapi"

# Matches {!path!}, {!> path!}, and either with a [ln:a-b] line range.
INCLUDE_RE = re.compile(r"\{!>?\s*(?P<path>[^\[!]+?)\s*(?:\[ln:(?P<start>\d+)-(?P<end>\d+)\])?\s*!\}")

# A tab opener looks like "//// tab | Python 3.10+"; a closer is a bare "////".
TAB_OPEN_RE = re.compile(r"^/{3,4} ?tab \|")
TAB_CLOSE_RE = re.compile(r"^/{3,4}\s*$")


def resolve_includes(text):
    """Replace every include directive with the contents of the file it names.

    Paths are relative to the docs root (docs/en/docs/), not to the markdown
    file - that's how the mkdocs include extension resolves them. Three "../"
    from docs/en/docs/ lands at the repo root, where docs_src/ lives.
    """

    def replace(match):
        target = (DOCS_DIR / match["path"]).resolve()
        if not target.exists():
            # Leave the directive in place rather than silently dropping content.
            return match.group(0)
        lines = target.read_text(encoding="utf-8").splitlines()
        if match["start"]:
            lines = lines[int(match["start"]) - 1 : int(match["end"])]
        return "\n".join(lines)

    return INCLUDE_RE.sub(replace, text)


def collapse_tabs(text):
    """Keep only the first tab in each run of consecutive tabs.

    A small state machine over three facts:
      in_tab        - are we inside a //// tab ... //// block?
      keep_tab      - is the current tab the one we're keeping?
      after_a_tab   - did a tab just close, with only blank lines since?
                      If so, the next tab opener is a sibling variant.
    """
    out = []
    in_tab = keep_tab = after_a_tab = False

    for line in text.splitlines():
        if TAB_OPEN_RE.match(line):
            in_tab = True
            keep_tab = not after_a_tab  # first of its group
            continue
        if in_tab and TAB_CLOSE_RE.match(line):
            in_tab = False
            after_a_tab = True
            continue
        if in_tab:
            if keep_tab:
                out.append(line)
            continue
        # Outside any tab: real content ends the current tab group.
        if line.strip():
            after_a_tab = False
        out.append(line)

    return "\n".join(out)


def rel(path):
    """Repo-relative path string, the canonical form used for provenance."""
    return str(Path(path).relative_to(CORPUS_ROOT))


def load_docs():
    """Yield (repo_relative_path, preprocessed_markdown) for every English doc."""
    for md_path in sorted(DOCS_DIR.rglob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        text = collapse_tabs(text)
        text = resolve_includes(text)
        yield rel(md_path), text


def load_source():
    """Yield (repo_relative_path, python_source) for every FastAPI source file."""
    for py_path in sorted(SOURCE_DIR.rglob("*.py")):
        yield rel(py_path), py_path.read_text(encoding="utf-8")

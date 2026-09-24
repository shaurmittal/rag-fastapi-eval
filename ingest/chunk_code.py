"""Strategy 2, code half - AST-aware Python chunking.

Parse each source file into its syntax tree and chunk along the units a
programmer thinks in: a top-level function, a class, a method. A fixed-size
window happily cuts a function signature away from its body; this never does.

Rules:
  - A top-level function or class that fits in MAX_CHARS is one chunk.
  - A class too big for one chunk is split into its methods. Each method chunk
    is prefixed with the class header line, so "def get(self, ...)" still says
    which class it belongs to - without that, a method named `get` is
    indistinguishable from the dozen other `get`s in the codebase.
  - A single definition bigger than MAX_CHARS is split by lines, each part
    carrying the definition's first line as context. FastAPI has several of
    these: APIRouter.get() carries hundreds of lines of Doc() annotations.
  - Module-level statements between definitions (imports, constants) are
    grouped into their own chunks.

Every chunk is prefixed with "# <file> :: <qualified name>" so the embedding
sees where the code lives, not just what it says.
"""

import ast

from ingest.chunk import Chunk, make_chunk_id

MAX_CHARS = 1500


def _segment(lines, node):
    """Source text for a node, including any decorators above it."""
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(lines[start - 1 : node.end_lineno])


def _split_long(text, header, max_chars):
    """Line-split an oversized unit; parts after the first repeat its header."""
    parts, current = [], ""
    for line in text.splitlines():
        if current and len(current) + len(line) + 1 > max_chars:
            parts.append(current)
            current = f"{header}\n    # ...continued"
        current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


def code_units(source):
    """Yield (qualified_name, text) units from one Python file."""
    lines = source.splitlines()
    tree = ast.parse(source)
    pending = []  # module-level statements waiting to be grouped

    def flush_pending():
        # Module-level code can be large too: fastapi/_compat.py wraps dozens
        # of functions in a module-level `if PYDANTIC_V2:` block, which the AST
        # sees as one statement rather than as definitions.
        if pending:
            text = "\n".join(_segment(lines, n) for n in pending)
            pending.clear()
            for part in _split_long(text, "# (module level)", MAX_CHARS):
                yield "<module>", part

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from flush_pending()
            text = _segment(lines, node)
            if len(text) <= MAX_CHARS:
                yield node.name, text
            else:
                header = lines[node.lineno - 1]
                for part in _split_long(text, header, MAX_CHARS):
                    yield node.name, part

        elif isinstance(node, ast.ClassDef):
            yield from flush_pending()
            text = _segment(lines, node)
            if len(text) <= MAX_CHARS:
                yield node.name, text
                continue

            class_header = lines[node.lineno - 1]
            # The class "shell": signature, docstring, class-level attributes.
            shell = [n for n in node.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            shell_text = "\n".join([class_header] + [_segment(lines, n) for n in shell])
            for part in _split_long(shell_text, class_header, MAX_CHARS):
                yield node.name, part

            for method in node.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                name = f"{node.name}.{method.name}"
                method_text = f"{class_header}\n    ...\n{_segment(lines, method)}"
                if len(method_text) <= MAX_CHARS:
                    yield name, method_text
                else:
                    header = f"{class_header}\n    {lines[method.lineno - 1].strip()}"
                    for part in _split_long(method_text, header, MAX_CHARS):
                        yield name, part
        else:
            pending.append(node)

    yield from flush_pending()


def chunk_python(source, source_file):
    chunks = []
    for name, text in code_units(source):
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                id=make_chunk_id("structured", source_file, len(chunks)),
                text=f"# {source_file} :: {name}\n{text}",
                source_file=source_file,
                section=name,
                chunk_type="code",
                strategy="structured",
            )
        )
    return chunks

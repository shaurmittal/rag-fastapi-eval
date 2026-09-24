"""The Chunk contract shared by every chunking strategy.

Both strategies (naive and structured) emit exactly this shape. That is what
makes them swappable in the ablation: everything downstream - embedding,
the vector store, retrieval, evaluation - only ever sees a Chunk.

`source_file` is the provenance field the evaluation relies on. A retrieved chunk
counts as relevant if its source_file is one of the question's expected
sources. Chunk ids differ between strategies; file paths do not.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    source_file: str  # repo-relative, e.g. "docs/en/docs/tutorial/body.md"
    section: str  # heading path or qualified name, e.g. "Request Body > Import"
    chunk_type: str  # "docs" or "code"
    strategy: str  # "naive" or "structured"
    metadata: dict = field(default_factory=dict)

    def to_json(self):
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line):
        return cls(**json.loads(line))


def make_chunk_id(strategy, source_file, index):
    """A stable id: the same input always produces the same id.

    Stability matters because Qdrant upserts by id - re-running ingestion
    overwrites chunks in place instead of duplicating them.
    """
    key = f"{strategy}:{source_file}:{index}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]

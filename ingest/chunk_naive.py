"""Strategy 1 - naive fixed-size chunking (the baseline).

Slide a fixed-size character window across each file, with overlap, ignoring
all structure. A window may start mid-sentence and end mid-function.

This exists to be beaten. It answers the question "does structure-awareness
actually help, or would any chunker do?" - without a baseline, an ablation has
nothing to be measured against.

The overlap means a sentence cut at one window's edge appears whole in the
next, which is the standard mitigation for boundary damage. Sizes are chosen
so naive chunks are roughly the same size as structured ones: that keeps the
ablation about WHERE the boundaries fall, not how big the chunks are.
"""

from ingest.chunk import Chunk, make_chunk_id

CHUNK_SIZE = 1000  # characters (~250 tokens)
OVERLAP = 200


def chunk_text(text, source_file, chunk_type, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    step = size - overlap
    for index, start in enumerate(range(0, max(len(text) - overlap, 1), step)):
        window = text[start : start + size]
        if not window.strip():
            continue
        chunks.append(
            Chunk(
                id=make_chunk_id("naive", source_file, index),
                text=window,
                source_file=source_file,
                # Naive chunks have no structural position, only a character offset.
                section=f"chars {start}-{start + len(window)}",
                chunk_type=chunk_type,
                strategy="naive",
                metadata={"start_char": start},
            )
        )
    return chunks

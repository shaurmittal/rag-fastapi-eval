"""Generate a cited answer from retrieved chunks with Claude.

Each chunk is sent as its own `document` content block with citations
enabled. Claude's answer then comes back split into text blocks, each carrying
the exact passages (cited_text) it drew on and which document they came from.
That makes grounding checkable: every citation points at a specific chunk,
so an answer's claims can be traced back to - or caught diverging from - the
retrieved context.

Citations reduce hallucination; they don't eliminate it. A model can cite a
passage and still overstate what it says, which is what the faithfulness
judge in evaluation/generation_metrics.py measures.

Requests use server-side refusal fallbacks (fallbacks="default"): if a safety
classifier declines a request, the API re-runs it on a fallback model instead
of failing, so one decline can't abort an evaluation sweep.
"""

from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens (input, output). Cache reads/writes are priced
# relative to input; the Batch API halves everything. Used to report cost
# per question and per sweep.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
}

SYSTEM_PROMPT = """You answer developer questions about the FastAPI web framework (version 0.115.0).

Answer using only the documents provided with the question. They are excerpts from FastAPI's documentation and source code.

- If the documents contain the answer, give it directly and concisely, including a short code example when the question is about how to do something.
- If the documents don't contain enough information to answer, say so plainly rather than filling the gap from general knowledge. A partial answer is fine if you're clear about which part the documents don't cover.
- The question is often a real GitHub issue and may include the asker's own code or environment details. Answer the underlying question about FastAPI."""


@dataclass
class Citation:
    chunk_index: int  # position in the context passed to generate()
    source_file: str
    section: str
    cited_text: str


@dataclass
class Answer:
    text: str
    citations: list = field(default_factory=list)
    stop_reason: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # Share of answer text blocks that carry at least one citation.
    cited_block_share: float = 0.0


_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def build_documents(hits):
    """One citable document block per retrieved chunk."""
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": hit.chunk.text},
            "title": f"{hit.chunk.source_file} :: {hit.chunk.section}",
            "citations": {"enabled": True},
        }
        for hit in hits
    ]


def cost(usage, model, batch=False):
    price_in, price_out = PRICES.get(model, PRICES[MODEL])
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    total = (
        usage.input_tokens * price_in
        + cache_read * price_in * 0.1
        + cache_write * price_in * 1.25
        + usage.output_tokens * price_out
    ) / 1_000_000
    return total * 0.5 if batch else total


def parse_response(response, hits):
    """Collect the answer text and its citations from the response blocks."""
    parts, citations = [], []
    text_blocks = cited_blocks = 0
    for block in response.content:
        if block.type != "text":
            continue  # thinking blocks, fallback markers
        parts.append(block.text)
        if not block.text.strip():
            continue
        text_blocks += 1
        block_citations = getattr(block, "citations", None) or []
        if block_citations:
            cited_blocks += 1
        for c in block_citations:
            chunk = hits[c.document_index].chunk
            citations.append(Citation(c.document_index, chunk.source_file, chunk.section, c.cited_text))
    return "".join(parts).strip(), citations, (cited_blocks / text_blocks if text_blocks else 0.0)


def request_params(question, hits):
    """The Messages API request for one answer. Shared by the live call and
    the batch runner, so both paths send exactly the same prompt."""
    return {
        "model": MODEL,
        "max_tokens": 16000,
        "system": SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": build_documents(hits) + [{"type": "text", "text": question}],
        }],
    }


def answer_from_message(response, hits, batch=False):
    """Turn an API response into an Answer."""
    spent = cost(response.usage, response.model, batch)
    if response.stop_reason == "refusal":
        # Every model in the fallback chain (if any) declined. Record it; don't crash the sweep.
        return Answer(text="", stop_reason="refusal", model=response.model,
                      input_tokens=response.usage.input_tokens,
                      output_tokens=response.usage.output_tokens, cost_usd=spent)
    text, citations, cited_share = parse_response(response, hits)
    return Answer(
        text=text,
        citations=citations,
        stop_reason=response.stop_reason,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=spent,
        cited_block_share=cited_share,
    )


def generate(question, hits):
    """Answer `question` from the retrieved `hits`, with citations (live call)."""
    response = client().beta.messages.create(
        **request_params(question, hits),
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    return answer_from_message(response, hits)

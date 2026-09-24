"""Generation metrics: faithfulness, relevance, correctness, and citations.

LLM-AS-JUDGE
    Faithfulness and correctness have no formula, so a model grades them. The
    judge is a measuring instrument with known failure modes, and the design
    below is chosen against them:

    - Binary verdicts only. On a 1-5 scale judges pile onto 3 and 4 and the
      metric loses resolution; "is this claim supported, yes or no" is a much
      easier question to answer consistently.
    - Faithfulness is judged per claim. The answer is first decomposed into
      atomic claims (one call), then each claim is checked against the
      retrieved context (a second call). Length can't inflate a per-claim
      verdict, and a failure points at the exact unsupported sentence.
    - Evidence before verdict. The verifier must quote the passage that
      supports a claim; a claim with no quotable support is unsupported.
    - Structured outputs, so verdicts parse as validated objects, not prose.
    - A frozen, versioned prompt. Changing it invalidates earlier numbers, so
      every row records JUDGE_VERSION.

    Two limitations are reported rather than hidden. The judge is the same
    model family as the generator (self-preference bias is possible), and
    Claude Opus 5 does not accept a temperature setting, so the judge cannot
    be pinned to greedy decoding. evaluation/judge_validation.py measures
    both concerns empirically: agreement with human labels, and agreement
    with itself on a re-run.

DETERMINISTIC METRICS (no LLM)
    has_citation            whether the answer cites any retrieved passage
    cited_source_precision  share of citations pointing into an expected source

    Citation coverage by block was dropped: native citations attach to quoted
    passages, and the model writes the connecting prose and code between them
    as separate uncited blocks, so the share of cited blocks sits near 0.5 by
    construction. Uncited claims are what faithfulness catches.

    Judge revision 2026-09-24.2: the first decomposition prompt asked for
    claims made by code examples, and extracted ~19 claims per answer,
    mostly boilerplate ("FastAPI is imported from fastapi") that retrieved
    chunks naturally don't state - so faithfulness was measuring boilerplate
    coverage, not hallucination. Found by reading smoke-test items, fixed
    before the full run, applied to every configuration.
"""

import anthropic
from pydantic import BaseModel, Field

from pipeline.generate import FALLBACK_BETA, cost

JUDGE_MODEL = "claude-opus-5"
JUDGE_VERSION = "2026-09-24.2"


class Claims(BaseModel):
    claims: list[str] = Field(description="Atomic factual claims, one verifiable assertion each.")


class ClaimVerdict(BaseModel):
    claim: str
    evidence: str = Field(description="Verbatim quote from the context supporting the claim, or empty if none.")
    supported: bool


class Verification(BaseModel):
    verdicts: list[ClaimVerdict]


class AnswerJudgement(BaseModel):
    relevance_reasoning: str
    addresses_question: bool
    correctness_reasoning: str
    consistent_with_reference: bool


DECOMPOSE_PROMPT = """Extract the factual claims this answer makes that bear on the question it answers.

Include statements about FastAPI's behaviour, APIs and parameters, and what the answer's code example demonstrates about solving the problem (for example, "passing include_in_schema=False to Query hides the parameter from the OpenAPI schema"). Make each claim a single assertion that can be checked on its own.

Leave out boilerplate that any FastAPI program contains - imports, creating the app, generic Python syntax like `async def` or type annotations - as well as hedges and statements that the documentation doesn't cover something. Prefer a few meaningful claims over many trivial ones.

If the answer makes no such claims, return an empty list.

<question>
{question}
</question>

<answer>
{answer}
</answer>"""

VERIFY_PROMPT = """For each claim, decide whether the context supports it.

A claim is supported only if the context states it or directly shows it (for example in a code sample). Background knowledge does not count, even if the claim is true. For each claim, first copy the exact passage from the context that supports it into `evidence`; if there is no such passage, leave `evidence` empty and mark the claim unsupported.

<context>
{context}
</context>

<claims>
{claims}
</claims>"""

ANSWER_PROMPT = """Judge an answer to a developer question about FastAPI on two separate points.

1. addresses_question: does the answer respond to what was actually asked? An answer that is accurate but sidesteps the question does not address it. An answer that correctly states the documentation doesn't cover the question does address it.

2. consistent_with_reference: is the answer's main point consistent with the reference answer? Different wording, extra detail, or an alternative approach that achieves the same thing is fine. Missing the reference's central point, or contradicting it, is not.

Reason briefly before each verdict.

<question>
{question}
</question>

<answer>
{answer}
</answer>

<reference_answer>
{reference}
</reference_answer>"""

_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def judge(prompt, schema):
    """One structured judge call. Returns (parsed_output or None, cost_usd)."""
    response = client().beta.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    spent = cost(response.usage, response.model)
    if response.stop_reason == "refusal":
        return None, spent
    return response.parsed_output, spent


def format_context(hits):
    return "\n\n".join(
        f'<document index="{i}" source="{h.chunk.source_file}">\n{h.chunk.text}\n</document>'
        for i, h in enumerate(hits)
    )


def score_generation(example, hits, generated):
    """All generation metrics for one answer, plus the details needed to audit them."""
    expected = set(example["expected_sources"])
    cited = [c.source_file for c in generated.citations]
    row = {
        "judge_version": JUDGE_VERSION,
        "has_citation": float(bool(generated.citations)),
        # Undefined (None) when the answer cites nothing; excluded from means.
        "cited_source_precision": (sum(s in expected for s in cited) / len(cited)) if cited else None,
        "refused": generated.stop_reason == "refusal",
    }
    judge_cost = 0.0

    if not generated.text:
        row.update(faithfulness=None, relevance=0.0, correctness=0.0, claims=[], judge_cost_usd=0.0)
        return row

    # Faithfulness: decompose, then verify each claim against the context.
    decomposed, spent = judge(DECOMPOSE_PROMPT.format(question=example["question"], answer=generated.text), Claims)
    judge_cost += spent
    claims = decomposed.claims if decomposed else []
    verdicts = []
    if claims:
        verified, spent = judge(
            VERIFY_PROMPT.format(context=format_context(hits), claims="\n".join(f"- {c}" for c in claims)),
            Verification,
        )
        judge_cost += spent
        verdicts = verified.verdicts if verified else []
    # An answer with no factual claims (e.g. "the docs don't cover this") is
    # neither faithful nor unfaithful - it's excluded from the faithfulness mean.
    row["faithfulness"] = (sum(v.supported for v in verdicts) / len(verdicts)) if verdicts else None
    row["claims"] = [v.model_dump() for v in verdicts]

    # Relevance and correctness, judged against the question and the reference answer.
    judged, spent = judge(
        ANSWER_PROMPT.format(question=example["question"], answer=generated.text,
                             reference=example["expected_answer"]),
        AnswerJudgement,
    )
    judge_cost += spent
    row["relevance"] = float(judged.addresses_question) if judged else None
    row["correctness"] = float(judged.consistent_with_reference) if judged else None
    row["judge_reasoning"] = judged.model_dump() if judged else None
    row["judge_cost_usd"] = judge_cost
    return row

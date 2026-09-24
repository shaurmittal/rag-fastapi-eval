"""Streamlit demo: ask a question about FastAPI, get a cited answer.

A thin consumer of pipeline.rag - it contains no retrieval or generation
logic of its own, so the demo runs exactly the code path that was evaluated.

    uv run streamlit run app.py

Without ANTHROPIC_API_KEY (env var or Streamlit secret) it shows retrieval
results only.
"""

import json
import os
from pathlib import Path

import streamlit as st

# On Streamlit Community Cloud the key arrives as a secret, not an env var.
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except FileNotFoundError:
    pass  # no secrets file locally; .env is used instead

from pipeline.rag import Config, answer, search  # noqa: E402

st.set_page_config(page_title="FastAPI Q&A", page_icon="⚡", layout="wide")


@st.cache_resource(show_spinner="Loading the index and models...")
def warm_up():
    # First search loads the vector index, the embedding model, and the reranker.
    search("warm up", Config(strategy="structured", use_rerank=True))
    return True


@st.cache_data
def example_questions():
    rows = [json.loads(line) for line in open("data/golden_eval_set.jsonl")]
    return [r["question"].split("\n")[0] for r in rows]


def has_api_key():
    from dotenv import load_dotenv

    load_dotenv()
    return bool(os.getenv("ANTHROPIC_API_KEY"))


warm_up()

st.title("FastAPI Q&A")
st.caption("Answers grounded in the FastAPI v0.115.0 docs and source, with citations to the passages used.")

with st.sidebar:
    st.header("Pipeline")
    strategy = st.radio("Chunking", ["structured", "naive"],
                        help="Structured splits docs at headings and code at functions/classes. "
                             "Naive uses fixed 1000-character windows.")
    use_rerank = st.toggle("Cross-encoder reranking", value=True,
                           help="Re-score the top 20 vector matches with a cross-encoder. Adds ~1s.")
    config = Config(strategy=strategy, use_rerank=use_rerank)
    generating = has_api_key()
    if not generating:
        st.info("No ANTHROPIC_API_KEY set: showing retrieval results only.")

ask_tab, results_tab = st.tabs(["Ask", "Evaluation results"])

with ask_tab:
    example = st.selectbox("Try a real question from a FastAPI GitHub issue", [""] + example_questions())
    question = st.text_area("Your question", value=example, height=100)

    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving" + (" and generating..." if generating else "...")):
            if generating:
                result, generated = answer(question, config)
            else:
                result, generated = search(question, config), None

        if generated is not None:
            if generated.stop_reason == "refusal":
                st.warning("The model declined to answer this question.")
            else:
                st.markdown(generated.text)
                if generated.citations:
                    st.markdown("**Cited passages**")
                    for c in generated.citations:
                        st.markdown(f"- `{c.source_file}` — “{c.cited_text.strip()[:220]}”")

        timings = " · ".join(f"{stage} {secs:.2f}s" for stage, secs in result.timings.items())
        cost = f" · ${generated.cost_usd:.4f}" if generated is not None else ""
        st.caption(f"{config.name} · {timings}{cost}")

        st.markdown(f"**Retrieved context** (top {config.k_context} are sent to the model)")
        for rank, hit in enumerate(result.hits, 1):
            used = "✅ " if rank <= config.k_context else ""
            with st.expander(f"{used}{rank}. {hit.chunk.source_file} — {hit.chunk.section[:70]}  ·  score {hit.score:.3f}"):
                st.code(hit.chunk.text, language="python" if hit.chunk.chunk_type == "code" else "markdown")

with results_tab:
    results = Path("results/retrieval.md")
    if results.exists():
        st.markdown(results.read_text())

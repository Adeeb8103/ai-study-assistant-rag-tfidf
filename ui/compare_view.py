"""Side-by-side screen: one question, both approaches, every metric visible."""

import altair as alt
import pandas as pd
import streamlit as st

from core import evaluation
from ui import state, theme


def render() -> None:
    theme.hero(
        "Modern vs Traditional",
        "Ask one question and watch both pipelines answer it from the same chunks.",
    )

    doc = state.document_picker("Study material", key="cmp_doc")
    if not doc:
        st.warning("Upload a document first.", icon="📄")
        return

    question = st.text_input(
        "Question",
        placeholder="e.g. What are the main advantages discussed in this chapter?",
    )

    with st.expander("Optional: give a reference answer to measure accuracy"):
        st.caption(
            "Accuracy is measured as the semantic similarity between each answer "
            "and this reference. Leave it blank to compare speed, cost and "
            "naturalness only."
        )
        reference = st.text_area("Reference answer", height=100,
                                 label_visibility="collapsed")

    if not st.button("Run both approaches", type="primary", disabled=not question):
        _show_previous()
        return

    with st.spinner("Running both pipelines..."):
        try:
            traditional = state.traditional_engine(doc["id"]).answer(
                question, top_k=st.session_state["top_k"]
            )
            modern = state.modern_engine(doc["id"]).answer(
                question, memory=None, top_k=st.session_state["top_k"],
                use_memory=False,
            )
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

        scored = {
            "modern": evaluation.score(modern, question, reference),
            "traditional": evaluation.score(traditional, question, reference),
        }

    st.session_state["cmp_last"] = {
        "question": question,
        "reference": reference,
        "results": {"modern": modern, "traditional": traditional},
        "scored": scored,
    }
    _show_previous()


# ---------------------------------------------------------------------------
def _show_previous() -> None:
    payload = st.session_state.get("cmp_last")
    if not payload:
        st.info("Enter a question above and press **Run both approaches**.", icon="⚖️")
        return

    results, scored = payload["results"], payload["scored"]
    has_reference = bool(payload["reference"].strip())

    left, right = st.columns(2)
    _panel(left, results["modern"], scored["modern"], has_reference)
    _panel(right, results["traditional"], scored["traditional"], has_reference)

    st.divider()
    _charts(scored, has_reference)
    _overlap(results)


def _panel(column, result, marks, has_reference: bool) -> None:
    with column:
        st.markdown(theme.badge(result.approach), unsafe_allow_html=True)
        st.caption(result.model)

        if not result.ok:
            st.error(result.error)
            return

        st.markdown(result.answer)
        st.divider()

        metrics = [
            ("Time", f"{result.total_time:.2f}s",
             f"retrieval {result.retrieval_time * 1000:.0f}ms"),
            ("Cost", f"${result.cost:.5f}",
             f"{result.tokens_in + result.tokens_out} tokens"
             if result.tokens_in else "runs locally"),
            ("Natural", f"{marks.naturalness:.2f}", "readability 0-1"),
        ]
        if has_reference:
            metrics.append(("Accuracy", f"{marks.accuracy:.2f}", "vs reference"))
        theme.cards(metrics)

        label = "cosine" if result.approach == "traditional" else "similarity"
        with st.expander(f"Retrieved chunks ({len(result.sources)})"):
            theme.show_sources(result.sources, score_label=label)


def _charts(scored: dict, has_reference: bool) -> None:
    st.markdown("##### Metric comparison")

    metrics = [("Naturalness", "naturalness"), ("Groundedness", "groundedness")]
    if has_reference:
        metrics = [("Accuracy", "accuracy"), ("Keyword hit", "keyword_hit")] + metrics

    rows = [
        {
            "Metric": label,
            "Approach": "Modern (RAG + LLM)" if key == "modern" else "Traditional (TF-IDF)",
            "Score": getattr(scored[key], attribute),
        }
        for label, attribute in metrics
        for key in ("modern", "traditional")
    ]

    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Score:Q", title="Score (0-1)", scale=alt.Scale(domain=[0, 1])),
            xOffset="Approach:N",
            color=alt.Color(
                "Approach:N",
                scale=alt.Scale(
                    domain=["Modern (RAG + LLM)", "Traditional (TF-IDF)"],
                    range=["#4f46e5", "#0d9488"],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["Approach", "Metric", "Score"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")

    speed_rows = [
        {"Approach": "Modern (RAG + LLM)", "Stage": "Retrieval",
         "Seconds": round(scored["modern"].retrieval_time, 3)},
        {"Approach": "Modern (RAG + LLM)", "Stage": "LLM generation",
         "Seconds": round(scored["modern"].generation_time, 3)},
        {"Approach": "Traditional (TF-IDF)", "Stage": "Retrieval",
         "Seconds": round(scored["traditional"].retrieval_time, 3)},
        {"Approach": "Traditional (TF-IDF)", "Stage": "LLM generation", "Seconds": 0.0},
    ]

    st.markdown("##### Where the time goes")
    speed_chart = (
        alt.Chart(pd.DataFrame(speed_rows))
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("Approach:N", title=None),
            x=alt.X("Seconds:Q", title="Seconds"),
            color=alt.Color(
                "Stage:N",
                scale=alt.Scale(domain=["Retrieval", "LLM generation"],
                                range=["#0d9488", "#4f46e5"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["Approach", "Stage", "Seconds"],
        )
        .properties(height=140)
    )
    st.altair_chart(speed_chart, width="stretch")
    st.caption(
        "Retrieval is fast in both systems. Almost all of the modern approach's "
        "waiting time is the network round-trip to the language model - the price "
        "paid for a natural, written answer."
    )


def _overlap(results: dict) -> None:
    """Did the two retrieval methods actually find the same passages?"""
    modern_ids = {s["index"] for s in results["modern"].sources}
    traditional_ids = {s["index"] for s in results["traditional"].sources}

    if not modern_ids or not traditional_ids:
        return

    shared = modern_ids & traditional_ids
    union = modern_ids | traditional_ids

    st.markdown("##### Retrieval agreement")
    theme.cards([
        ("Shared chunks", str(len(shared)),
         f"of {len(union)} distinct chunks found"),
        ("Jaccard overlap", f"{len(shared) / len(union):.2f}",
         "1.00 = identical retrieval"),
        ("Modern only", str(len(modern_ids - traditional_ids)), "found by meaning"),
        ("Traditional only", str(len(traditional_ids - modern_ids)), "found by words"),
    ])
    st.caption(
        "A low overlap is the clearest evidence of the difference: embeddings "
        "match on meaning, so they can find the right passage even when it uses "
        "none of the question's words. TF-IDF only matches on shared vocabulary."
    )

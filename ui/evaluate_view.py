"""Evaluation screen: run a whole question set through both approaches."""

import altair as alt
import pandas as pd
import streamlit as st

from config import EXPORT_DIR, has_api_key
from core import database as db
from core import evaluation
from ui import state, theme

STARTER_QUESTIONS = [
    {"Question": "What is the main topic of this document?", "Reference answer": ""},
    {"Question": "List the key points covered.", "Reference answer": ""},
    {"Question": "Explain the most important concept in simple words.", "Reference answer": ""},
]


def render() -> None:
    theme.hero(
        "Evaluation and results",
        "Section 3 of the methodology: test both approaches on the same questions "
        "and compare accuracy, speed, cost and naturalness.",
    )

    run_tab, history_tab = st.tabs(["Run an experiment", "Past runs"])

    with run_tab:
        _run_experiment()
    with history_tab:
        _history()


# ---------------------------------------------------------------------------
def _run_experiment() -> None:
    doc = state.document_picker("Study material", key="eval_doc")
    if not doc:
        st.warning("Upload a document first.", icon="📄")
        return

    st.markdown("##### 1. Build the question set")
    st.caption(
        "Add a reference answer wherever you can - accuracy is measured against "
        "it. Rows without one still contribute speed, cost and naturalness."
    )

    if "eval_questions" not in st.session_state:
        st.session_state["eval_questions"] = pd.DataFrame(STARTER_QUESTIONS)

    col_gen, col_count, _ = st.columns([1.6, 1, 2])
    count = col_count.number_input("How many", 3, 10, 5, key="gen_count")

    if col_gen.button("Draft questions from the document", width="stretch",
                      disabled=not has_api_key()):
        with st.spinner("Reading the document and drafting questions..."):
            try:
                store = state.tfidf_store(state.user_id(), doc["id"])
                drafted = evaluation.generate_questions(
                    store.chunks, st.session_state["model"], int(count)
                )
            except Exception as exc:
                st.error(f"Could not draft questions: {exc}")
                drafted = []

        if drafted:
            st.session_state["eval_questions"] = pd.DataFrame([
                {"Question": d["question"], "Reference answer": d["answer"]}
                for d in drafted
            ])
            st.rerun()
        else:
            st.warning("The model did not return a usable question set. Try again.")

    if not has_api_key():
        st.caption("Question drafting needs an OpenRouter key. Type your own below.")

    edited = st.data_editor(
        st.session_state["eval_questions"],
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Question": st.column_config.TextColumn(width="medium", required=True),
            "Reference answer": st.column_config.TextColumn(width="large"),
        },
        key="eval_editor",
    )

    st.markdown("##### 2. Choose what to run")
    col_a, col_b, col_c = st.columns(3)
    run_modern = col_a.checkbox("Modern (RAG + LLM)", value=has_api_key(),
                                disabled=not has_api_key())
    run_traditional = col_b.checkbox("Traditional (TF-IDF)", value=True)
    use_judge = col_c.checkbox(
        "Add LLM judge", value=False, disabled=not has_api_key(),
        help="A second model marks each answer 0-5 for correctness and fluency. "
             "Slower and costs extra tokens, but it is a stronger accuracy measure "
             "than similarity alone.",
    )

    if not has_api_key():
        st.info(
            "No OpenRouter key configured, so only the traditional approach can "
            "run. Add OPENROUTER_API_KEY to the .env file to enable the modern one.",
            icon="🔑",
        )

    rows = [
        (str(r["Question"]).strip(), str(r.get("Reference answer") or "").strip())
        for _, r in edited.iterrows()
        if str(r["Question"]).strip()
    ]

    ready = rows and (run_modern or run_traditional)
    if not st.button("Run evaluation", type="primary", disabled=not ready):
        _show_results()
        return

    st.session_state["eval_questions"] = edited
    _execute(doc, rows, run_modern, run_traditional, use_judge)
    _show_results()


# ---------------------------------------------------------------------------
def _execute(doc: dict, rows: list[tuple[str, str]], run_modern: bool,
             run_traditional: bool, use_judge: bool) -> None:
    model = st.session_state["model"]
    judge_model = model if use_judge else None
    top_k = st.session_state["top_k"]

    try:
        traditional_engine = state.traditional_engine(doc["id"]) if run_traditional else None
        modern_engine = state.modern_engine(doc["id"]) if run_modern else None
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    total = len(rows) * sum([run_modern, run_traditional])
    bar = st.progress(0.0, text="Starting...")
    done = 0
    scored: list[evaluation.Scored] = []

    for question, reference in rows:
        if traditional_engine:
            bar.progress(done / total, text=f"Traditional: {question[:60]}")
            result = traditional_engine.answer(question, top_k=top_k)
            scored.append(evaluation.score(result, question, reference, judge_model))
            done += 1

        if modern_engine:
            bar.progress(done / total, text=f"Modern: {question[:60]}")
            result = modern_engine.answer(
                question, memory=None, top_k=top_k, use_memory=False
            )
            scored.append(evaluation.score(result, question, reference, judge_model))
            done += 1

    bar.progress(1.0, text="Scoring complete.")
    bar.empty()

    # Persist so the run can be shown again later or put in the report.
    run_id = db.create_eval_run(
        state.user_id(), doc["id"], f"{doc['title']} - {len(rows)} questions", model
    )
    for entry in scored:
        db.add_eval_result(
            run_id, question=entry.question, reference=entry.reference,
            approach=entry.approach, answer=entry.answer, latency=entry.latency,
            tokens_in=entry.tokens_in, tokens_out=entry.tokens_out,
            cost=entry.cost, accuracy=entry.accuracy,
            keyword_hit=entry.keyword_hit, naturalness=entry.naturalness,
        )

    st.session_state["eval_scored"] = scored
    st.session_state["eval_run_id"] = run_id


# ---------------------------------------------------------------------------
def _show_results() -> None:
    scored: list[evaluation.Scored] = st.session_state.get("eval_scored", [])
    if not scored:
        st.info("Run an evaluation to see results here.", icon="📊")
        return

    errors = [s for s in scored if s.error]
    if errors:
        st.warning(
            f"{len(errors)} answer(s) failed - for example: {errors[0].error}",
            icon="⚠️",
        )

    summary = evaluation.aggregate(scored)

    st.divider()
    st.markdown("### Results")

    _summary_cards(summary)
    _summary_table(summary)
    _summary_charts(scored, summary)

    st.markdown("##### Findings")
    for line in evaluation.verdict(summary):
        st.markdown(f"- {line}")

    st.markdown("##### Per-question detail")
    table = pd.DataFrame([s.to_row() for s in scored])
    st.dataframe(table, width="stretch", hide_index=True)

    run_id = st.session_state.get("eval_run_id", 0)
    csv_path = EXPORT_DIR / f"evaluation_run_{run_id}.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8")

    st.download_button(
        "Download results as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"evaluation_run_{run_id}.csv",
        mime="text/csv",
    )
    st.caption(f"Also saved to `{csv_path}` for your report appendix.")


def _summary_cards(summary: dict) -> None:
    modern = summary.get("modern")
    traditional = summary.get("traditional")

    if modern and traditional:
        speedup = modern["latency"] / max(traditional["latency"], 1e-6)
        theme.cards([
            ("Questions", str(modern["questions"]), "run through both"),
            ("Speed gap", f"{speedup:.0f}x",
             "traditional is faster" if speedup > 1 else "modern is faster"),
            ("Accuracy gap", f"{modern['accuracy'] - traditional['accuracy']:+.2f}",
             "modern minus traditional"),
            ("Total cost", f"${modern['cost']:.4f}", "modern only"),
        ])
    elif traditional:
        theme.cards([
            ("Questions", str(traditional["questions"]), "traditional only"),
            ("Avg time", f"{traditional['latency']:.3f}s", "per question"),
            ("Naturalness", f"{traditional['naturalness']:.2f}", "0-1"),
            ("Cost", "$0.0000", "runs locally"),
        ])


def _summary_table(summary: dict) -> None:
    names = {"modern": "Modern (RAG + LLM)", "traditional": "Traditional (TF-IDF)"}
    metrics = [
        ("Accuracy (vs reference)", "accuracy", "{:.3f}"),
        ("Keyword coverage", "keyword_hit", "{:.3f}"),
        ("Groundedness", "groundedness", "{:.3f}"),
        ("Naturalness", "naturalness", "{:.3f}"),
        ("Avg time per question (s)", "latency", "{:.3f}"),
        ("Avg retrieval time (s)", "retrieval_time", "{:.4f}"),
        ("Total tokens", "tokens", "{:,}"),
        ("Total cost (USD)", "cost", "${:.5f}"),
        ("Judge - correctness /5", "judge_correctness", "{:.2f}"),
        ("Judge - fluency /5", "judge_fluency", "{:.2f}"),
    ]

    rows = []
    for label, key, fmt in metrics:
        if not any(key in entry for entry in summary.values()):
            continue
        row = {"Metric": label}
        for approach, entry in summary.items():
            value = entry.get(key)
            row[names[approach]] = fmt.format(value) if value is not None else "-"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _summary_charts(scored: list, summary: dict) -> None:
    names = {"modern": "Modern (RAG + LLM)", "traditional": "Traditional (TF-IDF)"}
    quality = [
        ("Accuracy", "accuracy"), ("Keyword hit", "keyword_hit"),
        ("Groundedness", "groundedness"), ("Naturalness", "naturalness"),
    ]

    rows = [
        {"Metric": label, "Approach": names[approach], "Score": entry[key]}
        for label, key in quality
        for approach, entry in summary.items()
    ]

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("**Quality metrics (higher is better)**")
        chart = (
            alt.Chart(pd.DataFrame(rows))
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
                xOffset="Approach:N",
                color=alt.Color("Approach:N", scale=_scale(), legend=alt.Legend(
                    orient="top", title=None)),
                tooltip=["Approach", "Metric", "Score"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, width="stretch")

    with right:
        st.markdown("**Response time per question**")
        time_rows = [
            {"Approach": names[s.approach], "Question": s.question[:35],
             "Seconds": round(s.latency, 3)}
            for s in scored if not s.error
        ]
        time_chart = (
            alt.Chart(pd.DataFrame(time_rows))
            .mark_circle(size=90, opacity=0.75)
            .encode(
                y=alt.Y("Approach:N", title=None),
                x=alt.X("Seconds:Q", title="Seconds", scale=alt.Scale(type="symlog")),
                color=alt.Color("Approach:N", scale=_scale(), legend=None),
                tooltip=["Approach", "Question", "Seconds"],
            )
            .properties(height=300)
        )
        st.altair_chart(time_chart, width="stretch")


def _scale():
    return alt.Scale(
        domain=["Modern (RAG + LLM)", "Traditional (TF-IDF)"],
        range=["#4f46e5", "#0d9488"],
    )


# ---------------------------------------------------------------------------
def _history() -> None:
    runs = db.list_eval_runs(state.user_id())
    if not runs:
        st.info("No saved evaluation runs yet.", icon="🗂️")
        return

    labels = {r["id"]: f"#{r['id']} · {r['name']} · {r['created_at']}" for r in runs}
    run_id = st.selectbox("Saved run", list(labels), format_func=lambda i: labels[i])

    results = db.list_eval_results(run_id)
    if not results:
        st.caption("This run has no rows.")
        return

    table = pd.DataFrame([
        {
            "Question": r["question"],
            "Approach": "Modern" if r["approach"] == "modern" else "Traditional",
            "Answer": r["answer"],
            "Accuracy": r["accuracy"],
            "Naturalness": r["naturalness"],
            "Time (s)": round(r["latency"], 3),
            "Cost (USD)": round(r["cost"], 6),
        }
        for r in results
    ])
    st.dataframe(table, width="stretch", hide_index=True)

    col_download, col_delete = st.columns([1, 1])
    col_download.download_button(
        "Download CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"evaluation_run_{run_id}.csv",
        mime="text/csv",
        width="stretch",
    )
    if col_delete.button("Delete this run", width="stretch"):
        db.delete_eval_run(run_id, state.user_id())
        st.rerun()

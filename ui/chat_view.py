"""Chat screen - the actual study assistant."""

import streamlit as st

from core import database as db
from core.memory import ConversationMemory
from ui import state, theme

APPROACHES = {
    "Modern - RAG + LLM (OpenRouter)": "modern",
    "Traditional - TF-IDF + cosine similarity": "traditional",
}


def render() -> None:
    theme.hero(
        "Ask your notes",
        "Answers come only from the material you uploaded, not from the internet.",
    )

    doc = _controls()
    if not doc:
        return

    session_id = _ensure_session(doc)
    messages = db.list_messages(session_id)

    _render_history(messages)
    _handle_input(doc, session_id, messages)


# ---------------------------------------------------------------------------
def _controls() -> dict | None:
    docs = state.my_documents()
    if not docs:
        st.warning("Upload a document first.", icon="📄")
        if st.button("Go to library", type="primary"):
            state.goto("Library")
            st.rerun()
        return None

    col_doc, col_mode, col_new = st.columns([2.2, 2.2, 1])

    with col_doc:
        doc = state.document_picker("Study material", key="chat_doc")

    with col_mode:
        label = st.selectbox(
            "Answering approach", list(APPROACHES),
            help="Switch between the two methods this dissertation compares.",
        )
        st.session_state["chat_approach"] = APPROACHES[label]

    with col_new:
        st.write("")
        st.write("")
        if st.button("New chat", width="stretch"):
            st.session_state["session_id"] = None
            st.rerun()

    return doc


def _ensure_session(doc: dict) -> int:
    """Reuse the current chat session, or open one for this document."""
    session_id = st.session_state.get("session_id")

    if session_id:
        row = db.query_one(
            "SELECT * FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, state.user_id()),
        )
        if row:
            return session_id

    session_id = db.create_session(
        state.user_id(), doc["id"], f"Chat about {doc['title']}"
    )
    st.session_state["session_id"] = session_id
    return session_id


# ---------------------------------------------------------------------------
def _render_history(messages: list[dict]) -> None:
    if not messages:
        st.caption(
            "Try: *What is the main topic of this document?* then follow up with "
            "*explain that in simpler words* - the follow-up only works because "
            "the assistant remembers the conversation."
        )
        return

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(theme.badge(message["approach"]), unsafe_allow_html=True)
            st.markdown(message["content"])

            if message["role"] == "assistant":
                _answer_footer(message)


def _answer_footer(message: dict) -> None:
    bits = [f"{message['latency']:.2f}s"]
    if message["tokens_in"] or message["tokens_out"]:
        bits.append(f"{message['tokens_in']} in / {message['tokens_out']} out tokens")
        bits.append(f"${message['cost']:.5f}")
    else:
        bits.append("no tokens, no cost (runs locally)")
    st.caption(" · ".join(bits))

    if message["sources"]:
        label = "cosine" if message["approach"] == "traditional" else "similarity"
        with st.expander(f"Sources used ({len(message['sources'])} chunks)"):
            theme.show_sources(message["sources"], score_label=label)


# ---------------------------------------------------------------------------
def _handle_input(doc: dict, session_id: int, messages: list[dict]) -> None:
    question = st.chat_input("Ask a question about this document...")
    if not question:
        return

    approach = st.session_state.get("chat_approach", "modern")

    with st.chat_message("user"):
        st.markdown(question)

    db.add_message(session_id, "user", question)

    with st.chat_message("assistant"):
        st.markdown(theme.badge(approach), unsafe_allow_html=True)
        placeholder = st.empty()
        placeholder.markdown("_Searching your notes..._")

        try:
            result = _run(doc, question, approach, messages, session_id)
        except FileNotFoundError as exc:
            placeholder.error(str(exc))
            return

        if not result.ok:
            placeholder.error(result.error)
            if approach == "modern":
                st.info(
                    "The traditional TF-IDF approach needs no API key - switch to "
                    "it in the dropdown above to keep working.",
                    icon="💡",
                )
            return

        placeholder.markdown(result.answer)

        if result.standalone_question and result.standalone_question != question:
            st.caption(
                f"Memory rewrote the question for search: *{result.standalone_question}*"
            )

        db.add_message(
            session_id, "assistant", result.answer, approach=approach,
            latency=result.total_time, tokens_in=result.tokens_in,
            tokens_out=result.tokens_out, cost=result.cost,
            sources=result.sources,
        )

    st.rerun()


def _run(doc: dict, question: str, approach: str,
         messages: list[dict], session_id: int):
    """Dispatch to the chosen engine, wiring memory in for the modern one."""
    if approach == "traditional":
        engine = state.traditional_engine(doc["id"])
        return engine.answer(question, top_k=st.session_state["top_k"])

    session = db.query_one(
        "SELECT summary FROM chat_sessions WHERE id = ?", (session_id,)
    ) or {}
    memory = ConversationMemory.from_messages(messages, session.get("summary", ""))

    engine = state.modern_engine(doc["id"])

    # Once the conversation gets long, fold the older turns into a summary so
    # the prompt stays small without losing the thread.
    if st.session_state["use_memory"] and memory.needs_summary() and not memory.summary:
        summary = engine.summarise(memory)
        if summary:
            memory.summary = summary
            db.update_session_summary(session_id, summary)

    return engine.answer(
        question,
        memory=memory,
        top_k=st.session_state["top_k"],
        use_memory=st.session_state["use_memory"],
    )

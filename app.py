"""
Personal AI Study Assistant - Streamlit entry point.

Run with:  streamlit run app.py
"""

import streamlit as st

from config import APP_TITLE, AVAILABLE_MODELS, has_api_key
from ui import (
    auth_view,
    chat_view,
    compare_view,
    evaluate_view,
    library_view,
    methodology_view,
    state,
    theme,
)

PAGES = {
    "Chat": ("💬", chat_view),
    "Library": ("📚", library_view),
    "Compare": ("⚖️", compare_view),
    "Evaluation": ("📊", evaluate_view),
    "How it works": ("🧭", methodology_view),
}


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    theme.inject()
    state.init()

    if not state.logged_in():
        auth_view.render()
        return

    state.warm_embedder()
    _sidebar()

    page = st.session_state.get("page", "Chat")
    PAGES.get(page, PAGES["Chat"])[1].render()


# ---------------------------------------------------------------------------
def _sidebar() -> None:
    user = state.user()

    with st.sidebar:
        st.markdown(f"### 📘 {APP_TITLE}")
        st.caption(f"Signed in as **{user['name']}**")
        st.divider()

        current = st.session_state.get("page", "Chat")
        for name, (icon, _) in PAGES.items():
            if st.button(
                f"{icon}  {name}",
                key=f"nav_{name}",
                width="stretch",
                type="primary" if name == current else "secondary",
            ):
                state.goto(name)
                st.rerun()

        st.divider()
        st.markdown("##### Settings")

        st.session_state["model"] = st.selectbox(
            "Language model",
            list(AVAILABLE_MODELS),
            index=list(AVAILABLE_MODELS).index(st.session_state["model"]),
            help="Used by the modern approach through OpenRouter. Prices differ, "
                 "which shows up directly in the cost comparison.",
        )

        price = AVAILABLE_MODELS[st.session_state["model"]]
        st.caption(
            f"${price['input']:.3f} per 1M input tokens · "
            f"${price['output']:.3f} per 1M output tokens"
        )

        st.session_state["top_k"] = st.slider(
            "Chunks to retrieve (top-k)", 1, 10, st.session_state["top_k"],
            help="Both approaches retrieve the same number, keeping the "
                 "comparison fair.",
        )

        st.session_state["use_memory"] = st.toggle(
            "Conversation memory", value=st.session_state["use_memory"],
            help="Turn off to see how much worse follow-up questions get "
                 "without it.",
        )

        st.divider()
        _connection_status()

        st.divider()
        docs = state.my_documents()
        st.caption(f"{len(docs)} document(s) · "
                   f"{sum(d['num_chunks'] for d in docs)} chunks indexed")

        if st.button("Sign out", width="stretch"):
            state.sign_out()
            st.rerun()


def _connection_status() -> None:
    if not has_api_key():
        st.error("No OpenRouter key", icon="🔑")
        st.caption(
            "Add `OPENROUTER_API_KEY` to `.env` to enable the modern approach. "
            "The traditional TF-IDF approach works without it."
        )
        return

    st.success("OpenRouter key loaded", icon="🔑")

    if st.button("Test connection", width="stretch"):
        from core import llm
        with st.spinner("Contacting OpenRouter..."):
            ok, message = llm.check_connection(st.session_state["model"])
        (st.success if ok else st.error)(message)


if __name__ == "__main__":
    main()

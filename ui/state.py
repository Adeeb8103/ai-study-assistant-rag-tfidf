"""Session-state helpers and cached loaders shared by every screen."""

import streamlit as st

from config import DEFAULT_MODEL, DEFAULT_TOP_K
from core import database as db
from core import documents
from core.engines import ModernRAG, TraditionalQA

DEFAULTS = {
    "user": None,
    "page": "Chat",
    "model": DEFAULT_MODEL,
    "top_k": DEFAULT_TOP_K,
    "use_memory": True,
    "active_doc": None,
    "session_id": None,
    "auth_tab": "login",
}


def init() -> None:
    db.init_db()
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def user() -> dict | None:
    return st.session_state.get("user")


def user_id() -> int:
    current = user()
    return current["id"] if current else 0


def logged_in() -> bool:
    return user() is not None


def sign_in(record: dict) -> None:
    st.session_state["user"] = record
    st.session_state["page"] = "Library"


def sign_out() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init()


def goto(page: str) -> None:
    st.session_state["page"] = page


# ---------------------------------------------------------------------------
# Cached index loading - a FAISS index and a TF-IDF model are expensive to read,
# so they are cached per (user, document) for the life of the server process.
# ---------------------------------------------------------------------------
@st.cache_resource(
    show_spinner="Loading the embedding model (first run only, ~30 seconds)..."
)
def warm_embedder():
    """Load the sentence-transformer up front so no later action stalls."""
    from core.vector_store import get_embedder
    return get_embedder()


@st.cache_resource(show_spinner=False)
def vector_store(uid: int, doc_id: int):
    return documents.load_vector_store(uid, doc_id)


@st.cache_resource(show_spinner=False)
def tfidf_store(uid: int, doc_id: int):
    return documents.load_tfidf_store(uid, doc_id)


def clear_caches() -> None:
    vector_store.clear()
    tfidf_store.clear()


def modern_engine(doc_id: int, model: str | None = None) -> ModernRAG:
    return ModernRAG(
        vector_store(user_id(), doc_id),
        model=model or st.session_state["model"],
    )


def traditional_engine(doc_id: int) -> TraditionalQA:
    return TraditionalQA(tfidf_store(user_id(), doc_id))


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def my_documents() -> list[dict]:
    return db.list_documents(user_id())


def document_picker(label: str = "Study material", key: str = "doc_pick",
                    help_text: str | None = None) -> dict | None:
    """Sidebar/inline selector that remembers the last chosen document."""
    docs = my_documents()
    if not docs:
        return None

    ids = [d["id"] for d in docs]
    titles = {d["id"]: f"{d['title']}  ({d['num_chunks']} chunks)" for d in docs}

    active = st.session_state.get("active_doc")
    index = ids.index(active) if active in ids else 0

    chosen = st.selectbox(
        label, ids, index=index, format_func=lambda i: titles[i],
        key=key, help=help_text,
    )
    st.session_state["active_doc"] = chosen
    return next(d for d in docs if d["id"] == chosen)

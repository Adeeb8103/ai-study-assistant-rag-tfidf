"""Library screen: upload study material and inspect what the pipeline built."""

import pandas as pd
import streamlit as st

from config import CHUNK_OVERLAP, CHUNK_SIZE, SUPPORTED_EXTENSIONS
from core import documents
from ui import state, theme


def render() -> None:
    theme.hero(
        "Your study library",
        "Upload notes, books or PDFs. Each file is cleaned, chunked and indexed "
        "for both approaches at once.",
    )

    upload_tab, manage_tab, inspect_tab = st.tabs(
        ["Upload material", "Manage documents", "Inspect chunks"]
    )

    with upload_tab:
        _upload()
    with manage_tab:
        _manage()
    with inspect_tab:
        _inspect()


# ---------------------------------------------------------------------------
def _upload() -> None:
    files = st.file_uploader(
        "Choose one or more files",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        help="PDF, Word (.docx), plain text and Markdown are supported.",
    )

    with st.expander("Chunking settings (Methodology, step 2)"):
        st.caption(
            "Long documents are cut into overlapping pieces. Smaller chunks give "
            "sharper retrieval; a larger overlap stops a sentence being lost at a "
            "boundary. Both approaches use whatever you set here."
        )
        col_a, col_b = st.columns(2)
        chunk_size = col_a.slider("Chunk size (characters)", 400, 2000, CHUNK_SIZE, 100)
        overlap = col_b.slider("Chunk overlap (characters)", 0, 400, CHUNK_OVERLAP, 25)

    if not files:
        st.info(
            "No file selected yet. Any PDF of your notes will do - a lecture "
            "handout is a good demo.",
            icon="📄",
        )
        return

    if not st.button("Process and index", type="primary"):
        return

    for file in files:
        st.markdown(f"**{file.name}**")
        bar = st.progress(0.0, text="Starting...")

        def report(fraction: float, label: str, _bar=bar) -> None:
            _bar.progress(min(fraction, 1.0), text=label)

        try:
            result = documents.ingest(
                user_id=state.user_id(),
                filename=file.name,
                data=file.getvalue(),
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                progress=report,
            )
        except Exception as exc:
            bar.empty()
            st.error(f"Could not process {file.name}: {exc}")
            continue

        bar.empty()
        st.success(f"Indexed {file.name}.")

        timings = result["timings"]
        theme.cards([
            ("Pages", str(result["pages"]), "extracted"),
            ("Characters", f"{result['clean_chars']:,}",
             f"from {result['raw_chars']:,} raw"),
            ("Chunks", str(result["num_chunks"]), f"~{chunk_size} chars each"),
            ("Vector index", f"{timings.get('vector_index', 0):.1f}s", "FAISS build"),
            ("TF-IDF model", f"{timings.get('tfidf_index', 0):.2f}s", "fit time"),
        ])

        st.caption(
            "Notice the difference in build time - embedding every chunk with a "
            "neural model is far slower than fitting a TF-IDF matrix. That is the "
            "first trade-off the dissertation reports."
        )
        st.session_state["active_doc"] = result["document_id"]

    state.clear_caches()


# ---------------------------------------------------------------------------
def _manage() -> None:
    docs = state.my_documents()

    if not docs:
        st.info("Nothing uploaded yet.", icon="📭")
        return

    table = pd.DataFrame([
        {
            "Title": d["title"],
            "Type": d["filetype"].upper(),
            "Pages": d["num_pages"],
            "Characters": d["num_chars"],
            "Chunks": d["num_chunks"],
            "Uploaded": d["uploaded_at"],
        }
        for d in docs
    ])
    st.dataframe(table, width="stretch", hide_index=True)

    st.markdown("##### Actions")
    for doc in docs:
        col_title, col_chat, col_compare, col_delete = st.columns([4, 1.2, 1.2, 1.2])
        col_title.markdown(
            f"**{doc['title']}**  \n"
            f"<span style='color:#64748b;font-size:.8rem'>"
            f"{doc['num_chunks']} chunks · {doc['num_pages']} pages</span>",
            unsafe_allow_html=True,
        )

        if col_chat.button("Chat", key=f"chat_{doc['id']}", width="stretch"):
            st.session_state["active_doc"] = doc["id"]
            state.goto("Chat")
            st.rerun()

        if col_compare.button("Compare", key=f"cmp_{doc['id']}", width="stretch"):
            st.session_state["active_doc"] = doc["id"]
            state.goto("Compare")
            st.rerun()

        if col_delete.button("Delete", key=f"del_{doc['id']}", width="stretch"):
            documents.remove(state.user_id(), doc["id"])
            state.clear_caches()
            if st.session_state.get("active_doc") == doc["id"]:
                st.session_state["active_doc"] = None
            st.rerun()


# ---------------------------------------------------------------------------
def _inspect() -> None:
    doc = state.document_picker("Document to inspect", key="inspect_pick")
    if not doc:
        st.info("Upload something first.", icon="📭")
        return

    try:
        tfidf = state.tfidf_store(state.user_id(), doc["id"])
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.markdown("##### The chunks both approaches search")
    st.caption(
        "This is what the system actually stores. The modern approach turns each "
        "of these into a 384-dimension vector; the traditional approach turns each "
        "into a sparse TF-IDF word vector."
    )

    total = len(tfidf.chunks)
    position = st.slider("Chunk", 0, max(total - 1, 0), 0) if total > 1 else 0
    chunk = tfidf.chunks[position]

    st.markdown(
        f"<div class='source'><div class='meta'>chunk {chunk['index']} of {total - 1} "
        f"· page {chunk['page']} · {len(chunk['text'])} characters</div>"
        f"{chunk['text']}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("##### Highest-weighted TF-IDF terms in this document")
    st.caption(
        "TF-IDF gives a high weight to words that are frequent in this document "
        "but rare across the rest of it - these are the words that carry meaning."
    )
    terms = pd.DataFrame(tfidf.top_terms(20), columns=["Term", "Total TF-IDF weight"])
    st.bar_chart(terms.set_index("Term"), height=280)

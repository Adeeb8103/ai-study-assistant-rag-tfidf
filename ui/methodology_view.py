"""Methodology screen - the page to present during the viva."""

import numpy as np
import pandas as pd
import streamlit as st

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from core.engines import SYSTEM_PROMPT
from ui import state, theme

PIPELINE = [
    ("1. Collect the material",
     "Notes, books and PDFs are uploaded. Text is pulled out page by page with "
     "pypdf (or python-docx for Word files)."),
    ("2. Clean the text",
     "Unicode is normalised, words hyphenated across line breaks are stitched "
     "back together, and header/footer page numbers are stripped out."),
    (f"3. Chunk it ({CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap)",
     "The document is split on the largest natural boundary that fits - "
     "paragraph, then line, then sentence - so chunks never cut a word in half. "
     "An overlap carries the tail of each chunk into the next."),
    ("4a. Modern: embed and store",
     f"Each chunk becomes a 384-dimension vector using {EMBEDDING_MODEL}. The "
     "vectors are L2-normalised and written to a FAISS inner-product index, so "
     "the score FAISS returns is the cosine similarity."),
    ("4b. Traditional: fit TF-IDF",
     "The same chunks are fitted with a TF-IDF vectoriser (unigrams and bigrams, "
     "English stop words removed, words reduced to their stem, sublinear term "
     "frequency), producing a sparse word-weight matrix. Each chunk is also "
     "indexed as short overlapping sentence windows in that same space."),
    ("5. Retrieve",
     "The question is projected into the same space as the chunks. The modern "
     "path searches FAISS by meaning; the traditional path computes cosine "
     "similarity against every TF-IDF row and ranks them, scoring a chunk by its "
     "best-matching sentence window as well as the chunk as a whole, so a long "
     "chunk is not penalised for the words it holds outside the relevant part."),
    ("6a. Modern: generate",
     "The top chunks plus the conversation memory are packed into a prompt and "
     "sent to a language model through OpenRouter, which writes the answer and "
     "cites the page."),
    ("6b. Traditional: extract",
     "There is no language model. Sentences inside the best chunks are re-scored "
     "against the question and the closest ones are returned in their original "
     "order."),
    ("7. Compare",
     "Both answers are scored on accuracy, speed, cost and naturalness, and the "
     "results are charted and exported."),
]

COMPARISON_ROWS = [
    ("How a passage is found", "Meaning (dense embeddings)", "Shared words (sparse TF-IDF)"),
    ("Where the answer comes from", "Written by a language model", "Copied from the notes"),
    ("Handles reworded questions", "Yes - synonyms still match", "Poorly - needs word overlap"),
    ("Conversation memory", "Yes - buffer plus summary", "No - each question is isolated"),
    ("Typical response time", "1-4 seconds (network bound)", "Milliseconds"),
    ("Running cost", "Per token, billed by OpenRouter", "Free - runs on the local machine"),
    ("Works offline", "No", "Yes"),
    ("Main weakness", "Cost, latency, external dependency", "Basic phrasing, literal matching"),
]


def render() -> None:
    theme.hero(
        "How the system works",
        "Aim, problem, methodology and the trade-off between the two approaches.",
    )

    overview, pipeline, trace, prompt = st.tabs(
        ["Aim and problem", "Methodology", "Live pipeline trace", "Prompt design"]
    )

    with overview:
        _overview()
    with pipeline:
        _pipeline()
    with trace:
        _trace()
    with prompt:
        _prompt()


# ---------------------------------------------------------------------------
def _overview() -> None:
    st.markdown("#### Aim")
    st.markdown(
        "Build a personal AI study assistant that works like ChatGPT but answers "
        "from **the student's own notes, books and study material** rather than "
        "from general public knowledge - and give it a memory so it can follow a "
        "conversation and give personalised help."
    )

    st.markdown("#### The problem being solved")
    st.markdown(
        "Students have a lot of material and no quick way to search or understand "
        "it. Re-reading everything to find one small point wastes time. This "
        "assistant reads the uploaded notes and answers only from what is "
        "actually in them."
    )

    st.markdown("#### Two solutions, built and compared")
    left, right = st.columns(2)

    with left:
        st.markdown(theme.badge("modern"), unsafe_allow_html=True)
        st.markdown(
            """
**OpenRouter API + RAG + vector store**

Notes are broken into chunks, converted into vectors and stored. A question is
converted the same way, the most relevant chunks are found, and a language model
turns them into a clear, human-like answer. A memory component keeps track of
the ongoing conversation.
            """
        )

    with right:
        st.markdown(theme.badge("traditional"), unsafe_allow_html=True)
        st.markdown(
            """
**TF-IDF + cosine similarity**

No large external model. TF-IDF works out which words matter in the documents,
cosine similarity measures how closely the question matches each piece of text,
and the closest passage is returned. Simpler, cheaper, fully under our control.
            """
        )

    st.markdown("#### Side by side")
    st.dataframe(
        pd.DataFrame(COMPARISON_ROWS,
                     columns=["", "Modern (RAG + LLM)", "Traditional (TF-IDF)"]),
        width="stretch", hide_index=True,
    )


def _pipeline() -> None:
    st.caption("Every stage below is a real function in the codebase.")
    for title, description in PIPELINE:
        traditional = title.startswith(("4b", "6b"))
        css = "step step-t" if traditional else "step"
        st.markdown(
            f'<div class="{css}"><h4>{title}</h4><p>{description}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("##### Where each stage lives")
    st.dataframe(
        pd.DataFrame([
            ("Extract, clean, chunk", "core/ingestion.py"),
            ("Embeddings + FAISS vector store", "core/vector_store.py"),
            ("TF-IDF + cosine similarity", "core/tfidf_store.py"),
            ("OpenRouter client and cost tracking", "core/llm.py"),
            ("Conversation memory", "core/memory.py"),
            ("The two answering engines", "core/engines.py"),
            ("Accuracy / speed / cost / naturalness metrics", "core/evaluation.py"),
        ], columns=["Stage", "File"]),
        width="stretch", hide_index=True,
    )


# ---------------------------------------------------------------------------
def _trace() -> None:
    """Show what actually happens to one question, stage by stage."""
    doc = state.document_picker("Study material", key="trace_doc")
    if not doc:
        st.warning("Upload a document first.", icon="📄")
        return

    question = st.text_input(
        "Question to trace", value="What is this document about?", key="trace_q"
    )
    if not st.button("Trace the pipeline", type="primary"):
        return

    try:
        vectors = state.vector_store(state.user_id(), doc["id"])
        tfidf = state.tfidf_store(state.user_id(), doc["id"])
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.markdown("##### Stage 1 - the question becomes a vector")

    from core.vector_store import embed
    query_vector = embed([question])[0]

    col_a, col_b = st.columns([1, 2])
    with col_a:
        theme.cards([
            ("Dimensions", str(len(query_vector)), "dense embedding"),
            ("Vector length", f"{np.linalg.norm(query_vector):.3f}", "L2-normalised"),
        ])
    with col_b:
        st.caption("First 40 of the 384 dimensions:")
        st.line_chart(pd.DataFrame({"value": query_vector[:40]}), height=140)

    sparse = tfidf.query_vector(question)
    vocabulary = tfidf.vectorizer.get_feature_names_out()
    active = [(tfidf.readable(vocabulary[i]), round(float(sparse[0, i]), 3))
              for i in sparse.nonzero()[1]]

    st.caption(
        f"The same question as a TF-IDF vector: {sparse.shape[1]:,} possible terms, "
        f"but only {len(active)} are non-zero."
    )
    if active:
        st.dataframe(
            pd.DataFrame(sorted(active, key=lambda x: -x[1]),
                         columns=["Term", "TF-IDF weight"]),
            width="stretch", hide_index=True, height=180,
        )
    else:
        st.warning(
            "None of the question's words appear in this document's vocabulary, "
            "so TF-IDF cannot match anything - a concrete example of the "
            "traditional approach's main weakness.",
            icon="⚠️",
        )

    st.markdown("##### Stage 2 - retrieval, side by side")
    top_k = st.session_state["top_k"]
    left, right = st.columns(2)

    with left:
        st.markdown(theme.badge("modern"), unsafe_allow_html=True)
        theme.show_sources(vectors.search(question, top_k), "cosine")
    with right:
        st.markdown(theme.badge("traditional"), unsafe_allow_html=True)
        theme.show_sources(tfidf.search(question, top_k), "cosine")

    st.markdown("##### Stage 3 - what the language model is sent")
    context = "\n\n".join(
        f"[page {s['page']}] {s['text'][:300]}..."
        for s in vectors.search(question, top_k)
    )
    st.code(
        f"SYSTEM:\n{SYSTEM_PROMPT}\n\n"
        f"USER:\nCONTEXT:\n{context}\n\nQUESTION: {question}",
        language="text",
    )
    st.caption(
        "Only the retrieved chunks reach the model. That is what stops it from "
        "answering out of general knowledge - and why it can cite a page."
    )


def _prompt() -> None:
    st.markdown("#### The system prompt")
    st.caption(
        "Retrieval alone does not guarantee the answer stays inside the notes - "
        "the instructions do the rest of that work."
    )
    st.code(SYSTEM_PROMPT, language="text")

    st.markdown("#### How memory is built")
    st.markdown(
        """
Three mechanisms work together so a follow-up question makes sense:

1. **Question rewriting.** Before retrieval, a follow-up such as *"explain that
   in simpler words"* is rewritten into a standalone question using the recent
   history. Without this the vector search would have nothing meaningful to
   match against.
2. **Buffer memory.** The last few question/answer pairs are replayed to the
   model as real chat turns.
3. **Summary memory.** Once the conversation grows past that window, the older
   turns are compressed by the model into a short paragraph, so a long session
   still fits in the prompt without losing the thread.

The traditional approach has none of this - each question is answered in
isolation, which is one of the clearest practical differences between the two.
        """
    )

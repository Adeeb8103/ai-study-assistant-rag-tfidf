"""Shared styling and small reusable UI pieces."""

import streamlit as st

CSS = """
<style>
  :root {
    --brand:        #4f46e5;
    --brand-soft:   #eef2ff;
    --modern:       #4f46e5;
    --traditional:  #0d9488;
    --ink:          #0f172a;
    --muted:        #64748b;
    --line:         #e2e8f0;
    --surface:      #ffffff;
    --surface-alt:  #f8fafc;
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --brand-soft:  #1e1b4b;
      --ink:         #e2e8f0;
      --muted:       #94a3b8;
      --line:        #334155;
      --surface:     #1e293b;
      --surface-alt: #0f172a;
    }
  }

  .block-container { padding-top: 2.2rem; max-width: 1200px; }

  /* ---- hero ---- */
  .hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #0d9488 100%);
    color: #fff;
    padding: 1.6rem 1.8rem;
    border-radius: 16px;
    margin-bottom: 1.4rem;
  }
  .hero h1 { margin: 0; font-size: 1.65rem; font-weight: 700; }
  .hero p  { margin: .35rem 0 0; opacity: .9; font-size: .95rem; }

  /* ---- metric cards ---- */
  .cards { display: flex; gap: .8rem; flex-wrap: wrap; margin: .6rem 0 1rem; }
  .card {
    flex: 1 1 150px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: .85rem 1rem;
  }
  .card .label {
    font-size: .72rem; text-transform: uppercase; letter-spacing: .05em;
    color: var(--muted); font-weight: 600;
  }
  .card .value { font-size: 1.5rem; font-weight: 700; color: var(--ink); line-height: 1.3; }
  .card .hint  { font-size: .75rem; color: var(--muted); }

  /* ---- approach badges ---- */
  .badge {
    display: inline-block; padding: .18rem .6rem; border-radius: 999px;
    font-size: .74rem; font-weight: 600; letter-spacing: .02em;
  }
  .badge-modern      { background: #eef2ff; color: #4338ca; }
  .badge-traditional { background: #ccfbf1; color: #0f766e; }

  /* ---- source chunk ---- */
  .source {
    border-left: 3px solid var(--brand);
    background: var(--surface-alt);
    padding: .6rem .85rem; margin: .45rem 0;
    border-radius: 0 8px 8px 0; font-size: .85rem;
  }
  .source .meta { font-size: .72rem; color: var(--muted); font-weight: 600; margin-bottom: .25rem; }

  /* ---- pipeline steps ---- */
  .step {
    background: var(--surface); border: 1px solid var(--line);
    border-left: 4px solid var(--brand);
    border-radius: 10px; padding: .8rem 1rem; margin-bottom: .6rem;
  }
  .step h4 { margin: 0 0 .2rem; font-size: .95rem; }
  .step p  { margin: 0; font-size: .85rem; color: var(--muted); }
  .step-t  { border-left-color: var(--traditional); }

  /* ---- misc ---- */
  .stTabs [data-baseweb="tab"] { font-weight: 600; }
  div[data-testid="stSidebarNav"] { display: none; }
  footer { visibility: hidden; }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def cards(items: list[tuple[str, str, str]]) -> None:
    """Render a row of metric cards from (label, value, hint) tuples."""
    html = "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="hint">{hint}</div></div>'
        for label, value, hint in items
    )
    st.markdown(f'<div class="cards">{html}</div>', unsafe_allow_html=True)


def badge(approach: str) -> str:
    if approach == "modern":
        return '<span class="badge badge-modern">Modern &middot; RAG + LLM</span>'
    return '<span class="badge badge-traditional">Traditional &middot; TF-IDF</span>'


def show_sources(sources: list[dict], score_label: str = "similarity") -> None:
    """Render the retrieved chunks that an answer was built from."""
    if not sources:
        st.caption("No chunks were retrieved.")
        return

    for rank, source in enumerate(sources, start=1):
        text = source["text"].strip().replace("\n", " ")
        if len(text) > 700:
            text = text[:700] + " ..."
        st.markdown(
            f'<div class="source"><div class="meta">'
            f'#{rank} &middot; page {source.get("page", "?")} &middot; '
            f'chunk {source.get("index", "?")} &middot; '
            f'{score_label} {source.get("score", 0):.3f}</div>{text}</div>',
            unsafe_allow_html=True,
        )

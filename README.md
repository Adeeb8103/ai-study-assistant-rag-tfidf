# Personal AI Study Assistant

A study assistant that answers from **your own** notes, books and PDFs instead of
from general internet knowledge — built two different ways so the methods can be
compared.

| | Approach 1 — Modern | Approach 2 — Traditional |
|---|---|---|
| Retrieval | Sentence-transformer embeddings in a **FAISS vector store** | **TF-IDF** + **cosine similarity** |
| Answering | Language model via the **OpenRouter API** (RAG) | Extractive — the best passage from the notes |
| Memory | Buffer + summary + follow-up question rewriting | None |
| Cost | Per token | Free, runs locally |
| Speed | ~1–5 s (network bound) | ~10 ms |

---

## 1. Quick start

```bash
cd study_assistant

# use the venv that already exists one level up
..\venv\Scripts\activate            # Windows
# source ../venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

Or just double-click **`run.bat`** on Windows.

The app opens at <http://localhost:8501>. Create an account on the sign-up tab,
then sign in.

> The first upload takes about 30 seconds while the embedding model loads. Every
> upload after that is fast.

### API key

`.env` holds the OpenRouter key:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Get a free one at <https://openrouter.ai/keys>. **Approach 2 works without any
key** — only the modern approach needs it.

### Check everything works without opening a browser

```bash
python selftest.py              # uses built-in sample notes
python selftest.py notes.pdf    # or your own PDF
```

---

## 2. Screens

| Screen | What it is for |
|---|---|
| **Chat** | Ask questions, switch approach, sources and cost shown under every answer |
| **Library** | Upload material, tune chunk size, inspect chunks and TF-IDF term weights |
| **Compare** | One question → both approaches side by side, with charts and retrieval overlap |
| **Evaluation** | Run a whole question set through both, get the results table + CSV export |
| **How it works** | Aim, methodology, and a live trace of the pipeline — the page to present |

---

## 3. Methodology, as implemented

```
        study material (PDF / DOCX / TXT / MD)
                        |
        [1] extract text            core/ingestion.py
        [2] clean it                core/ingestion.py
        [3] chunk it (1000 chars, 150 overlap)
                        |
          +-------------+-------------+
          |                           |
   [4a] embed each chunk       [4b] fit TF-IDF
        384-dim vectors             sparse word weights
        FAISS index                 core/tfidf_store.py
        core/vector_store.py
          |                           |
   [5a] similarity search      [5b] cosine similarity
          |                           |
   [6a] chunks + memory        [6b] re-score sentences
        -> OpenRouter LLM            inside the best chunks
        core/engines.py              core/tfidf_store.py
          |                           |
          +-------------+-------------+
                        |
        [7] score both: accuracy, speed, cost, naturalness
                        core/evaluation.py
```

Both approaches consume the **same chunks** and the **same top-k**, so any
measured difference comes from the method, not the pre-processing.

### Retrieval tuning in step 4b/5b

The traditional path applies three standard IR refinements, all switchable from
the `TFIDF_*` settings in `config.py`:

| Refinement | Why | Effect on the cosine score |
|---|---|---|
| **Stemming** (dependency-free Porter-style, `TFIDF_USE_STEMMING`) | "evaluate", "evaluating" and "evaluation" become one term, so a question worded differently from the notes still overlaps them | Small on its own (~2%), but it is what stops a reworded question scoring 0 |
| **Question-word stripping** | "what is", "explain", "tell me about" state intent, not content; left in, they lengthen the query vector without ever matching | Removes the dilution; falls back to the raw question if stripping would leave nothing |
| **Passage (MaxP) scoring** (`TFIDF_PASSAGE_WEIGHT`) | A 1000-character chunk holds far more vocabulary than a one-line question, so their cosine is diluted by chunk length. Each chunk is also indexed as overlapping 2-sentence windows, and scored `0.85 × best window + 0.15 × whole chunk`, floored at the plain whole-chunk cosine | The dominant gain — a chunk can never score *lower* than before |

Measured over 10 questions against a 122-chunk OOP textbook, mean top-1 cosine
rose from **0.151 to 0.272 (1.81×)**, with every question improving and
retrieval still taking ~2 ms. Because these change only how a score is
computed, run `python rebuild_tfidf.py` after editing any `TFIDF_*` setting to
re-fit the stored indexes without re-uploading the material.

---

## 4. How each metric is calculated

| Metric | Method |
|---|---|
| **Accuracy** | Cosine similarity between the answer's embedding and a reference answer's embedding (0–1) |
| **Keyword coverage** | Fraction of the reference's meaningful words that appear in the answer — the word-overlap view of accuracy |
| **Groundedness** | Similarity between the answer and the retrieved chunks; low means the model drifted away from the notes |
| **Naturalness** | Readability heuristic: sentence completeness, length band, length variety, freedom from PDF artefacts, readable structure |
| **Speed** | Wall-clock seconds, split into retrieval and generation |
| **Cost** | Real token counts from the API response × the model's published per-million price |
| **LLM judge** *(optional)* | A model marks each answer 0–5 for correctness and fluency |

---

## 5. Project layout

```
study_assistant/
├── app.py                  Streamlit entry point and sidebar
├── config.py               models, prices, chunk size, paths — all tunables
├── selftest.py             headless end-to-end check
├── rebuild_tfidf.py        re-fit Approach 2 after a TFIDF_* config change
├── .env                    OPENROUTER_API_KEY
├── core/
│   ├── database.py         SQLite schema and queries
│   ├── auth.py             PBKDF2 registration / login
│   ├── ingestion.py        extract → clean → chunk
│   ├── vector_store.py     embeddings + FAISS        (Approach 1)
│   ├── tfidf_store.py      TF-IDF + cosine           (Approach 2)
│   ├── llm.py              OpenRouter client + cost tracking
│   ├── memory.py           buffer + summary memory
│   ├── engines.py          the two answering engines
│   ├── documents.py        upload → index both ways
│   └── evaluation.py       all metrics + aggregation
├── ui/                     one module per screen
└── data/                   uploads, indexes, SQLite db, CSV exports
```

---

## 6. Demo script for the presentation

1. **How it works → Aim and problem** — state the aim and the two approaches.
2. **Library** — upload a PDF. Point at the build times: TF-IDF fits in
   milliseconds, embedding every chunk takes seconds. First trade-off.
3. **Library → Inspect chunks** — show a real chunk and the top TF-IDF terms.
4. **Chat** — ask a question on *Modern*, then ask *"explain that in simpler
   words"*. Show the caption revealing how memory rewrote the question.
   Switch to *Traditional* and ask the same follow-up — it fails, because it has
   no memory.
5. **Compare** — ask a question worded **differently from the notes**. The
   retrieval-agreement panel shows a low overlap: embeddings match on meaning,
   TF-IDF only on shared words.
6. **Evaluation** — draft questions from the document, run both, show the
   results table, the charts and the findings. Download the CSV for the report.
7. **How it works → Live pipeline trace** — show the question as a dense vector
   and as a sparse TF-IDF vector, then the exact prompt sent to the model.

---

## 7. Findings this project demonstrates

- **Speed** — the traditional approach is hundreds of times faster, because it
  never leaves the machine. Almost all of the modern approach's time is the
  network round-trip to the language model.
- **Naturalness** — the modern approach writes an explanation; the traditional
  one returns sentences lifted from the notes, which read as choppy and often
  start mid-thought.
- **Accuracy** — the modern approach handles reworded and conceptual questions
  because it matches on meaning. On questions phrased with the notes' own
  wording, TF-IDF can match it and occasionally beat it, since copying the
  source sentence is hard to improve on.
- **Cost and control** — the modern approach costs money per question and
  depends on an external service; the traditional one is free, offline and fully
  under our control.

Neither approach wins outright, which is the point: the trade-off is between
natural, meaning-aware answers and speed, cost and independence.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `No OPENROUTER_API_KEY found` | Add the key to `.env` and restart. Approach 2 still works without it. |
| First upload hangs ~30 s | The embedding model is downloading/loading. Once only. |
| `No readable text could be extracted` | The PDF is a scanned image. It would need OCR — use a text-based PDF. |
| TF-IDF returns nothing relevant | Expected behaviour — it needs shared words. Reword the question using the notes' vocabulary, and show this in the demo as its documented weakness. |
| Port already in use | `streamlit run app.py --server.port 8502` |

"""
Headless end-to-end check of the whole pipeline.

Run with:  python selftest.py [path-to-a-pdf]

It exercises registration, ingestion, both retrieval methods, the metrics and
(if a key is present) the OpenRouter call - without needing a browser.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import has_api_key                      # noqa: E402
from core import auth, database as db, documents, evaluation  # noqa: E402
from core.engines import ModernRAG, TraditionalQA   # noqa: E402
from core.memory import ConversationMemory          # noqa: E402

SAMPLE = """Machine Learning Study Notes

Supervised learning uses labelled data to train a model. The model learns a
mapping from inputs to a known output, and is then asked to predict the output
for data it has never seen. Classification and regression are the two main
kinds of supervised learning.

Unsupervised learning works without labels. The algorithm looks for structure
in the data on its own. Clustering, where similar records are grouped together,
is the most common example.

Overfitting happens when a model memorises the training data instead of
learning the underlying pattern. Such a model scores very well on the training
set but performs badly on new data. Cross-validation and regularisation are the
usual remedies.

A support vector machine finds the boundary, called a hyperplane, that best
separates two classes. The data points closest to that boundary are the support
vectors, and they alone determine where the boundary sits.

A decision tree splits the data with a series of if-else questions. It is easy
to read and needs no feature scaling, but a single deep tree tends to overfit.
A random forest fixes this by combining many trees and taking a majority vote.
"""

STEP = 0


def step(label: str) -> None:
    global STEP
    STEP += 1
    print(f"\n[{STEP}] {label}")


def main() -> int:
    step("Initialising the database")
    db.init_db()
    print("    ok")

    step("Registering a test user")
    email = "selftest@example.com"
    ok, message = auth.register("Self Test", email, "test123")
    print(f"    {message}")
    ok, message, user = auth.login(email, "test123")
    if not ok:
        print(f"    FAILED: {message}")
        return 1
    print(f"    logged in as user id {user['id']}")

    step("Preparing study material")
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
        data, filename = source.read_bytes(), source.name
    else:
        data, filename = SAMPLE.encode("utf-8"), "ml_notes.txt"
    print(f"    using {filename} ({len(data):,} bytes)")

    step("Running the ingestion pipeline (extract, clean, chunk, index both ways)")
    result = documents.ingest(user["id"], filename, data,
                              progress=lambda f, l: print(f"    {f:>5.0%} {l}"))
    doc_id = result["document_id"]
    print(f"    pages={result['pages']} chars={result['clean_chars']:,} "
          f"chunks={result['num_chunks']}")
    print(f"    vector index {result['timings'].get('vector_index', 0):.2f}s | "
          f"tfidf {result['timings'].get('tfidf_index', 0):.3f}s")

    question = "What is overfitting and how is it fixed?"
    reference = ("Overfitting is when a model memorises the training data instead "
                 "of the pattern, so it does well on training data but badly on "
                 "new data. Cross-validation and regularisation fix it.")

    step(f"Traditional approach: {question!r}")
    tfidf = TraditionalQA(documents.load_tfidf_store(user["id"], doc_id))
    traditional = tfidf.answer(question, top_k=3)
    print(f"    {traditional.total_time * 1000:.1f}ms, "
          f"top cosine {traditional.top_score:.3f}")
    print(f"    answer: {traditional.answer[:220]}...")
    assert traditional.ok, traditional.error
    assert traditional.sources, "traditional retrieval returned nothing"

    step("Scoring the traditional answer")
    marks = evaluation.score(traditional, question, reference)
    print(f"    accuracy={marks.accuracy:.3f} keyword_hit={marks.keyword_hit:.3f} "
          f"grounded={marks.groundedness:.3f} natural={marks.naturalness:.3f}")

    scored = [marks]

    if has_api_key():
        step(f"Modern approach (RAG + OpenRouter): {question!r}")
        rag = ModernRAG(documents.load_vector_store(user["id"], doc_id))
        modern = rag.answer(question, memory=None, top_k=3, use_memory=False)

        if modern.ok:
            print(f"    {modern.total_time:.2f}s "
                  f"(retrieval {modern.retrieval_time * 1000:.0f}ms, "
                  f"llm {modern.generation_time:.2f}s)")
            print(f"    tokens {modern.tokens_in} in / {modern.tokens_out} out, "
                  f"cost ${modern.cost:.6f}")
            print(f"    answer: {modern.answer[:220]}...")

            step("Scoring the modern answer")
            modern_marks = evaluation.score(modern, question, reference)
            print(f"    accuracy={modern_marks.accuracy:.3f} "
                  f"keyword_hit={modern_marks.keyword_hit:.3f} "
                  f"grounded={modern_marks.groundedness:.3f} "
                  f"natural={modern_marks.naturalness:.3f}")
            scored.append(modern_marks)

            step("Memory: rewriting a follow-up question")
            memory = ConversationMemory()
            memory.add(question, modern.answer)
            rewritten = rag.rewrite_question("explain that in simpler words", memory)
            print(f"    'explain that in simpler words' -> {rewritten!r}")
        else:
            print(f"    SKIPPED - {modern.error}")
    else:
        step("Modern approach skipped - no OPENROUTER_API_KEY in .env")

    step("Aggregate comparison")
    summary = evaluation.aggregate(scored)
    for approach, entry in summary.items():
        print(f"    {approach:12} time={entry['latency']:.3f}s "
              f"acc={entry['accuracy']:.3f} nat={entry['naturalness']:.3f} "
              f"cost=${entry['cost']:.6f}")
    for line in evaluation.verdict(summary):
        print(f"    - {line}")

    step("Cleaning up the test document")
    documents.remove(user["id"], doc_id)
    print("    removed")

    print("\nSELF TEST PASSED\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

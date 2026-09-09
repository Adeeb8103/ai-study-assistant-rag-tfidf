"""
The comparison layer.

Section 3 of the dissertation ends with "we test both approaches with sample
questions and compare them on accuracy, speed, cost, and how natural the
answers feel". This module turns each of those four words into a number.

    accuracy     semantic similarity between the answer and a reference answer
                 (embedding cosine), plus a keyword-coverage check
    groundedness how much of the answer is actually supported by the retrieved
                 chunks - a low score means the model drifted from the notes
    speed        wall-clock seconds, split into retrieval and generation
    cost         estimated USD from real token counts
    naturalness  a readability heuristic, optionally replaced by an LLM judge
"""

import json
import re
import statistics
from dataclasses import dataclass, field

import numpy as np

from core import llm
from core.engines import AnswerResult

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]+")

# Markdown the language model adds for presentation. It is stripped before the
# readability heuristic runs, otherwise a well-formatted bulleted answer would
# be marked down for "starting with a dash" - which says nothing about how
# naturally it reads.
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`)")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s{0,4}(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to",
    "in", "on", "for", "and", "or", "but", "with", "as", "by", "that", "this",
    "it", "its", "at", "from", "we", "you", "they", "he", "she", "not", "can",
    "will", "would", "should", "which", "what", "how", "why", "when", "who",
}


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity of two sentence embeddings, clipped to 0..1."""
    if not a.strip() or not b.strip():
        return 0.0

    from core.vector_store import embed

    vectors = embed([a, b])
    score = float(np.dot(vectors[0], vectors[1]))  # already unit-normalised
    return round(max(0.0, min(1.0, score)), 4)


def keyword_coverage(answer: str, expected: str) -> float:
    """
    Fraction of the reference answer's meaningful words that appear in the
    answer. This is the cheap, word-overlap style of accuracy - deliberately
    included because it is the same idea TF-IDF is built on.
    """
    if not expected.strip():
        return 0.0

    wanted = {w.lower() for w in _WORD.findall(expected)} - _STOP
    if not wanted:
        return 0.0

    produced = {w.lower() for w in _WORD.findall(answer)}
    return round(len(wanted & produced) / len(wanted), 4)


def groundedness(answer: str, sources: list[dict]) -> float:
    """How much of the answer is supported by the retrieved chunks."""
    if not answer.strip() or not sources:
        return 0.0
    context = " ".join(source["text"] for source in sources)
    return semantic_similarity(answer, context)


# ---------------------------------------------------------------------------
# Naturalness
# ---------------------------------------------------------------------------
def naturalness(answer: str) -> float:
    """
    Heuristic 0..1 score for "does this read like a person wrote it".

    The traditional approach returns sentences copied out of the notes, so it
    tends to score lower on completeness and on starting mid-thought, which is
    exactly the weakness the dissertation reports.
    """
    # Judge the prose, not the markdown it is wrapped in.
    text = _MD_BULLET.sub("", answer)
    text = _MD_HEADING.sub("", text)
    text = _MD_EMPHASIS.sub("", text).strip()
    if not text:
        return 0.0

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return 0.0

    words = _WORD.findall(text)
    if not words:
        return 0.0

    scores: list[float] = []

    # 1. Does it open like a proper sentence rather than mid-phrase?
    scores.append(1.0 if text[0].isupper() else 0.3)

    # 2. Does it close with terminal punctuation?
    scores.append(1.0 if text[-1] in ".!?)" else 0.4)

    # 3. Average sentence length inside a comfortable reading band (8-26 words).
    lengths = [len(_WORD.findall(s)) for s in sentences]
    average = statistics.mean(lengths)
    if 8 <= average <= 26:
        scores.append(1.0)
    elif average < 8:
        scores.append(max(0.2, average / 8))
    else:
        scores.append(max(0.2, 26 / average))

    # 4. Sentence-length variety - copied text is often uniform and choppy.
    if len(lengths) > 1:
        spread = statistics.pstdev(lengths) / max(average, 1)
        scores.append(min(1.0, 0.4 + spread))
    else:
        scores.append(0.6)

    # 5. Freedom from extraction artefacts (stray symbols, broken spacing,
    #    runs of capitals - the fingerprints of raw PDF text).
    artefacts = len(re.findall(r"[|•■□]|\s{3,}|\b[A-Z]{4,}\b", text))
    scores.append(max(0.0, 1.0 - artefacts / 8))

    # 6. Structure the reader can follow: bullets, numbering or linking words.
    #    Checked against the ORIGINAL answer, since the markers were stripped above.
    structured = bool(_MD_BULLET.search(answer)) or bool(re.search(
        r"\b(because|therefore|however|for example|in other words|firstly|"
        r"finally|which means|so that)\b", text, re.IGNORECASE))
    scores.append(1.0 if structured else 0.65)

    return round(sum(scores) / len(scores), 4)


# ---------------------------------------------------------------------------
# Optional LLM judge
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are marking two answers produced by a student study \
assistant. Judge only what is written.

QUESTION: {question}

REFERENCE ANSWER (ground truth from the notes): {reference}

ANSWER TO MARK: {answer}

Give two integer scores from 0 to 5:
- correctness: does the answer match the reference and stay factual?
- fluency: does it read naturally and clearly for a student?

Reply with JSON only, in this exact shape:
{{"correctness": <0-5>, "fluency": <0-5>, "comment": "<one short sentence>"}}"""


def llm_judge(question: str, answer: str, reference: str,
              model: str) -> dict | None:
    """Ask a language model to mark an answer. Returns None if unavailable."""
    if not answer.strip():
        return None

    try:
        response = llm.chat(
            [{"role": "user", "content": JUDGE_PROMPT.format(
                question=question,
                reference=reference or "(none provided - judge on the question alone)",
                answer=answer)}],
            model=model,
            max_tokens=160,
        )
    except llm.LLMError:
        return None

    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return {
        "correctness": max(0, min(5, int(payload.get("correctness", 0)))),
        "fluency": max(0, min(5, int(payload.get("fluency", 0)))),
        "comment": str(payload.get("comment", ""))[:200],
    }


# ---------------------------------------------------------------------------
# Scoring one answer
# ---------------------------------------------------------------------------
@dataclass
class Scored:
    question: str
    reference: str
    approach: str
    answer: str
    latency: float
    retrieval_time: float
    generation_time: float
    tokens_in: int
    tokens_out: int
    cost: float
    accuracy: float
    keyword_hit: float
    groundedness: float
    naturalness: float
    top_score: float
    judge: dict | None = field(default=None)
    error: str = ""

    def to_row(self) -> dict:
        """Flat dictionary for a pandas DataFrame."""
        row = {
            "Question": self.question,
            "Approach": "Modern (RAG + LLM)" if self.approach == "modern"
                        else "Traditional (TF-IDF)",
            "Answer": self.answer,
            "Accuracy": self.accuracy,
            "Keyword hit": self.keyword_hit,
            "Groundedness": self.groundedness,
            "Naturalness": self.naturalness,
            "Time (s)": round(self.latency, 3),
            "Retrieval (s)": round(self.retrieval_time, 4),
            "Tokens in": self.tokens_in,
            "Tokens out": self.tokens_out,
            "Cost (USD)": round(self.cost, 6),
            "Retrieval score": self.top_score,
        }
        if self.judge:
            row["Judge correctness"] = self.judge["correctness"]
            row["Judge fluency"] = self.judge["fluency"]
        return row


def score(result: AnswerResult, question: str, reference: str = "",
          judge_model: str | None = None) -> Scored:
    """Attach every metric to a single engine answer."""
    answer = result.answer or ""

    accuracy = semantic_similarity(answer, reference) if reference.strip() else 0.0

    return Scored(
        question=question,
        reference=reference,
        approach=result.approach,
        answer=answer,
        latency=result.total_time,
        retrieval_time=result.retrieval_time,
        generation_time=result.generation_time,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost=result.cost,
        accuracy=accuracy,
        keyword_hit=keyword_coverage(answer, reference),
        groundedness=groundedness(answer, result.sources),
        naturalness=naturalness(answer),
        top_score=result.top_score,
        judge=llm_judge(question, answer, reference, judge_model)
              if judge_model and answer else None,
        error=result.error,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(scored: list[Scored]) -> dict[str, dict]:
    """Average every metric per approach, ready for the summary table."""
    summary: dict[str, dict] = {}

    for approach in ("modern", "traditional"):
        rows = [s for s in scored if s.approach == approach and not s.error]
        if not rows:
            continue

        entry = {
            "questions": len(rows),
            "accuracy": round(statistics.mean(r.accuracy for r in rows), 4),
            "keyword_hit": round(statistics.mean(r.keyword_hit for r in rows), 4),
            "groundedness": round(statistics.mean(r.groundedness for r in rows), 4),
            "naturalness": round(statistics.mean(r.naturalness for r in rows), 4),
            "latency": round(statistics.mean(r.latency for r in rows), 3),
            "retrieval_time": round(statistics.mean(r.retrieval_time for r in rows), 4),
            "tokens": sum(r.tokens_in + r.tokens_out for r in rows),
            "cost": round(sum(r.cost for r in rows), 6),
        }

        judged = [r.judge for r in rows if r.judge]
        if judged:
            entry["judge_correctness"] = round(
                statistics.mean(j["correctness"] for j in judged), 2)
            entry["judge_fluency"] = round(
                statistics.mean(j["fluency"] for j in judged), 2)

        summary[approach] = entry

    return summary


def verdict(summary: dict[str, dict]) -> list[str]:
    """Plain-English findings, the kind that belong in the conclusion chapter."""
    modern, traditional = summary.get("modern"), summary.get("traditional")
    if not modern or not traditional:
        return ["Run both approaches over the same questions to see a comparison."]

    lines: list[str] = []

    faster = "Traditional" if traditional["latency"] < modern["latency"] else "Modern"
    times = max(modern["latency"], traditional["latency"]) / \
            max(min(modern["latency"], traditional["latency"]), 1e-6)
    lines.append(
        f"**Speed** - {faster} is about {times:.1f}x faster "
        f"({traditional['latency']:.2f}s vs {modern['latency']:.2f}s per question)."
    )

    if modern["accuracy"] or traditional["accuracy"]:
        better = "Modern" if modern["accuracy"] >= traditional["accuracy"] else "Traditional"
        lines.append(
            f"**Accuracy** - {better} matches the reference answers more closely "
            f"({modern['accuracy']:.2f} vs {traditional['accuracy']:.2f} semantic similarity)."
        )

    nicer = "Modern" if modern["naturalness"] >= traditional["naturalness"] else "Traditional"
    lines.append(
        f"**Naturalness** - {nicer} reads better "
        f"({modern['naturalness']:.2f} vs {traditional['naturalness']:.2f})."
    )

    lines.append(
        f"**Cost** - the modern approach used {modern['tokens']:,} tokens costing about "
        f"${modern['cost']:.4f}; the traditional approach cost $0.0000 because it "
        f"runs entirely on the local machine."
    )

    lines.append(
        f"**Grounding** - modern {modern['groundedness']:.2f} vs traditional "
        f"{traditional['groundedness']:.2f} similarity to the retrieved notes, "
        f"showing how closely each answer stays to the source material."
    )

    return lines


# ---------------------------------------------------------------------------
# Test-question generation (a convenience for building the question set)
# ---------------------------------------------------------------------------
QUESTION_PROMPT = """Read the study material below and write {n} exam-style \
questions a student might ask about it, together with a short correct answer \
taken only from this material.

MATERIAL:
{context}

Reply with JSON only: a list of objects shaped \
{{"question": "...", "answer": "..."}}"""


def generate_questions(chunks: list[dict], model: str, count: int = 5) -> list[dict]:
    """Draft a question/reference-answer set straight from the document."""
    context = "\n\n".join(chunk["text"] for chunk in chunks[:6])[:6000]

    response = llm.chat(
        [{"role": "user", "content": QUESTION_PROMPT.format(n=count, context=context)}],
        model=model,
        max_tokens=900,
    )

    match = re.search(r"\[.*\]", response.text, re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    return [
        {"question": str(i.get("question", "")).strip(),
         "answer": str(i.get("answer", "")).strip()}
        for i in items
        if isinstance(i, dict) and i.get("question")
    ]

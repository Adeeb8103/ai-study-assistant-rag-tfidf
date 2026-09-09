"""
Approach 2, retrieval half: the traditional TF-IDF + cosine similarity model.

No external service and no neural network. Every chunk is represented by a
sparse vector where each dimension is a word weighted by TF-IDF (how often the
word appears in this chunk, damped by how common it is across all chunks). The
question is projected into the same space and the closest chunk by cosine
similarity is returned.

Three classical IR refinements are applied so the cosine numbers reflect real
word overlap rather than accidents of wording or chunk length:

1. **Stemming.** "evaluate", "evaluating" and "evaluation" collapse to one term,
   so a question phrased differently from the notes still overlaps them.
2. **Stop-word removal before stemming**, plus removal of question boilerplate
   ("what is", "explain", "tell me about") from the *query only*. Those words
   describe intent, not content; leaving them in lengthens the query vector and
   drags the cosine down without ever matching anything useful.
3. **Passage (MaxP) scoring.** A 1000-character chunk holds far more vocabulary
   than a one-line question, so the cosine between them is diluted by the
   chunk's length. Each chunk is therefore also indexed as short overlapping
   sentence windows, and its score blends the whole-chunk cosine with the
   cosine of its best-matching window. A chunk can only ever score higher than
   it did before, never lower.
"""

import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    TFIDF_MIN_SCORE,
    TFIDF_PASSAGE_SENTENCES,
    TFIDF_PASSAGE_STRIDE,
    TFIDF_PASSAGE_WEIGHT,
    TFIDF_USE_STEMMING,
)
from core.ingestion import Chunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PASSAGE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN = re.compile(r"[a-z][a-z'-]*|\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Text normalisation: stop words + a lightweight suffix stemmer
# ---------------------------------------------------------------------------
# Words that carry the *intent* of a question rather than its subject. They are
# stripped from the query but kept in the index, because a chunk that genuinely
# discusses "the main difference" should still match on "difference".
# Only genuine boilerplate is listed: words such as "list", "state" or "class"
# carry meaning in real study material, so they stay.
_QUESTION_WORDS = frozenset("""
    what which who whom whose when where why how explain describe
    summarise summarize summary outline tell discuss mention
    please kindly briefly shortly according
""".split())

_STOP = frozenset(ENGLISH_STOP_WORDS) | frozenset("""
    also using use used within across via etc ie eg
""".split())

# Irregular forms the suffix rules below cannot reach.
_IRREGULAR = {
    "children": "child", "men": "man", "women": "woman", "people": "person",
    "criteria": "criterion", "analyses": "analysis", "data": "data",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
}

# Derivational suffixes, longest first. Applied repeatedly (max two passes) so
# that "computation" -> "computate" -> "comput" lands on the same stem as
# "computing" -> "comput".
_DERIVATIONAL = (
    ("ational", "ate"), ("tional", "tion"), ("ization", "ize"),
    ("isation", "ize"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("ability", "able"), ("ibility", "able"),
    ("ically", "ic"), ("iously", "ious"), ("fully", "ful"),
    ("ation", "ate"), ("ities", "ity"), ("ments", ""), ("ment", ""),
    ("ness", ""), ("ance", ""), ("ence", ""), ("able", ""), ("ible", ""),
    ("ally", "al"), ("isms", ""), ("ism", ""), ("ity", ""), ("ive", ""),
    ("ize", ""), ("ise", ""), ("ate", ""), ("ly", ""), ("er", ""),
    ("or", ""), ("al", ""), ("ic", ""),
)

_UNDOUBLE = ("bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt")

# After -ing/-ed comes off, these endings had a silent "e" before it
# ("evaluating" -> "evaluat" -> "evaluate"), which the rules above then reduce
# to the same stem as "evaluate" and "evaluation".
_RESTORE_E = {"at": "ate", "bl": "ble", "iz": "ize"}

_MIN_STEM = 4          # never strip a word down below this many characters
_stem_cache: dict[str, str] = {}


def _has_vowel(word: str) -> bool:
    return any(character in "aeiouy" for character in word)


def _stem(word: str) -> str:
    """A compact Porter-style stemmer - deliberately dependency-free."""
    if not TFIDF_USE_STEMMING:
        return word

    cached = _stem_cache.get(word)
    if cached is not None:
        return cached

    stem = _IRREGULAR.get(word, word)

    # Short words are left alone; there is nothing safe to strip from them.
    if stem == word and len(word) > _MIN_STEM:
        # 1. Plurals.
        if stem.endswith("ies"):
            stem = stem[:-3] + "y"
        elif stem.endswith("sses"):
            stem = stem[:-2]
        elif stem.endswith("s") and not stem.endswith(("ss", "us", "is")):
            stem = stem[:-1]

        # 2. Verb endings, then undouble the consonant they exposed.
        for suffix in ("ing", "edly", "ed"):
            if stem.endswith(suffix) and len(stem) - len(suffix) >= _MIN_STEM - 1:
                trimmed = stem[: -len(suffix)]
                if _has_vowel(trimmed):
                    if trimmed.endswith(_UNDOUBLE):
                        trimmed = trimmed[:-1]
                    elif trimmed[-2:] in _RESTORE_E:
                        trimmed = trimmed[:-2] + _RESTORE_E[trimmed[-2:]]
                    stem = trimmed
                break

        # 3. Derivational suffixes, up to two passes.
        for _ in range(2):
            for suffix, replacement in _DERIVATIONAL:
                if stem.endswith(suffix):
                    candidate = stem[: -len(suffix)] + replacement
                    if len(candidate) >= _MIN_STEM:
                        stem = candidate
                        break
            else:
                break

        # 4. A trailing silent "e" so "evaluate"/"evaluating" agree.
        if stem.endswith("e") and len(stem) > _MIN_STEM:
            stem = stem[:-1]

    _stem_cache[word] = stem
    return stem


def _normalise(text: str) -> str:
    """
    Vectoriser preprocessor: lowercase, drop stop words, stem what is left.

    Returned as a plain string so scikit-learn's own tokeniser and n-gram
    machinery still run on top of it - the vocabulary is simply made of stems.
    """
    return " ".join(
        _stem(token) for token in _TOKEN.findall(text.lower())
        if token not in _STOP and len(token) > 1
    )


def _normalise_question(question: str) -> str:
    """Same as `_normalise`, minus the words that only phrase the question."""
    tokens = [
        token for token in _TOKEN.findall(question.lower())
        if token not in _STOP and len(token) > 1
    ]
    content = [token for token in tokens if token not in _QUESTION_WORDS]
    # "Summarise this" is all boilerplate - fall back rather than search for "".
    return " ".join(_stem(token) for token in (content or tokens))


# ---------------------------------------------------------------------------
# Passage windows
# ---------------------------------------------------------------------------
def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _PASSAGE_SPLIT.split(text) if part.strip()]
    merged: list[str] = []
    for part in parts:
        # Glue headings and stray fragments onto their neighbour so a window is
        # always a meaningful amount of text.
        if merged and len(part) < 40:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _windows(text: str,
             size: int = TFIDF_PASSAGE_SENTENCES,
             stride: int = TFIDF_PASSAGE_STRIDE) -> list[str]:
    """Overlapping sentence windows - the units MaxP scoring ranks."""
    sentences = _sentences(text)
    if not sentences:
        return []
    if len(sentences) <= size:
        return [" ".join(sentences)]
    return [
        " ".join(sentences[start:start + size])
        for start in range(0, len(sentences) - size + 1, max(1, stride))
    ]


class TfidfStore:
    """A fitted TF-IDF vectoriser plus the chunk matrix it produced."""

    def __init__(self, vectorizer: TfidfVectorizer, matrix, chunks: list[dict],
                 passages: list[str] | None = None, passage_matrix=None,
                 passage_owner: list[int] | None = None,
                 display: dict[str, str] | None = None):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks
        self.display = display or {}

        if passage_matrix is None or passages is None or passage_owner is None:
            passages, passage_owner, passage_matrix = self._index_passages()

        self.passages = passages
        self.passage_owner = np.asarray(passage_owner, dtype=int)
        self.passage_matrix = passage_matrix

    def _index_passages(self):
        """Cut every chunk into sentence windows and project them into the
        *existing* TF-IDF space, so window and chunk scores stay comparable."""
        passages: list[str] = []
        owners: list[int] = []

        for position, chunk in enumerate(self.chunks):
            for window in _windows(chunk["text"]):
                passages.append(window)
                owners.append(position)

        if not passages:
            return [], [], None
        return passages, owners, self.vectorizer.transform(passages)

    # -- construction -------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[Chunk]) -> "TfidfStore":
        texts = [chunk.text for chunk in chunks]
        if not texts:
            raise ValueError("There are no chunks to build a TF-IDF model from.")

        # max_df is a *proportion* of chunks. On a short document 0.9 rounds
        # down below min_df and scikit-learn refuses to fit, so only prune
        # very common terms once the corpus is large enough for it to mean
        # anything.
        max_df = 0.9 if len(texts) >= 10 else 1.0

        def make(**overrides) -> TfidfVectorizer:
            params = dict(
                lowercase=True,
                preprocessor=_normalise,  # stop words + stemming happen here
                stop_words=None,          # already handled by the preprocessor
                ngram_range=(1, 2),       # single stems and two-stem phrases
                sublinear_tf=True,        # 1 + log(tf) damps very frequent words
                min_df=1,
                max_df=max_df,
            )
            params.update(overrides)
            return TfidfVectorizer(**params)

        try:
            vectorizer = make()
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Very short material can be made up almost entirely of stop words,
            # leaving an empty vocabulary. Fall back to keeping every word.
            vectorizer = make(preprocessor=None, ngram_range=(1, 1), max_df=1.0)
            matrix = vectorizer.fit_transform(texts)

        return cls(
            vectorizer, matrix, [chunk.to_dict() for chunk in chunks],
            # The fallback vectoriser indexes raw words, so there are no stems
            # left to translate back.
            display=_display_map(texts) if vectorizer.preprocessor else None,
        )

    # -- persistence --------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "tfidf.pkl", "wb") as handle:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "passages": self.passages,
                "passage_owner": self.passage_owner,
                "passage_matrix": self.passage_matrix,
                "display": self.display,
            }, handle)
        (directory / "tfidf_chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "TfidfStore":
        model_path = directory / "tfidf.pkl"
        chunks_path = directory / "tfidf_chunks.json"

        if not model_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No TF-IDF model found in {directory}. Re-upload the document."
            )

        with open(model_path, "rb") as handle:
            payload = pickle.load(handle)
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

        # Indexes written before passage scoring existed simply have their
        # windows rebuilt on load - no re-upload needed.
        return cls(
            payload["vectorizer"], payload["matrix"], chunks,
            passages=payload.get("passages"),
            passage_matrix=payload.get("passage_matrix"),
            passage_owner=payload.get("passage_owner"),
            display=payload.get("display"),
        )

    # -- search -------------------------------------------------------------
    def query_vector(self, question: str):
        """The sparse vector `search` actually compares against the chunks."""
        vector = self.vectorizer.transform([_normalise_question(question)])
        if vector.nnz == 0:
            # Everything the student typed was boilerplate or absent from the
            # vocabulary. Retry with the question untouched rather than search
            # for nothing - stripping must never cost us a match.
            vector = self.vectorizer.transform([question])
        return vector

    def search(self, question: str, top_k: int = 4) -> list[dict]:
        """Rank every chunk against the question by cosine similarity."""
        if not self.chunks:
            return []

        query_vector = self.query_vector(question)
        chunk_similarity = cosine_similarity(query_vector, self.matrix)[0]
        similarities = chunk_similarity
        best_passage = np.full(len(self.chunks), -1)

        if self.passage_matrix is not None and self.passage_matrix.shape[0]:
            passage_similarity = cosine_similarity(
                query_vector, self.passage_matrix)[0]

            # Assign ascending so the last write per chunk is its best window.
            order = np.argsort(passage_similarity)
            best = np.zeros(len(self.chunks))
            best[self.passage_owner[order]] = passage_similarity[order]
            best_passage[self.passage_owner[order]] = order

            blended = (TFIDF_PASSAGE_WEIGHT * best
                       + (1 - TFIDF_PASSAGE_WEIGHT) * chunk_similarity)
            # A chunk never scores worse than plain whole-chunk cosine.
            similarities = np.maximum(chunk_similarity, blended)

        top_k = min(top_k, len(self.chunks))
        # argsort ascending, so take the tail and reverse it for descending order.
        ranked = similarities.argsort()[-top_k:][::-1]

        results = []
        for position in ranked:
            position = int(position)
            chunk = dict(self.chunks[position])
            chunk["score"] = round(float(similarities[position]), 4)
            chunk["chunk_score"] = round(float(chunk_similarity[position]), 4)

            window = int(best_passage[position])
            if window >= 0 and similarities[position] > 0:
                chunk["passage"] = self.passages[window]
            results.append(chunk)

        return results

    def top_terms(self, limit: int = 25) -> list[tuple[str, float]]:
        """Highest-weighted vocabulary terms - useful for the methodology page."""
        vocabulary = self.vectorizer.get_feature_names_out()
        weights = self.matrix.sum(axis=0).A1
        order = weights.argsort()[-limit:][::-1]
        return [(self.readable(vocabulary[i]), round(float(weights[i]), 3))
                for i in order]

    def readable(self, term: str) -> str:
        """Turn an indexed stem such as 'evaluat model' back into real words."""
        return " ".join(self.display.get(part, part) for part in term.split())


def _display_map(texts: list[str]) -> dict[str, str]:
    """Map each stem to the most common word in the notes that produced it."""
    counts: dict[str, Counter] = {}
    for text in texts:
        for token in _TOKEN.findall(text.lower()):
            if token in _STOP or len(token) <= 1:
                continue
            counts.setdefault(_stem(token), Counter())[token] += 1
    return {stem: words.most_common(1)[0][0] for stem, words in counts.items()}


# ---------------------------------------------------------------------------
# Extractive answering
# ---------------------------------------------------------------------------
def extractive_answer(question: str, results: list[dict],
                      max_sentences: int = 4) -> str:
    """
    Build the traditional system's answer.

    There is no language model here, so the "answer" is the passage from the
    notes that matches best. To keep it readable we re-run cosine similarity at
    sentence level inside the top chunks and return only the closest sentences
    in their original order.
    """
    if not results or results[0].get("score", 0) <= TFIDF_MIN_SCORE:
        return (
            "No matching content was found in this document for that question. "
            "TF-IDF relies on word overlap, so try rephrasing using the same "
            "wording that appears in your notes."
        )

    # Pool sentences from the best two chunks, leading with the window that the
    # retriever actually matched on.
    sentences: list[str] = []
    seen: set[str] = set()
    for chunk in results[:2]:
        source = f"{chunk.get('passage', '')} {chunk['text']}".strip()
        for sentence in _SENTENCE_SPLIT.split(source):
            sentence = " ".join(sentence.split())
            # The matched window is a copy of text that also appears in the
            # chunk, so compare on words alone - line breaks differ between them.
            key = sentence.lower()
            if len(sentence) > 25 and key not in seen:
                seen.add(key)
                sentences.append(sentence)

    if not sentences:
        return results[0]["text"].strip()

    try:
        vectorizer = TfidfVectorizer(preprocessor=_normalise, stop_words=None,
                                     ngram_range=(1, 2), sublinear_tf=True)
        matrix = vectorizer.fit_transform(sentences)
        scores = cosine_similarity(
            vectorizer.transform([_normalise_question(question)]), matrix)[0]
    except ValueError:
        # Every word was a stop word - fall back to the raw chunk.
        return results[0]["text"].strip()

    chosen = sorted(scores.argsort()[-max_sentences:][::-1])
    selected = [sentences[i] for i in chosen if scores[i] > 0]

    if not selected:
        return results[0]["text"].strip()

    return " ".join(selected)

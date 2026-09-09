"""
The two question-answering engines that this dissertation compares.

Both share the same chunks and the same top-k setting; the only thing that
changes is how a chunk is found and how the answer is produced.

    ModernRAG        embeddings -> FAISS -> OpenRouter LLM  (+ memory)
    TraditionalQA    TF-IDF     -> cosine similarity -> extractive answer
"""

import time
from dataclasses import dataclass, field

from config import DEFAULT_MODEL, DEFAULT_TOP_K
from core import llm
from core.memory import REWRITE_PROMPT, SUMMARY_PROMPT, ConversationMemory
from core.tfidf_store import TfidfStore, extractive_answer
from core.vector_store import VectorStore

SYSTEM_PROMPT = """You are a personal study assistant. You answer only from the \
CONTEXT below, which comes from the student's own notes and books.

Rules:
- Use only the CONTEXT. Do not add outside knowledge.
- If the CONTEXT does not contain the answer, say exactly that the notes do not \
cover it, and suggest what the student could search for instead.
- Explain clearly and simply, as if teaching a classmate.
- Cite the page you used in square brackets, for example [page 4].
- Keep the answer focused; use short bullet points when listing things."""


@dataclass
class AnswerResult:
    """Everything the UI and the evaluation screens need about one answer."""
    approach: str                      # 'modern' | 'traditional'
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    model: str = ""
    standalone_question: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def top_score(self) -> float:
        return self.sources[0].get("score", 0.0) if self.sources else 0.0


# ---------------------------------------------------------------------------
# Approach 1 - Modern: RAG with a vector store, an LLM and memory
# ---------------------------------------------------------------------------
class ModernRAG:
    def __init__(self, store: VectorStore, model: str = DEFAULT_MODEL):
        self.store = store
        self.model = model

    # -- memory helpers -----------------------------------------------------
    def rewrite_question(self, question: str,
                         memory: ConversationMemory | None) -> str:
        """Turn a follow-up such as 'explain that again' into a full question."""
        if not memory or not memory.turns:
            return question

        try:
            response = llm.chat(
                [{"role": "user",
                  "content": REWRITE_PROMPT.format(
                      history=memory.as_text(), question=question)}],
                model=self.model,
                max_tokens=120,
            )
            rewritten = response.text.strip().strip('"')
            # Guard against the model returning something odd.
            return rewritten if 3 < len(rewritten) < 400 else question
        except llm.LLMError:
            return question

    def summarise(self, memory: ConversationMemory) -> str:
        """Compress the older part of a long conversation."""
        older = memory.older_turns()
        if not older:
            return memory.summary

        history = "\n".join(
            f"Student: {t.question}\nAssistant: {t.answer[:400]}" for t in older
        )
        try:
            response = llm.chat(
                [{"role": "user", "content": SUMMARY_PROMPT.format(history=history)}],
                model=self.model,
                max_tokens=250,
            )
            return response.text.strip()
        except llm.LLMError:
            return memory.summary

    # -- main entry point ---------------------------------------------------
    def answer(self, question: str, memory: ConversationMemory | None = None,
               top_k: int = DEFAULT_TOP_K,
               use_memory: bool = True) -> AnswerResult:
        started = time.perf_counter()

        search_query = question
        if use_memory and memory:
            search_query = self.rewrite_question(question, memory)

        # --- Retrieval ---
        retrieval_started = time.perf_counter()
        try:
            sources = self.store.search(search_query, top_k=top_k)
        except Exception as exc:
            return AnswerResult(
                approach="modern", answer="", model=self.model,
                error=f"Vector search failed: {exc}",
                total_time=time.perf_counter() - started,
            )
        retrieval_time = time.perf_counter() - retrieval_started

        if not sources:
            return AnswerResult(
                approach="modern",
                answer="This document has no indexed content to search.",
                model=self.model, retrieval_time=retrieval_time,
                total_time=time.perf_counter() - started,
            )

        context = "\n\n".join(
            f"[page {s['page']}] {s['text']}" for s in sources
        )

        # --- Generation ---
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if use_memory and memory:
            if memory.summary:
                messages.append({
                    "role": "system",
                    "content": f"Summary of the earlier conversation: {memory.summary}",
                })
            messages.extend(memory.as_messages())

        messages.append({
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
        })

        try:
            response = llm.chat(messages, model=self.model)
        except llm.LLMError as exc:
            return AnswerResult(
                approach="modern", answer="", sources=sources, model=self.model,
                retrieval_time=retrieval_time,
                total_time=time.perf_counter() - started,
                standalone_question=search_query,
                error=str(exc),
            )

        return AnswerResult(
            approach="modern",
            answer=response.text,
            sources=sources,
            retrieval_time=retrieval_time,
            generation_time=response.latency,
            total_time=time.perf_counter() - started,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost=response.cost,
            model=self.model,
            standalone_question=search_query,
        )


# ---------------------------------------------------------------------------
# Approach 2 - Traditional: TF-IDF, cosine similarity, no external service
# ---------------------------------------------------------------------------
class TraditionalQA:
    def __init__(self, store: TfidfStore):
        self.store = store

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K,
               max_sentences: int = 4, **_ignored) -> AnswerResult:
        started = time.perf_counter()

        retrieval_started = time.perf_counter()
        try:
            sources = self.store.search(question, top_k=top_k)
        except Exception as exc:
            return AnswerResult(
                approach="traditional", answer="", model="tf-idf",
                error=f"TF-IDF search failed: {exc}",
                total_time=time.perf_counter() - started,
            )
        retrieval_time = time.perf_counter() - retrieval_started

        text = extractive_answer(question, sources, max_sentences=max_sentences)

        return AnswerResult(
            approach="traditional",
            answer=text,
            sources=sources,
            retrieval_time=retrieval_time,
            generation_time=0.0,          # nothing is generated - it is extracted
            total_time=time.perf_counter() - started,
            tokens_in=0,
            tokens_out=0,
            cost=0.0,                     # runs locally, so there is no API cost
            model="TF-IDF + cosine similarity",
            standalone_question=question,
        )

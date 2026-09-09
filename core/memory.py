"""
Conversational memory for the modern assistant.

A plain chatbot forgets everything after each question, so a follow-up like
"explain that in simpler words" has no meaning. Two complementary mechanisms
are used here:

* Buffer memory  - the last N question/answer pairs are replayed verbatim.
* Summary memory - anything older is compressed by the language model into a
  short paragraph, so a long session still fits in the prompt without losing
  the thread.

The memory is also used to rewrite follow-up questions into standalone ones
before retrieval, otherwise searching the vector store for "explain that"
returns nothing useful.
"""

from dataclasses import dataclass, field

from config import MEMORY_SUMMARY_AFTER, MEMORY_WINDOW_TURNS


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ConversationMemory:
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    window: int = MEMORY_WINDOW_TURNS

    # -- writing ------------------------------------------------------------
    def add(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""

    @classmethod
    def from_messages(cls, messages: list[dict],
                      summary: str = "") -> "ConversationMemory":
        """Rebuild memory from rows loaded out of the database."""
        memory = cls(summary=summary)
        pending: str | None = None

        for message in messages:
            if message["role"] == "user":
                pending = message["content"]
            elif message["role"] == "assistant" and pending is not None:
                memory.add(pending, message["content"])
                pending = None

        return memory

    # -- reading ------------------------------------------------------------
    @property
    def recent(self) -> list[Turn]:
        return self.turns[-self.window:]

    def as_messages(self) -> list[dict]:
        """The recent turns formatted as chat messages for the LLM."""
        messages: list[dict] = []
        for turn in self.recent:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def as_text(self, limit: int = 3) -> str:
        """Compact plain-text history, used for question rewriting."""
        lines = []
        if self.summary:
            lines.append(f"Earlier in the conversation: {self.summary}")
        for turn in self.turns[-limit:]:
            lines.append(f"Student: {turn.question}")
            lines.append(f"Assistant: {turn.answer[:300]}")
        return "\n".join(lines)

    def needs_summary(self) -> bool:
        return len(self.turns) > MEMORY_SUMMARY_AFTER

    def older_turns(self) -> list[Turn]:
        return self.turns[:-self.window] if self.needs_summary() else []


SUMMARY_PROMPT = (
    "Summarise the study conversation below in at most four sentences. Keep "
    "the topics discussed and any facts the student seemed unsure about. Write "
    "it as notes, not as a reply to the student.\n\n{history}"
)

REWRITE_PROMPT = (
    "Rewrite the student's latest question so that it can be understood on its "
    "own, without the earlier conversation. Resolve words like 'it', 'that' or "
    "'this' using the history. If the question is already self-contained, "
    "repeat it unchanged. Reply with the rewritten question only.\n\n"
    "Conversation so far:\n{history}\n\nLatest question: {question}"
)

"""
Thin wrapper around the OpenRouter chat-completions endpoint.

We call the HTTP API through the official OpenAI SDK (OpenRouter is
wire-compatible with it) because the response carries a `usage` block, which is
what lets the comparison screens report real token counts and an estimated cost
rather than a guess.
"""

import time
from dataclasses import dataclass

from openai import OpenAI

from config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


class LLMError(RuntimeError):
    """Raised when the language model cannot be reached."""


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost: float
    latency: float


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimated USD cost from the published per-million-token prices."""
    price = AVAILABLE_MODELS.get(model)
    if not price:
        return 0.0
    return (tokens_in / 1_000_000) * price["input"] + \
           (tokens_out / 1_000_000) * price["output"]


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENROUTER_API_KEY:
            raise LLMError(
                "No OPENROUTER_API_KEY found. Add it to the .env file to use "
                "the modern (RAG) approach. The traditional TF-IDF approach "
                "works without any key."
            )
        _client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            timeout=LLM_TIMEOUT_SECONDS,
        )
    return _client


def chat(messages: list[dict], model: str = DEFAULT_MODEL,
         temperature: float = LLM_TEMPERATURE,
         max_tokens: int = LLM_MAX_TOKENS) -> LLMResponse:
    """Send a chat request and return the answer plus its usage statistics."""
    client = get_client()
    started = time.perf_counter()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # network error, bad key, rate limit, ...
        raise LLMError(f"OpenRouter request failed: {exc}") from exc

    latency = time.perf_counter() - started

    usage = getattr(completion, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", 0) or 0
    tokens_out = getattr(usage, "completion_tokens", 0) or 0

    choices = getattr(completion, "choices", None) or []
    text = (choices[0].message.content or "").strip() if choices else ""

    if not text:
        raise LLMError("The model returned an empty response.")

    return LLMResponse(
        text=text,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=estimate_cost(model, tokens_in, tokens_out),
        latency=latency,
    )


def check_connection(model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    """Used by the sidebar status light."""
    try:
        response = chat(
            [{"role": "user", "content": "Reply with the single word: ready"}],
            model=model,
            max_tokens=5,
        )
        return True, f"Connected ({response.model})"
    except LLMError as exc:
        return False, str(exc)

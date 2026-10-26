"""Translation backend registry."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from ..types import BackendError
from .base import (
    BaseTranslator,
    ProgressFn,
    TextUnit,
    TranslationOptions,
    Translator,
    noop_progress,
)

__all__ = [
    "BaseTranslator",
    "ProgressFn",
    "TextUnit",
    "TranslationOptions",
    "Translator",
    "noop_progress",
    "get_translator",
    "available_backends",
    "BACKENDS",
]


def _llm(**kwargs: Any) -> BaseTranslator:
    from .llm import LLMTranslator

    return LLMTranslator(**kwargs)


def _nllb(**kwargs: Any) -> BaseTranslator:
    from .nllb import NLLBTranslator

    return NLLBTranslator(**kwargs)


BACKENDS: dict[str, Callable[..., BaseTranslator]] = {
    "llm": _llm,
    "anthropic": lambda **kw: _llm(provider="anthropic", **kw),
    "openai": lambda **kw: _llm(provider="openai", **kw),
    "nllb": _nllb,
    "local": _nllb,
}

_AUTO_ORDER = ("llm", "nllb")


def get_translator(name: str = "auto", **kwargs: Any) -> BaseTranslator:
    name = (name or "auto").strip().lower()
    if name == "auto":
        if os.environ.get("SUBTRANS_MT_BACKEND"):
            return get_translator(os.environ["SUBTRANS_MT_BACKEND"], **kwargs)
        failures: list[str] = []
        for candidate in _AUTO_ORDER:
            backend = BACKENDS[candidate](**_filter_kwargs(candidate, kwargs))
            ok, reason = backend.available()
            if ok:
                return backend
            failures.append(f"{candidate}: {reason}")
        raise BackendError(
            "No usable translation backend. "
            + "; ".join(failures)
            + ". Set ANTHROPIC_API_KEY (best quality) or install the local "
            "model with: pip install 'subtrans[local-mt]'"
        )
    if name not in BACKENDS:
        raise BackendError(
            f"Unknown translation backend {name!r}. "
            f"Available: {', '.join(sorted(BACKENDS))}"
        )
    return BACKENDS[name](**_filter_kwargs(name, kwargs))


def _filter_kwargs(backend: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    if backend in ("llm", "anthropic", "openai"):
        allowed = {"model", "provider", "api_key", "base_url", "max_retries", "temperature"}
        return {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if backend in ("nllb", "local"):
        allowed = {"model", "device", "max_new_tokens", "batch_size"}
        return {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    return {}


def available_backends() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in ("llm", "nllb"):
        try:
            backend = BACKENDS[key]()
            ok, reason = backend.available()
            label = getattr(backend, "model", None) or getattr(backend, "model_name", "")
        except Exception as exc:  # pragma: no cover - defensive
            ok, reason, label = False, str(exc), ""
        out.append({"name": key, "available": ok, "reason": reason, "model": label})
    return out

"""ASR backend registry.

`get_transcriber("auto")` picks the best usable backend: a cloud API if one is
configured (fast), otherwise the local model (always works). Nothing else in
the codebase imports a concrete backend.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from ..types import BackendError
from .base import ASROptions, BaseTranscriber, ProgressFn, Transcriber, noop_progress

__all__ = [
    "ASROptions",
    "BaseTranscriber",
    "ProgressFn",
    "Transcriber",
    "noop_progress",
    "get_transcriber",
    "available_backends",
    "BACKENDS",
]


def _faster_whisper(**kwargs: Any) -> BaseTranscriber:
    from .faster_whisper_backend import FasterWhisperTranscriber

    return FasterWhisperTranscriber(**kwargs)


def _openai(**kwargs: Any) -> BaseTranscriber:
    from .openai_backend import OpenAITranscriber

    return OpenAITranscriber(**kwargs)


BACKENDS: dict[str, Callable[..., BaseTranscriber]] = {
    "faster-whisper": _faster_whisper,
    "local": _faster_whisper,
    "whisper": _faster_whisper,
    "openai": _openai,
    "cloud": _openai,
}

# Order matters: cloud first when configured, because it is ~20x faster.
_AUTO_ORDER = ("openai", "faster-whisper")


def get_transcriber(name: str = "auto", **kwargs: Any) -> BaseTranscriber:
    name = (name or "auto").strip().lower()
    if name == "auto":
        if os.environ.get("SUBTRANS_ASR_BACKEND"):
            return get_transcriber(os.environ["SUBTRANS_ASR_BACKEND"], **kwargs)
        failures: list[str] = []
        for candidate in _AUTO_ORDER:
            try:
                backend = BACKENDS[candidate](**_filter_kwargs(candidate, kwargs))
            except TypeError:
                backend = BACKENDS[candidate]()
            ok, reason = backend.available()
            if ok:
                return backend
            failures.append(f"{candidate}: {reason}")
        raise BackendError(
            "No usable ASR backend. " + "; ".join(failures)
        )
    if name not in BACKENDS:
        raise BackendError(
            f"Unknown ASR backend {name!r}. Available: {', '.join(sorted(set(BACKENDS)))}"
        )
    return BACKENDS[name](**_filter_kwargs(name, kwargs))


def _filter_kwargs(backend: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pass only the options a given backend understands.

    Lets callers hand the same dict to every backend (`model`, `device`, ...)
    without each one choking on options meant for another.
    """
    if backend in ("faster-whisper", "local", "whisper"):
        allowed = {"model_size", "device", "compute_type", "download_root", "cpu_threads"}
        out = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if "model" in kwargs and kwargs["model"] and "model_size" not in out:
            out["model_size"] = kwargs["model"]
        return out
    if backend in ("openai", "cloud"):
        allowed = {"model", "api_key", "base_url", "chunk_seconds"}
        return {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    return {}


def available_backends() -> list[dict[str, Any]]:
    """For `--list-backends` and the web UI: what works right now, and why not."""
    seen: dict[str, dict[str, Any]] = {}
    for key in ("faster-whisper", "openai"):
        try:
            backend = BACKENDS[key]()
            ok, reason = backend.available()
        except Exception as exc:  # pragma: no cover - defensive
            ok, reason = False, str(exc)
        seen[key] = {"name": key, "available": ok, "reason": reason}
    return list(seen.values())

"""The translation contract.

The important design decision lives in this signature: translators receive a
*list* of units plus their surrounding context, never a single string. Subtitle
cues are sentence fragments; translating them one at a time is what produces
the disjointed output you see in most auto-translated subtitles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

ProgressFn = Callable[[float, str], None]


def noop_progress(fraction: float, message: str) -> None:  # pragma: no cover
    return None


@dataclass
class TextUnit:
    """One translatable item: a subtitle cue or a document paragraph."""

    id: int
    text: str
    speaker: str | None = None
    max_chars: int | None = None  # soft budget so the result fits on screen


@dataclass
class TranslationOptions:
    target: str                       # ISO code, required
    source: str | None = None         # None = let the model infer
    mode: str = "subtitle"            # "subtitle" | "document"
    window: int = 24                  # units translated per request
    context: int = 6                  # preceding units supplied as context
    formality: str | None = None      # "formal" | "informal" | None
    glossary: dict[str, str] = field(default_factory=dict)
    notes: str | None = None          # free-form domain guidance
    max_chars: int | None = None      # per-unit budget for subtitle mode


@runtime_checkable
class Translator(Protocol):
    name: str

    def translate(
        self,
        units: list[TextUnit],
        options: TranslationOptions,
        progress: ProgressFn = noop_progress,
    ) -> list[str]:
        ...


class BaseTranslator:
    name = "base"
    supports_glossary = False

    def translate(
        self,
        units: list[TextUnit],
        options: TranslationOptions,
        progress: ProgressFn = noop_progress,
    ) -> list[str]:
        raise NotImplementedError

    def available(self) -> tuple[bool, str]:
        return True, ""

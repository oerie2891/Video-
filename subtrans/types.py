"""Core data model shared by every backend.

Keeping these dataclasses free of backend specifics is what lets a cloud ASR
API and a local model be swapped without touching the cue or format layers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Word:
    """A single token with its own timing. Word timings are what make cue
    boundaries land on speech instead of mid-syllable."""

    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass
class Segment:
    """An ASR-native chunk of speech. Backends differ wildly in how long these
    are, which is exactly why we re-segment into cues downstream."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None
    no_speech_prob: float | None = None
    avg_logprob: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    language: str | None
    language_probability: float | None = None
    duration: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    source: str | None = None
    model: str | None = None
    backend: str | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def words(self) -> list[Word]:
        out: list[Word] = []
        for seg in self.segments:
            out.extend(seg.words)
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Cue:
    """A display-ready subtitle event: timing plus already-wrapped lines."""

    index: int
    start: float
    end: float
    lines: list[str]
    speaker: str | None = None
    source_text: str | None = None  # original text, when this cue was translated

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def flat_text(self) -> str:
        return " ".join(line.strip() for line in self.lines).strip()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def cps(self) -> float:
        """Characters per second — the reading-speed metric subtitlers use."""
        chars = len(self.flat_text)
        return chars / self.duration if self.duration > 0 else float("inf")


@dataclass
class Subtitles:
    cues: list[Cue]
    language: str | None = None
    source_language: str | None = None
    title: str | None = None
    # Populated when translated: the terminology the translator settled on.
    glossary: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cues)

    @property
    def text(self) -> str:
        return "\n".join(c.flat_text for c in self.cues)


class MediaError(RuntimeError):
    """Raised when ffmpeg cannot read or decode the input."""


class BackendError(RuntimeError):
    """Raised when an ASR or translation backend fails or is unavailable."""

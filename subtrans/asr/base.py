"""The ASR contract every backend implements.

Backends receive a path to 16 kHz mono WAV and return a `Transcript`. They must
not know about cues, formats or translation — that separation is what makes a
local model and a cloud API interchangeable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..types import Transcript

ProgressFn = Callable[[float, str], None]


def noop_progress(fraction: float, message: str) -> None:  # pragma: no cover
    return None


@dataclass
class ASROptions:
    """Knobs that are meaningful across backends."""

    language: str | None = None          # None = auto-detect
    task: str = "transcribe"             # or "translate" (to English, Whisper-native)
    beam_size: int = 5
    vad: bool = True
    word_timestamps: bool = True
    initial_prompt: str | None = None    # bias toward names/jargon
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    condition_on_previous_text: bool = False  # False markedly reduces loops
    compute_type: str | None = None
    device: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Transcriber(Protocol):
    name: str

    def transcribe(
        self,
        audio_path: str,
        options: ASROptions,
        progress: ProgressFn = noop_progress,
    ) -> Transcript:
        ...


class BaseTranscriber:
    name = "base"

    def transcribe(
        self,
        audio_path: str,
        options: ASROptions,
        progress: ProgressFn = noop_progress,
    ) -> Transcript:
        raise NotImplementedError

    def available(self) -> tuple[bool, str]:
        """(usable, reason). Lets the CLI/UI explain *why* a backend is off."""
        return True, ""

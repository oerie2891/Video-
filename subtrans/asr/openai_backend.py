"""Cloud transcription via an OpenAI-compatible /audio/transcriptions endpoint.

Works with api.openai.com (`whisper-1`, `gpt-4o-transcribe`) and any
compatible gateway (Groq, Azure OpenAI, local vLLM) via `base_url`.

Two things this handles that a naive implementation does not:

* The 25 MB request limit. Long media is split into chunks on silence where
  possible, and every returned timestamp is shifted by its chunk's offset.
* `gpt-4o-transcribe` does not return word timestamps. We ask for segment
  granularity and let the cue layer synthesise word timings, which is why
  `Transcript` tolerates word-less segments.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

from ..media import extract_audio, probe
from ..types import BackendError, Segment, Transcript, Word
from .base import ASROptions, BaseTranscriber, ProgressFn, noop_progress

# Keep comfortably under the 25 MB limit: 16 kHz mono PCM is 32 kB/s, so
# 10 minutes ≈ 19 MB.
CHUNK_SECONDS = 600.0
WORD_CAPABLE_MODELS = frozenset({"whisper-1"})


class OpenAITranscriber(BaseTranscriber):
    name = "openai"

    def __init__(
        self,
        model: str = "whisper-1",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        chunk_seconds: float = CHUNK_SECONDS,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.chunk_seconds = chunk_seconds

    def available(self) -> tuple[bool, str]:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "openai package not installed (pip install openai)"
        if not self.api_key:
            return False, "OPENAI_API_KEY is not set"
        return True, ""

    def _client(self) -> Any:
        ok, reason = self.available()
        if not ok:
            raise BackendError(reason)
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def transcribe(
        self,
        audio_path: str,
        options: ASROptions,
        progress: ProgressFn = noop_progress,
    ) -> Transcript:
        client = self._client()
        info = probe(audio_path)
        duration = info.duration
        chunks = max(1, math.ceil(duration / self.chunk_seconds)) if duration else 1

        granularities = (
            ["segment", "word"]
            if (options.word_timestamps and self.model in WORD_CAPABLE_MODELS)
            else ["segment"]
        )

        segments: list[Segment] = []
        detected = options.language
        tmpdir = tempfile.mkdtemp(prefix="subtrans-openai-")

        for i in range(chunks):
            offset = i * self.chunk_seconds
            span = min(self.chunk_seconds, duration - offset) if duration else None
            progress(i / chunks, f"Uploading chunk {i + 1}/{chunks}…")

            if chunks == 1:
                part = audio_path
            else:
                part = extract_audio(
                    audio_path,
                    dest=str(Path(tmpdir) / f"chunk-{i:04d}.wav"),
                    normalize=False,
                    start=offset,
                    duration=span,
                )

            try:
                with open(part, "rb") as fh:
                    kwargs: dict[str, Any] = {
                        "model": self.model,
                        "file": fh,
                        "response_format": "verbose_json",
                        "timestamp_granularities": granularities,
                    }
                    if options.language:
                        kwargs["language"] = options.language
                    if options.initial_prompt:
                        kwargs["prompt"] = options.initial_prompt
                    if options.task == "translate":
                        # The endpoint has a separate translations route that
                        # only ever targets English.
                        kwargs.pop("timestamp_granularities", None)
                        response = client.audio.translations.create(**kwargs)
                    else:
                        response = client.audio.transcriptions.create(**kwargs)
            except Exception as exc:
                raise BackendError(f"Cloud transcription failed on chunk {i + 1}: {exc}") from exc

            payload = _as_dict(response)
            detected = detected or payload.get("language")
            segments.extend(_parse_segments(payload, offset))

        progress(1.0, f"Transcribed {len(segments)} segments via {self.model}")
        return Transcript(
            language=_normalise_language(detected),
            duration=duration or (segments[-1].end if segments else 0.0),
            segments=segments,
            source=audio_path,
            model=self.model,
            backend=f"{self.name}:{self.model}",
        )


def _as_dict(response: Any) -> dict[str, Any]:
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(response, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    if isinstance(response, dict):
        return response
    return {"text": str(getattr(response, "text", response))}


def _parse_segments(payload: dict[str, Any], offset: float) -> list[Segment]:
    raw_segments = payload.get("segments") or []
    raw_words = payload.get("words") or []

    words = [
        Word(
            start=float(w["start"]) + offset,
            end=float(w["end"]) + offset,
            text=str(w.get("word") or w.get("text") or ""),
        )
        for w in raw_words
        if w.get("start") is not None and w.get("end") is not None
    ]

    if not raw_segments:
        text = (payload.get("text") or "").strip()
        if not text:
            return []
        end = words[-1].end if words else offset
        return [Segment(start=offset, end=end, text=text, words=words)]

    out: list[Segment] = []
    for seg in raw_segments:
        start = float(seg.get("start", 0.0)) + offset
        end = float(seg.get("end", start)) + offset
        owned = [w for w in words if start - 0.01 <= w.start <= end + 0.01]
        out.append(
            Segment(
                start=start,
                end=end,
                text=str(seg.get("text", "")),
                words=owned,
                no_speech_prob=seg.get("no_speech_prob"),
                avg_logprob=seg.get("avg_logprob"),
            )
        )
    return out


def _normalise_language(value: str | None) -> str | None:
    """The API sometimes returns a full name ('english') rather than a code."""
    if not value:
        return None
    from ..languages import normalize

    try:
        return normalize(value)
    except ValueError:
        return value

"""Local transcription via faster-whisper (CTranslate2).

Default backend: no API key, no network after the first model download, and
unlimited use. The cost is speed — on CPU expect roughly 1–3× realtime for
`large-v3` with int8, which is why the pipeline reports progress.

Model choice guidance (word error rate vs. speed, English and major languages):
  tiny/base        - draft quality only; useful for smoke tests
  small            - usable for clear single-speaker audio
  medium           - good; a reasonable CPU default
  large-v3         - best quality, slowest
  large-v3-turbo   - near-large quality at ~4x the speed; best default if
                     the extra download is acceptable
  distil-large-v3  - English-only, very fast, close to large-v3 on English
"""

from __future__ import annotations

import os
from typing import Any

from ..types import BackendError, Segment, Transcript, Word
from .base import ASROptions, BaseTranscriber, ProgressFn, noop_progress

MODEL_SIZES = (
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "turbo", "distil-small.en", "distil-medium.en",
    "distil-large-v2", "distil-large-v3",
)

DEFAULT_MODEL = os.environ.get("SUBTRANS_WHISPER_MODEL", "small")


class FasterWhisperTranscriber(BaseTranscriber):
    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        compute_type: str | None = None,
        download_root: str | None = None,
        cpu_threads: int = 0,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root or os.environ.get("SUBTRANS_MODEL_DIR")
        self.cpu_threads = cpu_threads
        self._model: Any = None
        self._resolved: tuple[str, str] | None = None

    # -- lifecycle ---------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False, "faster-whisper is not installed (pip install faster-whisper)"
        return True, ""

    def _resolve_device(self) -> tuple[str, str]:
        if self._resolved:
            return self._resolved
        device = self.device
        compute = self.compute_type
        if device is None:
            device = "cpu"
            try:  # CUDA is a large win when present; never required.
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
            except Exception:
                device = "cpu"
        if compute is None:
            compute = "float16" if device == "cuda" else "int8"
        self._resolved = (device, compute)
        return self._resolved

    def load(self, progress: ProgressFn = noop_progress) -> Any:
        if self._model is not None:
            return self._model
        ok, reason = self.available()
        if not ok:
            raise BackendError(reason)
        from faster_whisper import WhisperModel

        device, compute = self._resolve_device()
        progress(0.0, f"Loading {self.model_size} on {device} ({compute})…")
        try:
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute,
                download_root=self.download_root,
                cpu_threads=self.cpu_threads,
            )
        except Exception as exc:
            raise BackendError(
                f"Could not load Whisper model {self.model_size!r} on {device}: {exc}"
            ) from exc
        return self._model

    # -- transcription -----------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        options: ASROptions,
        progress: ProgressFn = noop_progress,
    ) -> Transcript:
        model = self.load(progress)
        device, compute = self._resolve_device()

        vad_params = {"min_silence_duration_ms": 500, "speech_pad_ms": 200}
        kwargs: dict[str, Any] = {
            "language": options.language,
            "task": options.task,
            "beam_size": options.beam_size,
            "word_timestamps": options.word_timestamps,
            "vad_filter": options.vad,
            "vad_parameters": vad_params if options.vad else None,
            "temperature": list(options.temperature),
            "condition_on_previous_text": options.condition_on_previous_text,
            "initial_prompt": options.initial_prompt,
            # Whisper's own guards against degenerate output.
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
        kwargs.update(options.extra or {})

        progress(0.02, "Detecting language…" if not options.language else "Transcribing…")
        try:
            iterator, info = model.transcribe(audio_path, **kwargs)
        except Exception as exc:
            raise BackendError(f"Transcription failed: {exc}") from exc

        total = float(getattr(info, "duration", 0.0) or 0.0)
        detected = getattr(info, "language", None) or options.language
        lang_prob = getattr(info, "language_probability", None)

        segments: list[Segment] = []
        try:
            for seg in iterator:  # lazy: this is where the work happens
                words = [
                    Word(
                        start=float(w.start),
                        end=float(w.end),
                        text=w.word,
                        probability=getattr(w, "probability", None),
                    )
                    for w in (getattr(seg, "words", None) or [])
                    if w.start is not None and w.end is not None
                ]
                segments.append(
                    Segment(
                        start=float(seg.start),
                        end=float(seg.end),
                        text=seg.text,
                        words=words,
                        no_speech_prob=getattr(seg, "no_speech_prob", None),
                        avg_logprob=getattr(seg, "avg_logprob", None),
                    )
                )
                if total > 0:
                    frac = min(0.98, max(0.02, float(seg.end) / total))
                    progress(frac, f"Transcribing… {_clock(seg.end)} / {_clock(total)}")
        except Exception as exc:
            raise BackendError(f"Transcription aborted after {len(segments)} segments: {exc}") from exc

        progress(1.0, f"Transcribed {len(segments)} segments")
        return Transcript(
            language=detected,
            language_probability=float(lang_prob) if lang_prob is not None else None,
            duration=total or (segments[-1].end if segments else 0.0),
            segments=segments,
            source=audio_path,
            model=self.model_size,
            backend=f"{self.name}:{device}/{compute}",
        )


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

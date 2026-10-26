"""End-to-end orchestration: media or text in, subtitles and transcripts out.

One entry point (`run`) is shared by the CLI and the web server so both behave
identically. Everything expensive reports progress, because on CPU a long file
takes long enough that a silent process looks broken.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import languages as lang_mod
from .asr import ASROptions, get_transcriber
from .cues import reading_speed_report, rewrap, segment_cues
from .formats import EXTENSIONS, read_subtitles, read_text_blocks, render, sniff
from .languages import style_for
from .media import classify, extract_audio, have_ffmpeg, probe
from .translate import TextUnit, TranslationOptions, get_translator
from .types import BackendError, Cue, MediaError, Subtitles, Transcript

ProgressFn = Callable[[float, str], None]


def _noop(fraction: float, message: str) -> None:
    return None


@dataclass
class JobSpec:
    """Everything one run needs. Defaults are the ones I'd actually use."""

    input_path: str
    targets: list[str] = field(default_factory=list)   # [] => transcribe only
    source_language: str | None = None                 # None => auto-detect
    formats: list[str] = field(default_factory=lambda: ["srt"])
    output_dir: str | None = None
    output_stem: str | None = None

    # ASR
    asr_backend: str = "auto"
    asr_model: str | None = None
    device: str | None = None
    compute_type: str | None = None
    vad: bool = True
    normalize_audio: bool = True
    initial_prompt: str | None = None
    beam_size: int = 5
    word_timestamps: bool = True

    # Translation
    mt_backend: str = "auto"
    mt_model: str | None = None
    window: int = 24
    context: int = 6
    formality: str | None = None
    glossary: dict[str, str] = field(default_factory=dict)
    notes: str | None = None

    # Output shaping
    include_source: bool = True
    timestamps_in_text: bool = True
    speakers_in_output: bool = True
    keep_audio: bool = False
    burn_in: bool = False


@dataclass
class Artifact:
    language: str
    format: str
    path: str
    size: int

    @property
    def filename(self) -> str:
        return Path(self.path).name


@dataclass
class JobResult:
    kind: str                                   # "media" | "subtitle" | "text"
    source_language: str | None = None
    transcript: Transcript | None = None
    subtitles: dict[str, Subtitles] = field(default_factory=dict)
    documents: dict[str, list[str]] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    glossary: dict[str, str] = field(default_factory=dict)


class _Stage:
    """Maps a sub-task's 0..1 progress onto a slice of the overall bar."""

    def __init__(self, progress: ProgressFn, start: float, end: float, label: str = "") -> None:
        self._progress = progress
        self._start = start
        self._end = end
        self._label = label

    def __call__(self, fraction: float, message: str) -> None:
        fraction = min(1.0, max(0.0, fraction))
        overall = self._start + (self._end - self._start) * fraction
        self._progress(overall, message or self._label)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def transcribe(
    input_path: str,
    spec: JobSpec | None = None,
    progress: ProgressFn = _noop,
) -> tuple[Transcript, Subtitles]:
    """Media file -> (raw transcript, segmented cues)."""
    spec = spec or JobSpec(input_path=input_path)
    if not have_ffmpeg():
        raise MediaError(
            "ffmpeg and ffprobe are required to read media. "
            "Install with: apt install ffmpeg (or brew install ffmpeg)"
        )

    info = probe(input_path)
    progress(0.01, f"Read {Path(input_path).name} ({_clock(info.duration)})")

    audio_path: str | None = None
    temp_audio = False
    try:
        progress(0.03, "Extracting audio…")
        audio_path = extract_audio(input_path, normalize=spec.normalize_audio)
        temp_audio = True

        backend = get_transcriber(
            spec.asr_backend,
            model=spec.asr_model,
            device=spec.device,
            compute_type=spec.compute_type,
        )
        options = ASROptions(
            language=lang_mod.normalize(spec.source_language),
            beam_size=spec.beam_size,
            vad=spec.vad,
            word_timestamps=spec.word_timestamps,
            initial_prompt=spec.initial_prompt,
        )
        transcript = backend.transcribe(
            audio_path, options, _Stage(progress, 0.05, 0.95, "Transcribing…")
        )
        transcript.source = input_path
        if not transcript.duration:
            transcript.duration = info.duration

        progress(0.96, "Building cues…")
        subs = segment_cues(transcript, language=transcript.language)
        subs.title = info.title or Path(input_path).stem
        progress(1.0, f"{len(subs.cues)} cues")
        return transcript, subs
    finally:
        if audio_path and temp_audio and not spec.keep_audio:
            try:
                os.unlink(audio_path)
            except OSError:
                pass


def translate_subtitles(
    subs: Subtitles,
    target: str,
    *,
    source: str | None = None,
    backend: str = "auto",
    model: str | None = None,
    window: int = 24,
    context: int = 6,
    formality: str | None = None,
    glossary: dict[str, str] | None = None,
    notes: str | None = None,
    progress: ProgressFn = _noop,
) -> Subtitles:
    """Translate cues with surrounding context, then re-wrap for the target.

    Timings are preserved exactly; only the text and the line breaks change.
    Reading speed is re-checked afterwards because translated text is usually a
    different length from the source.
    """
    target = lang_mod.normalize(target) or target
    source = lang_mod.normalize(source or subs.language)
    translator = get_translator(backend, model=model)
    ok, reason = translator.available()
    if not ok:
        raise BackendError(reason)

    target_style = style_for(target)
    units = [
        TextUnit(
            id=cue.index,
            text=cue.flat_text,
            speaker=cue.speaker,
            max_chars=target_style.max_chars,
        )
        for cue in subs.cues
    ]
    options = TranslationOptions(
        target=target,
        source=source,
        mode="subtitle",
        window=window,
        context=context,
        formality=formality,
        glossary=dict(glossary or {}),
        notes=notes,
        max_chars=target_style.max_chars,
    )
    texts = translator.translate(units, options, progress)

    out = Subtitles(
        cues=[
            Cue(
                index=cue.index,
                start=cue.start,
                end=cue.end,
                lines=[text],
                speaker=cue.speaker,
                source_text=cue.flat_text,
            )
            for cue, text in zip(subs.cues, texts, strict=True)
        ],
        language=target,
        source_language=source,
        title=subs.title,
    )
    rewrap(out, language=target, style=target_style)
    out.glossary = dict(options.glossary)
    return out


def translate_blocks(
    blocks: list[str],
    target: str,
    *,
    source: str | None = None,
    backend: str = "auto",
    model: str | None = None,
    window: int = 12,
    context: int = 3,
    formality: str | None = None,
    glossary: dict[str, str] | None = None,
    notes: str | None = None,
    progress: ProgressFn = _noop,
) -> tuple[list[str], dict[str, str]]:
    """Translate document paragraphs. No character budget, structure preserved."""
    target = lang_mod.normalize(target) or target
    source = lang_mod.normalize(source)
    translator = get_translator(backend, model=model)
    ok, reason = translator.available()
    if not ok:
        raise BackendError(reason)

    units = [TextUnit(id=i + 1, text=text) for i, text in enumerate(blocks)]
    options = TranslationOptions(
        target=target,
        source=source,
        mode="document",
        window=window,
        context=context,
        formality=formality,
        glossary=dict(glossary or {}),
        notes=notes,
    )
    return translator.translate(units, options, progress), options.glossary


# --------------------------------------------------------------------------
# One-stop entry point
# --------------------------------------------------------------------------

def run(spec: JobSpec, progress: ProgressFn = _noop) -> JobResult:
    started = time.monotonic()
    src = Path(spec.input_path)
    if not src.exists():
        raise MediaError(f"No such file: {src}")

    kind = classify(src)
    if kind == "unknown":
        # The extension is unhelpful, so ask ffprobe, then the content sniffer.
        try:
            info = probe(src)
            kind = "media" if (info.has_audio or info.has_video) else "text"
        except MediaError:
            kind = "subtitle" if sniff(src) in ("srt", "vtt", "ass", "sbv") else "text"

    targets = [lang_mod.normalize(t) or t for t in spec.targets if t]
    result = JobResult(kind=kind)

    if kind == "media":
        _run_media(spec, result, targets, progress)
    elif kind == "subtitle":
        _run_subtitle_file(spec, result, targets, progress)
    else:
        _run_text_file(spec, result, targets, progress)

    result.elapsed = time.monotonic() - started
    progress(1.0, f"Done in {result.elapsed:.1f}s — {len(result.artifacts)} file(s)")
    return result


def _run_media(
    spec: JobSpec, result: JobResult, targets: list[str], progress: ProgressFn
) -> None:
    asr_share = 0.75 if targets else 0.95
    transcript, subs = transcribe(
        spec.input_path, spec, _Stage(progress, 0.0, asr_share)
    )

    result.transcript = transcript
    result.source_language = transcript.language
    source = transcript.language
    if source:
        tier = lang_mod.quality_tier(source)
        if tier != "good":
            result.warnings.append(
                f"Detected source language {lang_mod.display_name(source)} is in the "
                f"'{tier}' accuracy tier for Whisper — expect a higher error rate and "
                f"proofread before publishing."
            )
    if transcript.language_probability is not None and transcript.language_probability < 0.6:
        result.warnings.append(
            f"Language detection confidence was only "
            f"{transcript.language_probability:.0%}; pass --source to force it."
        )
    if not subs.cues:
        result.warnings.append("No speech was detected in this file.")

    if spec.include_source or not targets:
        result.subtitles[source or "source"] = subs
    result.metrics[source or "source"] = reading_speed_report(subs)

    _translate_into(spec, result, subs, targets, source, progress, start=asr_share)
    _write_outputs(spec, result)
    if spec.burn_in and result.subtitles:
        _burn(spec, result)


def _run_subtitle_file(
    spec: JobSpec, result: JobResult, targets: list[str], progress: ProgressFn
) -> None:
    progress(0.05, f"Parsing {Path(spec.input_path).name}…")
    subs = read_subtitles(spec.input_path)
    subs.language = lang_mod.normalize(spec.source_language) or subs.language
    result.source_language = subs.language
    if not subs.cues:
        raise MediaError(f"No cues found in {Path(spec.input_path).name}")
    progress(0.1, f"{len(subs.cues)} cues parsed")

    if spec.include_source or not targets:
        result.subtitles[subs.language or "source"] = subs
    result.metrics[subs.language or "source"] = reading_speed_report(subs)
    _translate_into(spec, result, subs, targets, subs.language, progress, start=0.1)
    _write_outputs(spec, result)


def _run_text_file(
    spec: JobSpec, result: JobResult, targets: list[str], progress: ProgressFn
) -> None:
    progress(0.05, f"Reading {Path(spec.input_path).name}…")
    blocks, _ = read_text_blocks(spec.input_path)
    if not blocks:
        raise MediaError(f"{Path(spec.input_path).name} contains no readable text")
    source = lang_mod.normalize(spec.source_language)
    result.source_language = source
    if spec.include_source or not targets:
        result.documents[source or "source"] = blocks
    if not targets:
        result.warnings.append(
            "A text file was given with no target language, so there was nothing "
            "to do but copy it. Pass --to <language> to translate."
        )

    span = 0.9 / max(1, len(targets))
    for i, target in enumerate(targets):
        stage = _Stage(progress, 0.1 + i * span, 0.1 + (i + 1) * span)
        translated, glossary = translate_blocks(
            blocks,
            target,
            source=source,
            backend=spec.mt_backend,
            model=spec.mt_model,
            formality=spec.formality,
            glossary={**spec.glossary, **result.glossary},
            notes=spec.notes,
            progress=stage,
        )
        result.documents[target] = translated
        result.glossary.update(glossary)
    _write_outputs(spec, result)


def _translate_into(
    spec: JobSpec,
    result: JobResult,
    subs: Subtitles,
    targets: list[str],
    source: str | None,
    progress: ProgressFn,
    *,
    start: float,
) -> None:
    if not targets:
        return
    span = (0.95 - start) / len(targets)
    for i, target in enumerate(targets):
        if source and target == source:
            result.warnings.append(
                f"Target {lang_mod.display_name(target)} matches the source; "
                f"skipped translation."
            )
            result.subtitles.setdefault(target, subs)
            continue
        stage = _Stage(progress, start + i * span, start + (i + 1) * span)
        try:
            translated = translate_subtitles(
                subs,
                target,
                source=source,
                backend=spec.mt_backend,
                model=spec.mt_model,
                window=spec.window,
                context=spec.context,
                formality=spec.formality,
                glossary={**spec.glossary, **result.glossary},
                notes=spec.notes,
                progress=stage,
            )
        except BackendError as exc:
            # Whisper can translate to English without any MT backend at all.
            if target == "en" and result.transcript is not None:
                result.warnings.append(
                    f"{exc} Falling back to Whisper's own speech-to-English "
                    f"translation, which is lower quality than the LLM path."
                )
                translated = _whisper_translate_to_english(spec, stage)
                if translated is None:
                    result.warnings.append("Whisper fallback also failed; skipped English.")
                    continue
            else:
                result.warnings.append(f"Could not translate into {target}: {exc}")
                continue
        result.subtitles[target] = translated
        result.metrics[target] = reading_speed_report(translated)
        result.glossary.update(getattr(translated, "glossary", {}) or {})

        report = result.metrics[target]
        if report.get("over_cps"):
            result.warnings.append(
                f"{report['over_cps']} of {report['cues']} {target} cues exceed the "
                f"reading-speed limit (max {report['max_cps']} cps). The text is "
                f"accurate but fast to read."
            )


def _whisper_translate_to_english(spec: JobSpec, progress: ProgressFn) -> Subtitles | None:
    """Whisper's built-in speech-to-English translation. English only."""
    audio = None
    try:
        audio = extract_audio(spec.input_path, normalize=spec.normalize_audio)
        backend = get_transcriber(
            spec.asr_backend, model=spec.asr_model, device=spec.device,
            compute_type=spec.compute_type,
        )
        options = ASROptions(
            language=lang_mod.normalize(spec.source_language),
            task="translate",
            beam_size=spec.beam_size,
            vad=spec.vad,
            word_timestamps=spec.word_timestamps,
        )
        transcript = backend.transcribe(audio, options, progress)
        subs = segment_cues(transcript, language="en")
        subs.language = "en"
        subs.source_language = spec.source_language
        return subs
    except Exception:
        return None
    finally:
        if audio:
            try:
                os.unlink(audio)
            except OSError:
                pass


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

_DOC_FORMATS = {"txt", "md", "docx", "json"}


def _write_outputs(spec: JobSpec, result: JobResult) -> None:
    out_dir = Path(spec.output_dir or Path(spec.input_path).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = spec.output_stem or Path(spec.input_path).stem

    for language, subs in result.subtitles.items():
        for fmt in spec.formats:
            fmt = fmt.lower().lstrip(".")
            if fmt == "docx":
                path = out_dir / f"{stem}.{language}.docx"
                _write_docx([c.flat_text for c in subs.cues], path, title=subs.title)
            else:
                ext = EXTENSIONS.get(fmt)
                if ext is None:
                    result.warnings.append(f"Unknown format {fmt!r}; skipped.")
                    continue
                body = render(
                    subs,
                    fmt,
                    timestamps=spec.timestamps_in_text,
                    speakers=spec.speakers_in_output,
                    transcript=result.transcript,
                    metrics=result.metrics.get(language),
                )
                path = out_dir / f"{stem}.{language}{ext}"
                path.write_text(body, encoding="utf-8")
            result.artifacts.append(
                Artifact(language=language, format=fmt, path=str(path), size=path.stat().st_size)
            )

    for language, blocks in result.documents.items():
        formats = [f for f in spec.formats if f.lower().lstrip(".") in _DOC_FORMATS] or ["txt"]
        for fmt in formats:
            fmt = fmt.lower().lstrip(".")
            path = out_dir / f"{stem}.{language}{'.docx' if fmt == 'docx' else EXTENSIONS.get(fmt, '.txt')}"
            if fmt == "docx":
                _write_docx(blocks, path, title=stem)
            elif fmt == "json":
                import json as _json

                path.write_text(
                    _json.dumps({"language": language, "blocks": blocks}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            elif fmt == "md":
                path.write_text(f"# {stem}\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
            else:
                path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
            result.artifacts.append(
                Artifact(language=language, format=fmt, path=str(path), size=path.stat().st_size)
            )


def _write_docx(blocks: list[str], path: Path, *, title: str | None = None) -> None:
    try:
        import docx  # type: ignore
    except ImportError as exc:
        raise BackendError("Writing .docx needs python-docx (pip install python-docx)") from exc
    document = docx.Document()
    if title:
        document.add_heading(title, level=1)
    for block in blocks:
        document.add_paragraph(block)
    document.save(str(path))


def _burn(spec: JobSpec, result: JobResult) -> None:
    from .media import burn_in

    srt_artifacts = [a for a in result.artifacts if a.format == "srt"]
    if not srt_artifacts:
        result.warnings.append("Burn-in needs an SRT output; add 'srt' to --format.")
        return
    chosen = srt_artifacts[-1]
    out = Path(chosen.path).with_suffix("")
    dest = out.parent / f"{out.name}.burned.mp4"
    try:
        burn_in(spec.input_path, chosen.path, dest)
    except MediaError as exc:
        result.warnings.append(f"Burn-in failed: {exc}")
        return
    result.artifacts.append(
        Artifact(language=chosen.language, format="mp4", path=str(dest), size=dest.stat().st_size)
    )


def _clock(seconds: float) -> str:
    m, s = divmod(int(max(0.0, seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

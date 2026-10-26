"""Subtitle and transcript writers."""

from __future__ import annotations

import html
import json
from typing import Any

from ..types import Subtitles, Transcript

__all__ = [
    "write_srt", "write_vtt", "write_ass", "write_txt", "write_md",
    "write_json", "write_tsv", "write_lrc", "render", "EXTENSIONS",
]

EXTENSIONS = {
    "srt": ".srt", "vtt": ".vtt", "ass": ".ass", "txt": ".txt",
    "md": ".md", "json": ".json", "tsv": ".tsv", "lrc": ".lrc",
}


def _ts(seconds: float, *, sep: str = ",", hours_digits: int = 2) -> str:
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:0{hours_digits}d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _speaker_prefix(cue: Any, show: bool) -> str:
    return f"{cue.speaker}: " if (show and cue.speaker) else ""


def write_srt(subs: Subtitles, *, speakers: bool = True) -> str:
    blocks: list[str] = []
    for i, cue in enumerate(subs.cues, start=1):
        lines = list(cue.lines) or [""]
        if speakers and cue.speaker:
            lines = [f"{cue.speaker}: {lines[0]}", *lines[1:]]
        blocks.append(
            f"{i}\n{_ts(cue.start)} --> {_ts(cue.end)}\n" + "\n".join(lines)
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_vtt(subs: Subtitles, *, speakers: bool = True) -> str:
    head = "WEBVTT"
    if subs.language:
        head += f"\nLanguage: {subs.language}"
    blocks = [head]
    for i, cue in enumerate(subs.cues, start=1):
        lines = list(cue.lines) or [""]
        body = "\n".join(html.escape(ln, quote=False) for ln in lines)
        if speakers and cue.speaker:
            body = f"<v {html.escape(cue.speaker, quote=False)}>{body}"
        blocks.append(
            f"{i}\n{_ts(cue.start, sep='.')} --> {_ts(cue.end, sep='.')}\n{body}"
        )
    return "\n\n".join(blocks) + "\n"


_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,54,&H00FFFFFF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,2.6,0,2,96,96,54,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_ass(subs: Subtitles, *, speakers: bool = True) -> str:
    rows: list[str] = []
    for cue in subs.cues:
        text = "\\N".join(cue.lines).replace("\n", "\\N")
        name = cue.speaker or ""
        if speakers and cue.speaker:
            text = f"{cue.speaker}: {text}"
        rows.append(
            f"Dialogue: 0,{_ts(cue.start, sep='.', hours_digits=1)[:-1]},"
            f"{_ts(cue.end, sep='.', hours_digits=1)[:-1]},Default,{name},0,0,0,,{text}"
        )
    return _ASS_HEADER + "\n".join(rows) + "\n"


def write_lrc(subs: Subtitles, **_: Any) -> str:
    """Lyric format — useful for music and for players that want a simple
    time-tagged line list."""
    rows = []
    for cue in subs.cues:
        total = max(0.0, cue.start)
        m, s = divmod(total, 60)
        rows.append(f"[{int(m):02d}:{s:05.2f}]{cue.flat_text}")
    return "\n".join(rows) + "\n"


def write_txt(
    subs: Subtitles,
    *,
    timestamps: bool = False,
    speakers: bool = True,
    paragraphs: bool = True,
    paragraph_gap: float = 1.6,
) -> str:
    """Readable transcript rather than a cue list.

    Cues are reflowed into paragraphs on speaker change or a real pause, which
    is what makes a transcript usable as a document instead of 700 short lines.
    """
    if not subs.cues:
        return ""
    out: list[str] = []
    buffer: list[str] = []
    current_speaker: str | None = subs.cues[0].speaker
    block_start = subs.cues[0].start

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(buffer).strip()
        prefix = ""
        if timestamps:
            prefix += f"[{_ts(block_start, sep='.')[:-4]}] "
        if speakers and current_speaker:
            prefix += f"{current_speaker}: "
        out.append(prefix + text)
        buffer.clear()

    previous_end = subs.cues[0].start
    for cue in subs.cues:
        gap = cue.start - previous_end
        breaks = (
            paragraphs
            and buffer
            and (cue.speaker != current_speaker or gap >= paragraph_gap)
        )
        if breaks:
            flush()
            current_speaker = cue.speaker
            block_start = cue.start
        if not buffer:
            block_start = cue.start
            current_speaker = cue.speaker
        buffer.append(cue.flat_text)
        previous_end = cue.end
    flush()
    return "\n\n".join(out) + "\n"


def write_md(
    subs: Subtitles,
    *,
    timestamps: bool = True,
    speakers: bool = True,
    paragraphs: bool = True,
    paragraph_gap: float = 1.6,
    **_: Any,
) -> str:
    title = subs.title or "Transcript"
    head = [f"# {title}", ""]
    meta = []
    if subs.language:
        meta.append(f"**Language:** {subs.language}")
    if subs.source_language and subs.source_language != subs.language:
        meta.append(f"**Translated from:** {subs.source_language}")
    if subs.cues:
        meta.append(f"**Duration:** {_ts(subs.cues[-1].end, sep='.')[:-4]}")
    if meta:
        head += [" · ".join(meta), ""]
    # Forward only what write_txt accepts: `render` may hand us extras such as
    # `transcript` and `metrics` that are meaningful to other writers.
    body = write_txt(
        subs,
        timestamps=timestamps,
        speakers=speakers,
        paragraphs=paragraphs,
        paragraph_gap=paragraph_gap,
    )
    return "\n".join(head) + "\n" + body


def write_tsv(subs: Subtitles, **_: Any) -> str:
    rows = ["start\tend\tspeaker\ttext"]
    for cue in subs.cues:
        rows.append(
            f"{cue.start:.3f}\t{cue.end:.3f}\t{cue.speaker or ''}\t"
            f"{cue.flat_text}"
        )
    return "\n".join(rows) + "\n"


def write_json(
    subs: Subtitles,
    *,
    transcript: Transcript | None = None,
    metrics: dict[str, Any] | None = None,
    **_: Any,
) -> str:
    payload: dict[str, Any] = {
        "language": subs.language,
        "source_language": subs.source_language,
        "title": subs.title,
        "cues": [
            {
                "index": cue.index,
                "start": round(cue.start, 3),
                "end": round(cue.end, 3),
                "text": cue.flat_text,
                "lines": cue.lines,
                "speaker": cue.speaker,
                "cps": round(cue.cps, 1) if cue.duration else None,
                **({"source_text": cue.source_text} if cue.source_text else {}),
            }
            for cue in subs.cues
        ],
    }
    if metrics:
        payload["metrics"] = metrics
    if transcript is not None:
        payload["asr"] = {
            "model": transcript.model,
            "backend": transcript.backend,
            "duration": round(transcript.duration, 3),
            "language_probability": transcript.language_probability,
        }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


_WRITERS = {
    "srt": write_srt,
    "vtt": write_vtt,
    "ass": write_ass,
    "txt": write_txt,
    "md": write_md,
    "json": write_json,
    "tsv": write_tsv,
    "lrc": write_lrc,
}


def render(subs: Subtitles, fmt: str, **kwargs: Any) -> str:
    fmt = fmt.lower().lstrip(".")
    if fmt == "text":
        fmt = "txt"
    if fmt == "markdown":
        fmt = "md"
    if fmt not in _WRITERS:
        raise ValueError(
            f"Unknown output format {fmt!r}. Available: {', '.join(sorted(_WRITERS))}"
        )
    writer = _WRITERS[fmt]
    # Writers accept different keyword sets; pass only what each one takes.
    import inspect

    accepted = inspect.signature(writer).parameters
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in accepted.values())
    if has_var_kw:
        return writer(subs, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return writer(subs, **filtered)

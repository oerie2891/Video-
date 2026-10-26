"""Readers for existing subtitle and text files.

These feed the "translate a file I already have" path: an SRT keeps its exact
timings and only the text is replaced, and a document is split into blocks that
survive a round trip.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..types import Cue, Subtitles

__all__ = ["read_subtitles", "read_text_blocks", "sniff", "load_any"]

_TIME_RE = re.compile(
    r"(?P<h>\d{1,3}):(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,](?P<ms>\d{1,3}))?"
)
_ARROW_RE = re.compile(r"-{1,3}>|=>")


def _parse_time(raw: str) -> float:
    m = _TIME_RE.search(raw.strip())
    if not m:
        # SBV / LRC style m:ss.cc
        alt = re.search(r"(?P<m>\d{1,3}):(?P<s>\d{1,2})(?:[.,](?P<ms>\d{1,3}))?", raw)
        if not alt:
            raise ValueError(f"Unparseable timestamp: {raw!r}")
        ms = (alt.group("ms") or "0").ljust(3, "0")
        return int(alt.group("m")) * 60 + int(alt.group("s")) + int(ms) / 1000
    ms = (m.group("ms") or "0").ljust(3, "0")
    return (
        int(m.group("h")) * 3600
        + int(m.group("m")) * 60
        + int(m.group("s"))
        + int(ms) / 1000
    )


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def sniff(path: str | Path) -> str:
    """Identify a text-ish input: 'srt', 'vtt', 'ass', 'sbv', 'json', 'docx', 'text'."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".docx":
        return "docx"
    if ext in (".ass", ".ssa"):
        return "ass"
    if ext == ".json":
        return "json"
    head = _read_text(p)[:4000]
    if head.lstrip().upper().startswith("WEBVTT"):
        return "vtt"
    if "[Script Info]" in head or "Dialogue:" in head:
        return "ass"
    if re.search(r"^\s*\d+\s*$", head, re.M) and "-->" in head:
        return "srt"
    if "-->" in head:
        return "vtt" if ext == ".vtt" else "srt"
    if re.search(r"^\d{1,2}:\d{2}:\d{2}\.\d{3},\d{1,2}:\d{2}:\d{2}\.\d{3}", head, re.M):
        return "sbv"
    if ext == ".json":
        return "json"
    return "text"


def read_subtitles(path: str | Path) -> Subtitles:
    """Parse any supported subtitle file into cues, preserving timings."""
    p = Path(path)
    kind = sniff(p)
    if kind == "ass":
        cues = _parse_ass(_read_text(p))
    elif kind == "sbv":
        cues = _parse_sbv(_read_text(p))
    elif kind == "json":
        cues = _parse_json(_read_text(p))
    elif kind in ("srt", "vtt"):
        cues = _parse_srt_vtt(_read_text(p))
    else:
        raise ValueError(
            f"{p.name} does not look like a subtitle file (detected: {kind}). "
            f"Use the text path for plain documents."
        )
    for i, cue in enumerate(cues, start=1):
        cue.index = i
    return Subtitles(cues=cues, title=p.stem)


def _parse_srt_vtt(text: str) -> list[Cue]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    blocks = re.split(r"\n{2,}", text)
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        upper = lines[0].strip().upper()
        if upper.startswith("WEBVTT") or upper.startswith("NOTE") or upper.startswith("STYLE"):
            continue
        timing_idx = next(
            (i for i, ln in enumerate(lines) if _ARROW_RE.search(ln)), None
        )
        if timing_idx is None:
            continue
        parts = _ARROW_RE.split(lines[timing_idx], maxsplit=1)
        if len(parts) != 2:
            continue
        left, right = parts
        try:
            start = _parse_time(left)
            end = _parse_time(right)
        except ValueError:
            continue
        body = lines[timing_idx + 1:]
        if not body:
            continue
        speaker: str | None = None
        cleaned: list[str] = []
        for ln in body:
            voice = re.match(r"\s*<v(?:\.\S+)?\s+([^>]+)>(.*)$", ln)
            if voice:
                speaker = voice.group(1).strip()
                ln = voice.group(2)
            ln = re.sub(r"</?(?:v|c|i|b|u|ruby|rt|lang)(?:[^>]*)>", "", ln)
            ln = _unescape(ln)
            if ln.strip():
                cleaned.append(ln.strip())
        if not cleaned:
            continue
        cues.append(
            Cue(index=len(cues) + 1, start=start, end=max(end, start), lines=cleaned, speaker=speaker)
        )
    return cues


def _unescape(text: str) -> str:
    return (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )


def _parse_ass(text: str) -> list[Cue]:
    cues: list[Cue] = []
    fields: list[str] = []
    section = ""
    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.lower()
            continue
        # A script has a Format line per section; only the [Events] one
        # describes Dialogue rows.
        if section.startswith("[events") and stripped.lower().startswith("format:"):
            fields = [f.strip().lower() for f in stripped.split(":", 1)[1].split(",")]
            continue
        if not stripped.lower().startswith("dialogue:"):
            continue
        payload = stripped.split(":", 1)[1]
        if not fields:
            fields = [
                "layer", "start", "end", "style", "name", "marginl",
                "marginr", "marginv", "effect", "text",
            ]
        values = payload.split(",", len(fields) - 1)
        row = dict(zip(fields, [v.strip() for v in values], strict=False))
        try:
            start = _parse_time(row.get("start", ""))
            end = _parse_time(row.get("end", ""))
        except ValueError:
            continue
        body = row.get("text", "")
        body = re.sub(r"\{[^}]*\}", "", body)  # drop override tags
        lines = [seg.strip() for seg in re.split(r"\\[Nn]", body) if seg.strip()]
        if not lines:
            continue
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=start,
                end=max(end, start),
                lines=lines,
                speaker=(row.get("name") or None),
            )
        )
    return cues


def _parse_sbv(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\n{2,}", text.replace("\r\n", "\n")):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if len(lines) < 2 or "," not in lines[0]:
            continue
        left, right = lines[0].split(",", 1)
        try:
            start, end = _parse_time(left), _parse_time(right)
        except ValueError:
            continue
        cues.append(
            Cue(index=len(cues) + 1, start=start, end=max(end, start), lines=lines[1:])
        )
    return cues


def _parse_json(text: str) -> list[Cue]:
    """Read back our own JSON output (round-trip support)."""
    data = json.loads(text)
    raw = data.get("cues") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("JSON does not contain a cue list")
    cues: list[Cue] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        lines = entry.get("lines") or [entry.get("text", "")]
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=float(entry.get("start", 0.0)),
                end=float(entry.get("end", 0.0)),
                lines=[str(ln) for ln in lines if str(ln).strip()],
                speaker=entry.get("speaker"),
            )
        )
    return cues


def read_text_blocks(path: str | Path) -> tuple[list[str], str]:
    """Split a document into translatable blocks.

    Returns (blocks, kind). Blocks are paragraphs — translating paragraph by
    paragraph with surrounding context preserves structure and lets the output
    be reassembled exactly.
    """
    p = Path(path)
    kind = sniff(p)
    if kind == "docx":
        return _read_docx(p), "docx"
    text = _read_text(p)
    if kind == "json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("cues"), list):
            raise ValueError("That JSON is a subtitle file; use read_subtitles")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks and text.strip():
        blocks = [text.strip()]
    return blocks, "text"


def _read_docx(path: Path) -> list[str]:
    try:
        import docx  # type: ignore
    except ImportError as exc:
        raise ValueError(
            "Reading .docx needs python-docx (pip install python-docx)"
        ) from exc
    document = docx.Document(str(path))
    blocks = [para.text.strip() for para in document.paragraphs if para.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    return blocks


def load_any(path: str | Path) -> tuple[Subtitles | list[str], str]:
    """Convenience: returns (Subtitles, 'subtitle') or (blocks, 'text')."""
    kind = sniff(path)
    if kind in ("srt", "vtt", "ass", "sbv"):
        return read_subtitles(path), "subtitle"
    if kind == "json":
        try:
            return read_subtitles(path), "subtitle"
        except ValueError:
            pass
    blocks, _ = read_text_blocks(path)
    return blocks, "text"

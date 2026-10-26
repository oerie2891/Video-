"""ffmpeg/ffprobe wrappers: probing, audio extraction and chunking.

Everything that touches the container format lives here. ASR backends only
ever see a 16 kHz mono WAV, which is what every speech model wants anyway.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .types import MediaError

AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma",
    ".aiff", ".aif", ".alac", ".amr", ".ac3", ".mka", ".caf", ".m4b", ".mp2",
}
VIDEO_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpg",
    ".mpeg", ".ts", ".m2ts", ".mts", ".3gp", ".3g2", ".ogv", ".vob", ".asf",
    ".rm", ".rmvb", ".divx", ".f4v", ".mxf",
}
SUBTITLE_EXTS = {".srt", ".vtt", ".sbv", ".ass", ".ssa"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".text", ".docx", ".rst", ".csv", ".json"}

SAMPLE_RATE = 16_000


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(
            f"{name} not found on PATH. Install ffmpeg (which ships {name}): "
            f"apt install ffmpeg / brew install ffmpeg"
        )
    return path


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@dataclass
class MediaInfo:
    path: str
    duration: float
    has_audio: bool
    has_video: bool
    audio_codec: str | None = None
    video_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    size_bytes: int = 0
    format_name: str | None = None
    title: str | None = None

    @property
    def kind(self) -> str:
        if self.has_video and self.has_audio:
            return "video"
        if self.has_video:
            return "video-silent"
        if self.has_audio:
            return "audio"
        return "unknown"


def classify(path: str | os.PathLike[str]) -> str:
    """Route an input by extension: 'media', 'subtitle', 'text' or 'unknown'.

    Extension-based on purpose: we only use this to decide which pipeline to
    try first, and `probe()` is the authority for anything claiming to be media.
    """
    ext = Path(path).suffix.lower()
    if ext in AUDIO_EXTS or ext in VIDEO_EXTS:
        return "media"
    if ext in SUBTITLE_EXTS:
        return "subtitle"
    if ext in TEXT_EXTS:
        return "text"
    return "unknown"


def probe(path: str | os.PathLike[str]) -> MediaInfo:
    """Read duration and stream layout. Raises MediaError on undecodable input."""
    p = Path(path)
    if not p.exists():
        raise MediaError(f"No such file: {p}")
    cmd = [
        _tool("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(p),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        hint = detail[-1] if detail else "unknown ffprobe error"
        raise MediaError(f"ffprobe could not read {p.name}: {hint}")
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover - ffprobe misbehaving
        raise MediaError(f"Unparseable ffprobe output for {p.name}: {exc}") from exc

    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next(
        (s for s in streams
         if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) != 1),
        None,
    )

    duration = _as_float(fmt.get("duration"))
    if duration <= 0:
        for s in (audio, video):
            if s is not None:
                duration = max(duration, _as_float(s.get("duration")))

    return MediaInfo(
        path=str(p),
        duration=duration,
        has_audio=audio is not None,
        has_video=video is not None,
        audio_codec=(audio or {}).get("codec_name"),
        video_codec=(video or {}).get("codec_name"),
        sample_rate=int(_as_float((audio or {}).get("sample_rate"))) or None,
        channels=(audio or {}).get("channels"),
        size_bytes=int(_as_float(fmt.get("size"))) or p.stat().st_size,
        format_name=fmt.get("format_name"),
        title=((fmt.get("tags") or {}).get("title")) or p.stem,
    )


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def extract_audio(
    path: str | os.PathLike[str],
    dest: str | os.PathLike[str] | None = None,
    *,
    normalize: bool = True,
    start: float | None = None,
    duration: float | None = None,
) -> str:
    """Decode any input to 16 kHz mono PCM WAV and return the output path.

    `normalize` applies a mild speech-oriented gain chain. It measurably helps
    quiet or unevenly recorded sources; disable it for already-mastered audio
    where you want the model to see the original dynamics.
    """
    src = Path(path)
    info = probe(src)
    if not info.has_audio:
        raise MediaError(
            f"{src.name} contains no audio stream ({info.kind}); there is "
            f"nothing to transcribe."
        )

    if dest is None:
        fd, tmp = tempfile.mkstemp(prefix="subtrans-", suffix=".wav")
        os.close(fd)
        dest = tmp
    out = Path(dest)

    cmd = [_tool("ffmpeg"), "-nostdin", "-y", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.3f}"]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{max(0.0, duration):.3f}"]
    cmd += ["-vn", "-sn", "-dn", "-map", "0:a:0"]
    if normalize:
        # highpass removes rumble/handling noise; speechnorm levels speech
        # without the two-pass cost of loudnorm.
        cmd += ["-af", "highpass=f=50,speechnorm=e=6.25:r=0.00001:l=1"]
    cmd += [
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        "-f", "wav", str(out),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        hint = detail[-1] if detail else "unknown ffmpeg error"
        raise MediaError(f"Audio extraction failed for {src.name}: {hint}")
    if not out.exists() or out.stat().st_size < 1024:
        raise MediaError(f"Audio extraction produced no usable audio for {src.name}")
    return str(out)


def burn_in(
    video: str | os.PathLike[str],
    subtitle_file: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    font_size: int = 24,
) -> str:
    """Hard-burn a subtitle file into a video. Optional convenience path."""
    vid, sub, out = Path(video), Path(subtitle_file), Path(dest)
    style = (
        f"FontSize={font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H90000000,"
        f"BorderStyle=3,Outline=1,Shadow=0,MarginV=28"
    )
    escaped = str(sub).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    cmd = [
        _tool("ffmpeg"), "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(vid),
        "-vf", f"subtitles='{escaped}':force_style='{style}'",
        "-c:a", "copy", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise MediaError(f"Burn-in failed: {detail[-1] if detail else 'ffmpeg error'}")
    return str(out)

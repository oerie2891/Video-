"""Reading and writing subtitle/transcript formats."""

from __future__ import annotations

from .readers import load_any, read_subtitles, read_text_blocks, sniff
from .writers import (
    EXTENSIONS,
    render,
    write_ass,
    write_json,
    write_lrc,
    write_md,
    write_srt,
    write_tsv,
    write_txt,
    write_vtt,
)

__all__ = [
    "EXTENSIONS",
    "load_any",
    "read_subtitles",
    "read_text_blocks",
    "render",
    "sniff",
    "write_ass",
    "write_json",
    "write_lrc",
    "write_md",
    "write_srt",
    "write_tsv",
    "write_txt",
    "write_vtt",
]

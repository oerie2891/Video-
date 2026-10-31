"""Write the playlists to disk as real folders, one per category.

Each folder holds one clickable shortcut per item. File managers sort names by
their own rules (Finder, Explorer and most Linux file managers all disagree),
so names are numbered `001 `, `002 `, ... to pin the playlist order. Colour
can't live in a file name, so the site goes in brackets and an `index.html`
alongside shows the coloured tags.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Any

from .library import Library

SHORTCUT_STYLES = ("url", "webloc", "desktop")

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_WIN = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                 *(f"lpt{i}" for i in range(1, 10))}


def default_style() -> str:
    if sys.platform == "darwin":
        return "webloc"
    if sys.platform.startswith("win"):
        return "url"
    return "desktop"


def safe_name(text: str, limit: int = 120) -> str:
    """A file/folder name valid on Windows, macOS and Linux."""
    name = _UNSAFE.sub("-", text)
    name = " ".join(name.split()).strip(" .")
    if name.split(".")[0].lower() in _RESERVED_WIN:
        name = "_" + name
    return name[:limit].rstrip(" .") or "untitled"


def _shortcut(style: str, url: str, title: str) -> tuple[str, str]:
    if style == "url":
        return ".url", f"[InternetShortcut]\r\nURL={url}\r\n"
    if style == "webloc":
        return ".webloc", (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0"><dict><key>URL</key>'
            f"<string>{html.escape(url)}</string></dict></plist>\n"
        )
    if style == "desktop":
        clean = title.replace("\n", " ")
        return ".desktop", f"[Desktop Entry]\nType=Link\nName={clean}\nURL={url}\nIcon=text-html\n"
    raise ValueError(f"style must be one of {', '.join(SHORTCUT_STYLES)}")


def export(library: Library, out_dir: str | Path, style: str | None = None,
           numbered: bool = True) -> dict[str, Any]:
    style = style or default_style()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    playlists = library.playlists()
    width = max(3, len(str(len(playlists))))
    written = 0
    sections = []
    for index, playlist in enumerate(playlists, start=1):
        if playlist.get("uncategorised"):
            items = library.items(uncategorised=True)
        else:
            items = library.items(category=playlist["name"])
        folder_name = safe_name(playlist["name"])
        if numbered:
            folder_name = f"{index:0{width}d} {folder_name}"
        folder = root / folder_name
        folder.mkdir(exist_ok=True)
        item_width = max(3, len(str(len(items))))
        for position, item in enumerate(items, start=1):
            title = item["title"] or item["url"]
            name = f"[{item['site']['name']}] {title}"
            if numbered:
                name = f"{position:0{item_width}d} {name}"
            ext, body = _shortcut(style, item["page_url"] or item["url"], title)
            (folder / (safe_name(name, 150) + ext)).write_text(body, encoding="utf-8")
            written += 1
        sections.append((playlist["name"], items))
    (root / "index.html").write_text(_index_html(sections), encoding="utf-8")
    return {"folders": len(playlists), "shortcuts": written, "path": str(root.resolve())}


def _index_html(sections: list[tuple[str, list[dict[str, Any]]]]) -> str:
    parts = []
    for name, items in sections:
        rows = "".join(
            f'<li><span class="tag" style="background:{i["site"]["color"]};'
            f'color:{i["site"]["text_color"]}">{html.escape(i["site"]["name"])}</span> '
            f'<a href="{html.escape(i["page_url"] or i["url"])}" rel="noreferrer">'
            f'{html.escape(i["title"] or i["url"])}</a></li>'
            for i in items
        )
        parts.append(f"<section><h2>{html.escape(name)} <small>{len(items)}</small></h2>"
                     f"<ol>{rows}</ol></section>")
    return (
        "<!doctype html><meta charset=utf-8><meta name=viewport "
        'content="width=device-width,initial-scale=1"><title>Playlists</title>'
        "<style>body{font:15px/1.5 system-ui,sans-serif;margin:24px auto;max-width:860px;"
        "padding:0 16px;background:#fff;color:#111}h2{margin:28px 0 6px}small{color:#777;"
        "font-weight:400}.tag{display:inline-block;font-size:12px;font-weight:600;"
        "padding:1px 8px;border-radius:999px;box-shadow:inset 0 0 0 1px rgba(127,127,127,.4)}"
        "a{color:inherit}@media(prefers-color-scheme:dark){body{background:#111;color:#eee}}"
        "</style><h1>Playlists</h1>" + "".join(parts)
    )

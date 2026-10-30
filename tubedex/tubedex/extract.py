"""Turn a link into a `Record`: title, kind, uploader, people, stats.

yt-dlp does the heavy lifting. It has maintained extractors for every site
with a fixed tag colour, handles age-gate cookies, redirects and mirror
domains, and is updated within days when a site changes. Its coverage is
uneven, though: Pornhub returns cast, categories, views, likes and comments,
while the XVideos extractor returns little beyond a title. So after yt-dlp
runs, any field still empty is filled from the page's own HTML (see
`htmlmeta`). Fields are never merged across sources; the first source that
has a value wins, in the order yt-dlp -> JSON-LD -> meta tags -> markup.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from . import htmlmeta
from .sites import identify

KINDS = ("video", "photo", "gif", "magazine", "link")

STAT_FIELDS = ("views", "likes", "dislikes", "comments")
TEXT_FIELDS = ("title", "channel", "channel_url", "published", "thumbnail", "description")
LIST_FIELDS = ("people", "categories", "tags")
NUMBER_FIELDS = STAT_FIELDS + ("duration", "height")
# If yt-dlp leaves any of these empty, the page itself is fetched and parsed.
WANTED = ("title", "channel", "people", "categories", "views", "likes", "comments", "published")

_PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".heic", ".jfif"}
_GIF_EXT = {".gif", ".gifv"}
_VIDEO_EXT = {".mp4", ".webm", ".mkv", ".m4v", ".mov", ".avi", ".wmv", ".flv"}
_MAGAZINE_EXT = {".pdf", ".cbz", ".cbr", ".epub"}

# URL path segments that say what a page is, independent of the site.
_PATH_KINDS = (
    (re.compile(r"/(?:gifs?|gif-[\w-]+)(?:/|$)", re.I), "gif"),
    (re.compile(r"/(?:photos?|albums?|pics?|pictures?|galler(?:y|ies)|images?)(?:/|$)", re.I),
     "photo"),
    (re.compile(r"/(?:magazines?|issues?|zines?)(?:/|$)", re.I), "magazine"),
)

_TRACKING_PARAMS = re.compile(r"^(?:utm_\w+|fbclid|gclid|igshid|ref_?src|si|s)$", re.I)

# Above this, a "page" is not HTML we want to parse.
MAX_HTML_BYTES = 4 * 1024 * 1024


class FetchError(RuntimeError):
    pass


@dataclass
class Record:
    url: str
    page_url: str | None = None
    media_key: str | None = None
    kind: str = "link"
    title: str | None = None
    channel: str | None = None
    channel_url: str | None = None
    people: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    views: int | None = None
    likes: int | None = None
    dislikes: int | None = None
    comments: int | None = None
    published: str | None = None
    duration: int | None = None
    height: int | None = None
    thumbnail: str | None = None
    description: str | None = None
    # Which layer supplied each field, so a wrong value can be traced.
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def missing(self) -> list[str]:
        return [name for name in WANTED if getattr(self, name) in (None, [], "")]


def normalize_url(raw: str) -> str:
    """Canonical form used for de-duplication: https, lower-case host, no
    fragment, no tracking parameters. Anything else (Pornhub's `viewkey`)
    is kept, because it is what identifies the video."""
    text = raw.strip().strip("<>\"'")
    if not text:
        raise ValueError("empty link")
    if "://" not in text:
        text = "https://" + text.lstrip("/")
    parts = urlsplit(text)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError(f"not a web link: {raw!r}")
    host = parts.hostname.lower()
    netloc = host if parts.port in (None, 80, 443) else f"{host}:{parts.port}"
    query = urlencode(
        [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
         if not _TRACKING_PARAMS.match(k)]
    )
    return urlunsplit(("https" if parts.scheme.lower() == "https" else "http",
                       netloc, parts.path or "/", query, ""))


def _extension(url: str) -> str:
    path = urlsplit(url).path.lower()
    return os.path.splitext(path)[1]


def guess_kind(url: str) -> str | None:
    """Kind from the URL alone: file extension, then path, then site."""
    ext = _extension(url)
    if ext in _GIF_EXT:
        return "gif"
    if ext in _PHOTO_EXT:
        return "photo"
    if ext in _MAGAZINE_EXT:
        return "magazine"
    if ext in _VIDEO_EXT:
        return "video"
    path = urlsplit(url).path
    for pattern, kind in _PATH_KINDS:
        if pattern.search(path):
            return kind
    return identify(url).default_kind


def is_direct_file(url: str) -> bool:
    return _extension(url) in _PHOTO_EXT | _GIF_EXT | _VIDEO_EXT | _MAGAZINE_EXT


def _title_from_filename(url: str) -> str:
    name = unquote(os.path.basename(urlsplit(url).path))
    stem = os.path.splitext(name)[0]
    return " ".join(re.split(r"[_\s]+", stem)).strip() or name


# --------------------------------------------------------------------------
# yt-dlp
# --------------------------------------------------------------------------

class _Log:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        self.warnings.append(_clean_message(msg))

    def error(self, msg: str) -> None:
        self.warnings.append(_clean_message(msg))


def _clean_message(msg: str) -> str:
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(msg))
    msg = re.sub(r"^(?:ERROR|WARNING):\s*", "", msg)
    # yt-dlp appends a bug-report plea and the exception chain; neither helps here.
    msg = re.split(r";\s*please report this issue", msg, maxsplit=1)[0]
    msg = re.sub(r"\s*\(caused by .*\)\s*$", "", msg)
    return msg.strip()


@dataclass
class FetchOptions:
    cookies: str | None = None
    cookies_from_browser: str | None = None
    timeout: float = 25.0

    @classmethod
    def from_env(cls) -> FetchOptions:
        return cls(
            cookies=os.environ.get("TUBEDEX_COOKIES") or None,
            cookies_from_browser=os.environ.get("TUBEDEX_COOKIES_FROM_BROWSER") or None,
        )


def _ydl_params(options: FetchOptions, log: _Log) -> dict[str, Any]:
    params: dict[str, Any] = {
        "quiet": True,
        "no_warnings": False,
        "logger": log,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": "in_playlist",
        "ignore_no_formats_error": True,
        "socket_timeout": options.timeout,
        "noprogress": True,
    }
    if options.cookies:
        params["cookiefile"] = options.cookies
    if options.cookies_from_browser:
        browser, _, profile = options.cookies_from_browser.partition(":")
        params["cookiesfrombrowser"] = (browser, profile or None, None, None)
    return params


def _from_ytdlp(info: dict[str, Any]) -> dict[str, Any]:
    published = htmlmeta.parse_date(info.get("upload_date") or info.get("release_date"))
    if not published:
        stamp = info.get("timestamp") or info.get("release_timestamp")
        if isinstance(stamp, (int, float)):
            published = datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d")
    heights = [f.get("height") for f in info.get("formats") or [] if isinstance(f, dict)]
    heights = [h for h in heights if isinstance(h, int)]
    height = max(heights) if heights else info.get("height")
    people = info.get("cast") or []
    if isinstance(people, str):
        people = [people]
    channel = info.get("channel") or info.get("uploader") or info.get("creator")
    channel_url = info.get("channel_url") or info.get("uploader_url")
    out = {
        "title": info.get("title"),
        "channel": channel,
        "channel_url": channel_url if isinstance(channel_url, str) else None,
        "people": [p for p in people if isinstance(p, str) and p.strip()],
        "categories": [c for c in info.get("categories") or [] if isinstance(c, str)],
        "tags": [t for t in info.get("tags") or [] if isinstance(t, str)],
        "views": htmlmeta.parse_count(info.get("view_count")),
        "likes": htmlmeta.parse_count(info.get("like_count")),
        "dislikes": htmlmeta.parse_count(info.get("dislike_count")),
        "comments": htmlmeta.parse_count(info.get("comment_count")),
        "published": published,
        "duration": htmlmeta.parse_duration(info.get("duration")),
        "height": height if isinstance(height, int) else None,
        "thumbnail": info.get("thumbnail"),
        "description": info.get("description"),
    }
    return {k: v for k, v in out.items() if v not in (None, [], "")}


# --------------------------------------------------------------------------
# Fetcher
# --------------------------------------------------------------------------

Fetcher = Callable[[str, Any], Record]


class Extractor:
    """Fetches one link at a time. Reuses a yt-dlp session so cookies set by
    an age gate on the first link apply to the next."""

    def __init__(self, options: FetchOptions | None = None) -> None:
        self.options = options or FetchOptions.from_env()

    def __call__(self, url: str, kind: str | None = None) -> Record:
        return self.fetch(url, kind)

    def fetch(self, url: str, kind: str | None = None) -> Record:
        try:
            import yt_dlp
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise FetchError("yt-dlp is not installed: pip install yt-dlp") from exc

        record = Record(url=url)
        guessed = guess_kind(url)
        layers: list[tuple[str, dict[str, Any]]] = []
        log = _Log()

        with yt_dlp.YoutubeDL(_ydl_params(self.options, log)) as ydl:
            if is_direct_file(url):
                layers.append(("file", {"title": _title_from_filename(url)}))
                guessed = self._probe_file(ydl, url, record) or guessed
            else:
                # Photo and magazine pages never have a video for yt-dlp to
                # find; skip the wasted request and go straight to the page.
                if guessed not in ("photo", "magazine"):
                    info = self._run_ytdlp(ydl, url, record, log)
                    if info:
                        layers.append(("yt-dlp", _from_ytdlp(info)))
                        if info.get("formats") or info.get("url"):
                            layers.append(("yt-dlp", {"kind": "video"}))
                        record.page_url = info.get("webpage_url") or record.page_url
                        if info.get("id") and info.get("extractor_key") not in (None, "Generic"):
                            record.media_key = f"{info['extractor_key'].lower()}:{info['id']}"
                if _missing(layers):
                    layers += self._page_layers(ydl, record.page_url or url, record)

        merged = _merge(layers, record)
        kind = kind or (guessed if guessed and guessed != "video" else None) \
            or merged.get("kind") or guessed or "link"
        record.kind = kind if kind in KINDS else "link"
        site = identify(record.page_url or url)
        if merged.get("channel") and merged["channel"].casefold() == site.name.casefold():
            # JSON-LD often names the site itself as author/publisher.
            merged.pop("channel")
            record.sources.pop("channel", None)
        for name, value in merged.items():
            if name != "kind" and hasattr(record, name):
                setattr(record, name, value)
        record.warnings = _dedupe_messages(record.warnings + log.warnings)
        if not layers and record.error is None:
            record.error = record.warnings[-1] if record.warnings else "nothing could be read"
        return record

    def _run_ytdlp(self, ydl: Any, url: str, record: Record, log: _Log) -> dict | None:
        from yt_dlp.utils import DownloadError

        try:
            info = ydl.extract_info(url, download=False)
        except DownloadError as exc:
            message = _clean_message(str(exc))
            if "Unsupported URL" not in message:
                record.warnings.append(f"yt-dlp: {message}")
            log.warnings.clear()
            return None
        except Exception as exc:  # noqa: BLE001 - extractor bugs must not kill the import
            record.warnings.append(f"yt-dlp crashed: {exc}")
            return None
        if not isinstance(info, dict):
            return None
        if info.get("_type") == "playlist":
            record.warnings.append("this link is a playlist or channel page, not a single item")
            return {k: info.get(k) for k in ("title", "uploader", "channel", "webpage_url")}
        return info

    def _probe_file(self, ydl: Any, url: str, record: Record) -> str | None:
        from yt_dlp.networking import Request

        try:
            with ydl.urlopen(Request(url, method="HEAD")) as response:
                record.page_url = response.url
                return _kind_from_content_type(response.headers.get("Content-Type", ""))
        except Exception as exc:  # noqa: BLE001
            record.warnings.append(f"could not reach file: {_clean_message(str(exc))}")
            return None

    def _page_layers(self, ydl: Any, url: str, record: Record) -> list[tuple[str, dict]]:
        try:
            with ydl.urlopen(url) as response:
                record.page_url = record.page_url or response.url
                content_type = response.headers.get("Content-Type", "")
                file_kind = _kind_from_content_type(content_type)
                if file_kind:
                    return [("file", {"kind": file_kind, "title": _title_from_filename(url)})]
                raw = response.read(MAX_HTML_BYTES)
        except Exception as exc:  # noqa: BLE001
            record.warnings.append(f"page: {_clean_message(str(exc))}")
            return []
        html = _decode(raw, content_type)
        page = htmlmeta.parse(html)
        return [
            ("json-ld", htmlmeta.from_json_ld(page)),
            ("meta-tags", htmlmeta.from_meta_tags(page)),
            ("markup", htmlmeta.from_markup(page, html)),
        ]


def _kind_from_content_type(content_type: str) -> str | None:
    ct = content_type.split(";")[0].strip().lower()
    if ct == "image/gif":
        return "gif"
    if ct.startswith("image/"):
        return "photo"
    if ct.startswith("video/"):
        return "video"
    if ct in ("application/pdf", "application/epub+zip", "application/vnd.comicbook+zip"):
        return "magazine"
    return None


def _decode(raw: bytes, content_type: str) -> str:
    match = re.search(r"charset=([\w-]+)", content_type, re.I)
    if not match:
        match = re.search(rb'<meta[^>]+charset=["\']?([\w-]+)', raw[:4096], re.I)
    charset = match.group(1) if match else "utf-8"
    if isinstance(charset, bytes):
        charset = charset.decode("ascii", "ignore")
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _missing(layers: list[tuple[str, dict[str, Any]]]) -> bool:
    merged = _merge(layers, Record(url=""))
    return any(merged.get(name) in (None, [], "") for name in WANTED)


def _merge(layers: list[tuple[str, dict[str, Any]]], record: Record) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source, values in layers:
        for name, value in values.items():
            if name in merged or value in (None, [], ""):
                continue
            if name in LIST_FIELDS:
                value = _clean_list(value)
                if not value:
                    continue
            if name in NUMBER_FIELDS and not isinstance(value, int):
                continue
            merged[name] = value
            record.sources[name] = source
    merged.pop("site_name", None)
    record.sources.pop("site_name", None)
    return merged


def _clean_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in values:
        text = " ".join(str(value).split())
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text)
    return out


def _dedupe_messages(messages: list[str]) -> list[str]:
    return list(dict.fromkeys(m for m in messages if m))

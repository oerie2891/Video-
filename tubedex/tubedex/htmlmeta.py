"""Metadata from a page's own HTML, for whatever yt-dlp did not supply.

Three layers, most trustworthy first:

1. JSON-LD (`<script type="application/ld+json">`): structured data sites
   publish for search engines. When present it usually carries the upload
   date, view/like/comment counts and sometimes the cast.
2. OpenGraph / meta tags: title, thumbnail, `video:tag`, `video:actor`.
3. Markup hints: class names some tube sites put on performer, tag and
   uploader links. These are site-specific and break when a site redesigns,
   so they only ever fill gaps and `tubedex inspect` shows what each layer saw.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import Any

_VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

# Class names that mark a link as a performer / category / tag / uploader.
# Only distinctive names are listed: a generic one like "tag" or "model" would
# also match navigation menus and "popular performers" sidebars.
PERSON_CLASSES = {"is-pornstar", "pornstar_link", "pornstar-link", "vit-pornstar"}
CATEGORY_CLASSES = {"vit-category", "is-category", "video-category-link"}
TAG_CLASSES = {"is-keyword", "vit-tag"}
CHANNEL_CLASSES = {"main-uploader", "uploader-tag", "vit-uploader"}
_HINT_CLASSES = PERSON_CLASSES | CATEGORY_CLASSES | TAG_CLASSES | CHANNEL_CLASSES

_SKIP_TEXT_IN = {"script", "style", "noscript", "template"}


@dataclass
class PageMeta:
    title: str | None = None
    canonical: str | None = None
    meta: dict[str, list[str]] = field(default_factory=dict)
    json_ld: list[dict[str, Any]] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    channel: list[str] = field(default_factory=list)

    def first(self, *names: str) -> str | None:
        for name in names:
            for value in self.meta.get(name.lower(), []):
                if value.strip():
                    return value.strip()
        return None

    def all(self, name: str) -> list[str]:
        return [v.strip() for v in self.meta.get(name.lower(), []) if v.strip()]


class _Parser(HTMLParser):
    """Collects meta tags, JSON-LD blocks and text of hint-classed links.

    Real-world HTML is not well formed, so the element stack is maintained
    leniently: an end tag pops back to its matching open tag if there is one
    and is ignored otherwise.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page = PageMeta()
        self.stack: list[tuple[str, frozenset[str]]] = []
        self._in_title = False
        self._title: list[str] = []
        self._script: list[str] | None = None
        # Open hint-classed links: [classes in scope, text parts, name-span parts]
        self._links: list[tuple[frozenset[str], list[str], list[str]]] = []
        self._name_depth: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        classes = frozenset(a.get("class", "").split())
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and "content" in a:
                self.page.meta.setdefault(key, []).append(unescape(a["content"]))
            return
        if tag == "link" and "canonical" in a.get("rel", "").lower().split():
            self.page.canonical = a.get("href") or None
            return
        if tag in _VOID:
            return
        if tag == "title" and self.page.title is None:
            self._in_title = True
        if tag == "script" and "ld+json" in a.get("type", "").lower():
            self._script = []
        if tag == "a":
            in_scope = classes | {c for _, cs in self.stack[-3:] for c in cs}
            hits = frozenset(in_scope & _HINT_CLASSES)
            if hits:
                self._links.append((hits, [], []))
        if self._links and "name" in classes and self._name_depth is None:
            self._name_depth = len(self.stack)
        self.stack.append((tag, classes))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            if self.page.title is None:
                self.page.title = " ".join("".join(self._title).split()) or None
        if tag == "script" and self._script is not None:
            self._add_json_ld("".join(self._script))
            self._script = None
        for depth in range(len(self.stack) - 1, -1, -1):
            if self.stack[depth][0] == tag:
                del self.stack[depth:]
                if self._name_depth is not None and depth <= self._name_depth:
                    self._name_depth = None
                if tag == "a" and self._links:
                    self._finish_link()
                return

    def handle_data(self, data: str) -> None:
        if self._script is not None:
            self._script.append(data)
            return
        if self._in_title:
            self._title.append(data)
        if self._links and not (self.stack and self.stack[-1][0] in _SKIP_TEXT_IN):
            _, text, name = self._links[-1]
            text.append(data)
            if self._name_depth is not None:
                name.append(data)

    def _finish_link(self) -> None:
        hits, text, name = self._links.pop()
        label = " ".join("".join(name or text).split())
        if not label:
            return
        if hits & PERSON_CLASSES:
            self.page.people.append(label)
        elif hits & CHANNEL_CLASSES:
            self.page.channel.append(label)
        elif hits & CATEGORY_CLASSES:
            self.page.categories.append(label)
        elif hits & TAG_CLASSES:
            self.page.tags.append(label)

    def _add_json_ld(self, raw: str) -> None:
        try:
            data = json.loads(raw.strip())
        except ValueError:
            # Some sites emit trailing commas or raw newlines in strings.
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", raw.strip()), strict=False)
            except ValueError:
                return
        for node in _walk_json_ld(data):
            self.page.json_ld.append(node)


def _walk_json_ld(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [n for item in data for n in _walk_json_ld(item)]
    if not isinstance(data, dict):
        return []
    nodes = [data] if "@type" in data else []
    if isinstance(data.get("@graph"), list):
        nodes += _walk_json_ld(data["@graph"])
    return nodes


def parse(html: str) -> PageMeta:
    parser = _Parser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - a malformed page must not lose what was parsed
        pass
    page = parser.page
    page.people = _dedupe(page.people)
    page.categories = _dedupe(page.categories)
    page.tags = _dedupe(page.tags)
    page.channel = _dedupe(page.channel)
    return page


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


# --------------------------------------------------------------------------
# Interpreting JSON-LD
# --------------------------------------------------------------------------

MEDIA_TYPES = {
    "VideoObject": "video",
    "Movie": "video",
    "Clip": "video",
    "ImageObject": "photo",
    "ImageGallery": "photo",
    "Photograph": "photo",
    "Periodical": "magazine",
    "PublicationIssue": "magazine",
    "PublicationVolume": "magazine",
}

_COUNT = re.compile(r"^\s*([\d.,\s]*\d)\s*([kKmMbB])?\b")


def parse_count(value: Any) -> int | None:
    """1234, '1,234', '1.2K', '3.4M views', 'UserPlays:5678' -> int."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value >= 0 else None
    text = str(value)
    if ":" in text:
        text = text.rsplit(":", 1)[1]
    match = _COUNT.match(text)
    if not match:
        return None
    number, suffix = match.group(1).replace(" ", ""), (match.group(2) or "").lower()
    if suffix:
        number = number.replace(",", ".")
        if number.count(".") > 1:
            return None
        try:
            return int(round(float(number) * {"k": 1e3, "m": 1e6, "b": 1e9}[suffix]))
        except ValueError:
            return None
    digits = re.sub(r"[.,\s]", "", number)
    return int(digits) if digits.isdigit() else None


_ISO_DURATION = re.compile(
    r"^P(?:(?P<d>\d+)D)?(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?)?$", re.I
)


def parse_duration(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    match = _ISO_DURATION.match(text)
    if not match or not any(match.groupdict().values()):
        return None
    d, h, m, s = (float(match.group(k) or 0) for k in "dhms")
    return int(d * 86400 + h * 3600 + m * 60 + s)


def parse_date(value: Any) -> str | None:
    """'2024-03-05T10:00:00+00:00' / '20240305' -> '2024-03-05'."""
    if not value:
        return None
    text = str(value).strip()
    match = re.match(r"^(\d{4})-?(\d{2})-?(\d{2})", text)
    if not match:
        return None
    year, month, day = match.groups()
    if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
        return None
    return f"{year}-{month}-{day}"


def _names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [n for item in value for n in _names(item)]
    if isinstance(value, dict):
        name = value.get("name") or value.get("alternateName")
        return _names(name)
    text = unescape(str(value)).strip()
    return [text] if text else []


def _keywords(value: Any) -> list[str]:
    if isinstance(value, list):
        return [k for item in value for k in _keywords(item)]
    if isinstance(value, str):
        return [k.strip() for k in unescape(value).split(",") if k.strip()]
    return []


def _interaction(node: dict[str, Any], action: str) -> int | None:
    stats = node.get("interactionStatistic")
    if isinstance(stats, dict):
        stats = [stats]
    for stat in stats if isinstance(stats, list) else []:
        if not isinstance(stat, dict):
            continue
        kind = stat.get("interactionType")
        if isinstance(kind, dict):
            kind = kind.get("@type") or kind.get("@id")
        if isinstance(kind, str) and kind.rsplit("/", 1)[-1] == action:
            return parse_count(stat.get("userInteractionCount"))
    return None


_KIND_PRIORITY = ("video", "magazine", "photo")


def media_node(page: PageMeta) -> dict[str, Any] | None:
    """The JSON-LD node describing the page's main item, if any.

    Video pages often also carry an ImageObject (a logo, the thumbnail), so a
    video node is preferred over a magazine node over an image node.
    """
    for wanted in _KIND_PRIORITY:
        for node in page.json_ld:
            types = node.get("@type")
            types = types if isinstance(types, list) else [types]
            if any(MEDIA_TYPES.get(t) == wanted for t in types if isinstance(t, str)):
                return node
    return None


def from_json_ld(page: PageMeta) -> dict[str, Any]:
    node = media_node(page)
    if node is None:
        return {}
    types = node.get("@type")
    types = types if isinstance(types, list) else [types]
    kind = next((MEDIA_TYPES[t] for t in types if isinstance(t, str) and t in MEDIA_TYPES), None)
    views = _interaction(node, "WatchAction")
    if views is None and node.get("interactionCount") is not None:
        views = parse_count(node.get("interactionCount"))
    thumb = node.get("thumbnailUrl") or node.get("image")
    if isinstance(thumb, list):
        thumb = thumb[0] if thumb else None
    if isinstance(thumb, dict):
        thumb = thumb.get("url")
    out = {
        "kind": kind,
        "title": next(iter(_names(node.get("name") or node.get("headline"))), None),
        "description": node.get("description") if isinstance(node.get("description"), str) else None,
        "published": parse_date(node.get("uploadDate") or node.get("datePublished")),
        "duration": parse_duration(node.get("duration")),
        "thumbnail": thumb if isinstance(thumb, str) else None,
        "channel": next(iter(_names(node.get("author") or node.get("creator")
                                    or node.get("publisher"))), None),
        "people": _names(node.get("actor") or node.get("actors")),
        "categories": _keywords(node.get("genre")),
        "tags": _keywords(node.get("keywords")),
        "views": views,
        "likes": _interaction(node, "LikeAction"),
        "dislikes": _interaction(node, "DislikeAction"),
        "comments": _interaction(node, "CommentAction")
        or parse_count(node.get("commentCount")),
    }
    return {k: v for k, v in out.items() if v not in (None, [], "")}


def from_meta_tags(page: PageMeta) -> dict[str, Any]:
    og_type = (page.first("og:type") or "").lower()
    kind = None
    if og_type.startswith("video"):
        kind = "video"
    elif og_type in ("book", "article") and page.first("book:isbn", "book:release_date"):
        kind = "magazine"
    tags = page.all("video:tag") or page.all("article:tag")
    if not tags and page.first("keywords"):
        tags = _keywords(page.first("keywords"))
    out = {
        "kind": kind,
        "title": page.first("og:title", "twitter:title") or page.title,
        "description": page.first("og:description", "description", "twitter:description"),
        "thumbnail": page.first("og:image", "og:image:url", "twitter:image"),
        "published": parse_date(page.first("video:release_date", "article:published_time",
                                           "uploaddate", "datepublished")),
        "duration": parse_duration(page.first("video:duration", "og:video:duration")),
        "people": page.all("video:actor"),
        "tags": tags,
        "channel": page.first("og:video:uploader", "author"),
        "site_name": page.first("og:site_name"),
    }
    return {k: v for k, v in out.items() if v not in (None, [], "")}


# Counters that some sites render as plain HTML instead of structured data.
_COUNTER_PATTERNS = {
    "views": (
        r'id="v-views"[^>]*>(?:(?!</div>).){0,400}?<strong[^>]*>\s*([\d.,]+\s*[kKmMbB]?)\s*<',
    ),
    "likes": (r'class="rating-good-nbr"[^>]*>\s*([\d.,]+\s*[kKmMbB]?)\s*<',),
    "dislikes": (r'class="rating-bad-nbr"[^>]*>\s*([\d.,]+\s*[kKmMbB]?)\s*<',),
}


def from_markup(page: PageMeta, html: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "people": page.people,
        "categories": page.categories,
        "tags": page.tags,
        "channel": page.channel[0] if page.channel else None,
    }
    for field_name, patterns in _COUNTER_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, html, re.S)
            if match:
                out[field_name] = parse_count(match.group(1))
                break
    return {k: v for k, v in out.items() if v not in (None, [], "")}

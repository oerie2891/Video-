"""Which website a link belongs to, and the colour of its tag.

Eight sites have fixed colours. Every other site gets a generated colour the
first time it is seen, chosen to be as far as possible (in CIELAB, i.e. as the
eye sees it) from every colour already in use. Early sites get clearly
different colours; once there are dozens, new ones are necessarily only
slightly different, which is the best any palette can do.
"""

from __future__ import annotations

import colorsys
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class KnownSite:
    key: str
    name: str
    color: str
    # Matched against the registrable domain's first label ("xvideos2" for
    # "www.xvideos2.com"), so mirrors and country domains map to one site.
    label: str = ""
    domains: tuple[str, ...] = ()
    default_kind: str | None = None


KNOWN_SITES: tuple[KnownSite, ...] = (
    KnownSite("pornhub", "Pornhub", "#FF9000", label=r"pornhub(?:premium)?", default_kind="video"),
    KnownSite("xvideos", "XVideos", "#FF0000", label=r"xvideos\d*", default_kind="video"),
    KnownSite("xnxx", "XNXX", "#0066FF", label=r"xnxx\d*", default_kind="video"),
    KnownSite("redtube", "RedTube", "#8A0303", label=r"redtube", default_kind="video"),
    KnownSite(
        "x",
        "X",
        "#000000",
        # Exact domains only: a label rule of "x" would swallow unrelated sites.
        domains=("x.com", "twitter.com", "fxtwitter.com", "vxtwitter.com", "fixupx.com"),
    ),
    KnownSite("eporner", "Eporner", "#36454F", label=r"eporner", default_kind="video"),
    KnownSite("youporn", "YouPorn", "#FF69B4", label=r"youporn", default_kind="video"),
    KnownSite("xhamster", "xHamster", "#800080", label=r"xhamster\d*", default_kind="video"),
)

# Sites without a fixed colour whose links are almost always one kind of media.
DEFAULT_KINDS = {"redgifs.com": "gif", "gfycat.com": "gif", "imgur.com": "photo"}

_TWO_PART_SUFFIXES = {"co", "com", "net", "org", "gov", "ac", "edu", "ne", "or"}


def host_of(url: str) -> str:
    if "://" not in url:
        url = "https://" + url
    return (urlsplit(url).hostname or "").lower().rstrip(".")


def registrable_domain(host: str) -> str:
    """'de.pornhub.com' -> 'pornhub.com'; 'www.example.co.uk' -> 'example.co.uk'."""
    labels = [part for part in host.lower().rstrip(".").split(".") if part]
    if len(labels) <= 2:
        return ".".join(labels)
    if len(labels[-1]) == 2 and labels[-2] in _TWO_PART_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


@dataclass(frozen=True)
class SiteId:
    key: str
    name: str
    known: KnownSite | None = None

    @property
    def default_kind(self) -> str | None:
        if self.known:
            return self.known.default_kind
        return DEFAULT_KINDS.get(self.key)


def identify(url: str) -> SiteId:
    """Map any URL to a stable site key. Unknown sites are keyed by domain."""
    domain = registrable_domain(host_of(url))
    label = domain.split(".", 1)[0]
    for site in KNOWN_SITES:
        if domain in site.domains or (site.label and re.fullmatch(site.label, label)):
            return SiteId(site.key, site.name, site)
    return SiteId(domain or "unknown", domain or "Unknown site")


# --------------------------------------------------------------------------
# Colour maths
# --------------------------------------------------------------------------

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def normalize_hex(value: str) -> str:
    """'#ff9000' / 'FF9000' -> '#FF9000'. Raises ValueError otherwise."""
    match = _HEX.match(value.strip())
    if not match:
        raise ValueError(f"not a #RRGGBB colour: {value!r}")
    return "#" + match.group(1).upper()


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    digits = normalize_hex(value)[1:]
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def _linear(channel: int) -> float:
    c = channel / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def to_lab(value: str) -> tuple[float, float, float]:
    r, g, b = (_linear(c) for c in hex_to_rgb(value))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def distance(a: str, b: str) -> float:
    """Perceptual difference (CIE76 delta-E). ~2.3 is the just-noticeable step."""
    return math.dist(to_lab(a), to_lab(b))


def text_color(background: str) -> str:
    """Black or white, whichever has more WCAG contrast on `background`."""
    r, g, b = (_linear(c) for c in hex_to_rgb(background))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    on_white = 1.05 / (luminance + 0.05)
    on_black = (luminance + 0.05) / 0.05
    return "#FFFFFF" if on_white >= on_black else "#000000"


# Saturation/lightness tiers. Every tier is saturated enough that a generated
# tag can never be mistaken for X's black or Eporner's charcoal.
_TIERS = ((0.75, 0.50), (0.60, 0.36), (0.85, 0.66), (0.50, 0.27), (0.65, 0.78))
_GOLDEN_ANGLE = 137.50776


def _candidates() -> list[str]:
    out = []
    for saturation, lightness in _TIERS:
        for i in range(72):
            hue = (i * _GOLDEN_ANGLE) % 360 / 360
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            out.append(rgb_to_hex(round(r * 255), round(g * 255), round(b * 255)))
    return out


_CANDIDATES = _candidates()
_CANDIDATE_LAB = [to_lab(c) for c in _CANDIDATES]


def generate_color(in_use: Iterable[str]) -> str:
    """The candidate colour farthest from everything in `in_use`.

    Deterministic: the same set of existing colours always yields the same
    pick, so rebuilding a library reproduces its palette.
    """
    used = [to_lab(c) for c in in_use]
    if not used:
        return _CANDIDATES[0]
    best, best_gap = _CANDIDATES[0], -1.0
    for colour, lab in zip(_CANDIDATES, _CANDIDATE_LAB, strict=True):
        gap = min(math.dist(lab, other) for other in used)
        if gap > best_gap:
            best, best_gap = colour, gap
    return best

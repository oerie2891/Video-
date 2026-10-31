"""The ordering rule used for playlists and for the titles inside them.

Symbols first, then titles that start with a number (counted up numerically, so
"2" comes before "10"), then letters A-Z ignoring case and accents.

Python's default string order gets all three wrong: it sorts "10" before "2",
"Zoe" before "anna", and "Émilie" after "zebra".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any, TypeVar

T = TypeVar("T")

SYMBOL, NUMBER, LETTER, EMPTY = 0, 1, 2, 3

_CHUNKS = re.compile(r"(\d+)")


def _fold(text: str) -> str:
    """Case- and accent-insensitive form: "Émilie" -> "emilie"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def group_of(text: str) -> int:
    """Which block a title sorts into: SYMBOL, NUMBER, LETTER or EMPTY."""
    folded = _fold(text.strip())
    if not folded:
        return EMPTY
    first = folded[0]
    if first.isdecimal():
        return NUMBER
    if first.isalpha():
        return LETTER
    return SYMBOL


def sort_key(text: str | None) -> tuple[Any, ...]:
    """Key for `sorted()` implementing the symbols -> numbers -> A-Z rule.

    Runs of digits compare by value everywhere in the title, not just at the
    start, so "Part 2" sorts before "Part 10". The raw title is the final
    tie-break so the order is total and stable across runs.
    """
    raw = (text or "").strip()
    folded = _fold(raw)
    chunks = tuple(
        (0, int(part), len(part)) if part.isdecimal() else (1, part, 0)
        for part in _CHUNKS.split(folded)
        if part
    )
    return (group_of(raw), chunks, raw)


def sort_titles(items: Iterable[T], key: Any = None) -> list[T]:
    """Sort `items` by the rule; `key` extracts the title (default: the item)."""
    get = key or (lambda item: item)
    return sorted(items, key=lambda item: sort_key(get(item)))

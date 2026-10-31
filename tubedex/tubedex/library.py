"""The SQLite library: items, their people/categories/tags, sites and colours.

One file, no server. Every access goes through one connection guarded by a
lock, so the web UI's background fetchers and request handlers can share it;
the slow part (network) always happens outside the lock.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .extract import (
    KINDS,
    LIST_FIELDS,
    NUMBER_FIELDS,
    TEXT_FIELDS,
    Fetcher,
    Record,
    guess_kind,
    normalize_url,
)
from .sites import KNOWN_SITES, generate_color, identify, normalize_hex, text_color
from .sorting import sort_key

PENDING, OK, PARTIAL, ERROR = "pending", "ok", "partial", "error"
LABEL_TYPES = {"people": "person", "categories": "category", "tags": "tag"}
EDITABLE = ("kind",) + TEXT_FIELDS + NUMBER_FIELDS + LIST_FIELDS
UNCATEGORISED = "Uncategorised"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sites (
    key     TEXT PRIMARY KEY,
    name    TEXT NOT NULL,
    color   TEXT NOT NULL,
    custom  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS items (
    id          INTEGER PRIMARY KEY,
    url         TEXT NOT NULL UNIQUE,
    page_url    TEXT,
    media_key   TEXT UNIQUE,
    kind        TEXT NOT NULL,
    site_key    TEXT NOT NULL REFERENCES sites(key),
    title       TEXT,
    channel     TEXT,
    channel_url TEXT,
    views       INTEGER,
    likes       INTEGER,
    dislikes    INTEGER,
    comments    INTEGER,
    published   TEXT,
    duration    INTEGER,
    height      INTEGER,
    thumbnail   TEXT,
    description TEXT,
    status      TEXT NOT NULL,
    error       TEXT,
    warnings    TEXT NOT NULL DEFAULT '[]',
    sources     TEXT NOT NULL DEFAULT '{}',
    locked      TEXT NOT NULL DEFAULT '[]',
    added_at    TEXT NOT NULL,
    fetched_at  TEXT
);
CREATE TABLE IF NOT EXISTS labels (
    item_id  INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    type     TEXT NOT NULL CHECK (type IN ('person', 'category', 'tag')),
    key      TEXT NOT NULL,
    name     TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (item_id, type, key)
);
CREATE INDEX IF NOT EXISTS labels_by_key ON labels (type, key);
CREATE TABLE IF NOT EXISTS stat_history (
    item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    fetched_at TEXT NOT NULL,
    views      INTEGER,
    likes      INTEGER,
    comments   INTEGER
);
"""


def default_path() -> Path:
    configured = os.environ.get("TUBEDEX_DB")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".tubedex" / "library.db"


def label_key(name: str) -> str:
    """'Big-Booty', 'big booty', 'BIG_BOOTY' -> 'big booty'.

    Sites spell the same category differently; without this every spelling
    would become its own playlist.
    """
    return " ".join(re.sub(r"[-_]+", " ", name).split()).casefold()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class NotFound(KeyError):
    pass


class Library:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        with self._tx() as db:
            db.executescript(SCHEMA)
            for site in KNOWN_SITES:
                db.execute(
                    "INSERT OR IGNORE INTO sites (key, name, color) VALUES (?, ?, ?)",
                    (site.key, site.name, site.color),
                )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._db
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    # -- sites --------------------------------------------------------------

    def _ensure_site(self, db: sqlite3.Connection, url: str) -> str:
        site = identify(url)
        if db.execute("SELECT 1 FROM sites WHERE key = ?", (site.key,)).fetchone():
            return site.key
        in_use = [row[0] for row in db.execute("SELECT color FROM sites")]
        db.execute(
            "INSERT INTO sites (key, name, color) VALUES (?, ?, ?)",
            (site.key, site.name, generate_color(in_use)),
        )
        return site.key

    def sites(self, used_only: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT s.key, s.name, s.color, s.custom, COUNT(i.id) AS count "
                "FROM sites s LEFT JOIN items i ON i.site_key = s.key "
                "GROUP BY s.key"
            ).fetchall()
        out = [_site_dict(row) | {"count": row["count"], "custom": bool(row["custom"])}
               for row in rows]
        if used_only:
            out = [s for s in out if s["count"]]
        return sorted(out, key=lambda s: sort_key(s["name"]))

    def set_site_color(self, key: str, color: str | None) -> dict[str, Any]:
        """Override a site's colour; `None` restores the default/generated one."""
        with self._tx() as db:
            row = db.execute("SELECT * FROM sites WHERE key = ?", (key,)).fetchone()
            if row is None:
                raise NotFound(f"no site {key!r}")
            if color is None:
                known = next((s for s in KNOWN_SITES if s.key == key), None)
                if known:
                    new = known.color
                else:
                    others = [r[0] for r in db.execute(
                        "SELECT color FROM sites WHERE key != ?", (key,))]
                    new = generate_color(others)
                db.execute("UPDATE sites SET color = ?, custom = 0 WHERE key = ?", (new, key))
            else:
                db.execute("UPDATE sites SET color = ?, custom = 1 WHERE key = ?",
                           (normalize_hex(color), key))
        return next(s for s in self.sites() if s["key"] == key)

    # -- adding and fetching -----------------------------------------------

    def add(self, raw_url: str, kind: str | None = None) -> tuple[dict[str, Any], bool]:
        """Insert a link as pending. Returns (item, created)."""
        url = normalize_url(raw_url)
        if kind is not None and kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        with self._tx() as db:
            row = db.execute("SELECT id FROM items WHERE url = ?", (url,)).fetchone()
            if row:
                return self._item(db, row["id"]), False
            site_key = self._ensure_site(db, url)
            locked = ["kind"] if kind else []
            cur = db.execute(
                "INSERT INTO items (url, kind, site_key, status, locked, added_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (url, kind or guess_kind(url) or "link", site_key, PENDING,
                 json.dumps(locked), _now()),
            )
            return self._item(db, cur.lastrowid), True

    def fetch(self, item_id: int, fetcher: Fetcher) -> dict[str, Any]:
        """Fetch metadata for an item and store it. Hand-edited fields are kept.

        If the link turns out to be a video already in the library under a
        different URL, the new row is dropped and the existing item returned
        with `duplicate_of` set.
        """
        item = self.get(item_id)
        locked = set(item["locked"])
        record = fetcher(item["url"], item["kind"] if "kind" in locked else None)
        return self.apply(item_id, record)

    def apply(self, item_id: int, record: Record) -> dict[str, Any]:
        with self._tx() as db:
            row = db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise NotFound(item_id)
            locked = set(json.loads(row["locked"]))
            if record.media_key:
                dup = db.execute(
                    "SELECT id FROM items WHERE media_key = ? AND id != ?",
                    (record.media_key, item_id),
                ).fetchone()
                if dup:
                    db.execute("DELETE FROM items WHERE id = ?", (item_id,))
                    return self._item(db, dup["id"]) | {"duplicate_of": dup["id"]}

            values: dict[str, Any] = {
                "page_url": record.page_url,
                "media_key": record.media_key,
                "error": record.error,
                "warnings": json.dumps(record.warnings),
                "sources": json.dumps(record.sources),
                "fetched_at": _now(),
            }
            got_something = bool(record.title or record.views is not None or record.people
                                 or record.categories)
            # A failed refresh must not wipe good data: keep every field and the
            # previous status, and only record what went wrong.
            failed = bool(record.error) and not got_something
            if record.page_url and not failed:
                values["site_key"] = self._ensure_site(db, record.page_url)
            for name in ("kind",) + TEXT_FIELDS + NUMBER_FIELDS:
                value = getattr(record, name)
                if name in locked or failed or (value is None and record.error):
                    continue
                values[name] = value
            if failed:
                values["status"] = ERROR if row["status"] in (PENDING, ERROR) else row["status"]
            else:
                values["status"] = PARTIAL if record.missing() else OK
            assignments = ", ".join(f"{name} = ?" for name in values)
            db.execute(f"UPDATE items SET {assignments} WHERE id = ?",
                       [*values.values(), item_id])
            for name in LIST_FIELDS:
                if name in locked or failed or (record.error and not getattr(record, name)):
                    continue
                self._set_labels(db, item_id, name, getattr(record, name))
            if any(getattr(record, s) is not None for s in ("views", "likes", "comments")):
                db.execute(
                    "INSERT INTO stat_history (item_id, fetched_at, views, likes, comments) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (item_id, values["fetched_at"], record.views, record.likes, record.comments),
                )
            return self._item(db, item_id)

    def _set_labels(self, db: sqlite3.Connection, item_id: int, field_name: str,
                    names: Iterable[str]) -> None:
        label_type = LABEL_TYPES[field_name]
        db.execute("DELETE FROM labels WHERE item_id = ? AND type = ?", (item_id, label_type))
        seen: set[str] = set()
        for position, name in enumerate(names):
            name = " ".join(str(name).split())
            key = label_key(name)
            if not key or key in seen:
                continue
            seen.add(key)
            db.execute(
                "INSERT INTO labels (item_id, type, key, name, position) VALUES (?, ?, ?, ?, ?)",
                (item_id, label_type, key, name, position),
            )

    # -- editing ------------------------------------------------------------

    def edit(self, item_id: int, changes: dict[str, Any]) -> dict[str, Any]:
        """Set fields by hand. Each edited field is locked so refreshes keep it."""
        unknown = set(changes) - set(EDITABLE)
        if unknown:
            raise ValueError(f"cannot edit: {', '.join(sorted(unknown))}")
        with self._tx() as db:
            row = db.execute("SELECT locked FROM items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise NotFound(item_id)
            locked = set(json.loads(row["locked"]))
            for name, value in changes.items():
                locked.add(name)
                if name in LIST_FIELDS:
                    if isinstance(value, str):
                        value = value.split(",")
                    self._set_labels(db, item_id, name, value or [])
                    continue
                if name == "kind" and value not in KINDS:
                    raise ValueError(f"kind must be one of {', '.join(KINDS)}")
                if name in NUMBER_FIELDS and value not in (None, ""):
                    value = int(value)
                if name == "published" and value:
                    value = date.fromisoformat(str(value)).isoformat()
                if value == "":
                    value = None
                db.execute(f"UPDATE items SET {name} = ? WHERE id = ?", (value, item_id))
            db.execute("UPDATE items SET locked = ? WHERE id = ?",
                       (json.dumps(sorted(locked)), item_id))
            return self._item(db, item_id)

    def unlock(self, item_id: int, fields: Iterable[str] | None = None) -> dict[str, Any]:
        """Let the next refresh overwrite hand-edited fields again."""
        with self._tx() as db:
            row = db.execute("SELECT locked FROM items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise NotFound(item_id)
            locked = set(json.loads(row["locked"]))
            locked = set() if fields is None else locked - set(fields)
            db.execute("UPDATE items SET locked = ? WHERE id = ?",
                       (json.dumps(sorted(locked)), item_id))
            return self._item(db, item_id)

    def delete(self, item_id: int) -> None:
        with self._tx() as db:
            if db.execute("DELETE FROM items WHERE id = ?", (item_id,)).rowcount == 0:
                raise NotFound(item_id)

    # -- reading ------------------------------------------------------------

    def get(self, item_id: int) -> dict[str, Any]:
        with self._lock:
            return self._item(self._db, item_id)

    def _item(self, db: sqlite3.Connection, item_id: int) -> dict[str, Any]:
        items = self._hydrate(db, db.execute(
            "SELECT i.*, s.name AS site_name, s.color AS site_color "
            "FROM items i JOIN sites s ON s.key = i.site_key WHERE i.id = ?", (item_id,)
        ).fetchall())
        if not items:
            raise NotFound(item_id)
        item = items[0]
        item["history"] = [dict(r) for r in db.execute(
            "SELECT fetched_at, views, likes, comments FROM stat_history "
            "WHERE item_id = ? ORDER BY fetched_at", (item_id,))]
        return item

    def _hydrate(self, db: sqlite3.Connection, rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        if not rows:
            return []
        ids = [row["id"] for row in rows]
        labels: dict[int, dict[str, list[str]]] = {i: {f: [] for f in LIST_FIELDS} for i in ids}
        reverse = {v: k for k, v in LABEL_TYPES.items()}
        placeholders = ",".join("?" * len(ids))
        for lab in db.execute(
            f"SELECT item_id, type, name FROM labels WHERE item_id IN ({placeholders}) "
            "ORDER BY position", ids,
        ):
            labels[lab["item_id"]][reverse[lab["type"]]].append(lab["name"])
        out = []
        for row in rows:
            item = {k: row[k] for k in row.keys() if k not in ("site_name", "site_color")}
            for name in ("warnings", "sources", "locked"):
                item[name] = json.loads(item[name])
            item.update(labels[row["id"]])
            item["site"] = _site_dict({"key": row["site_key"], "name": row["site_name"],
                                       "color": row["site_color"]})
            item["views_per_day"] = _per_day(item["views"], item["published"])
            total = (item["likes"] or 0) + (item["dislikes"] or 0)
            item["rating"] = round(100 * item["likes"] / total, 1) if item["likes"] is not None \
                and item["dislikes"] is not None and total else None
            out.append(item)
        return out

    def items(
        self,
        *,
        category: str | None = None,
        uncategorised: bool = False,
        person: str | None = None,
        tag: str | None = None,
        site: str | None = None,
        kind: str | None = None,
        search: str | None = None,
        status: str | None = None,
        sort: str = "title",
    ) -> list[dict[str, Any]]:
        where, args = [], []
        if uncategorised:
            where.append("NOT EXISTS (SELECT 1 FROM labels l WHERE l.item_id = i.id "
                         "AND l.type = 'category')")
        for label_type, value in (("category", category), ("person", person), ("tag", tag)):
            if value is None:
                continue
            where.append("EXISTS (SELECT 1 FROM labels l WHERE l.item_id = i.id "
                         "AND l.type = ? AND l.key = ?)")
            args += [label_type, label_key(value)]
        if site:
            where.append("i.site_key = ?")
            args.append(site)
        if kind:
            where.append("i.kind = ?")
            args.append(kind)
        if status:
            where.append("i.status = ?")
            args.append(status)
        if search:
            like = f"%{search.strip()}%"
            where.append("(i.title LIKE ? OR i.channel LIKE ? OR i.url LIKE ? OR EXISTS "
                         "(SELECT 1 FROM labels l WHERE l.item_id = i.id AND l.name LIKE ?))")
            args += [like, like, like, like]
        sql = ("SELECT i.*, s.name AS site_name, s.color AS site_color FROM items i "
               "JOIN sites s ON s.key = i.site_key")
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._lock:
            items = self._hydrate(self._db, self._db.execute(sql, args).fetchall())
        return _sorted(items, sort)

    def playlists(self) -> list[dict[str, Any]]:
        """One playlist per category, A-Z by the sorting rule.

        A category's display name is the spelling most items use. Items
        inside each playlist are ordered by title with the same rule.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT key, name, COUNT(*) AS n FROM labels WHERE type = 'category' "
                "GROUP BY key, name"
            ).fetchall()
            uncategorised = self._db.execute(
                "SELECT COUNT(*) FROM items i WHERE NOT EXISTS (SELECT 1 FROM labels l "
                "WHERE l.item_id = i.id AND l.type = 'category')"
            ).fetchone()[0]
        spellings: dict[str, Counter[str]] = {}
        for row in rows:
            spellings.setdefault(row["key"], Counter())[row["name"]] += row["n"]
        lists = [
            {"key": key, "name": names.most_common(1)[0][0], "count": sum(names.values())}
            for key, names in spellings.items()
        ]
        lists.sort(key=lambda p: sort_key(p["name"]))
        if uncategorised:
            lists.append({"key": None, "name": UNCATEGORISED, "count": uncategorised,
                          "uncategorised": True})
        return lists

    def labels(self, field_name: str) -> list[dict[str, Any]]:
        """All distinct people / categories / tags with item counts."""
        label_type = LABEL_TYPES[field_name]
        with self._lock:
            rows = self._db.execute(
                "SELECT key, name, COUNT(*) AS n FROM labels WHERE type = ? GROUP BY key, name",
                (label_type,),
            ).fetchall()
        merged: dict[str, Counter[str]] = {}
        for row in rows:
            merged.setdefault(row["key"], Counter())[row["name"]] += row["n"]
        out = [{"name": c.most_common(1)[0][0], "count": sum(c.values())} for c in merged.values()]
        return sorted(out, key=lambda x: sort_key(x["name"]))

    def summary(self) -> dict[str, Any]:
        with self._lock:
            db = self._db
            total = db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            kinds = dict(db.execute("SELECT kind, COUNT(*) FROM items GROUP BY kind").fetchall())
            statuses = dict(db.execute(
                "SELECT status, COUNT(*) FROM items GROUP BY status").fetchall())
        return {"items": total, "kinds": kinds, "statuses": statuses}


def _site_dict(row: Any) -> dict[str, Any]:
    return {"key": row["key"], "name": row["name"], "color": row["color"],
            "text_color": text_color(row["color"])}


def _per_day(views: int | None, published: str | None) -> float | None:
    if views is None or not published:
        return None
    try:
        age = (date.today() - date.fromisoformat(published)).days
    except ValueError:
        return None
    return round(views / max(age, 1), 1)


_SORTS = {
    "title": (lambda i: sort_key(i["title"] or i["url"]), False),
    "views": (lambda i: (i["views"] is not None, i["views"] or 0), True),
    "likes": (lambda i: (i["likes"] is not None, i["likes"] or 0), True),
    "comments": (lambda i: (i["comments"] is not None, i["comments"] or 0), True),
    "published": (lambda i: (i["published"] is not None, i["published"] or ""), True),
    "added": (lambda i: i["id"], True),
}
SORTS = tuple(_SORTS)


def _sorted(items: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
    if sort not in _SORTS:
        raise ValueError(f"sort must be one of {', '.join(SORTS)}")
    key, reverse = _SORTS[sort]
    return sorted(items, key=key, reverse=reverse)

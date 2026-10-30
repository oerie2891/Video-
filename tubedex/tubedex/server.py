"""FastAPI app behind the web UI.

Adding a link returns immediately with the item marked `pending`; a small
thread pool fetches metadata in the background and the page polls. A slow or
geo-blocked site therefore never holds up the request or the other links.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .extract import KINDS, Extractor, Fetcher, FetchOptions, Record
from .library import EDITABLE, SORTS, Library, NotFound

WEB_DIR = Path(__file__).parent / "web"


class AddRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=500)
    kind: str | None = None


class ColorRequest(BaseModel):
    color: str | None = None


def create_app(
    library: Library | None = None,
    fetcher: Fetcher | None = None,
    workers: int | None = None,
) -> FastAPI:
    lib = library or Library()
    fetch = fetcher or Extractor(FetchOptions.from_env())
    pool = ThreadPoolExecutor(
        max_workers=int(workers or os.environ.get("TUBEDEX_WORKERS", "3")),
        thread_name_prefix="tubedex-fetch",
    )
    app = FastAPI(title="tubedex", version=__version__)
    app.state.library = lib
    app.state.pool = pool

    @app.on_event("shutdown")
    def _stop() -> None:  # pragma: no cover - lifecycle
        pool.shutdown(wait=False, cancel_futures=True)

    def _fetch_in_background(item_id: int) -> None:
        def job() -> None:
            try:
                lib.fetch(item_id, fetch)
            except NotFound:
                pass  # deleted while queued
            except Exception as exc:  # noqa: BLE001 - surface on the item, never crash
                try:
                    item = lib.get(item_id)
                    lib.apply(item_id, Record(url=item["url"], kind=item["kind"],
                                              error=f"unexpected error: {exc}"))
                except NotFound:
                    pass

        pool.submit(job)

    # The queue lives in memory, so links still pending when the server last
    # stopped would otherwise never be fetched.
    for item in lib.items(status="pending", sort="added"):
        _fetch_in_background(item["id"])

    def _get(item_id: int) -> dict[str, Any]:
        try:
            return lib.get(item_id)
        except NotFound:
            raise HTTPException(404, f"no item {item_id}") from None

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {"version": __version__, "kinds": list(KINDS), "sorts": list(SORTS),
                "editable": list(EDITABLE), "summary": lib.summary()}

    @app.get("/api/items")
    def list_items(
        category: str | None = None,
        uncategorised: bool = False,
        person: str | None = None,
        tag: str | None = None,
        site: str | None = None,
        kind: str | None = None,
        q: str | None = None,
        status: str | None = None,
        sort: str = "title",
    ) -> list[dict[str, Any]]:
        try:
            return lib.items(category=category, uncategorised=uncategorised, person=person,
                             tag=tag, site=site, kind=kind, search=q, status=status, sort=sort)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/items")
    def add_items(body: AddRequest) -> dict[str, Any]:
        if body.kind is not None and body.kind not in KINDS:
            raise HTTPException(400, f"kind must be one of {', '.join(KINDS)}")
        added, existing, rejected = [], [], []
        for raw in body.urls:
            if not raw.strip():
                continue
            try:
                item, created = lib.add(raw, body.kind)
            except ValueError as exc:
                rejected.append({"url": raw, "error": str(exc)})
                continue
            if created:
                added.append(item)
                _fetch_in_background(item["id"])
            else:
                existing.append(item)
        return {"added": added, "existing": existing, "rejected": rejected}

    @app.get("/api/items/{item_id}")
    def get_item(item_id: int) -> dict[str, Any]:
        return _get(item_id)

    @app.patch("/api/items/{item_id}")
    def edit_item(item_id: int, changes: dict[str, Any]) -> dict[str, Any]:
        _get(item_id)
        try:
            return lib.edit(item_id, changes)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/items/{item_id}/unlock")
    def unlock_item(item_id: int) -> dict[str, Any]:
        _get(item_id)
        return lib.unlock(item_id)

    @app.post("/api/items/{item_id}/refresh")
    def refresh_item(item_id: int) -> dict[str, Any]:
        item = _get(item_id)
        _fetch_in_background(item_id)
        return item | {"refreshing": True}

    @app.delete("/api/items/{item_id}")
    def delete_item(item_id: int) -> dict[str, Any]:
        try:
            lib.delete(item_id)
        except NotFound:
            raise HTTPException(404, f"no item {item_id}") from None
        return {"deleted": item_id}

    @app.get("/api/playlists")
    def playlists() -> list[dict[str, Any]]:
        return lib.playlists()

    @app.get("/api/labels/{field_name}")
    def labels(field_name: str) -> list[dict[str, Any]]:
        if field_name not in ("people", "categories", "tags"):
            raise HTTPException(404, "labels are people, categories or tags")
        return lib.labels(field_name)

    @app.get("/api/sites")
    def sites(used: bool = False) -> list[dict[str, Any]]:
        return lib.sites(used_only=used)

    @app.patch("/api/sites/{key}")
    def set_site_color(key: str, body: ColorRequest) -> dict[str, Any]:
        try:
            return lib.set_site_color(key, body.color)
        except NotFound:
            raise HTTPException(404, f"no site {key!r}") from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    return app

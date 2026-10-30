"""Extraction pipeline (with yt-dlp faked), the library, export and the API."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import yt_dlp
from fastapi.testclient import TestClient
from yt_dlp.utils import DownloadError

from tubedex.export import export
from tubedex.extract import Extractor, Record, guess_kind, normalize_url
from tubedex.library import Library
from tubedex.server import create_app

from .test_core import VIDEO_PAGE

PORNHUB_INFO = {
    "id": "ph123", "extractor_key": "PornHub", "title": "Full Info Video",
    "uploader": "Big Channel", "cast": ["Ava Example", "Bo Sample"],
    "categories": ["Amateur", "HD Porn"], "tags": ["fun"], "view_count": 5000,
    "like_count": 90, "dislike_count": 10, "comment_count": 7, "upload_date": "20230102",
    "duration": 600, "formats": [{"height": 720}, {"height": 1080}],
    "webpage_url": "https://www.pornhub.com/view_video.php?viewkey=ph123",
}
XVIDEOS_INFO = {
    "id": "xv1", "extractor_key": "XVideos", "title": "Thin Info Video", "duration": 750,
    "formats": [{"height": 480}],
}


class FakeResponse:
    def __init__(self, url: str, body: str = "", content_type: str = "text/html; charset=utf-8"):
        self.url = url
        self._body = body.encode()
        self.headers = {"Content-Type": content_type}

    def read(self, limit: int = -1) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL. `infos` maps URL -> info dict or exception;
    `pages` maps URL -> FakeResponse. Records which URLs were requested."""

    infos: dict[str, Any] = {}
    pages: dict[str, FakeResponse] = {}
    requested: list[str] = []

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def extract_info(self, url: str, download: bool = False) -> dict[str, Any]:
        FakeYDL.requested.append(("ytdlp", url))
        result = self.infos.get(url, DownloadError(f"ERROR: Unsupported URL: {url}"))
        if isinstance(result, Exception):
            raise result
        return result

    def urlopen(self, request: Any) -> FakeResponse:
        url = request if isinstance(request, str) else request.url
        FakeYDL.requested.append(("page", url))
        if url not in self.pages:
            raise OSError("HTTP Error 404: Not Found")
        return self.pages[url]


@pytest.fixture()
def fake_ydl(monkeypatch):
    FakeYDL.infos, FakeYDL.pages, FakeYDL.requested = {}, {}, []
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return FakeYDL


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def test_complete_ytdlp_result_skips_page_fetch(fake_ydl):
    url = "https://www.pornhub.com/view_video.php?viewkey=ph123"
    fake_ydl.infos[url] = PORNHUB_INFO
    record = Extractor().fetch(url)
    assert record.kind == "video"
    assert record.title == "Full Info Video"
    assert record.channel == "Big Channel"
    assert record.people == ["Ava Example", "Bo Sample"]
    assert record.categories == ["Amateur", "HD Porn"]
    assert (record.views, record.likes, record.dislikes, record.comments) == (5000, 90, 10, 7)
    assert record.published == "2023-01-02"
    assert record.height == 1080
    assert record.media_key == "pornhub:ph123"
    assert ("page", url) not in fake_ydl.requested
    assert set(record.sources.values()) == {"yt-dlp"}


def test_thin_ytdlp_result_is_filled_from_the_page(fake_ydl):
    url = "https://www.xvideos.com/video.xv1/thin"
    fake_ydl.infos[url] = XVIDEOS_INFO
    fake_ydl.pages[url] = FakeResponse(url, VIDEO_PAGE)
    record = Extractor().fetch(url)
    assert record.title == "Thin Info Video"           # yt-dlp wins where it has a value
    assert record.sources["title"] == "yt-dlp"
    assert record.people == ["Ava Example", "Bo Sample"]  # JSON-LD fills the gaps
    assert record.views == 1234567 and record.sources["views"] == "json-ld"
    assert record.categories == ["Outdoor", "Romantic"]
    assert record.channel == "Studio One"
    assert record.dislikes == 310 and record.sources["dislikes"] == "markup"


def test_unsupported_url_falls_back_to_page_quietly(fake_ydl):
    url = "https://unknown.example/watch/1"
    fake_ydl.pages[url] = FakeResponse(url, VIDEO_PAGE)
    record = Extractor().fetch(url)
    assert record.title == "Sunset & Waves"
    assert record.kind == "video"
    assert not any("Unsupported" in w for w in record.warnings)


def test_direct_image_link(fake_ydl):
    url = "https://cdn.example/pics/My_Holiday_Shot.jpg"
    fake_ydl.pages[url] = FakeResponse(url, content_type="image/jpeg")
    record = Extractor().fetch(url)
    assert record.kind == "photo"
    assert record.title == "My Holiday Shot"
    assert not [r for r in fake_ydl.requested if r[0] == "ytdlp"]


def test_gif_and_gallery_detection_from_url():
    assert guess_kind("https://www.pornhub.com/gif/12345") == "gif"
    assert guess_kind("https://www.pornhub.com/album/999") == "photo"
    assert guess_kind("https://xhamster.com/photos/gallery/abc-1") == "photo"
    assert guess_kind("https://example.com/files/issue.pdf") == "magazine"
    assert guess_kind("https://www.redgifs.com/watch/somename") == "gif"
    assert guess_kind("https://www.pornhub.com/view_video.php?viewkey=1") == "video"
    assert guess_kind("https://x.com/someone/status/1") is None


def test_unreachable_link_is_an_error_not_a_crash(fake_ydl):
    url = "https://www.eporner.com/video-gone/x/"
    fake_ydl.infos[url] = DownloadError("ERROR: [Eporner] gone: HTTP Error 404: Not Found")
    record = Extractor().fetch(url)
    assert record.error
    assert any("404" in w for w in record.warnings)


def test_site_name_is_not_used_as_channel(fake_ydl):
    url = "https://www.redtube.com/555"
    page = VIDEO_PAGE.replace('"name": "Studio One"', '"name": "RedTube"').replace(
        '<li class="main-uploader">', "<li>")
    fake_ydl.pages[url] = FakeResponse(url, page)
    assert Extractor().fetch(url).channel is None


def test_normalize_url():
    assert normalize_url("WWW.Pornhub.com/view_video.php?viewkey=abc&utm_source=x#t=10") == \
        "https://www.pornhub.com/view_video.php?viewkey=abc"
    with pytest.raises(ValueError):
        normalize_url("ftp://example.com/file")
    with pytest.raises(ValueError):
        normalize_url("   ")


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------


def fake_fetcher(records: dict[str, Record]):
    def fetch(url: str, kind: str | None = None) -> Record:
        record = records[url]
        if kind:
            record.kind = kind
        return record
    return fetch


def _rec(url: str, title: str, categories: list[str], **kw: Any) -> Record:
    return Record(url=url, kind="video", title=title, categories=categories, channel="C",
                  people=["P"], views=10, likes=1, comments=0, published="2024-01-01", **kw)


@pytest.fixture()
def lib(tmp_path: Path):
    library = Library(tmp_path / "lib.db")
    yield library
    library.close()


def test_add_fetch_and_dedupe(lib):
    url = normalize_url("https://www.pornhub.com/view_video.php?viewkey=a")
    item, created = lib.add(url)
    assert created and item["status"] == "pending"
    again, created = lib.add(url + "&utm_source=newsletter")
    assert not created and again["id"] == item["id"]

    item = lib.apply(item["id"], _rec(url, "First", ["Amateur"], media_key="pornhub:a"))
    assert item["status"] == "ok" and item["site"]["color"] == "#FF9000"

    other = "https://de.pornhub.com/view_video.php?viewkey=a"
    second, _ = lib.add(other)
    result = lib.apply(second["id"], _rec(other, "First", ["Amateur"], media_key="pornhub:a"))
    assert result["duplicate_of"] == item["id"]
    assert len(lib.items()) == 1


def test_hand_edits_survive_refresh_and_failed_refresh_keeps_data(lib):
    url = "https://www.xvideos.com/video1/a"
    item, _ = lib.add(url)
    lib.apply(item["id"], _rec(url, "Original", ["Amateur"]))
    lib.edit(item["id"], {"people": "Ava, Bo", "title": "My Title", "views": 0})
    item = lib.apply(item["id"], _rec(url, "Site Title", ["Amateur"], ))
    assert item["title"] == "My Title"
    assert item["people"] == ["Ava", "Bo"]
    assert item["views"] == 0
    assert sorted(item["locked"]) == ["people", "title", "views"]

    failed = lib.apply(item["id"], Record(url=url, kind="link", error="HTTP Error 403"))
    assert failed["categories"] == ["Amateur"] and failed["published"] == "2024-01-01"
    assert failed["kind"] == "video"
    assert failed["status"] == "ok" and failed["error"] == "HTTP Error 403"

    assert lib.unlock(item["id"])["locked"] == []


def test_playlists_order_merge_spellings_and_sort_titles(lib):
    records = [
        ("https://www.pornhub.com/v?viewkey=1", "Zed", ["Big-Tits", "Amateur"]),
        ("https://www.xvideos.com/v2/b", "10 Minutes", ["big tits", "3D"]),
        ("https://www.xnxx.com/v3/c", "2 Hours", ["Big Tits", "#Trending"]),
        ("https://www.eporner.com/v4/d/", "!Wow", ["Big Tits"]),
        ("https://www.youporn.com/watch/5/e/", "apple", []),
    ]
    for url, title, cats in records:
        item, _ = lib.add(url)
        lib.apply(item["id"], _rec(url, title, cats))
    lists = lib.playlists()
    assert [p["name"] for p in lists] == ["#Trending", "3D", "Amateur", "Big Tits",
                                          "Uncategorised"]
    big = next(p for p in lists if p["name"] == "Big Tits")
    assert big["count"] == 4  # three spellings, one playlist
    titles = [i["title"] for i in lib.items(category="big-tits")]
    assert titles == ["!Wow", "2 Hours", "10 Minutes", "Zed"]
    assert [i["title"] for i in lib.items(uncategorised=True)] == ["apple"]


def test_unknown_sites_get_distinct_persistent_colours(lib):
    a, _ = lib.add("https://spankbang.com/a/video/x")
    b, _ = lib.add("https://www.example-tube.net/v/1")
    colours = {s["key"]: s["color"] for s in lib.sites()}
    assert colours["spankbang.com"] != colours["example-tube.net"]
    assert colours["spankbang.com"] not in {"#FF9000", "#FF0000", "#000000"}
    assert lib.get(a["id"])["site"]["color"] == colours["spankbang.com"]

    lib.set_site_color("pornhub", "#ffa31a")
    assert next(s for s in lib.sites() if s["key"] == "pornhub")["color"] == "#FFA31A"
    lib.set_site_color("pornhub", None)
    assert next(s for s in lib.sites() if s["key"] == "pornhub")["color"] == "#FF9000"


def test_export_writes_ordered_folders(lib, tmp_path):
    for url, title, cats in [
        ("https://www.pornhub.com/v?viewkey=1", "Beta", ["Amateur"]),
        ("https://www.xvideos.com/v2/b", "Alpha: Part/2", ["Amateur", "3D"]),
    ]:
        item, _ = lib.add(url)
        lib.apply(item["id"], _rec(url, title, cats))
    result = export(lib, tmp_path / "out", style="url")
    assert result == {"folders": 2, "shortcuts": 3, "path": str((tmp_path / "out").resolve())}
    folders = sorted(p.name for p in (tmp_path / "out").iterdir() if p.is_dir())
    assert folders == ["001 3D", "002 Amateur"]
    files = sorted(p.name for p in (tmp_path / "out" / "002 Amateur").iterdir())
    assert files == ["001 [XVideos] Alpha- Part-2.url", "002 [Pornhub] Beta.url"]
    body = (tmp_path / "out" / "002 Amateur" / files[1]).read_text()
    assert "URL=https://www.pornhub.com/v?viewkey=1" in body
    assert "#FF9000" in (tmp_path / "out" / "index.html").read_text()


# --------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------


def test_api_add_poll_edit_and_colour(tmp_path):
    url = "https://www.pornhub.com/view_video.php?viewkey=zz"
    records = {url: _rec(url, "API Video", ["Solo"], media_key="pornhub:zz")}
    lib = Library(tmp_path / "api.db")
    with TestClient(create_app(lib, fetcher=fake_fetcher(records), workers=1)) as client:
        res = client.post("/api/items", json={"urls": [url, "not a link at all ::", url]})
        body = res.json()
        assert len(body["added"]) == 1 and len(body["existing"]) == 1
        assert len(body["rejected"]) == 1
        item_id = body["added"][0]["id"]

        deadline = time.time() + 10
        while client.get(f"/api/items/{item_id}").json()["status"] == "pending":
            assert time.time() < deadline
            time.sleep(0.02)
        item = client.get(f"/api/items/{item_id}").json()
        assert item["title"] == "API Video" and item["site"]["name"] == "Pornhub"

        assert client.get("/api/playlists").json()[0]["name"] == "Solo"
        assert client.get("/api/items", params={"category": "solo"}).json()[0]["id"] == item_id

        edited = client.patch(f"/api/items/{item_id}", json={"categories": ["Duo", "Solo"]})
        assert edited.json()["categories"] == ["Duo", "Solo"]
        assert client.patch(f"/api/items/{item_id}", json={"url": "x"}).status_code == 400

        assert client.patch("/api/sites/xvideos", json={"color": "nope"}).status_code == 400
        assert client.patch("/api/sites/xvideos", json={"color": "#CC0000"}).json()["color"] == "#CC0000"
        assert "tubedex" in client.get("/").text

        assert client.delete(f"/api/items/{item_id}").status_code == 200
        assert client.get(f"/api/items/{item_id}").status_code == 404


def test_pending_items_are_fetched_after_a_restart(tmp_path):
    url = "https://www.youporn.com/watch/9/x/"
    lib = Library(tmp_path / "restart.db")
    item, _ = lib.add(url)  # added, then the server "stopped" before fetching
    with TestClient(create_app(lib, fetcher=fake_fetcher({url: _rec(url, "Later", [])}),
                               workers=1)) as client:
        deadline = time.time() + 10
        while client.get(f"/api/items/{item['id']}").json()["status"] == "pending":
            assert time.time() < deadline
            time.sleep(0.02)
        assert client.get(f"/api/items/{item['id']}").json()["title"] == "Later"

"""HTTP API tests.

The subtitle-in/subtitle-out path exercises upload, the job queue, progress,
artifact listing and download without needing any model weights or API key.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from subtrans.server import create_app

SRT = (
    "1\n00:00:01,000 --> 00:00:03,000\nHello there.\n\n"
    "2\n00:00:04,000 --> 00:00:06,500\nThis is the second cue.\n"
)


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(data_dir=tmp_path / "jobs", workers=2)
    with TestClient(app) as c:
        yield c


def _wait(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def test_index_serves_the_ui(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    assert "subtrans" in res.text
    assert "<form" in res.text


def test_meta_describes_capabilities(client: TestClient):
    data = client.get("/api/meta").json()
    assert data["ffmpeg"] is True
    assert len(data["languages"]) > 90
    assert {"srt", "vtt", "txt", "json"} <= set(data["formats"])
    assert any(lang["code"] == "es" for lang in data["languages"])
    assert all("tier" in lang for lang in data["languages"])
    assert data["asr_backends"] and data["mt_backends"]


def test_health(client: TestClient):
    data = client.get("/api/health").json()
    assert data["ok"] is True
    assert "asr_ready" in data and "mt_ready" in data


def test_subtitle_conversion_job_completes_and_downloads(client: TestClient):
    res = client.post(
        "/api/jobs",
        files={"file": ("sample.srt", io.BytesIO(SRT.encode()), "application/x-subrip")},
        data={"formats": "srt,vtt,txt", "source_language": "en"},
    )
    assert res.status_code == 202, res.text
    job_id = res.json()["id"]

    job = _wait(client, job_id)
    assert job["status"] == "done", job
    assert job["source_language"] == "en"
    assert len(job["artifacts"]) == 3
    formats = {a["format"] for a in job["artifacts"]}
    assert formats == {"srt", "vtt", "txt"}
    assert job["metrics"]["en"]["cues"] == 2

    for artifact in job["artifacts"]:
        got = client.get(f"/api/jobs/{job_id}/artifacts/{artifact['index']}")
        assert got.status_code == 200
        assert len(got.content) > 0
        assert artifact["filename"].endswith(artifact["format"])


def test_preview_returns_cues_with_reading_speed(client: TestClient):
    res = client.post(
        "/api/jobs",
        files={"file": ("sample.srt", io.BytesIO(SRT.encode()), "application/x-subrip")},
        data={"formats": "srt", "source_language": "en"},
    )
    job_id = res.json()["id"]
    _wait(client, job_id)

    data = client.get(f"/api/jobs/{job_id}/preview").json()
    assert data["kind"] == "subtitle"
    assert data["total"] == 2
    assert data["cues"][0]["lines"] == ["Hello there."]
    assert data["cues"][0]["cps"] > 0


def test_unsupported_extension_is_rejected(client: TestClient):
    res = client.post(
        "/api/jobs",
        files={"file": ("virus.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
        data={"formats": "srt"},
    )
    assert res.status_code == 415
    assert "Unsupported file type" in res.json()["detail"]


def test_empty_upload_is_rejected(client: TestClient):
    res = client.post(
        "/api/jobs",
        files={"file": ("empty.srt", io.BytesIO(b""), "application/x-subrip")},
        data={"formats": "srt"},
    )
    assert res.status_code == 400


def test_unknown_format_is_rejected(client: TestClient):
    res = client.post(
        "/api/jobs",
        files={"file": ("sample.srt", io.BytesIO(SRT.encode()), "application/x-subrip")},
        data={"formats": "mp3"},
    )
    assert res.status_code == 400
    assert "Unknown output format" in res.json()["detail"]


def test_unknown_language_is_rejected(client: TestClient):
    res = client.post(
        "/api/jobs",
        files={"file": ("sample.srt", io.BytesIO(SRT.encode()), "application/x-subrip")},
        data={"formats": "srt", "targets": "Klingon"},
    )
    assert res.status_code == 400


def test_missing_job_is_404(client: TestClient):
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    assert client.get("/api/jobs/doesnotexist/artifacts/0").status_code == 404
    assert client.delete("/api/jobs/doesnotexist").status_code == 404


def test_artifact_index_out_of_range_is_404(client: TestClient):
    res = client.post(
        "/api/jobs",
        files={"file": ("sample.srt", io.BytesIO(SRT.encode()), "application/x-subrip")},
        data={"formats": "srt", "source_language": "en"},
    )
    job_id = res.json()["id"]
    _wait(client, job_id)
    assert client.get(f"/api/jobs/{job_id}/artifacts/99").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/artifacts/-1").status_code == 404


def test_job_without_translation_backend_reports_the_reason(client: TestClient, monkeypatch):
    """A translation target with no usable backend must surface a clear warning
    rather than silently producing only the source language."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    res = client.post(
        "/api/jobs",
        files={"file": ("sample.srt", io.BytesIO(SRT.encode()), "application/x-subrip")},
        data={"formats": "srt", "source_language": "en", "targets": "es"},
    )
    job_id = res.json()["id"]
    job = _wait(client, job_id)
    assert job["status"] == "done"
    # The source file is still produced, and the failure is explained.
    assert any(a["language"] == "en" for a in job["artifacts"])
    assert any("es" in w or "translat" in w.lower() for w in job["warnings"]), job["warnings"]


def test_delete_removes_job_and_files(client: TestClient):
    res = client.post(
        "/api/jobs",
        files={"file": ("sample.srt", io.BytesIO(SRT.encode()), "application/x-subrip")},
        data={"formats": "srt", "source_language": "en"},
    )
    job_id = res.json()["id"]
    _wait(client, job_id)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_job_list_includes_submitted_jobs(client: TestClient):
    client.post(
        "/api/jobs",
        files={"file": ("a.srt", io.BytesIO(SRT.encode()), "application/x-subrip")},
        data={"formats": "srt", "source_language": "en"},
    )
    jobs = client.get("/api/jobs").json()["jobs"]
    assert len(jobs) >= 1
    assert jobs[0]["filename"] == "a.srt"

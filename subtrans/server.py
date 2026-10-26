"""FastAPI app: upload a file, watch progress, download the results.

The browser polls `/api/jobs/{id}` rather than holding a long request open,
because a transcription can easily outlive any sensible HTTP timeout.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import __version__
from .asr import available_backends as asr_backends
from .asr.faster_whisper_backend import DEFAULT_MODEL, MODEL_SIZES
from .formats import EXTENSIONS
from .jobs import JobManager
from .languages import LANGUAGES, quality_tier
from .media import AUDIO_EXTS, SUBTITLE_EXTS, TEXT_EXTS, VIDEO_EXTS, have_ffmpeg
from .pipeline import JobSpec
from .translate import available_backends as mt_backends

MAX_UPLOAD_BYTES = int(os.environ.get("SUBTRANS_MAX_UPLOAD_MB", "4096")) * 1024 * 1024
ALLOWED_EXTS = AUDIO_EXTS | VIDEO_EXTS | SUBTITLE_EXTS | TEXT_EXTS
WEB_DIR = Path(__file__).parent / "web"


def _data_dir() -> Path:
    configured = os.environ.get("SUBTRANS_DATA_DIR")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "subtrans-jobs"


def create_app(data_dir: str | Path | None = None, workers: int | None = None) -> FastAPI:
    manager = JobManager(
        data_dir or _data_dir(),
        workers=int(workers or os.environ.get("SUBTRANS_WORKERS", "1")),
    )
    app = FastAPI(title="subtrans", version=__version__)
    app.state.manager = manager

    @app.on_event("shutdown")
    def _stop() -> None:  # pragma: no cover - lifecycle
        manager.shutdown()

    # -- UI ---------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = WEB_DIR / "index.html"
        if not page.exists():  # pragma: no cover
            return HTMLResponse("<h1>subtrans</h1><p>UI assets missing.</p>", status_code=500)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    # -- metadata ---------------------------------------------------------

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "version": __version__,
            "ffmpeg": have_ffmpeg(),
            "languages": [
                {"code": code, "name": name, "tier": quality_tier(code)}
                for code, name in sorted(LANGUAGES.items(), key=lambda kv: kv[1])
            ],
            "formats": sorted(set(EXTENSIONS) | {"docx"}),
            "models": list(MODEL_SIZES),
            "default_model": DEFAULT_MODEL,
            "asr_backends": asr_backends(),
            "mt_backends": mt_backends(),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "accept": sorted(ALLOWED_EXTS),
        }

    # -- jobs -------------------------------------------------------------

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        targets: str = Form(""),
        source_language: str = Form(""),
        formats: str = Form("srt"),
        model: str = Form(""),
        asr_backend: str = Form("auto"),
        mt_backend: str = Form("auto"),
        notes: str = Form(""),
        prompt: str = Form(""),
        formality: str = Form(""),
        include_source: bool = Form(True),
        vad: bool = Form(True),
        normalize_audio: bool = Form(True),
        timestamps_in_text: bool = Form(True),
    ) -> JSONResponse:
        name = Path(file.filename or "upload").name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type {ext or '(none)'}. Accepted: "
                    + ", ".join(sorted(ALLOWED_EXTS))
                ),
            )

        # Reserve the job directory first and stream the upload straight into
        # it, so the worker never sees a half-written or relocated input.
        job_id, work_dir = manager.allocate()
        dest = work_dir / name

        written = 0
        try:
            with dest.open("wb") as fh:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
                                f"limit. Raise SUBTRANS_MAX_UPLOAD_MB to allow more."
                            ),
                        )
                    fh.write(chunk)
        except HTTPException:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        finally:
            await file.close()

        if written == 0:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Uploaded file was empty.")

        try:
            spec = _spec_from_form(
                str(dest),
                targets=targets,
                source_language=source_language,
                formats=formats,
                model=model,
                asr_backend=asr_backend,
                mt_backend=mt_backend,
                notes=notes,
                prompt=prompt,
                formality=formality,
                include_source=include_source,
                vad=vad,
                normalize_audio=normalize_audio,
                timestamps_in_text=timestamps_in_text,
            )
        except ValueError as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job = manager.submit(spec, filename=name, job_id=job_id)
        return JSONResponse({"id": job.id, **job.to_dict(include_result=False)}, status_code=202)

    @app.get("/api/jobs")
    def list_jobs(limit: int = 50) -> dict[str, Any]:
        return {"jobs": [j.to_dict() for j in manager.list(limit=limit)]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/preview")
    def preview(job_id: str, language: str = "", limit: int = 40) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None or job.result is None:
            raise HTTPException(status_code=404, detail="No result yet")
        subtitles = job.result.subtitles
        documents = job.result.documents
        if language and language in subtitles:
            subs = subtitles[language]
        elif subtitles:
            subs = next(iter(subtitles.values()))
        elif documents:
            blocks = documents.get(language) or next(iter(documents.values()))
            return {"kind": "text", "language": language, "blocks": blocks[:limit]}
        else:
            raise HTTPException(status_code=404, detail="Nothing to preview")
        return {
            "kind": "subtitle",
            "language": subs.language,
            "cues": [
                {
                    "index": c.index,
                    "start": round(c.start, 3),
                    "end": round(c.end, 3),
                    "lines": c.lines,
                    "cps": round(c.cps, 1) if c.duration else None,
                    "source_text": c.source_text,
                }
                for c in subs.cues[:limit]
            ],
            "total": len(subs.cues),
        }

    @app.get("/api/jobs/{job_id}/artifacts/{index}")
    def download(job_id: str, index: int) -> FileResponse:
        path = manager.artifact_path(job_id, index)
        if path is None:
            raise HTTPException(status_code=404, detail="No such artifact")
        return FileResponse(
            str(path),
            filename=path.name,
            media_type=_media_type(path.suffix),
        )

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, Any]:
        if not manager.cancel(job_id):
            raise HTTPException(status_code=409, detail="Job is not cancellable")
        return {"ok": True}

    @app.delete("/api/jobs/{job_id}")
    def delete(job_id: str) -> dict[str, Any]:
        if not manager.delete(job_id):
            raise HTTPException(status_code=404, detail="No such job")
        return {"ok": True}

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "ffmpeg": have_ffmpeg(),
            "asr_ready": any(b["available"] for b in asr_backends()),
            "mt_ready": any(b["available"] for b in mt_backends()),
        }

    return app


def _spec_from_form(input_path: str, **form: Any) -> JobSpec:
    from .languages import normalize

    targets = [t for t in _split(form.get("targets", "")) if t]
    normalized = [normalize(t) or t for t in targets]
    formats = _split(form.get("formats", "srt")) or ["srt"]
    unknown = [f for f in formats if f not in (set(EXTENSIONS) | {"docx"})]
    if unknown:
        raise ValueError(f"Unknown output format(s): {', '.join(unknown)}")

    return JobSpec(
        input_path=input_path,
        targets=normalized,
        source_language=normalize(form.get("source_language") or None),
        formats=formats,
        asr_backend=form.get("asr_backend") or "auto",
        asr_model=form.get("model") or None,
        mt_backend=form.get("mt_backend") or "auto",
        notes=form.get("notes") or None,
        initial_prompt=form.get("prompt") or None,
        formality=(form.get("formality") or None) or None,
        include_source=bool(form.get("include_source", True)),
        vad=bool(form.get("vad", True)),
        normalize_audio=bool(form.get("normalize_audio", True)),
        timestamps_in_text=bool(form.get("timestamps_in_text", True)),
    )


def _split(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()]


def _media_type(suffix: str) -> str:
    return {
        ".srt": "application/x-subrip",
        ".vtt": "text/vtt",
        ".ass": "text/x-ssa",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".json": "application/json",
        ".tsv": "text/tab-separated-values",
        ".lrc": "text/plain; charset=utf-8",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".mp4": "video/mp4",
    }.get(suffix.lower(), "application/octet-stream")


def __getattr__(name: str):  # pragma: no cover - uvicorn convenience
    """Lets `uvicorn subtrans.server:app` work without building the app at
    import time (which would spin up a thread pool during test collection)."""
    if name == "app":
        return create_app()
    raise AttributeError(name)

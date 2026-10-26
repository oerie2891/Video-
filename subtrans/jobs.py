"""In-process job queue.

Transcription takes minutes, not milliseconds, so the HTTP layer must never do
the work inside the request. Jobs run on a small thread pool — CTranslate2 and
ffmpeg both release the GIL, so threads give real parallelism here without
dragging in a broker.

Deliberately not durable: restarting the server loses the queue. That is the
right trade for a single-machine tool; swap this module for Celery/RQ if you
need persistence.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pipeline import JobResult, JobSpec, run

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"


class JobCancelled(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    spec: JobSpec
    filename: str
    status: str = QUEUED
    progress: float = 0.0
    message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: JobResult | None = None
    error: str | None = None
    work_dir: str | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    def to_dict(self, *, include_result: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "created_at": self.created_at,
            "elapsed": round(self.elapsed, 2),
            "targets": self.spec.targets,
            "formats": self.spec.formats,
            "source_language": (
                self.result.source_language if self.result else self.spec.source_language
            ),
            "error": self.error,
        }
        if include_result and self.result is not None:
            payload["warnings"] = self.result.warnings
            payload["metrics"] = self.result.metrics
            payload["glossary"] = self.result.glossary
            payload["artifacts"] = [
                {
                    "index": i,
                    "language": a.language,
                    "format": a.format,
                    "filename": a.filename,
                    "bytes": a.size,
                }
                for i, a in enumerate(self.result.artifacts)
            ]
            payload["languages"] = sorted(
                set(self.result.subtitles) | set(self.result.documents)
            )
        return payload


class JobManager:
    """Thread-safe registry plus executor."""

    def __init__(self, data_dir: str | Path, workers: int = 1, max_jobs: int = 200) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="subtrans")
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    # -- queue ------------------------------------------------------------

    def allocate(self) -> tuple[str, Path]:
        """Reserve a job id and its directory before the input exists.

        Callers that receive an upload must write the file into this directory
        and only then call `submit`. Doing it the other way round races the
        worker, which may start reading before the file is in place.
        """
        job_id = uuid.uuid4().hex[:12]
        work_dir = self.data_dir / job_id
        (work_dir / "out").mkdir(parents=True, exist_ok=True)
        return job_id, work_dir

    def submit(self, spec: JobSpec, filename: str, job_id: str | None = None) -> Job:
        if job_id is None:
            job_id, work_dir = self.allocate()
        else:
            work_dir = self.data_dir / job_id
            (work_dir / "out").mkdir(parents=True, exist_ok=True)
        spec.output_dir = spec.output_dir or str(work_dir / "out")
        job = Job(id=job_id, spec=spec, filename=filename, work_dir=str(work_dir))
        with self._lock:
            self._jobs[job_id] = job
            self._evict_locked()
        self._pool.submit(self._execute, job)
        return job

    def _execute(self, job: Job) -> None:
        if job._cancel.is_set():
            job.status = CANCELLED
            job.message = "Cancelled before start"
            job.finished_at = time.time()
            return
        job.status = RUNNING
        job.started_at = time.time()
        job.message = "Starting…"

        def progress(fraction: float, message: str) -> None:
            if job._cancel.is_set():
                raise JobCancelled("cancelled by user")
            job.progress = max(job.progress, min(1.0, fraction))
            if message:
                job.message = message

        try:
            job.result = run(job.spec, progress)
            job.status = DONE
            job.progress = 1.0
            job.message = (
                f"Finished — {len(job.result.artifacts)} file(s) in "
                f"{job.result.elapsed:.1f}s"
            )
        except JobCancelled:
            job.status = CANCELLED
            job.message = "Cancelled"
        except Exception as exc:  # surfaced to the user verbatim
            job.status = ERROR
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "Failed"
        finally:
            job.finished_at = time.time()
            self._discard_upload(job)

    def _discard_upload(self, job: Job) -> None:
        """Delete the uploaded source once the job ends; outputs are kept."""
        try:
            source = Path(job.spec.input_path)
            if job.work_dir and str(source).startswith(str(Path(job.work_dir))):
                source.unlink(missing_ok=True)
        except OSError:
            pass

    # -- access -----------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            return list(reversed(list(self._jobs.values())))[:limit]

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in (DONE, ERROR, CANCELLED):
            return False
        job._cancel.set()
        job.message = "Cancelling…"
        return True

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        job._cancel.set()
        if job.work_dir:
            shutil.rmtree(job.work_dir, ignore_errors=True)
        return True

    def artifact_path(self, job_id: str, index: int) -> Path | None:
        """Resolve an artifact index to a path, refusing anything outside the
        job's own directory."""
        job = self.get(job_id)
        if job is None or job.result is None or job.work_dir is None:
            return None
        if index < 0 or index >= len(job.result.artifacts):
            return None
        candidate = Path(job.result.artifacts[index].path).resolve()
        root = Path(job.work_dir).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate if candidate.exists() else None

    def _evict_locked(self) -> None:
        while len(self._jobs) > self._max_jobs:
            old_id, old = self._jobs.popitem(last=False)
            if old.status in (RUNNING, QUEUED):
                # Never evict live work; put it back and stop pruning.
                self._jobs[old_id] = old
                self._jobs.move_to_end(old_id, last=False)
                break
            if old.work_dir:
                shutil.rmtree(old.work_dir, ignore_errors=True)

    def shutdown(self, wait: bool = False) -> None:
        for job in self.list(limit=10_000):
            if job.status in (QUEUED, RUNNING):
                job._cancel.set()
        self._pool.shutdown(wait=wait, cancel_futures=True)

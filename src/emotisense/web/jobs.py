"""Background analysis jobs with progress reporting (one at a time: the model is the bottleneck)."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    session_id: str
    status: str = "queued"          # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    detail: str = "Waiting for the model"
    error: str | None = None
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"id": self.id, "session_id": self.session_id, "status": self.status, "stage": self.stage,
                "progress": round(self.progress, 3), "detail": self.detail, "error": self.error}


class JobManager:
    def __init__(self, keep_seconds: float = 3600.0):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._keep = keep_seconds

    def submit(self, session_id: str, fn) -> Job:
        """``fn(progress)`` does the work; ``progress(stage, fraction, detail)`` updates the job."""
        job = Job(id=uuid.uuid4().hex[:12], session_id=session_id)
        with self._lock:
            self._prune()
            self._jobs[job.id] = job

        def run():
            job.status, job.stage, job.detail = "running", "decode", "Starting"
            try:
                def progress(stage: str, fraction: float, detail: str = ""):
                    job.stage, job.progress, job.detail = stage, max(job.progress, float(fraction)), detail or job.detail

                fn(progress)
                job.status, job.stage, job.progress, job.detail = "done", "done", 1.0, "Done"
            except Exception as exc:  # surfaced to the UI; never crash the worker
                job.status, job.error = "error", str(exc) or exc.__class__.__name__
                job.detail = "Something went wrong"
                traceback.print_exc()

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _prune(self) -> None:
        cutoff = time.time() - self._keep
        for jid in [j for j, job in self._jobs.items() if job.created < cutoff and job.status in ("done", "error")]:
            del self._jobs[jid]

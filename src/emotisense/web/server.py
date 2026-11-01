"""FastAPI backend for the React app: uploads, background analysis, audio streaming, notes, settings.

Everything runs locally; audio and results never leave the machine.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..data.audio import probe_duration
from .jobs import JobManager
from .registry import ModelRegistry
from .store import DEFAULT_SETTINGS, SessionStore, new_id

ALLOWED_SUFFIXES = {".wav", ".mp3", ".ogg", ".oga", ".opus", ".flac", ".m4a", ".aac", ".webm", ".mp4", ".wma"}
MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
        ".flac": "audio/flac", ".m4a": "audio/mp4", ".aac": "audio/aac", ".webm": "audio/webm", ".mp4": "audio/mp4",
        ".wma": "audio/x-ms-wma"}
MAX_UPLOAD_MB = 400
CHUNK = 1 << 20


def ranged_file(path: str, request: Request, mime: str) -> StreamingResponse:
    """Serve a file with HTTP Range support so the browser can seek instantly while playing."""
    size = os.path.getsize(path)
    start, end, status = 0, size - 1, 200
    header = request.headers.get("range")
    if header:
        m = re.match(r"bytes=(\d*)-(\d*)$", header.strip())
        if not m or (not m.group(1) and not m.group(2)):
            raise HTTPException(status_code=416, detail="Bad range")
        if m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
        else:                                   # suffix range: the last N bytes
            start = max(size - int(m.group(2)), 0)
        end = min(end, size - 1)
        if start > end or start >= size:
            raise HTTPException(status_code=416, detail="Range not satisfiable")
        status = 206

    def body():
        remaining = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            while remaining > 0:
                data = f.read(min(CHUNK, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1), "Cache-Control": "no-cache"}
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(body(), status_code=status, media_type=mime, headers=headers)


def seed_samples(store: SessionStore, samples_dir: Path) -> int:
    """Register pre-analysed sample sessions (``<name>.json`` + ``<name>.<audio>``) found in ``samples_dir``."""
    added = 0
    if not samples_dir.is_dir():
        return added
    for meta_file in sorted(samples_dir.glob("*.json")):
        audio = next((p for p in samples_dir.glob(meta_file.stem + ".*") if p.suffix.lower() in ALLOWED_SUFFIXES), None)
        sid = f"sample-{meta_file.stem}"
        if audio is None or store.exists(sid):
            continue
        payload = json.loads(meta_file.read_text(encoding="utf-8"))
        store.add(payload["result"], title=payload.get("title", meta_file.stem), tag=payload.get("tag", "Sample"),
                  filename=audio.name, audio_file=str(audio.resolve()), session_id=sid, is_sample=True)
        added += 1
    return added


def create_web_app(analyzer=None, analyzer_factory=None, data_dir: str | Path = "outputs/web",
                   static_dir: str | Path | None = None, samples_dir: str | Path | None = "data/samples",
                   dev_cors: bool = True, registry: ModelRegistry | None = None,
                   default_model: str | None = None) -> FastAPI:
    """``analyzer`` (ready) or ``analyzer_factory`` (loaded in a background thread) provides ``analyze``."""
    data_dir = Path(data_dir)
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    store = SessionStore(data_dir / "sessions.sqlite")
    registry = registry or ModelRegistry(None)       # no model folder: analyses use whatever the analyzer was built with

    def active_model_id() -> str | None:
        preferred = store.get_meta("active_model")
        return preferred if registry.get(preferred) else registry.default_id(default_model)
    jobs = JobManager()
    state = {"analyzer": analyzer, "error": None}
    ready = threading.Event()
    if analyzer is not None:
        ready.set()

    def load_model():
        try:
            state["analyzer"] = analyzer_factory()
        except Exception as exc:   # reported by /api/health instead of crashing the server
            state["error"] = str(exc)
        finally:
            ready.set()

    @asynccontextmanager
    async def lifespan(_app):
        if samples_dir:
            seed_samples(store, Path(samples_dir))
        if analyzer is None and analyzer_factory is not None:
            threading.Thread(target=load_model, daemon=True, name="model-loader").start()
        elif analyzer is None:
            state["error"] = "No model configured"
            ready.set()
        yield

    app = FastAPI(title="EmotiSense", version="0.2.0", lifespan=lifespan)
    if dev_cors:
        app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                           allow_methods=["*"], allow_headers=["*"])

    def model_status() -> str:
        if state["analyzer"] is not None:
            return "ready"
        return "error" if state["error"] else ("loading" if analyzer_factory else "absent")

    # ----------------------------------------------------------------- health
    @app.get("/api/health")
    def health():
        current = getattr(state["analyzer"], "timeline_id", None)
        return {"status": "ok", "model": model_status(), "model_error": state["error"], "version": "0.2.0",
                "timeline_model": current,
                "formats": sorted(s.lstrip(".") for s in ALLOWED_SUFFIXES)}

    # --------------------------------------------------------------- sessions
    @app.post("/api/sessions", status_code=202)
    async def create_session(file: UploadFile = File(...), title: str = Form(""), tag: str = Form(""),
                             model: str = Form("")):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(415, f"Unsupported file type {suffix or '(none)'}. Use wav, mp3, ogg, flac, m4a or webm.")
        wanted = registry.get(model or active_model_id())
        if model and wanted is None:
            raise HTTPException(422, f"Unknown model {model!r}. See /api/models for the available ones.")
        sid = new_id()
        dest = audio_dir / f"{sid}{suffix}"
        size = 0
        with open(dest, "wb") as out:
            while block := await file.read(CHUNK):
                size += len(block)
                if size > MAX_UPLOAD_MB * 2 ** 20:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
                out.write(block)
        if size == 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(422, "The file is empty")
        if probe_duration(str(dest)) is None and suffix in {".wav", ".mp3", ".ogg", ".flac", ".oga", ".opus"}:
            dest.unlink(missing_ok=True)
            raise HTTPException(422, "This file could not be read as audio")
        name = Path(file.filename or sid).stem
        shown_title = title.strip() or name

        def work(progress):
            try:
                if not ready.is_set():
                    progress("model", 0.01, "Warming up the model")
                    ready.wait()
                model = state["analyzer"]
                if model is None:
                    raise RuntimeError(state["error"] or "The model is not available")
                if wanted is not None and hasattr(model, "use_timeline") and getattr(model, "timeline_id", None) != wanted.id:
                    progress("model", 0.03, f"Switching to the {wanted.kind} model")
                    model.use_timeline(wanted.path)       # shares the loaded encoder, so this is near-instant
                result = model.analyze(str(dest), name=file.filename, progress=progress)
                if not result["chunks"]:
                    raise RuntimeError("No speech was found in this recording")
                store.add(result, title=shown_title, tag=tag, filename=file.filename or dest.name,
                          audio_file=str(dest), session_id=sid)
            except Exception:
                dest.unlink(missing_ok=True)      # failed analyses leave nothing behind
                raise

        job = jobs.submit(sid, work)
        return {"job_id": job.id, "session_id": sid}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job.to_dict()

    @app.get("/api/sessions")
    def list_sessions(tag: str | None = None, q: str | None = None):
        return store.list(tag=tag, query=q)

    @app.get("/api/sessions/{sid}")
    def get_session(sid: str):
        result = store.get(sid)
        if result is None:
            raise HTTPException(404, "Session not found")
        result["notes"] = store.notes(sid)
        return result

    @app.patch("/api/sessions/{sid}")
    async def patch_session(sid: str, request: Request):
        body = await request.json()
        if not store.update(sid, body.get("title"), body.get("tag")):
            raise HTTPException(404, "Session not found")
        return {"ok": True}

    @app.delete("/api/sessions/{sid}")
    def delete_session(sid: str):
        if not store.exists(sid):
            raise HTTPException(404, "Session not found")
        path = store.delete(sid)
        if path and Path(path).parent == audio_dir:   # never delete files that belong to bundled samples
            Path(path).unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/api/sessions/{sid}/audio")
    def session_audio(sid: str, request: Request):
        path = store.audio_path(sid)
        if path is None or not Path(path).is_file():
            raise HTTPException(404, "Audio not found")
        return ranged_file(path, request, MIME.get(Path(path).suffix.lower(), "application/octet-stream"))

    # ------------------------------------------------------------------ notes
    @app.post("/api/sessions/{sid}/notes", status_code=201)
    async def add_note(sid: str, request: Request):
        if not store.exists(sid):
            raise HTTPException(404, "Session not found")
        body = await request.json()
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(422, "Note text is empty")
        return store.add_note(sid, float(body.get("t", 0.0)), text)

    @app.delete("/api/notes/{note_id}")
    def delete_note(note_id: int):
        if not store.delete_note(note_id):
            raise HTTPException(404, "Note not found")
        return {"ok": True}

    # ----------------------------------------------------------------- models
    @app.get("/api/models")
    def list_models():
        return registry.describe(active_model_id())

    @app.put("/api/models/active")
    async def set_active_model(request: Request):
        body = await request.json()
        entry = registry.get(str(body.get("id", "")))
        if entry is None:
            raise HTTPException(422, "Unknown model")
        store.set_meta("active_model", entry.id)
        return {"active": entry.id}

    # --------------------------------------------------------------- settings
    @app.get("/api/settings")
    def get_settings():
        return {"values": store.get_settings(), "defaults": DEFAULT_SETTINGS}

    @app.put("/api/settings")
    async def put_settings(request: Request):
        body = await request.json()
        try:
            return {"values": store.set_settings(body), "defaults": DEFAULT_SETTINGS}
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, f"Invalid settings: {exc}")

    # ------------------------------------------------------------ built React UI
    if static_dir is not None and (Path(static_dir) / "index.html").exists():
        root = Path(static_dir).resolve()
        if (root / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "Not found")
            candidate = (root / path).resolve()
            if path and root in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(root / "index.html")

    app.state.store = store
    app.state.jobs = jobs
    app.state.audio_dir = audio_dir
    return app


def build_analyzer_factory(timeline: str, ser: str | None = None):
    def factory():
        from ..app.analyzer import CallAnalyzer
        return CallAnalyzer(timeline, ser)
    return factory

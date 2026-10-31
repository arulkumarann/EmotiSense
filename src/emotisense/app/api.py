"""FastAPI service: upload a call, get the emotion timeline, alerts and summary back."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .store import CallStore

ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg"}
MAX_UPLOAD_MB = 200


def analyzer_from_env():
    """Build the analyzer from EMOTISENSE_TIMELINE_CKPT (+ optional EMOTISENSE_SER_CKPT / EMOTISENSE_APP_CONFIG)."""
    from .analyzer import CallAnalyzer

    timeline = os.environ.get("EMOTISENSE_TIMELINE_CKPT")
    if not timeline:
        raise RuntimeError("Set EMOTISENSE_TIMELINE_CKPT to a timeline_deploy.pt checkpoint.")
    app_cfg = {}
    if os.environ.get("EMOTISENSE_APP_CONFIG"):
        from ..config import load_config
        app_cfg = load_config(os.environ["EMOTISENSE_APP_CONFIG"]).to_dict()
    return CallAnalyzer(timeline, os.environ.get("EMOTISENSE_SER_CKPT"),
                        vad=app_cfg.get("vad"), escalation=app_cfg.get("escalation"))


def create_app(analyzer=None, db_path: str | Path | None = None) -> FastAPI:
    """``analyzer`` is any object with ``analyze(path, name=...) -> dict`` (lazy-built from env if omitted)."""
    app = FastAPI(title="EmotiSense", version="0.1.0")
    store = CallStore(db_path or os.environ.get("EMOTISENSE_DB", "outputs/calls.sqlite"))
    state = {"analyzer": analyzer}

    def get_analyzer():
        if state["analyzer"] is None:
            try:
                state["analyzer"] = analyzer_from_env()
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Model not available: {exc}")
        return state["analyzer"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": state["analyzer"] is not None}

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...), agent: str = Form("unassigned")):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"Unsupported audio type {suffix!r}; use {sorted(ALLOWED_SUFFIXES)}")
        data = await file.read()
        if len(data) > MAX_UPLOAD_MB * 2 ** 20:
            raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB")
        model = get_analyzer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            try:
                result = model.analyze(path, name=file.filename)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Could not analyse audio: {exc}")
        result["id"] = store.add(result, agent)
        result["agent"] = agent
        return result

    @app.get("/calls")
    def calls(agent: str | None = None, limit: int = 100):
        return store.list_calls(agent, min(limit, 1000))

    @app.get("/calls/{call_id}")
    def call(call_id: int):
        result = store.get(call_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Call not found")
        return result

    @app.get("/agents")
    def agents():
        return store.agent_summary()

    return app

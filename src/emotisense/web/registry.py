"""Finds the trained models on this computer and describes them for the UI."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch

from . import catalog

# Timeline checkpoints are a few MB; anything much larger is a full encoder checkpoint and is never loaded here.
MAX_TIMELINE_BYTES = 100 * 1024 * 1024
REQUIRED_KEYS = {"model", "model_cfg", "emotions", "embedder", "in_dim"}


@dataclass
class TimelineEntry:
    id: str
    path: Path
    kind: str
    hidden: int
    layers: int
    params: int
    size_mb: float
    encoder: str
    in_dim: int
    val_ccc: float | None


class ModelRegistry:
    """Scans ``models_dir`` (plus any explicit paths) for timeline checkpoints and ``ser_dirs`` for reference weights."""

    def __init__(self, models_dir: str | Path | None = "models", extra_paths: tuple = (), ser_dirs: tuple = ()):
        self.models_dir = Path(models_dir) if models_dir else None
        self.extra_paths = [Path(p) for p in extra_paths if p]
        self.ser_dirs = [Path(d) for d in ser_dirs if d]
        self._cache: dict[tuple[str, float], TimelineEntry | None] = {}

    # ------------------------------------------------------------------ scan
    def _read(self, path: Path) -> TimelineEntry | None:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime)
        if key in self._cache:
            return self._cache[key]
        entry = None
        if stat.st_size <= MAX_TIMELINE_BYTES:
            try:
                ckpt = torch.load(path, map_location="cpu")
                if isinstance(ckpt, dict) and REQUIRED_KEYS <= set(ckpt):
                    mc = ckpt["model_cfg"]
                    emb = ckpt["embedder"]
                    entry = TimelineEntry(
                        id=path.stem, path=path, kind=mc["kind"], hidden=mc["hidden"], layers=mc["layers"],
                        params=sum(int(v.numel()) for v in ckpt["model"].values()), size_mb=stat.st_size / 1e6,
                        encoder=emb.get("encoder") or emb.get("checkpoint", ""), in_dim=ckpt["in_dim"],
                        val_ccc=ckpt.get("val_ccc"),
                    )
            except Exception:       # a corrupt or foreign file is simply not a model
                entry = None
        self._cache[key] = entry
        return entry

    def timeline_models(self) -> list[TimelineEntry]:
        paths: list[Path] = []
        if self.models_dir and self.models_dir.is_dir():
            paths += sorted(self.models_dir.glob("*.pt"))
        paths += [p for p in self.extra_paths if p.is_file()]
        seen, out = set(), []
        for p in paths:
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            entry = self._read(p)
            if entry:
                out.append(entry)
        order = {"bilstm": 0, "transformer": 1, "mlp": 2}
        return sorted(out, key=lambda e: (order.get(e.kind, 9), e.id))

    def get(self, model_id: str | None) -> TimelineEntry | None:
        if not model_id:
            return None
        return next((e for e in self.timeline_models() if e.id == model_id), None)

    def default_id(self, preferred: str | None = None) -> str | None:
        models = self.timeline_models()
        if preferred and any(e.id == preferred for e in models):
            return preferred
        return models[0].id if models else None

    def reference_files(self) -> list[dict]:
        found = []
        for d in self.ser_dirs:
            if d.is_dir():
                for p in sorted(d.glob("*.pt")):
                    m = re.search(r"fold(\d+)", p.name)
                    found.append({"file": p.name, "fold": int(m.group(1)) + 1 if m else None,
                                  "size_mb": round(p.stat().st_size / 1e6)})
        return found

    # -------------------------------------------------------------- describe
    def describe(self, active_id: str | None) -> dict:
        timeline = []
        for e in self.timeline_models():
            info = catalog.TIMELINE.get(e.kind, {})
            timeline.append({
                "id": e.id, "name": info.get("name", e.kind.title()), "role": "timeline", "kind": e.kind,
                "tagline": info.get("tagline", "Custom timeline model"),
                "description": info.get("description", "A timeline model found in your models folder."),
                "strengths": info.get("strengths", []), "limits": info.get("limits", []),
                "stats": info.get("stats"), "selectable": True, "available": True, "active": e.id == active_id,
                "file": e.path.name, "size_mb": round(e.size_mb, 1), "params": e.params, "params_m": round(e.params / 1e6, 2),
                "details": [
                    ["Type", info.get("name", e.kind)], ["Parameters", f"{e.params / 1e6:.2f} million"],
                    ["File size", f"{e.size_mb:.1f} MB"], ["Hidden size / layers", f"{e.hidden} / {e.layers}"],
                    ["Reads", f"{e.in_dim}-number descriptions from {e.encoder}"], ["File", e.path.name],
                ],
            })
        files = self.reference_files()
        classifier = {**catalog.CLASSIFIER, "selectable": False, "active": False, "available": bool(files),
                      "files": files, "stats": catalog.CLASSIFIER["stats"]}
        encoder = {**catalog.ENCODER, "selectable": False, "active": False, "available": True, "stats": None}
        return {
            "active": active_id,
            "timeline": timeline,
            "encoder": encoder,
            "classifier": classifier,
            "glossary": catalog.GLOSSARY,
            "supported": catalog.SUPPORTED,
            "data_note": catalog.DATA_NOTE,
        }

"""SQLite persistence for sessions, notes and settings."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..timeline.escalation import EscalationConfig, analyze_timeline

DEFAULT_SETTINGS = {
    "alert_threshold": 0.6, "rise_delta": 0.3, "rise_window_s": 20.0, "rise_floor": 0.35,
    "smooth_window": 3, "merge_gap_s": 5.0,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, tag TEXT NOT NULL DEFAULT '', created TEXT NOT NULL,
    duration REAL NOT NULL, filename TEXT NOT NULL, audio_file TEXT NOT NULL, is_sample INTEGER NOT NULL DEFAULT 0,
    meta TEXT NOT NULL, result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, t REAL NOT NULL, text TEXT NOT NULL,
    created TEXT NOT NULL, FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def _timeline(meta: dict, cfg: EscalationConfig) -> dict:
    times, ends = np.array(meta["times"]), np.array(meta["ends"])
    scores = {e: np.array(v) for e, v in meta["scores"].items()}
    return analyze_timeline(times, scores, cfg, ends)


class SessionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ---------------------------------------------------------------- settings
    def get_settings(self) -> dict:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        merged = dict(DEFAULT_SETTINGS)
        merged.update({r["key"]: json.loads(r["value"]) for r in rows if r["key"] in DEFAULT_SETTINGS})
        return merged

    def set_settings(self, values: dict) -> dict:
        clean = {k: type(DEFAULT_SETTINGS[k])(v) for k, v in values.items() if k in DEFAULT_SETTINGS}
        with self._conn() as conn:
            for k, v in clean.items():
                conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                             "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (k, json.dumps(v)))
        return self.get_settings()

    def get_meta(self, key: str, default=None):
        """Free-form preferences such as the selected model (kept apart from the alert settings)."""
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (f"meta:{key}",)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_meta(self, key: str, value) -> None:
        with self._conn() as conn:
            conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                         "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (f"meta:{key}", json.dumps(value)))

    def escalation_config(self) -> EscalationConfig:
        return EscalationConfig.from_dict(self.get_settings())

    # ---------------------------------------------------------------- sessions
    def add(self, result: dict, *, title: str, tag: str, filename: str, audio_file: str,
            session_id: str | None = None, is_sample: bool = False) -> str:
        sid = session_id or new_id()
        meta = {
            "times": [round(c["start"], 2) for c in result["chunks"]],
            "ends": [round(c["end"], 2) for c in result["chunks"]],
            "scores": {e: [round(v, 3) for v in vals] for e, vals in result["scores"].items()},
        }
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (id, title, tag, created, duration, filename, audio_file, is_sample, meta, result)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (sid, title.strip()[:120] or "Untitled", tag.strip()[:40],
                 datetime.now(timezone.utc).isoformat(timespec="seconds"), result["duration"],
                 filename, audio_file, int(is_sample), json.dumps(meta), json.dumps(result)))
        return sid

    @staticmethod
    def _card_summary(meta: dict, cfg: EscalationConfig) -> dict:
        out = _timeline(meta, cfg)
        n = max(len(meta["times"]), 1)
        out["emotion_means"] = {e: round(float(sum(v) / n), 3) for e, v in meta["scores"].items()}
        step = max(1, len(out["negative"]) // 48)
        out["sparkline"] = [round(v, 3) for v in out["negative"][::step]]
        out.pop("negative", None)
        out.pop("alerts", None)
        return out

    def list(self, tag: str | None = None, query: str | None = None) -> list[dict]:
        cfg = self.escalation_config()
        sql = "SELECT id, title, tag, created, duration, filename, is_sample, meta FROM sessions"
        where, args = [], []
        if tag:
            where.append("tag = ?")
            args.append(tag)
        if query:
            where.append("(title LIKE ? OR tag LIKE ?)")
            args += [f"%{query}%", f"%{query}%"]
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created DESC, rowid DESC"
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
            note_counts = {r["session_id"]: r["n"] for r in conn.execute(
                "SELECT session_id, COUNT(*) AS n FROM notes GROUP BY session_id")}
        out = []
        for r in rows:
            item = {k: r[k] for k in ("id", "title", "tag", "created", "duration", "filename")}
            item["is_sample"] = bool(r["is_sample"])
            item["notes"] = note_counts.get(r["id"], 0)
            item["summary"] = self._card_summary(json.loads(r["meta"]), cfg)
            out.append(item)
        return out

    def get(self, sid: str) -> dict | None:
        with self._conn() as conn:
            r = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        if r is None:
            return None
        result = json.loads(r["result"])
        cfg = self.escalation_config()
        meta = json.loads(r["meta"])
        full = _timeline(meta, cfg)   # recomputed with the saved settings so the viewer matches the Settings page
        result["alerts"] = full.pop("alerts")
        result["summary"] = full
        result["escalation_config"] = cfg.to_dict()
        result.update({"id": r["id"], "title": r["title"], "tag": r["tag"], "created": r["created"],
                       "filename": r["filename"], "is_sample": bool(r["is_sample"])})
        return result

    def audio_path(self, sid: str) -> str | None:
        with self._conn() as conn:
            r = conn.execute("SELECT audio_file FROM sessions WHERE id = ?", (sid,)).fetchone()
        return r["audio_file"] if r else None

    def exists(self, sid: str) -> bool:
        return self.audio_path(sid) is not None

    def update(self, sid: str, title: str | None = None, tag: str | None = None) -> bool:
        sets, args = [], []
        if title is not None:
            sets.append("title = ?")
            args.append(title.strip()[:120] or "Untitled")
        if tag is not None:
            sets.append("tag = ?")
            args.append(tag.strip()[:40])
        if not sets:
            return self.exists(sid)
        with self._conn() as conn:
            return conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", (*args, sid)).rowcount > 0

    def delete(self, sid: str) -> str | None:
        """Delete a session and return its audio path so the caller can remove the file."""
        path = self.audio_path(sid)
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        return path

    # ------------------------------------------------------------------- notes
    def notes(self, sid: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, t, text, created FROM notes WHERE session_id = ? ORDER BY t", (sid,)).fetchall()
        return [dict(r) for r in rows]

    def add_note(self, sid: str, t: float, text: str) -> dict:
        with self._conn() as conn:
            cur = conn.execute("INSERT INTO notes (session_id, t, text, created) VALUES (?,?,?,?)",
                               (sid, max(0.0, float(t)), text.strip()[:500],
                                datetime.now(timezone.utc).isoformat(timespec="seconds")))
            row = conn.execute("SELECT id, t, text, created FROM notes WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)

    def delete_note(self, note_id: int) -> bool:
        with self._conn() as conn:
            return conn.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount > 0

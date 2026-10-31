"""SQLite store of analysed calls for the agent / team analytics views."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Heuristic churn-risk blend: how negative calls are on average and how often they escalate.
RISK_WEIGHT_MEAN = 0.5
RISK_WEIGHT_ESCALATION = 0.5
HIGH_RISK = 0.5

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    agent TEXT NOT NULL,
    created TEXT NOT NULL,
    duration REAL,
    mean_negative REAL,
    peak REAL,
    peak_time REAL,
    trend TEXT,
    n_alerts INTEGER,
    result TEXT NOT NULL
)
"""


class CallStore:
    def __init__(self, path: str | Path = "outputs/calls.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, result: dict, agent: str = "unassigned") -> int:
        s = result["summary"]
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO calls (name, agent, created, duration, mean_negative, peak, peak_time, trend, n_alerts, result)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (result["name"], agent, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 result["duration"], s["mean"], s["peak"], s["peak_time"], s["trend"], s["n_alerts"],
                 json.dumps(result)),
            )
            return int(cur.lastrowid)

    def list_calls(self, agent: str | None = None, limit: int = 200) -> list[dict]:
        query = "SELECT id, name, agent, created, duration, mean_negative, peak, peak_time, trend, n_alerts FROM calls"
        args: tuple = ()
        if agent:
            query += " WHERE agent = ?"
            args = (agent,)
        query += " ORDER BY id DESC LIMIT ?"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, args + (limit,))]

    def get(self, call_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT result FROM calls WHERE id = ?", (call_id,)).fetchone()
        return json.loads(row["result"]) if row else None

    def agent_summary(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT agent, COUNT(*) AS n_calls, AVG(mean_negative) AS avg_negative, AVG(peak) AS avg_peak,"
                " AVG(CASE WHEN n_alerts > 0 THEN 1.0 ELSE 0.0 END) AS escalation_rate,"
                " SUM(CASE WHEN n_alerts > 0 THEN 1 ELSE 0 END) AS n_escalated"
                " FROM calls GROUP BY agent ORDER BY avg_negative DESC"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["risk_score"] = RISK_WEIGHT_MEAN * d["avg_negative"] + RISK_WEIGHT_ESCALATION * d["escalation_rate"]
            d["churn_risk"] = "high" if d["risk_score"] >= HIGH_RISK else "normal"
            out.append(d)
        return out

"""Plotly figures for the dashboard (pure functions, no Streamlit dependency)."""
from __future__ import annotations

import numpy as np

# Colour-blind-safe palette, one stable colour per emotion.
COLORS = {
    "anger": "#c0392b", "frustration": "#e67e22", "happiness": "#27ae60",
    "sadness": "#2980b9", "neutral": "#7f8c8d", "negative": "#111111",
}


def format_time(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def timeline_figure(result: dict):
    import plotly.graph_objects as go

    fig = go.Figure()
    chunks = result.get("chunks", [])
    mids = [(c["start"] + c["end"]) / 2 for c in chunks]
    names = result.get("display_names", {})
    for emotion, values in result["scores"].items():
        fig.add_trace(go.Scatter(x=mids, y=values, mode="lines+markers", name=names.get(emotion, emotion),
                                 line=dict(color=COLORS.get(emotion), width=2), marker=dict(size=5),
                                 hovertemplate="%{y:.2f}<extra>" + names.get(emotion, emotion) + "</extra>"))
    negative = result["summary"].get("negative")
    if negative:
        fig.add_trace(go.Scatter(x=mids, y=negative, mode="lines", name="negative (smoothed)",
                                 line=dict(color=COLORS["negative"], width=4, dash="dot")))
    threshold = result.get("escalation_config", {}).get("alert_threshold")
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color="#c0392b", opacity=0.5,
                      annotation_text="alert threshold", annotation_position="top left")
    for alert in result.get("alerts", []):
        fig.add_vrect(x0=alert["start"], x1=max(alert["end"], alert["start"] + 1.0), fillcolor="#c0392b",
                      opacity=0.12, line_width=0)
    fig.update_layout(
        xaxis_title="Call time (s)", yaxis_title="Intensity", yaxis=dict(range=[0, 1.02]),
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2), hovermode="x unified",
        height=420,
    )
    return fig


def agent_figure(summary: list[dict]):
    import plotly.graph_objects as go

    agents = [s["agent"] for s in summary]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agents, y=[s["avg_negative"] for s in summary], name="avg negative", marker_color="#e67e22"))
    fig.add_trace(go.Bar(x=agents, y=[s["escalation_rate"] for s in summary], name="escalation rate", marker_color="#c0392b"))
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.02]), margin=dict(l=10, r=10, t=30, b=10), height=360)
    return fig


def moments_to_review(result: dict, top: int = 3) -> list[dict]:
    """Highest-negativity chunks, for the supervisor's "listen to these moments" list."""
    negative = np.asarray(result["summary"].get("negative", []))
    chunks = result.get("chunks", [])
    if len(negative) == 0:
        return []
    order = np.argsort(-negative)[:top]
    return [{"time": format_time(chunks[i]["start"]), "start": chunks[i]["start"], "score": float(negative[i])}
            for i in sorted(order)]

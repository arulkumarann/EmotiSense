"""Streamlit supervisor dashboard.  Run with: ``emotisense dashboard``."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from emotisense.app.plots import agent_figure, format_time, moments_to_review, timeline_figure
from emotisense.app.store import CallStore


@st.cache_resource(show_spinner="Loading models...")
def load_analyzer(timeline_ckpt: str, ser_ckpt: str):
    from emotisense.app.analyzer import CallAnalyzer

    return CallAnalyzer(timeline_ckpt, ser_ckpt or None)


def show_result(result: dict) -> None:
    s = result["summary"]
    cols = st.columns(4)
    cols[0].metric("Overall negativity", f"{100 * s['mean']:.0f}%")
    cols[1].metric("Peak score", f"{100 * s['peak']:.0f}%")
    cols[2].metric("Peak time", format_time(s["peak_time"]))
    cols[3].metric("Trend", s["trend"].title())
    if result.get("warning"):
        st.warning(result["warning"])
        return
    st.plotly_chart(timeline_figure(result), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Live escalation alerts")
        if result["alerts"]:
            st.dataframe(pd.DataFrame([{
                "from": format_time(a["start"]), "to": format_time(a["end"]),
                "peak": f"{100 * a['peak']:.0f}%", "type": a["kind"]} for a in result["alerts"]]),
                hide_index=True, use_container_width=True)
        else:
            st.success("No escalation detected.")
    with right:
        st.subheader("Moments to review")
        for m in moments_to_review(result):
            st.write(f"**{m['time']}**: negativity {100 * m['score']:.0f}%")
    with st.expander("Raw result (JSON)"):
        st.download_button("Download JSON", json.dumps(result, indent=2), f"{result['name']}.json", "application/json")
        st.json(result, expanded=False)


def analyze_page(store: CallStore) -> None:
    st.header("Analyze a call")
    timeline = st.sidebar.text_input("Timeline checkpoint", os.environ.get("EMOTISENSE_TIMELINE_CKPT", ""))
    ser = st.sidebar.text_input("SER checkpoint (optional)", os.environ.get("EMOTISENSE_SER_CKPT", ""))
    tab_audio, tab_json = st.tabs(["Upload audio", "Open saved result"])
    with tab_audio:
        agent = st.text_input("Agent", "unassigned")
        audio = st.file_uploader("Call recording", type=["wav", "flac", "ogg"])
        if audio and st.button("Analyze", type="primary"):
            if not timeline or not Path(timeline).exists():
                st.error("Set a valid timeline checkpoint in the sidebar first.")
            else:
                with st.spinner("Analyzing call..."):
                    analyzer = load_analyzer(timeline, ser)
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / audio.name
                        path.write_bytes(audio.getvalue())
                        result = analyzer.analyze(path, name=audio.name)
                    result["id"] = store.add(result, agent)
                    st.session_state["result"] = result
    with tab_json:
        saved = st.file_uploader("Result JSON from `emotisense analyze`", type=["json"])
        if saved:
            st.session_state["result"] = json.load(saved)
    if "result" in st.session_state:
        show_result(st.session_state["result"])


def agents_page(store: CallStore) -> None:
    st.header("Agent and team analytics")
    summary = store.agent_summary()
    if not summary:
        st.info("No calls analysed yet. Analyze a call first.")
        return
    st.plotly_chart(agent_figure(summary), use_container_width=True)
    st.dataframe(pd.DataFrame(summary).round(3), hide_index=True, use_container_width=True)
    st.caption("Risk score = 0.5 x average negativity + 0.5 x escalation rate; >= 0.5 is flagged high (heuristic).")
    st.subheader("Recent calls")
    st.dataframe(pd.DataFrame(store.list_calls(limit=50)).round(3), hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="EmotiSense", page_icon=":headphones:", layout="wide")
    st.title("EmotiSense")
    st.caption("Moment-by-moment customer emotion, so supervisors can act before frustration turns into churn.")
    store = CallStore(os.environ.get("EMOTISENSE_DB", "outputs/calls.sqlite"))
    page = st.sidebar.radio("View", ["Analyze a call", "Agents and teams"])
    (analyze_page if page == "Analyze a call" else agents_page)(store)


if __name__ == "__main__":
    main()

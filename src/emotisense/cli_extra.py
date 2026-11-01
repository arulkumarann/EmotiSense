"""Timeline, analysis, ensembling and serving commands (registered onto the main CLI)."""
from __future__ import annotations

from pathlib import Path


def cmd_timeline(args) -> int:
    from .config import load_config
    from .timeline.train import run_timeline_cv

    cfg = load_config(args.config, args.overrides)
    run_timeline_cv(cfg, run_dir=args.run_dir, folds=args.folds)
    return 0


def cmd_ensemble(args) -> int:
    from .evaluation.ensemble import ensemble_runs

    ensemble_runs(args.runs, args.out)
    return 0


def cmd_analyze(args) -> int:
    from .app.analyzer import CallAnalyzer
    from .utils import save_json

    app_cfg = {}
    if args.config:
        from .config import load_config
        app_cfg = load_config(args.config).to_dict()
    analyzer = CallAnalyzer(args.timeline, args.ser, vad=app_cfg.get("vad"), escalation=app_cfg.get("escalation"))
    result = analyzer.analyze(args.audio)
    s = result["summary"]
    print(f"{result['name']}: {result['duration']:.1f}s | negativity mean {100 * s['mean']:.0f}% "
          f"peak {100 * s['peak']:.0f}% at {s['peak_time']:.1f}s | trend {s['trend']} | alerts {s['n_alerts']}")
    for a in result["alerts"]:
        print(f"  ALERT {a['start']:.1f}-{a['end']:.1f}s peak {100 * a['peak']:.0f}% ({a['kind']})")
    if args.out:
        save_json(result, args.out)
        print(f"Saved {args.out}")
    return 0


def cmd_web(args) -> int:
    import threading
    import webbrowser

    import uvicorn

    from .web.server import build_analyzer_factory, create_web_app

    static = Path(args.static)
    app = create_web_app(analyzer_factory=build_analyzer_factory(args.timeline, args.ser), data_dir=args.data_dir,
                         static_dir=static if static.is_dir() else None, samples_dir=args.samples)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    if not static.is_dir():
        print(f"[note] {static} not found: the API runs at {url}/docs. Build the UI with `npm run build` in web/, "
              "or use `npm run dev` for development.")
    if args.open and static.is_dir():
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    import logging
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)      # Windows: harmless "connection reset" noise when seeking audio
    print(f"EmotiSense is running at {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def register(sub) -> None:
    p = sub.add_parser("timeline", help="train/evaluate the conversation timeline model (BiLSTM / Transformer)")
    p.add_argument("--config", required=True)
    p.add_argument("--folds", type=int, nargs="+")
    p.add_argument("--run-dir")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, repeatable (e.g. --set model.kind=transformer)")
    p.set_defaults(func=cmd_timeline)

    p = sub.add_parser("ensemble", help="average the predictions of several finished SER runs")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_ensemble)

    p = sub.add_parser("analyze", help="analyze one recording from the command line")
    p.add_argument("audio")
    p.add_argument("--timeline", required=True, help="timeline_deploy.pt")
    p.add_argument("--ser", help="SER best.pt (only for fine-tuned embeddings)")
    p.add_argument("--config", help="analyzer settings, e.g. configs/app/default.yaml (VAD, escalation)")
    p.add_argument("--out", help="write the full result as JSON")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("web", help="start the EmotiSense web app (React UI + local model)")
    p.add_argument("--timeline", default="models/timeline_bilstm_deploy.pt")
    p.add_argument("--ser")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-dir", default="outputs/web")
    p.add_argument("--static", default="web/dist", help="built React app")
    p.add_argument("--samples", default="data/samples")
    p.add_argument("--open", action="store_true", help="open the browser when ready")
    p.set_defaults(func=cmd_web)

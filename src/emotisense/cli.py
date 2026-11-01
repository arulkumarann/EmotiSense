"""Command-line entry point: ``emotisense <command>`` (or ``python -m emotisense``)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_MANIFESTS = {
    "iemocap": ("data/raw/IEMOCAP_full_release", "data/processed/iemocap/manifest.csv"),
    "ravdess": ("data/raw/RAVDESS", "data/processed/ravdess/manifest.csv"),
    "synthetic": ("data/raw/synthetic", "data/processed/synthetic/manifest.csv"),
}


# ----------------------------------------------------------------- commands
def cmd_doctor(args) -> int:
    import platform

    import torch

    ok = True
    print(f"Python        {platform.python_version()} ({platform.system()})")
    print(f"PyTorch       {torch.__version__}")
    try:
        import transformers
        print(f"Transformers  {transformers.__version__}")
    except Exception as exc:
        print(f"Transformers  MISSING ({exc})")
        ok = False
    try:
        import soundfile
        print(f"soundfile     {soundfile.__version__}")
    except Exception as exc:
        print(f"soundfile     MISSING ({exc})")
        ok = False
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}         {p.name} | {p.total_memory / 2**30:.1f} GiB")
        print(f"bf16 support  {torch.cuda.is_bf16_supported()}")
    else:
        print("GPU           none detected (training will run on CPU: smoke tests only)")
    free = shutil.disk_usage(".").free / 2**30
    print(f"Free disk     {free:.1f} GiB" + ("  (low: WavLM-Large checkpoints need ~10 GiB)" if free < 15 else ""))
    for name, (root, manifest) in DEFAULT_MANIFESTS.items():
        status = ("ready" if Path(manifest).exists() else
                  ("raw data found, run `prepare`" if Path(root).exists()
                   else f"not set up (run `emotisense download {name}`)" if name != "synthetic" else "not set up"))
        print(f"Dataset {name:<10} {status}")
    return 0 if ok else 1


def prepare_dataset(dataset: str, root: str, out: str):
    """Build the manifest for ``dataset`` and record where its audio lives (meta.json)."""
    from .utils import save_json

    if dataset == "iemocap":
        from .data.iemocap import build_iemocap_manifest
        df = build_iemocap_manifest(root, out)
    elif dataset == "ravdess":
        from .data.ravdess import build_ravdess_manifest
        df = build_ravdess_manifest(root, out)
    else:
        from .data.synthetic import build_synthetic_corpus
        df = build_synthetic_corpus(root, out)
    source = df.attrs.get("label_source", "official")
    save_json({"dataset": dataset, "audio_root": str(Path(root).resolve()), "label_source": source},
              Path(out).with_name("meta.json"))
    print(f"Wrote {out}: {len(df)} utterances")
    print(df["label"].value_counts().to_string())
    if dataset == "iemocap":
        from .data.labels import scheme_targets
        keep, _, _ = scheme_targets(df, "iemocap4")
        print(f"iemocap4 benchmark subset: {int(keep.sum())} utterances (paper: 5,531)")
        if source == "csv":
            print("NOTE: these labels come from a hard-label CSV (no annotator votes, no valence/arousal). The 4-class "
                  "subset differs from the paper's 5,531-utterance protocol, so results are NOT directly comparable to "
                  "its 78.47 / 79.14; timeline targets become 0/1 per emotion. The official release (with "
                  "EmoEvaluation files) is picked up automatically if you point --root at it.")
    return df


def cmd_download(args) -> int:
    from .data.download import download_ravdess

    if args.dataset == "ravdess":
        root = download_ravdess(args.raw_dir)
        prepare_dataset("ravdess", str(root), DEFAULT_MANIFESTS["ravdess"][1])
    else:
        from .data.download import download_iemocap
        root = download_iemocap(args.raw_dir)
        prepare_dataset("iemocap", str(root), DEFAULT_MANIFESTS["iemocap"][1])
    print("Dataset ready. Next: `make train-ravdess`" if args.dataset == "ravdess"
          else "Dataset ready. Next: `make dry-run`, then `make train-paper`.")
    return 0


def cmd_prepare(args) -> int:
    root, out = DEFAULT_MANIFESTS[args.dataset]
    prepare_dataset(args.dataset, args.root or root, args.out or out)
    return 0


def cmd_train(args) -> int:
    from .config import load_config
    from .training import run_cv

    cfg = load_config(args.config, args.overrides)
    run_cv(cfg, run_dir=args.run_dir, folds=args.folds, resume=args.resume)
    return 0


def cmd_smoke(args) -> int:
    """Generate the synthetic corpus and run a 1-fold tiny-model training end to end."""
    from .config import load_config
    from .data.synthetic import build_synthetic_corpus
    from .training import run_cv

    root, out = DEFAULT_MANIFESTS["synthetic"]
    build_synthetic_corpus(root, out)
    cfg = load_config("configs/ser/smoke.yaml", args.overrides)
    summary = run_cv(cfg, run_dir=args.run_dir or "runs/smoke/latest", folds=[0], resume=False)
    print("Smoke test passed." if summary else "Smoke test produced no results.")
    return 0 if summary else 1


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emotisense", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check the environment, GPU and data").set_defaults(func=cmd_doctor)

    p = sub.add_parser("download", help="download a public dataset and build its manifest")
    p.add_argument("dataset", choices=["ravdess", "iemocap"])
    p.add_argument("--raw-dir", default="data/raw")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("prepare", help="build the manifest CSV for a dataset")
    p.add_argument("dataset", choices=["iemocap", "ravdess", "synthetic"])
    p.add_argument("--root", help="raw dataset directory")
    p.add_argument("--out", help="manifest output path")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="train the SER model with cross-validation")
    p.add_argument("--config", required=True)
    p.add_argument("--folds", type=int, nargs="+", help="subset of folds (default: all)")
    p.add_argument("--run-dir", help="reuse/continue this run directory")
    p.add_argument("--resume", action="store_true", help="resume unfinished folds from last.pt")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, repeatable (e.g. --set train.epochs=10)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("smoke", help="offline end-to-end check on synthetic data")
    p.add_argument("--run-dir")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=cmd_smoke)

    _register_extra_commands(sub)
    return parser


def _register_extra_commands(sub) -> None:
    """Later pipeline stages register themselves here when their modules are present."""
    try:
        from . import cli_extra
    except ImportError:
        return
    cli_extra.register(sub)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:     # missing data / manifests: print the fix, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

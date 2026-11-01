#!/usr/bin/env bash
# One-shot run for a rented GPU, in priority order so a budget cut-off still leaves the important results.
#
#   bash scripts/run_all.sh                   # 12-epoch cap, patience 4 (about 2.5 h on one A100 80GB)
#   bash scripts/run_all.sh 15 5              # longer runs
#   SKIP_BASELINE=1 bash scripts/run_all.sh   # skip the paper-recipe baseline (about 1.5 h on an A100)
#
# Re-running the same command is safe: finished folds are skipped, so an interrupted job resumes.
# Run it inside tmux / nohup so a dropped SSH connection does not stop it.
#
# Optional environment variables:
#   PY=python3            interpreter to use          OUT=runs/final       where results go
#   RESULTS=results.tgz   bundle to download          SER_EXTRA="--set k=v ..."  extra overrides for SER runs
#   TIMELINE_EXTRA="..."  extra overrides for timeline runs   SKIP_TESTS=1  skip the pytest step
set -euo pipefail

EPOCHS="${1:-12}"
PATIENCE="${2:-4}"
PY="${PY:-python}"
OUT="${OUT:-runs/final}"
RESULTS="${RESULTS:-results.tgz}"
read -r -a SER_EXTRA_ARR <<< "${SER_EXTRA:-}"
read -r -a TIMELINE_EXTRA_ARR <<< "${TIMELINE_EXTRA:-}"
export TOKENIZERS_PARALLELISM=false
START=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - START) / 60 )) min] $* ==="; }

step "environment"
"$PY" -m emotisense doctor

step "tests and offline smoke run"
if [ "${SKIP_TESTS:-0}" != "1" ]; then "$PY" -m pytest -q; fi
"$PY" -m emotisense smoke

step "data (public Kaggle IEMOCAP, 1.4 GB, no login needed)"
"$PY" -m emotisense download iemocap

step "pre-flight on real data (fails fast if anything is wrong)"
"$PY" -m emotisense train --config configs/ser/iemocap_wavlm_base.yaml --folds 0 --run-dir "${OUT}/preflight" \
    --set experiment=preflight --set data.limit_per_split=64 --set train.epochs=1 --set train.save_last=false \
    ${SER_EXTRA_ARR[@]+"${SER_EXTRA_ARR[@]}"}

COMMON=(--set "train.epochs=${EPOCHS}" --set "train.patience=${PATIENCE}" --set train.save_last=false)

step "1/3 main result: paper recipe + improvements (5-fold CV)"
"$PY" -m emotisense train --config configs/ser/iemocap_wavlm_large_plus.yaml --run-dir "${OUT}/plus" \
    "${COMMON[@]}" ${SER_EXTRA_ARR[@]+"${SER_EXTRA_ARR[@]}"}

step "2/3 timeline models: no-context MLP vs BiLSTM vs Transformer (frozen WavLM-Large embeddings)"
for kind in bilstm mlp transformer; do
    "$PY" -m emotisense timeline --config "configs/timeline/${kind}.yaml" --run-dir "${OUT}/timeline_${kind}" \
        ${TIMELINE_EXTRA_ARR[@]+"${TIMELINE_EXTRA_ARR[@]}"}
done

if [ "${SKIP_BASELINE:-0}" != "1" ]; then
    step "3/3 baseline: base-paper recipe (5-fold CV)"
    "$PY" -m emotisense train --config configs/ser/iemocap_wavlm_large.yaml --run-dir "${OUT}/paper" \
        "${COMMON[@]}" ${SER_EXTRA_ARR[@]+"${SER_EXTRA_ARR[@]}"}
fi

step "results"
for name in plus paper; do
    if [ -f "${OUT}/${name}/summary.md" ]; then echo "--- ${OUT}/${name}/summary.md"; cat "${OUT}/${name}/summary.md"; fi
done
for kind in bilstm mlp transformer; do
    if [ -f "${OUT}/timeline_${kind}/summary.json" ]; then
        "$PY" - "${OUT}/timeline_${kind}/summary.json" "$kind" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"timeline {sys.argv[2]:<12} CCC {s['ccc']['mean']:.4f} +/- {s['ccc']['std']:.4f} | RMSE {s['rmse']['mean']:.4f} | "
      f"alert F1 {s['alert_f1']['mean']:.3f} | trend acc {s['trend_accuracy']['mean']:.3f}")
PY
    fi
done

# Small bundle to download before stopping the machine (no multi-GB checkpoints).
"$PY" - "${OUT}" "${RESULTS}" <<'PY'
import pathlib, sys, tarfile

out, dest = pathlib.Path(sys.argv[1]), sys.argv[2]
skip = {"best.pt", "last.pt"}
with tarfile.open(dest, "w:gz") as tar:
    tar.add(out, arcname=out.name, filter=lambda info: None if pathlib.PurePath(info.name).name in skip else info)
print(f"wrote {dest} ({pathlib.Path(dest).stat().st_size / 1e6:.1f} MB)")
PY
step "done: download ${RESULTS}, then STOP the machine"

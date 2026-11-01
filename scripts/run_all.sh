#!/usr/bin/env bash
# One-shot run for a rented GPU, in priority order so a budget cut-off still leaves the important results.
#
#   bash scripts/run_all.sh                # 12-epoch cap, patience 4 (about 2.5 h on one A100 80GB)
#   bash scripts/run_all.sh 15 5           # longer runs
#   SKIP_BASELINE=1 bash scripts/run_all.sh   # skip the paper-recipe baseline (about 1.5 h on an A100)
#
# Re-running the same command is safe: finished folds are skipped, so an interrupted job resumes.
# Run it inside tmux / nohup so a dropped SSH connection does not stop it.
set -euo pipefail

EPOCHS="${1:-12}"
PATIENCE="${2:-4}"
export TOKENIZERS_PARALLELISM=false
START=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - START) / 60 )) min] $* ==="; }

step "environment"
python -m emotisense doctor

step "tests and offline smoke run"
python -m pytest -q
python -m emotisense smoke

step "data (public Kaggle IEMOCAP, 1.4 GB)"
python -m emotisense download iemocap

step "pre-flight on real data (fails fast if anything is wrong)"
python -m emotisense train --config configs/ser/iemocap_wavlm_base.yaml --folds 0 --run-dir runs/preflight \
    --set experiment=preflight --set data.limit_per_split=64 --set train.epochs=1 --set train.save_last=false

COMMON=(--set "train.epochs=${EPOCHS}" --set "train.patience=${PATIENCE}" --set train.save_last=false)

step "1/3 main result: paper recipe + improvements (5-fold CV)"
python -m emotisense train --config configs/ser/iemocap_wavlm_large_plus.yaml --run-dir runs/final/plus "${COMMON[@]}"

step "2/3 timeline models: no-context MLP vs BiLSTM vs Transformer (frozen WavLM-Large embeddings)"
for kind in bilstm mlp transformer; do
    python -m emotisense timeline --config "configs/timeline/${kind}.yaml" --run-dir "runs/final/timeline_${kind}"
done

if [ "${SKIP_BASELINE:-0}" != "1" ]; then
    step "3/3 baseline: base-paper recipe (5-fold CV)"
    python -m emotisense train --config configs/ser/iemocap_wavlm_large.yaml --run-dir runs/final/paper "${COMMON[@]}"
fi

step "results"
for f in runs/final/plus/summary.md runs/final/paper/summary.md; do [ -f "$f" ] && { echo "--- $f"; cat "$f"; }; done
for kind in bilstm mlp transformer; do
    f="runs/final/timeline_${kind}/summary.json"
    [ -f "$f" ] && python - "$f" "$kind" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"timeline {sys.argv[2]:<12} CCC {s['ccc']['mean']:.4f} +/- {s['ccc']['std']:.4f} | RMSE {s['rmse']['mean']:.4f} | "
      f"alert F1 {s['alert_f1']['mean']:.3f} | trend acc {s['trend_accuracy']['mean']:.3f}")
PY
done

# Small bundle to download before stopping the machine (no multi-GB checkpoints).
tar czf results.tgz --exclude='best.pt' --exclude='last.pt' runs/final
step "done: download results.tgz, then STOP the machine"

# EmotiSense

Multi-emotion intelligence from customer-call audio. EmotiSense tracks how a caller feels moment by
moment (anger, frustration, satisfaction, sadness, calm), flags escalation as it happens, and rolls the
results up into agent and team analytics.

It has two layers:

1. **Speech emotion recognition (SER)**: a WavLM model trained and evaluated on IEMOCAP with exactly the
   protocol of the base paper, so the numbers are directly comparable.
   *Wang et al., "Multi-Loss Learning for Speech Emotion Recognition with Energy-Adaptive Mixup and
   Frame-Level Attention", arXiv:2512.04551* reports **WA 78.47 / UA 79.14** on IEMOCAP (4 classes,
   5-fold session-independent CV).
2. **Conversation timeline**: a BiLSTM / Transformer over per-turn embeddings that outputs 0-1 intensity
   curves across a whole call, plus an escalation detector, REST API and supervisor dashboard.

> **Status:** the full pipeline is implemented and unit-tested (CPU, tiny model, synthetic data, plus a
> real WavLM-Base+ forward check). The WavLM-Large IEMOCAP numbers have **not been produced yet**; run
> `make train-paper` on a GPU to get them. Nothing in this repo claims a result that was not measured.

## Quick start (any GPU machine)

```bash
git clone https://github.com/arulkumarann/EmotiSense.git && cd EmotiSense
pip install -e ".[dev,app,track]"      # install PyTorch for your CUDA version first if needed
make doctor                            # GPU, dependencies, disk, dataset status
make smoke                             # offline end-to-end check in about a minute (no data needed)
```

### Data

| Dataset | Source | Command |
|---|---|---|
| **IEMOCAP** (benchmark + timeline) | public Kaggle copy `ourbucket/iemocap-audio-complete` via `kagglehub` | `make data-iemocap` |
| **RAVDESS** (8 emotions) | public Zenodo archive, MD5-verified | `make data-ravdess` |

`make data-iemocap` downloads IEMOCAP into `data/raw/kagglehub/` (about 10 GB, git-ignored), builds
`data/processed/iemocap/manifest.csv`, records where the audio lives in `meta.json` (so no path configuration is
needed) and prints the size of the 4-class benchmark subset, which should be **5,531** utterances.

The builder finds the `EmoEvaluation/*.txt` label files and the audio anywhere under the downloaded folder and
matches them by utterance id, so it does not depend on one particular directory layout. If Kaggle asks for
credentials, create an API token at https://www.kaggle.com/settings, then run
`python -c "import kagglehub; kagglehub.login()"` or set `KAGGLE_USERNAME` / `KAGGLE_KEY`.
Already have your own copy? `python -m emotisense prepare iemocap --root /path/to/IEMOCAP_full_release`.
It is a community copy of IEMOCAP; the original USC SAIL licence and citation requirements still apply.

### Train

```bash
make train-paper        # base-paper recipe, IEMOCAP 4-class, 5-fold CV
make train-plus         # + valence/arousal aux task, soft labels, EMA, layer-wise LR, SpecAugment
make train-frustration  # 5-class variant that includes "frustrated"
make ablate             # one improvement at a time, to see which ones actually help
```

Any config value can be overridden from the command line, and a subset of folds can be run:

```bash
python -m emotisense train --config configs/ser/iemocap_wavlm_large.yaml \
    --folds 0 1 --set train.epochs=20 --set encoder.name=microsoft/wavlm-base-plus
```

Runs are resumable: if a job is interrupted, re-run with `--run-dir runs/<experiment>/<timestamp> --resume`.
Finished folds are skipped automatically.

### Timeline, ensembling, serving

```bash
make timeline-all       # no-context MLP vs BiLSTM vs Transformer (answers the research question)
python -m emotisense ensemble --runs runs/a/<ts> runs/b/<ts> --out runs/ens   # average several runs

python -m emotisense analyze call.wav --timeline runs/timeline_bilstm/<ts>/timeline_deploy.pt --out result.json
python -m emotisense serve     --timeline runs/timeline_bilstm/<ts>/timeline_deploy.pt   # REST API :8000
python -m emotisense dashboard --timeline runs/timeline_bilstm/<ts>/timeline_deploy.pt   # Streamlit :8501
```

API: `POST /analyze` (multipart `file`, `agent`), `GET /calls`, `GET /calls/{id}`, `GET /agents`, `GET /health`.

## What each run produces

```
runs/<experiment>/<timestamp>/
  config.yaml  run.log  summary.json  summary.md      # CV table: WA/UA per fold, mean +/- std, pooled
  fold0/ ... fold4/
    best.pt         # weights (fp16) + config + encoder config: loads with no network access
    last.pt         # only while a fold is running (resume)
    metrics.csv     # per-epoch losses, val WA/UA, learning rate, seconds   (+ tb/ if enabled)
    preds.csv       # test probabilities per utterance (feeds `ensemble`)
    result.json  train.log  config.yaml
```

Logging backends: console + file always, CSV always, TensorBoard (`logging.tensorboard=true`), Weights & Biases
(`logging.wandb.enabled=true`).

## How the SER model works (and where it goes beyond the paper)

| Component | Paper | EmotiSense |
|---|---|---|
| Encoder | WavLM | WavLM-Large, **learnable weighted sum of all layers** |
| Pooling | residual 16-head MHSA + attention pooling | same |
| Losses | KL + focal + center + frame-level SupCon | same, weights configurable |
| Augmentation | energy-adaptive mixup (SNR -5..10 dB) | same, plus optional SpecAugment time masking |
| Labels | mixed one-hot | optional **annotator vote distribution** blend (`data.soft_label_alpha`) |
| Auxiliary task | none | optional **valence / arousal / dominance** regression |
| Weights | final | optional **EMA**, layer-wise LR decay, encoder warm-up |
| Beyond a single label | none | conversation timeline, escalation alerts, analytics |

The "plus" switches are principled but **unablated on this codebase**; `make ablate` measures each one on the
same folds. Where the paper leaves a detail open (epoch count, crop length, loss weights, temperature) the
configs choose a documented default.

### Protocol (matches the paper)

IEMOCAP, 4 classes (neutral, happy+excited, angry, sad), 5,531 utterances, 5-fold session-independent CV
(fold *k* tests on session *k+1*). The paper does not describe validation, so a **held-out speaker from the
training sessions** is used for model selection; the test session is never used for early stopping or
checkpoint choice. Report mean +/- std over folds (`summary.md`).

## The timeline layer

Targets are annotator-agreement intensities from IEMOCAP's per-evaluator votes for **anger, frustration,
happiness (shown as satisfaction), sadness, neutral (shown as calm)**. Confusion is not modelled: no public
dataset labels it. Dialogues are split by session, so both speakers of a conversation stay together.

Embeddings come from either a **frozen** pretrained encoder (leak-free, the default) or a **fine-tuned** SER
checkpoint (`embeddings.source=finetuned`, stronger but trained on the same sessions; use per-fold
checkpoints and treat results with care). Evaluation reports CCC / RMSE / MAE per emotion and how often the
predicted timeline raises the same alert and trend as the annotated one.

Escalation (`configs/timeline/base.yaml`): an alert fires when smoothed `max(anger, frustration)` crosses the
threshold or rises by `rise_delta` within `rise_window_s`; the call trend is the fitted change over the call.
The agent risk score (0.5 x mean negativity + 0.5 x escalation rate) is a heuristic, not a trained churn model.

## Repository layout

```
configs/        base.yaml, ser/*.yaml, timeline/*.yaml, app/default.yaml
src/emotisense/
  data/         iemocap, ravdess, labels, splits, datasets, augment (EAM), download, synthetic
  models/       encoder (WavLM + layer weighting), pooling, ser
  losses/       KL, focal, center, frame-level SupCon, multi-loss
  training/     trainer (AMP, EMA, resume), optimiser groups + schedules, cv driver
  evaluation/   WA/UA/CCC/RMSE/MAE, ensembling
  timeline/     timeline models, dialog data, escalation logic, CV training
  embedding.py  frozen / fine-tuned embedders shared by training and serving
  app/          VAD, analyzer, SQLite store, FastAPI, Streamlit dashboard, plots
  cli.py, cli_extra.py
tests/          70 tests (CPU only, no downloads)
Makefile        every common workflow as one command
```

## Honest limitations

- IEMOCAP is acted speech; behaviour on real call-centre audio is untested and needs your own labelled calls.
- Timeline labels are annotator agreement, not calibrated intensity.
- The energy VAD is simple; swap in a neural VAD for noisy audio.
- Base-size models (WavLM-Base+) use group norm in the CNN front-end, so padding slightly affects outputs;
  eval batches are length-sorted to minimise it. WavLM-Large is padding-invariant.

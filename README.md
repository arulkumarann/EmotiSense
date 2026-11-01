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

> **Status.** Verified on CPU with real data and real pretrained models: the test suite (83 tests), an offline smoke
> run, dry runs of WavLM-Base+ and **WavLM-Large** on real IEMOCAP and RAVDESS audio (loss, validation, checkpointing,
> test evaluation, CV summary), the timeline stage on real dialogues, `analyze`, the REST API and the dashboard.
> **Not done yet:** full-scale GPU training, so there are **no accuracy numbers**; see `make train-paper`.
> Nothing in this repo claims a result that was not measured.

## Quick start (any GPU machine)

```bash
git clone https://github.com/arulkumarann/EmotiSense.git && cd EmotiSense
pip install -e ".[dev,app,track]"      # install PyTorch for your CUDA version first if needed
make doctor                            # GPU, dependencies, disk, dataset status
make test                              # 83 tests, CPU only
make smoke                             # offline end-to-end check in about a minute (no data needed)
```

### Data

| Dataset | Source | Command |
|---|---|---|
| **IEMOCAP** (benchmark + timeline) | public Kaggle copy `ourbucket/iemocap-audio-complete` via `kagglehub` (1.4 GB) | `make data-iemocap` |
| **RAVDESS** (8 emotions) | public Zenodo archive, MD5-verified | `make data-ravdess` |

`make data-iemocap` downloads the audio, builds `data/processed/iemocap/manifest.csv` and records where the audio
lives in `meta.json`, so no path configuration is needed. Set `KAGGLEHUB_CACHE=/some/disk` first to choose where the
download goes (default: `data/raw/kagglehub/`). If Kaggle asks for credentials, create an API token at
https://www.kaggle.com/settings and run `python -c "import kagglehub; kagglehub.login()"`.

**What this copy contains, and what it means.** 10,039 utterances (5 sessions, 10 speakers, 151 dialogues) and one
`training_data.csv` with a single hard label per utterance. There are no annotator votes, no valence/arousal/dominance
and no timestamps (turn times are reconstructed from durations in dialogue order). Its labels also differ from the
official consensus: every utterance is labelled, so the 4-class subset has **7,536 utterances instead of the paper's
5,531**. Consequences:

- Scores on this copy are **not directly comparable** to the paper's 78.47 / 79.14. `prepare` prints a note saying so.
- Soft labels (`data.soft_label_alpha`) and the valence/arousal auxiliary loss have nothing to work with and are inert.
- Timeline targets are 0/1 per emotion rather than annotator-agreement fractions.

For the exact paper protocol, point the loader at the official IEMOCAP release (free licence request at
https://sail.usc.edu/iemocap/): `python -m emotisense prepare iemocap --root /path/to/IEMOCAP_full_release`. It detects
the `EmoEvaluation` files automatically and switches on votes, VAD and real timestamps; nothing else changes.
The original USC SAIL licence and citation requirements apply to either copy.

### Train

```bash
make dry-run            # 2-minute pre-flight on real IEMOCAP: 64-utterance sample, 1 epoch, 1 fold
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

### Renting a GPU on a budget

`bash scripts/run_all.sh` does everything in priority order (checks, data, pre-flight, then the main 5-fold run, the
timeline models and the baseline) and writes `results.tgz`. Run it inside `tmux`. Re-running resumes finished folds.
Batches are length-bucketed (`train.bucket_by_length`), which measured **47% less padded compute** on IEMOCAP.

WavLM-Large needs about 2.2 PFLOP per epoch on this data. Rough wall-clock for the whole script (setup about 20 min,
12-epoch cap) is about 2.5 h on one A100 80GB, so about $4.5 at $1.79/h; skip the baseline with `SKIP_BASELINE=1` for
about 1.5 h. These are estimates, not measurements: after the first fold of the first run, multiply that fold's time
by the remaining folds to see where you really are.

### Using the trained dashboard on your own PC

After a training run, keep `timeline_deploy.pt` (BiLSTM, about 12 MB) in `models/`, then double-click `run_dashboard.bat`
(Windows) or run `./run_dashboard.sh`. Your browser opens at http://localhost:8501: upload a `.wav` / `.flac` / `.ogg`
call and the page plots anger, frustration, satisfaction, sadness and calm over time, marks escalation alerts and lists
the moments to review. Any CPU works (about 40 s for a 2-minute call; the first run downloads WavLM-Large, 1.2 GB).

The dashboard embeds audio with the *frozen* pretrained WavLM-Large and scores it with the BiLSTM; it does not use the
fine-tuned classifier checkpoints (`best.pt`). Scores are indicative, not calibrated intensities, and agent and
customer speech are not separated.

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

IEMOCAP, 4 classes (neutral, happy+excited, angry, sad), 5-fold session-independent CV (5,531 utterances in the
official release; 7,536 in the Kaggle copy, see Data)
(fold *k* tests on session *k+1*). The paper does not describe validation, so a **held-out speaker from the
training sessions** is used for model selection; the test session is never used for early stopping or
checkpoint choice. Report mean +/- std over folds (`summary.md`).

## The timeline layer

With the official release, targets are annotator-agreement intensities from IEMOCAP's per-evaluator votes (0/1 with the
Kaggle copy) for **anger, frustration,
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
tests/          83 tests (CPU only, no downloads)
Makefile        every common workflow as one command
```

## Honest limitations

- The Kaggle copy has hard labels only and a different label set than the paper's protocol (see Data).
- IEMOCAP is acted speech; behaviour on real call-centre audio is untested and needs your own labelled calls.
- Timeline labels are annotator agreement, not calibrated intensity.
- The energy VAD is simple; swap in a neural VAD for noisy audio.
- Base-size models (WavLM-Base+) use group norm in the CNN front-end, so padding slightly affects outputs;
  eval batches are length-sorted to minimise it. WavLM-Large is padding-invariant.

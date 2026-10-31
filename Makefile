PY ?= python
IEMOCAP_ROOT ?= data/raw/IEMOCAP_full_release

.PHONY: install install-app test smoke doctor data-ravdess data-iemocap prepare-iemocap \
        train-paper train-plus train-frustration train-ravdess ablate timeline timeline-all dashboard

install:            ## core + dev dependencies
	$(PY) -m pip install -e ".[dev]"

install-app:        ## + API / dashboard / experiment tracking
	$(PY) -m pip install -e ".[dev,app,track]"

test:
	$(PY) -m pytest

doctor:             ## check GPU, dependencies, free disk and dataset status
	$(PY) -m emotisense doctor

smoke:              ## offline end-to-end check (tiny model, synthetic data, CPU ok)
	$(PY) -m emotisense smoke

data-ravdess:       ## fully public dataset: download + manifest
	$(PY) -m emotisense download ravdess

data-iemocap:       ## download the public Kaggle copy of IEMOCAP and build its manifest
	$(PY) -m emotisense download iemocap

prepare-iemocap:    ## build the manifest from your own extracted copy (IEMOCAP_ROOT=...)
	$(PY) -m emotisense prepare iemocap --root $(IEMOCAP_ROOT)

train-paper:        ## base-paper recipe, IEMOCAP 4-class, 5-fold CV
	$(PY) -m emotisense train --config configs/ser/iemocap_wavlm_large.yaml

train-plus:         ## paper recipe + EmotiSense improvements
	$(PY) -m emotisense train --config configs/ser/iemocap_wavlm_large_plus.yaml

train-frustration:  ## 5-class variant including "frustrated"
	$(PY) -m emotisense train --config configs/ser/iemocap5_frustration.yaml

train-ravdess:
	$(PY) -m emotisense train --config configs/ser/ravdess_wavlm_large.yaml

ablate:             ## one switch at a time on top of the paper recipe (each is a full 5-fold run)
	$(PY) -m emotisense train --config configs/ser/iemocap_wavlm_large.yaml --set experiment=abl_no_mixup --set augment.mixup_prob=0.0
	$(PY) -m emotisense train --config configs/ser/iemocap_wavlm_large.yaml --set experiment=abl_vad_aux --set loss.weights.vad=0.5
	$(PY) -m emotisense train --config configs/ser/iemocap_wavlm_large.yaml --set experiment=abl_soft_labels --set data.soft_label_alpha=0.3
	$(PY) -m emotisense train --config configs/ser/iemocap_wavlm_large.yaml --set experiment=abl_ema --set train.ema_decay=0.999
	$(PY) -m emotisense train --config configs/ser/iemocap_wavlm_large.yaml --set experiment=abl_layerwise_lr --set train.layerwise_lr_decay=0.95
	$(PY) -m emotisense train --config configs/ser/iemocap_wavlm_large.yaml --set experiment=abl_no_layer_weighting --set encoder.layer_weighting=false

timeline:           ## BiLSTM timeline on frozen WavLM-Large embeddings
	$(PY) -m emotisense timeline --config configs/timeline/bilstm.yaml

timeline-all:       ## the temporal-modelling comparison: no context vs BiLSTM vs Transformer
	$(PY) -m emotisense timeline --config configs/timeline/mlp.yaml
	$(PY) -m emotisense timeline --config configs/timeline/bilstm.yaml
	$(PY) -m emotisense timeline --config configs/timeline/transformer.yaml

dashboard:
	$(PY) -m emotisense dashboard

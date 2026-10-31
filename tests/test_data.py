import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from emotisense.data.augment import energy_adaptive_mixup
from emotisense.data.datasets import Collator, SERDataset
from emotisense.data.iemocap import build_iemocap_manifest, parse_evaluation_file
from emotisense.data.labels import INTENSITY_EMOTIONS, intensity_targets, scheme_targets
from emotisense.data.manifest import resolve_audio_root
from emotisense.data.ravdess import parse_ravdess_name
from emotisense.data.splits import split_fold

EVAL_TEXT = """%[START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
[6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
C-E2:\tNeutral;\t()
C-E3:\tNeutral;\t()
C-E4:\tNeutral;\t()
C-F1:\tNeutral;\t()
A-E3:\tval 3; act 3; dom  3;\t()
A-E4:\tval 2; act 2; dom  2;\t()

[10.0100 - 11.3900]\tSes01F_impro01_M001\tfru\t[2.0000, 3.5000, 3.0000]
C-E2:\tFrustration;\t()
C-E3:\tAnger; Frustration;\t(angry tone)
C-E4:\tFrustration;\t()

[12.0000 - 14.0000]\tSes01F_impro01_F002\txxx\t[3.0000, 3.0000, 3.0000]
C-E2:\tOther;\t()
C-E3:\tSurprise;\t()
"""


def _fake_iemocap(root):
    eval_dir = root / "Session1" / "dialog" / "EmoEvaluation"
    eval_dir.mkdir(parents=True)
    (eval_dir / "Ses01F_impro01.txt").write_text(EVAL_TEXT)
    wav_dir = root / "Session1" / "sentences" / "wav" / "Ses01F_impro01"
    wav_dir.mkdir(parents=True)
    for utt in ("Ses01F_impro01_F000", "Ses01F_impro01_M001", "Ses01F_impro01_F002"):
        sf.write(str(wav_dir / f"{utt}.wav"), np.zeros(1600, dtype=np.float32), 16000)
    for s in range(2, 6):
        d = root / f"Session{s}" / "dialog" / "EmoEvaluation"
        d.mkdir(parents=True)
        (d / f"Ses0{s}F_impro01.txt").write_text(EVAL_TEXT.replace("Ses01", f"Ses0{s}"))
        w = root / f"Session{s}" / "sentences" / "wav" / f"Ses0{s}F_impro01"
        w.mkdir(parents=True)
        for utt in ("F000", "M001", "F002"):
            sf.write(str(w / f"Ses0{s}F_impro01_{utt}.wav"), np.zeros(1600, dtype=np.float32), 16000)


def test_parse_evaluation_file(tmp_path):
    path = tmp_path / "e.txt"
    path.write_text(EVAL_TEXT)
    recs = parse_evaluation_file(path)
    assert [r["utt_id"] for r in recs] == ["Ses01F_impro01_F000", "Ses01F_impro01_M001", "Ses01F_impro01_F002"]
    assert recs[0]["votes"]["neu"] == 4 and recs[0]["n_eval"] == 4
    # An evaluator giving two labels splits their single vote.
    assert recs[1]["votes"]["fru"] == pytest.approx(2.5) and recs[1]["votes"]["ang"] == pytest.approx(0.5)
    assert recs[1]["val"] == 2.0 and recs[1]["act"] == 3.5
    assert recs[2]["label"] == "xxx"


def test_build_iemocap_manifest_and_schemes(tmp_path):
    _fake_iemocap(tmp_path)
    df = build_iemocap_manifest(tmp_path)
    assert len(df) == 15 and set(df["session"]) == {1, 2, 3, 4, 5}
    row = df[df["utt_id"] == "Ses01F_impro01_M001"].iloc[0]
    assert row["speaker"] == "1M" and row["dialog"] == "Ses01F_impro01"
    assert row["path"] == "Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_M001.wav"

    keep4, hard4, soft4 = scheme_targets(df, "iemocap4")
    assert keep4.sum() == 5          # only neutral rows are in the 4-class scheme
    assert np.allclose(soft4[keep4].sum(axis=1), 1.0)
    keep5, hard5, soft5 = scheme_targets(df, "iemocap5")
    assert keep5.sum() == 10         # frustration rows join in the 5-class scheme
    assert (hard5[keep5] >= 0).all() and (hard5[~keep5] == -1).all()
    # Soft label of the frustrated utterance mixes anger in.
    idx = np.where(df["utt_id"].eq("Ses01F_impro01_M001"))[0][0]
    assert soft5[idx].argmax() == 4 and soft5[idx][2] > 0


def test_intensity_targets_range_and_columns(tmp_path):
    _fake_iemocap(tmp_path)
    df = build_iemocap_manifest(tmp_path)
    y = intensity_targets(df)
    assert y.shape == (len(df), len(INTENSITY_EMOTIONS))
    assert (y >= 0).all() and (y <= 1).all()
    other = df[df["utt_id"] == "Ses01F_impro01_F002"].index[0]
    assert y[other].sum() == 0       # only "other"/"surprise" votes -> no timeline emotion


def test_iemocap_missing_root_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="IEMOCAP_full_release"):
        build_iemocap_manifest(tmp_path / "nope")


def test_iemocap_layout_is_discovered_anywhere(tmp_path):
    """Labels and audio may live in unrelated folders (e.g. a flat Kaggle copy)."""
    labels = tmp_path / "meta" / "EmoEvaluation"
    labels.mkdir(parents=True)
    audio = tmp_path / "all_wavs"
    audio.mkdir()
    (labels / "Ses01F_impro01.txt").write_text(EVAL_TEXT)
    for utt in ("Ses01F_impro01_F000", "Ses01F_impro01_M001", "Ses01F_impro01_F002"):
        sf.write(str(audio / f"{utt}.wav"), np.zeros(1600, dtype=np.float32), 16000)
    df = build_iemocap_manifest(tmp_path)
    assert len(df) == 3 and set(df["session"]) == {1}
    assert df["path"].str.startswith("all_wavs/").all()


def test_iemocap_without_labels_explains_itself(tmp_path):
    (tmp_path / "wavs").mkdir()
    sf.write(str(tmp_path / "wavs" / "Ses01F_impro01_F000.wav"), np.zeros(1600, dtype=np.float32), 16000)
    with pytest.raises(FileNotFoundError, match="EmoEvaluation"):
        build_iemocap_manifest(tmp_path)


def test_audio_root_falls_back_to_recorded_meta(tmp_path):
    real = tmp_path / "cache"
    real.mkdir()
    manifest = tmp_path / "proc" / "manifest.csv"
    manifest.parent.mkdir()
    (manifest.parent / "meta.json").write_text(json.dumps({"audio_root": str(real)}))
    assert resolve_audio_root(None, manifest) == str(real)
    assert resolve_audio_root(str(tmp_path / "missing"), manifest) == str(real)
    other = tmp_path / "other"
    other.mkdir()
    assert resolve_audio_root(str(other), manifest) == str(other)      # an existing configured folder wins
    assert resolve_audio_root(None, tmp_path / "nometa" / "m.csv") is None


def test_ravdess_name_parser():
    info = parse_ravdess_name("03-01-06-02-01-02-12.wav")
    assert info["emotion"] == "fearful" and info["actor"] == 12 and info["intensity"] == 2
    assert parse_ravdess_name("garbage.wav") is None


@pytest.mark.parametrize("dataset,n_groups", [("iemocap", 5), ("ravdess", 6)])
def test_splits_are_disjoint_and_speaker_independent(dataset, n_groups):
    rows = []
    for g in range(1, n_groups + 1):
        for spk in range(2):
            for i in range(3):
                rows.append({"session": g, "speaker": f"{g}{spk}"})
    df = pd.DataFrame(rows)
    for fold in range(n_groups):
        for level in ("speaker", "session"):
            m = split_fold(df, dataset, fold, level)
            assert not (m["train"] & m["val"]).any() and not (m["train"] & m["test"]).any()
            assert not (m["val"] & m["test"]).any()
            assert (m["train"] | m["val"] | m["test"]).all()
            tr, va, te = (set(df[m[k]]["speaker"]) for k in ("train", "val", "test"))
            assert not (tr & va) and not (tr & te) and not (va & te)
            assert set(df[m["test"]]["session"]) == {fold + 1}


def test_split_rejects_bad_fold():
    df = pd.DataFrame({"session": [1, 2, 3, 4, 5], "speaker": list("abcde")})
    with pytest.raises(ValueError):
        split_fold(df, "iemocap", 9)


def test_eam_label_weight_and_energy():
    rng = np.random.default_rng(0)
    x_i = rng.standard_normal(16000).astype(np.float32)
    x_j = rng.standard_normal(16000).astype(np.float32)
    for _ in range(50):
        res = energy_adaptive_mixup(x_i, x_j, rng)
        assert res is not None and 0.0 < res.lam < 1.0
        assert 0 <= res.start < res.end <= len(x_i) and (res.end - res.start) <= len(x_i) // 2
        # Outside the overlay the signal is untouched.
        assert np.array_equal(res.wave[:res.start], x_i[:res.start])
        assert np.array_equal(res.wave[res.end:], x_i[res.end:])
    assert energy_adaptive_mixup(np.zeros(1, np.float32), x_j, rng) is None


def test_eam_snr_controls_dominance():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8000).astype(np.float32)
    loud = energy_adaptive_mixup(x, x, rng, snr_range=(-5, -5))
    quiet = energy_adaptive_mixup(x, x, rng, snr_range=(10, 10))
    assert loud.interferer_dominates and not quiet.interferer_dominates


def test_dataset_and_collator(synthetic_corpus):
    root, df = synthetic_corpus
    df = df.copy()
    df["path"] = [str(root / p) for p in df["path"]]
    ds = SERDataset(df, "iemocap5", max_seconds=1.0, train=True, soft_label_alpha=0.5)
    assert len(ds) == len(df) and ds.num_classes == 5
    assert np.allclose(ds.soft.sum(axis=1), 1.0)
    item = ds[0]
    assert len(item["wave"]) <= 16000 and item["vad_mask"].all()

    batch = Collator(train=True, mixup_prob=1.0)([ds[i] for i in range(6)])
    assert batch["wave"].shape == batch["attn"].shape
    assert batch["soft"].shape == (6, 5) and torch_allclose_one(batch["soft"].sum(1))
    assert (batch["span_label"] >= -1).all()
    assert (batch["span"][:, 0] >= 0).any()           # mixup was applied to at least some items

    eval_batch = Collator(train=False)([ds[i] for i in range(3)])
    assert (eval_batch["span"] == -1).all()
    assert eval_batch["attn"].sum(1).tolist() == eval_batch["lengths"].tolist()
    # Per-utterance normalisation: valid samples have ~unit variance.
    first = eval_batch["wave"][0, :eval_batch["lengths"][0]]
    assert abs(float(first.std()) - 1.0) < 0.05


def torch_allclose_one(t):
    import torch

    return torch.allclose(t, torch.ones_like(t), atol=1e-5)

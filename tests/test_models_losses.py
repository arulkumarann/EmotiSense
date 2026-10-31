import numpy as np
import pytest
import torch

from emotisense.config import Config
from emotisense.evaluation.metrics import aggregate_folds, ccc, classification_metrics, regression_metrics
from emotisense.losses import MultiLoss, focal_loss, frame_labels, kl_loss, supcon_loss
from emotisense.models import FrameAttentionPool, SERModel, SpeechEncoder
from emotisense.training.optim import build_param_groups, lr_factor


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    enc = SpeechEncoder(tiny=True)
    return SERModel(5, enc, pool_heads=4, head_hidden=32, center_dim=8, supcon_dim=16).eval()


def test_pool_ignores_padding():
    torch.manual_seed(0)
    pool = FrameAttentionPool(16, heads=4, dropout=0.0).eval()
    frames = torch.randn(1, 7, 16)
    padded = torch.cat([frames, torch.randn(1, 5, 16)], dim=1)
    mask = torch.tensor([[True] * 7 + [False] * 5])
    a, _, alpha = pool(frames, torch.ones(1, 7, dtype=torch.bool))
    b, _, alpha_p = pool(padded, mask)
    assert torch.allclose(a, b, atol=1e-5)
    assert alpha_p[0, 7:].abs().sum() == 0 and torch.allclose(alpha_p.sum(), torch.tensor(1.0))


def test_pool_rejects_bad_head_count():
    with pytest.raises(ValueError):
        FrameAttentionPool(10, heads=4)


def test_ser_forward_shapes_and_padding_invariance(tiny_model):
    wave = torch.randn(2, 16000)
    attn = torch.ones(2, 16000, dtype=torch.long)
    attn[1, 9000:] = 0
    wave[1, 9000:] = 0
    out = tiny_model(wave, attn, attn.sum(1))
    b, t = out["mask"].shape
    assert out["logits"].shape == (2, 5) and out["z"].shape == (2, 8)
    assert out["h"].shape == (2, t, 16) and out["vad"].shape == (2, 3)
    assert out["vad"].min() >= 0 and out["vad"].max() <= 1
    assert out["mask"][0].all() and out["mask"][1].sum() < t
    norms = out["h"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    # The short utterance gives the same result alone as inside a padded batch.
    solo = tiny_model(wave[1:2, :9000], torch.ones(1, 9000, dtype=torch.long), torch.tensor([9000]))
    assert torch.allclose(solo["logits"], out["logits"][1:2], atol=1e-3)


def test_frozen_embedding_shape(tiny_model):
    wave = torch.randn(2, 8000)
    attn = torch.ones(2, 8000, dtype=torch.long)
    emb = tiny_model.encoder_embedding(wave, attn)
    assert emb.shape == (2, tiny_model.encoder.hidden_size)


def test_kl_and_focal():
    soft = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    logits = torch.log(soft.clamp(min=1e-9))
    assert kl_loss(logits, soft).item() == pytest.approx(0.0, abs=1e-5)
    assert kl_loss(torch.zeros(2, 2), soft).item() > 0
    easy = torch.tensor([[10.0, -10.0]])
    hard = torch.tensor([[-10.0, 10.0]])
    y = torch.tensor([[1.0, 0.0]])
    assert focal_loss(easy, y).item() < focal_loss(hard, y).item()
    assert focal_loss(torch.zeros(1, 2), y, gamma=0.0).item() == pytest.approx(np.log(2), abs=1e-5)


def test_supcon_prefers_clustered_features():
    torch.manual_seed(0)
    mask = torch.ones(4, 6, dtype=torch.bool)
    labels = torch.tensor([0, 0, 1, 1]).unsqueeze(1).expand(4, 6)
    centers = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    clustered = centers[labels] + 0.01 * torch.randn(4, 6, 8)
    clustered = torch.nn.functional.normalize(clustered, dim=-1)
    random = torch.nn.functional.normalize(torch.randn(4, 6, 8), dim=-1)
    assert supcon_loss(clustered, labels, mask) < supcon_loss(random, labels, mask)


def test_supcon_degenerate_cases_are_zero():
    h = torch.nn.functional.normalize(torch.randn(1, 1, 4), dim=-1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    assert supcon_loss(h, torch.zeros(1, 1, dtype=torch.long), mask).item() == 0.0
    h2 = torch.nn.functional.normalize(torch.randn(2, 3, 4), dim=-1)      # every frame a distinct class
    labels = torch.arange(6).view(2, 3)
    assert supcon_loss(h2, labels, torch.ones(2, 3, dtype=torch.bool)).item() == 0.0


def test_frame_labels_use_louder_interferer():
    mask = torch.ones(2, 10, dtype=torch.bool)
    hard = torch.tensor([0, 1])
    span = torch.tensor([[640, 1920], [-1, -1]])          # frames 2..5 for the first utterance
    span_label = torch.tensor([3, -1])
    labels = frame_labels(mask, hard, span, span_label)
    assert labels[0].tolist() == [0, 0, 3, 3, 3, 3, 0, 0, 0, 0]
    assert labels[1].tolist() == [1] * 10


def test_multiloss_backward_and_components(tiny_model):
    model = SERModel(5, SpeechEncoder(tiny=True), pool_heads=4, head_hidden=32, center_dim=8, supcon_dim=16).train()
    crit = MultiLoss(5, 8, {"kl": 1.0, "focal": 1.0, "center": 0.1, "supcon": 0.1, "vad": 0.5})
    wave = torch.randn(4, 12000)
    attn = torch.ones(4, 12000, dtype=torch.long)
    soft = torch.softmax(torch.randn(4, 5), dim=-1)
    batch = {"soft": soft, "hard": soft.argmax(1), "vad": torch.rand(4, 3), "vad_mask": torch.ones(4, 3),
             "span": torch.full((4, 2), -1), "span_label": torch.full((4,), -1)}
    loss, logs = crit(model(wave, attn, attn.sum(1)), batch)
    loss.backward()
    assert set(logs) == {"loss", "loss_kl", "loss_focal", "loss_center", "loss_supcon", "loss_vad"}
    assert torch.isfinite(loss) and crit.center.centers.grad is not None
    assert all(p.grad is not None for p in model.classifier.parameters())


def test_multiloss_requires_some_weight():
    crit = MultiLoss(3, 4, {"kl": 0.0})
    with pytest.raises(ValueError):
        crit({"logits": torch.zeros(1, 3)}, {"soft": torch.ones(1, 3) / 3, "hard": torch.zeros(1, dtype=torch.long)})


def test_param_groups_cover_each_parameter_once():
    model = SERModel(5, SpeechEncoder(tiny=True), pool_heads=4, head_hidden=32, center_dim=8, supcon_dim=16)
    crit = MultiLoss(5, 8, {"kl": 1.0, "center": 0.1})
    train_cfg = Config({"lr_encoder": 1e-5, "lr_head": 1e-4, "lr_center": 5e-3, "weight_decay": 0.01,
                        "layerwise_lr_decay": 0.9})
    groups = build_param_groups(model, crit, train_cfg)
    seen = [id(p) for g in groups for p in g["params"]]
    trainable = [id(p) for p in list(model.parameters()) + list(crit.parameters()) if p.requires_grad]
    assert sorted(seen) == sorted(trainable) and len(seen) == len(set(seen))
    enc_lrs = sorted({g["lr"] for g in groups if g["lr"] < 5e-5})
    assert len(enc_lrs) > 1 and max(enc_lrs) <= 1e-5 + 1e-12      # lower layers get smaller LRs
    assert any(g["lr"] == 5e-3 and g["weight_decay"] == 0.0 for g in groups)


def test_lr_schedules():
    exp = {"type": "exp", "gamma": 0.5, "decay_epochs": 2, "warmup_epochs": 0}
    assert lr_factor(0, 10, 100, exp) == 1.0
    assert lr_factor(10, 10, 100, exp) == 0.5 and lr_factor(95, 10, 100, exp) == 0.25   # capped at decay_epochs
    cos = {"type": "cosine", "warmup_epochs": 1, "min_factor": 0.0}
    assert lr_factor(0, 10, 100, cos) == pytest.approx(0.1)           # warm-up
    assert lr_factor(50, 10, 100, cos) == pytest.approx(0.5, abs=1e-6)
    with pytest.raises(ValueError):
        lr_factor(0, 1, 1, {"type": "nope"})


def test_classification_metrics_wa_ua():
    y_true = np.array([0, 0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0])
    m = classification_metrics(y_true, y_pred, ["a", "b", "c"])
    assert m["wa"] == pytest.approx(5 / 7)
    assert m["ua"] == pytest.approx((0.75 + 1.0 + 0.0) / 3)
    assert m["per_class_recall"]["c"] == 0.0 and m["n"] == 7


def test_ccc_and_regression_metrics():
    x = np.linspace(0, 1, 50)
    assert ccc(x, x) == pytest.approx(1.0, abs=1e-6)
    assert ccc(x, 1 - x) < -0.9
    assert ccc(x, x + 0.5) < 0.6            # CCC penalises bias, unlike Pearson correlation
    m = regression_metrics(np.stack([x, x], 1), np.stack([x, x + 0.1], 1), ["p", "q"])
    assert m["per_emotion"]["p"]["rmse"] == 0 and m["per_emotion"]["q"]["mae"] == pytest.approx(0.1)
    agg = aggregate_folds([{"wa": 0.5, "ua": 0.6, "macro_f1": 0.4}, {"wa": 0.7, "ua": 0.8, "macro_f1": 0.6}])
    assert agg["wa"]["mean"] == pytest.approx(0.6) and agg["ua"]["std"] == pytest.approx(0.1)

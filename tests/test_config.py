from emotisense.config import Config, apply_overrides, deep_merge, load_config


def test_deep_merge_is_recursive():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1}


def test_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("train:\n  epochs: 10\n  lr: 0.1\nname: base\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\ntrain:\n  lr: 0.5\n")
    cfg = load_config(tmp_path / "child.yaml", ["train.epochs=3", "extra.flag=true", "name=hello"])
    assert cfg.train.epochs == 3 and cfg.train.lr == 0.5
    assert cfg.extra.flag is True and cfg.name == "hello"


def test_attribute_access_and_roundtrip():
    cfg = Config({"a": {"b": [1, {"c": 2}]}})
    assert cfg.a.b[1].c == 2
    assert cfg.get_path("a.b") is not None and cfg.get_path("a.zzz", 7) == 7
    assert cfg.to_dict() == {"a": {"b": [1, {"c": 2}]}}


def test_overrides_reject_bad_syntax():
    import pytest

    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_shipped_configs_load():
    from pathlib import Path

    for path in Path("configs/ser").glob("*.yaml"):
        cfg = load_config(path)
        assert cfg.experiment and cfg.train.epochs > 0 and cfg.data.scheme

"""YAML configuration with inheritance and command-line overrides.

A config file may declare ``_base_: other.yaml`` (path relative to the file) to
inherit from another file. Nested dictionaries are merged recursively. Overrides
use dotted keys, e.g. ``train.epochs=30`` and are parsed as YAML scalars.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """A dict with attribute access that converts nested dicts recursively."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, _wrap(value))

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value):
        self[name] = value

    def to_dict(self) -> dict:
        return {k: _unwrap(v) for k, v in self.items()}

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _wrap(value):
    if isinstance(value, Config):
        return value
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value):
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path, _seen: tuple = ()) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular _base_ reference at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base_ref = data.pop("_base_", None)
    if base_ref is None:
        return data
    bases = base_ref if isinstance(base_ref, list) else [base_ref]
    merged: dict = {}
    for ref in bases:
        merged = deep_merge(merged, _read_yaml(path.parent / ref, _seen + (path,)))
    return deep_merge(merged, data)


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    data = _read_yaml(Path(path))
    data = apply_overrides(data, overrides)
    return Config(data)


def save_config(cfg: Config | dict, path: str | Path) -> None:
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

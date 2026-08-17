"""YAML configuration handling with dotted-key command line overrides."""

import json
import os

import yaml


class Config(dict):
    """Dictionary with attribute access, used for all experiment settings."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = _to_config(value)

    def __delattr__(self, name):
        del self[name]


def _to_config(obj):
    if isinstance(obj, Config):
        return obj
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def set_nested(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = Config()
        node = node[key]
    node[keys[-1]] = _to_config(value)


def get_nested(cfg, dotted_key, default=None):
    node = cfg
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def load_config(path, overrides=None):
    """Load a YAML config, resolve an optional ``_base_`` file, apply overrides.

    Overrides are strings of the form ``section.key=value`` where the value is
    parsed as YAML, so ``bridge.omega=0.5`` and ``eval.methods=[ddim50,tdpm]``
    both work.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_name = raw.pop("_base_", None)
    if base_name:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base_name)
        with open(base_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        raw = merge(base, raw)

    cfg = _to_config(raw)
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got: {item}")
        set_nested(cfg, key.strip(), yaml.safe_load(value))
    cfg.config_path = os.path.abspath(path)
    return cfg


def save_config(cfg, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_plain(cfg), f, indent=2, ensure_ascii=False)

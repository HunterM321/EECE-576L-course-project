"""YAML configs with `_base_` inheritance and `a.b=value` command-line overrides."""
import copy
from pathlib import Path

import yaml

from i2pflow.paths import resolve


class Config(dict):
    """Nested dict with attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def wrap(obj):
        if isinstance(obj, dict):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config.wrap(v) for v in obj]
        return obj

    def to_dict(self):
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o
        return unwrap(self)


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        cfg = deep_merge(_load_yaml(path.parent / base), cfg)
    return cfg


def apply_overrides(cfg, overrides):
    """Apply ["a.b=1", "c=[1,2]"] style overrides; values are parsed as YAML."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must be key=value, got {item!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path, overrides=None):
    cfg = _load_yaml(resolve(path))
    cfg = apply_overrides(cfg, overrides)
    return Config.wrap(cfg)


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict() if isinstance(cfg, Config) else cfg, f, sort_keys=False)

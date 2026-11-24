"""Layered YAML config: default.yaml <- <profile>.yaml <- --config file <- --set key=value."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from hri.platform_utils import detect_profile


def project_root() -> Path:
    """Repo root (folder containing configs/). Works with `pip install -e .`."""
    env = os.environ.get("HRI_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for cand in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (cand / "configs" / "default.yaml").exists():
            return cand
    raise FileNotFoundError("Cannot find configs/default.yaml - run from the repo root or set HRI_ROOT")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(raw: str) -> Any:
    # YAML parsing gives us ints, floats, bools, null, lists for free
    return yaml.safe_load(raw)


def apply_overrides(data: dict, overrides: list[str] | None) -> dict:
    data = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(raw)
    return data


class Config:
    """Thin wrapper over a nested dict with dotted access: cfg.get('audio.vad.threshold')."""

    def __init__(self, data: dict, profile: str, root: Path):
        self.data = data
        self.profile = profile
        self.root = root

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for p in dotted.split("."):
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    def __getitem__(self, key: str) -> Any:
        val = self.get(key, _MISSING)
        if val is _MISSING:
            raise KeyError(key)
        return val

    def section(self, dotted: str) -> dict:
        return self.get(dotted, {}) or {}

    def path(self, dotted: str) -> Path:
        """Resolve a path-valued key (~ and relative-to-root aware)."""
        p = Path(os.path.expandvars(str(self[dotted]))).expanduser()
        return p if p.is_absolute() else (self.root / p)

    @property
    def models_dir(self) -> Path:
        env = os.environ.get("HRI_MODELS_DIR")
        return Path(env).expanduser() if env else self.path("paths.models_dir")

    @property
    def data_dir(self) -> Path:
        return self.path("paths.data_dir")

    def dump(self) -> str:
        return yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False)


_MISSING = object()


def load_config(profile: str | None = None, extra: str | None = None,
                overrides: list[str] | None = None) -> Config:
    root = project_root()
    try:  # .env is optional
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
    except ImportError:
        pass

    profile = profile if profile and profile != "auto" else detect_profile()
    cfg_dir = root / "configs"
    data = yaml.safe_load((cfg_dir / "default.yaml").read_text(encoding="utf-8")) or {}
    prof_file = cfg_dir / f"{profile}.yaml"
    if not prof_file.exists():
        raise FileNotFoundError(f"Unknown profile {profile!r}: {prof_file} not found")
    data = deep_merge(data, yaml.safe_load(prof_file.read_text(encoding="utf-8")) or {})
    if extra:
        data = deep_merge(data, yaml.safe_load(Path(extra).read_text(encoding="utf-8")) or {})
    data = apply_overrides(data, overrides)
    return Config(data, profile, root)

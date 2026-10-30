"""File-system helpers: confined output paths, disk guard, hashes and small JSON manifests.

Every writer in slicemech goes through :func:`confine` so nothing is written outside the rewrite folder
(``ICC/rewrite_2026-10-18/``); caches live under :data:`DATA_DIR`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import shutil
import threading

import numpy as np

from .config import ROOT

#: Root of the rewrite project (``.../ICC/rewrite_2026-10-18``); no file is ever written outside it.
ROOT = pathlib.Path(ROOT).resolve()
#: Cache and prepared-data directory (``ROOT/data``).
DATA_DIR = ROOT / "data"
#: Read-only source tree of the earlier project (raw Puffer, TRACTOR and Raca files live below it).
ICC_DIR = ROOT.parent


def confine(path, *, root: pathlib.Path = ROOT) -> pathlib.Path:
    """Return ``path`` resolved, raising ValueError unless it lies inside ``root`` (default: the rewrite folder)."""
    p = pathlib.Path(path).resolve()
    if not p.is_relative_to(pathlib.Path(root).resolve()):
        raise ValueError(f"refusing to write outside {root}: {p}")
    return p


def free_gb(path=ROOT) -> float:
    """Free space (GB, 1e9 bytes) on the file system holding ``path``."""
    return shutil.disk_usage(pathlib.Path(path)).free / 1e9


def require_free_gb(gb: float = 3.0, path=ROOT) -> float:
    """Raise RuntimeError if fewer than ``gb`` GB are free at ``path``; return the free GB otherwise."""
    free = free_gb(path)
    if free < gb:
        raise RuntimeError(f"disk guard: only {free:.2f} GB free at {path}, need >= {gb} GB")
    return free


def file_sha256(path) -> str:
    """Hex SHA-256 of a file's stored bytes (streamed in 1 MiB blocks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_md5_b64(path) -> str:
    """Base64 MD5 of a file's stored bytes, in the format of the GCS ``md5Hash`` metadata field."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return base64.b64encode(h.digest()).decode()


def source_files(base) -> list[pathlib.Path]:
    """Every file below ``base`` that belongs to the source tree, sorted: dot-files and everything below a
    dot-directory (``.ruff_cache``, ``.mypy_cache``, ``.pytest_cache``, ...), ``__pycache__`` contents and ``*.pyc``
    are excluded (pre-freeze audit F1: a lint cache inside ``experiments/`` was part of the frozen code hash, so
    running ruff after the freeze looked like a code change)."""
    base = pathlib.Path(base)
    return sorted(f for f in base.rglob("*")
                  if f.is_file() and f.suffix != ".pyc"
                  and not any(p.startswith(".") or p == "__pycache__" for p in f.relative_to(base).parts))


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pathlib.Path):
        return str(x)
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    raise TypeError(f"not JSON serialisable: {type(x)}")


def digest(value) -> str:
    """Hex SHA-256 of the canonical JSON encoding of ``value`` (sorted keys; numpy values allowed)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_jsonable, allow_nan=False).encode()).hexdigest()


def unique_tmp(path) -> pathlib.Path:
    """Temp-file name next to ``path`` that is unique per process and thread (``<name>.<pid>.<thread>.tmp``), so
    concurrent writers of the same file never share a temp file (pre-freeze audit F2: the fixed ``<name>.tmp`` let
    one writer's ``os.replace`` move another's file away)."""
    p = pathlib.Path(path)
    return p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def write_json(path, value) -> pathlib.Path:
    """Atomically write ``value`` as indented JSON to a confined path (process- and thread-unique temp file, then
    ``os.replace``); returns the path."""
    p = confine(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = unique_tmp(p)
    try:
        with open(tmp, "w") as f:
            json.dump(value, f, indent=1, sort_keys=True, default=_jsonable)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
    return p


def read_json(path):
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)

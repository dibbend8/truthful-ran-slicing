"""Supplemental deadline/load study helpers; never change historical source or gates.

Put the isolated upstream/code directory on PYTHONPATH before importing. Shape
fitting reads TRAIN only. This module performs no experiment at import time.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from slicemech.config import Contract, Shape
from slicemech.data import calibrate as calibration_module
from slicemech.data import compose, puffer
from slicemech.model.shapes import fit_shape


DEADLINE_PROFILES = {
    "original": (2002, 150, 300),
    "video300": (300, 150, 300),
    "interactive50": (2002, 50, 300),
    "both": (300, 50, 300),
}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def object_sha256(obj: Any) -> str:
    payload = json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The caller supplies a supplemental destination; no historical file is written.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


@dataclass(frozen=True)
class DeadlineFit:
    profile: str
    contract: Contract
    shape: Shape
    provenance: dict

    def to_dict(self) -> dict:
        return _jsonable(dict(profile=self.profile, contract=asdict(self.contract),
                              shape=asdict(self.shape), provenance=self.provenance))

    @classmethod
    def from_dict(cls, payload: Mapping) -> "DeadlineFit":
        ct = dict(payload["contract"])
        ct["cls"] = tuple(ct["cls"])
        sh = payload["shape"]
        return cls(str(payload["profile"]), Contract(**ct),
                   Shape(beta=tuple(np.asarray(a, float) for a in sh["beta"]),
                         rho=tuple(np.asarray(a, float) for a in sh["rho"])),
                   dict(payload["provenance"]))


def refit_deadline_shapes(
    calibration,
    profiles: Mapping[str, tuple | list] | None = None,
    *,
    train_keys=None,
    n_atoms: int = 64,
    output_dir: str | Path | None = None,
    source_calibration_path: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, DeadlineFit]:
    """Refit only public burst shapes and deadline fields, on original TRAIN data.

    Counts, mu, g/M/C/pi, tenant forecasts and operator forecasts remain those of
    ``calibration``. By default use its stored TRAIN keys (usually all 180).
    Each TRAIN window is composed once, in original bytes, for all profiles.
    Shapes are pooled by (class, deadline), which equals historical class pooling
    when a class has one deadline. Remainders of 1 s are discarded exactly as in
    the historical fitter; e.g. 300 ms samples the first 900 ms of each interval.
    """
    profiles = dict(DEADLINE_PROFILES if profiles is None else profiles)
    if not profiles:
        raise ValueError("at least one deadline profile is required")
    if train_keys is None:
        train_keys = calibration.meta.get("train_windows") or calibration_module.train_window_keys()
    keys = [(str(day), int(offset)) for day, offset in train_keys]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("TRAIN keys must be nonempty and unique")
    if any(puffer.split_of_date(day) != "train" for day, _ in keys):
        raise ValueError("deadline shape fitting uses TRAIN dates only")
    if not (0 <= calibration.warmup < calibration.K):
        raise ValueError("warmup must leave scored TRAIN intervals")
    if int(n_atoms) != n_atoms or n_atoms < 1:
        raise ValueError("n_atoms must be a positive integer")

    delays = {}
    for name, values in profiles.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(name)):
            raise ValueError("profile names must be safe simple filename components")
        arr = np.asarray(values, float)
        if (arr.shape != (calibration.n,) or not np.all(np.isfinite(arr))
                or np.any(arr <= 0) or np.any(arr != np.floor(arr))):
            raise ValueError("each profile needs one positive whole-millisecond deadline per tenant")
        delays[str(name)] = arr

    # Common (tenant, deadline) samples are reused across profiles. Pooling order
    # remains window then tenant, as in the original calibration implementation.
    unique = sorted({(i, int(ds[i])) for ds in delays.values() for i in range(calibration.n)})
    pools = {(calibration.classes[i], deadline): [] for i, deadline in unique}
    training_digest = hashlib.sha256()
    counts_by_date = {}
    for index, (day, offset) in enumerate(keys, start=1):
        counts = tuple(calibration.counts_for("train", day))
        counts_by_date[day] = counts
        parts = compose.compose_parts(
            "train", day, offset, calibration.classes, counts, calibration.video_groups,
            lead_s=calibration.lead_s, report_lead_s=calibration.report_lead_s,
            K=calibration.K, seed=calibration.seed,
        )
        byte_arrivals = np.asarray(parts["bytes_ms"])
        theta = compose.truthful_theta(parts, calibration.classes, calibration.forecaster)
        training_digest.update(json.dumps([day, offset, counts]).encode())
        for label, array in (("bytes_ms", byte_arrivals), ("theta", theta)):
            contiguous = np.ascontiguousarray(array)
            training_digest.update(label.encode())
            training_digest.update(str((contiguous.shape, contiguous.dtype.str)).encode())
            training_digest.update(contiguous.tobytes())
        for tenant, deadline in unique:
            # Pass BYTES, not Window.arrivals_ms (which is already Mbit/ms).
            volumes = calibration_module.subwindow_volumes(
                byte_arrivals[:, tenant:tenant + 1], [deadline])
            samples = calibration_module.burst_samples(
                volumes, theta[:, tenant:tenant + 1], [deadline], calibration.warmup)[0]
            pools[(calibration.classes[tenant], deadline)].append(samples)
        if progress is not None:
            progress(index, len(keys))

    fitted, sample_counts = {}, {}
    for key, chunks in pools.items():
        samples = np.concatenate(chunks)
        fitted[key] = fit_shape(samples, n_atoms=n_atoms)
        sample_counts[f"{key[0]}|{key[1]}ms"] = int(samples.size)

    source_paths = [Path(__file__), Path(calibration_module.__file__), Path(compose.__file__),
                    Path(__import__(fit_shape.__module__, fromlist=["__file__"]).__file__)]
    common = dict(
        study="exploratory deadline/load sensitivity; not standards-compliance testing",
        fit_split="train", created_utc=datetime.now(timezone.utc).isoformat(),
        calibration_sha256=object_sha256(calibration.to_dict()),
        source_calibration_file_sha256=(None if source_calibration_path is None
                                        else file_sha256(source_calibration_path)),
        source_files_sha256={str(path.resolve()): file_sha256(path) for path in source_paths},
        composed_training_sha256=training_digest.hexdigest(),
        training_keys=keys, training_window_count=len(keys), training_date_count=len(counts_by_date),
        training_counts=counts_by_date, sample_counts=sample_counts, n_atoms=int(n_atoms),
        load_rho=float(calibration.rho), K=int(calibration.K), warmup=int(calibration.warmup),
        preserved=["mu", "g", "M", "C", "pi", "counts", "tenant_forecaster", "operator_forecasters"],
        sample_rule="nonoverlapping min(deadline,1000)-ms blocks within each 1-s interval; remainder discarded",
        zero_forecast_rule="scored intervals with theta > 0 only",
        classes=calibration.classes,
    )
    result = {}
    for name, ds in delays.items():
        atoms = [fitted[(c, int(deadline))] for c, deadline in zip(calibration.classes, ds)]
        shape = Shape(beta=tuple(a[0].copy() for a in atoms), rho=tuple(a[1].copy() for a in atoms))
        contract = replace(calibration.contract, delay_ms=ds.copy())
        provenance = dict(common, profile=name, deadlines_ms=ds.tolist())
        provenance["shape_sha256"] = object_sha256(asdict(shape))
        if np.array_equal(ds, calibration.contract.delay_ms):
            provenance["matches_original_shape"] = all(
                np.array_equal(a, b) for field in ("beta", "rho")
                for a, b in zip(getattr(shape, field), getattr(calibration.shape, field)))
        fit = DeadlineFit(name, contract, shape, provenance)
        result[name] = fit
        if output_dir is not None:
            write_json(Path(output_dir) / f"rho{calibration.rho:g}_{name}.json", fit.to_dict())
    return result


def with_deadlines(window, fit: DeadlineFit):
    """Share paired traffic arrays and give the altered contract a distinct ID."""
    if tuple(window.cls) != tuple(fit.contract.cls):
        raise ValueError("window and deadline fit have different tenant layouts")
    for name in ("mu", "g", "M", "C", "pi"):
        if not np.array_equal(getattr(window.contract, name), getattr(fit.contract, name)):
            raise ValueError(f"deadline fit changes {name}; use its matching load calibration")
    tag = object_sha256(fit.contract.delay_ms)[:12]
    return replace(window, id=f"{window.id}|deadline={fit.profile}-{tag}", contract=fit.contract,
                   meta=dict(window.meta, deadline_profile=fit.profile,
                             deadline_shape_sha256=fit.provenance["shape_sha256"]))


def safe_load_net(path: str | Path, *, expected_file_sha256: str | None = None,
                  expected_param_digest: str | None = None, config_path: str | Path | None = None):
    """Load tensor checkpoints without arbitrary pickle execution; verify hashes.

    ``config_path`` reads historical result.model_sha256 and param_digest unless
    explicit expected values are supplied. Never fall back to weights_only=False
    or add pickle globals. The returned inference net carries study_checkpoint
    provenance and the saved operator forecaster, like the upstream loader.
    """
    import torch
    from slicemech.control import features, nets

    path = Path(path)
    if config_path is not None:
        record = json.loads(Path(config_path).read_text())["result"]
        expected_file_sha256 = expected_file_sha256 or record["model_sha256"]
        expected_param_digest = expected_param_digest or record["param_digest"]
    file_digest = file_sha256(path)
    if expected_file_sha256 is not None and file_digest != expected_file_sha256:
        raise ValueError(f"checkpoint file SHA-256 mismatch: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, Mapping) or not isinstance(blob.get("spec"), Mapping):
        raise ValueError("checkpoint must contain a mapping spec and tensor state")
    spec = dict(blob["spec"])
    kind = spec.pop("kind")
    if kind not in nets._KINDS:
        raise ValueError(f"unknown checkpoint kind {kind!r}")
    state = blob.get("state")
    if not isinstance(state, Mapping) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("checkpoint state must contain tensors only")
    if kind == "unconstrained" and "report_prior" not in spec:
        spec["report_prior"] = None
    net = nets._KINDS[kind](**spec)
    net.load_state_dict(state, strict=True)
    param_digest = nets.param_digest(net)
    if expected_param_digest is not None and param_digest != expected_param_digest:
        raise ValueError(f"checkpoint parameter digest mismatch: {path}")
    forecaster = blob.get("forecaster")
    net.forecaster = None if forecaster is None else features.OperatorAR.from_dict(forecaster)
    net.eval()
    net.study_checkpoint = dict(path=str(path.resolve()), file_sha256=file_digest,
                                param_digest=param_digest, weights_only=True)
    return net

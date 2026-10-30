"""TRAIN-only regime diagnostic: the five regime targets as a function of the GBR fractions (decision support).

Run from ``code/``:

    PYTHONDONTWRITEBYTECODE=1 /home/darpan/anaconda3/bin/python experiments/diag_gbr_sweep.py [--workers 16]

The default calibration (``data/calibration/n3_default.json``) misses ``gbr_outage <= 0.05`` on all 180 TRAIN
windows (0.077). The pre-declared adjustment order is: GBR fractions first, then the load. This script keeps the
counts, mu, M, C, tariffs and burst shapes of the default calibration fixed and scales only the GBR fractions g.
For each option it re-evaluates ``data.calibrate.regime_diagnostics`` with truthful reports ``min(theta, C)``,
``r = rate_hat`` and equal weights, on the scored intervals of all 180 TRAIN windows. It also reports which tenant
dominates the reservation in overload intervals.

Nothing is changed in ``spec/params.yaml`` or the calibration files. Output: ``results/raw/diagnostics/gbr_sweep.json``.

History: the run recorded in code/STATUS.md §5.2 (the evidence for decision D1) used the pre-D2 calibration (one count
vector on every date, GBR fractions 0.5 / 0.8 / 0.2). The windows now use the calibration's per-date counts (D2,
``Calibration.counts_for``), and the option labels scale the GBR fractions currently in ``spec/params.yaml``.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from slicemech import io as sio  # noqa: E402

OPTIONS = {f"all x{s:g}": (s, s, s) for s in (1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5)}
OPTIONS.update({f"video x{s:g}": (s, 1.0, 1.0) for s in (0.8, 0.6, 0.5)})
OPTIONS.update({f"urllc x{s:g}": (1.0, s, 1.0) for s in (0.8, 0.6)})


def _window(key):
    from slicemech.data import compose
    from slicemech.data.calibrate import load_calibration
    cal = load_calibration("n3_default")
    d, o = key
    p = compose.compose_parts("train", d, o, cal.classes, cal.counts_for("train", d), cal.video_groups,
                              lead_s=cal.lead_s, report_lead_s=cal.report_lead_s, K=cal.K, seed=cal.seed)
    th = compose.truthful_theta(p, cal.classes, cal.forecaster)
    return np.minimum(th, cal.contract.C)[cal.warmup:], p["rate_hat"][cal.warmup:]


def main(argv=None):
    import multiprocessing as mp

    from slicemech.config import Contract, Mechanism
    from slicemech.data.calibrate import check_targets, load_calibration, regime_diagnostics, train_window_keys

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    cal = load_calibration("n3_default")
    ct, mech = cal.contract, Mechanism()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(_window, train_window_keys()))
    z = np.concatenate([a for a, _ in res])
    r = np.concatenate([b for _, b in res])
    out = dict(intervals=int(len(z)), options={})
    for tag, scale in OPTIONS.items():
        c2 = Contract(cls=ct.cls, g=ct.g * np.asarray(scale), M=ct.M, C=ct.C, pi=ct.pi, delay_ms=ct.delay_ms,
                      mu=ct.mu)
        t = check_targets(regime_diagnostics(z, r, c2, mech))
        out["options"][tag] = dict(gbr_fraction=[float(PARAMS_G[c]) * s for c, s in zip(ct.cls, scale)],
                                   all_ok=all(v["ok"] for v in t.values()),
                                   **{k: v["value"] for k, v in t.items()})
        print(tag, {k: round(v, 4) if isinstance(v, float) else v for k, v in out["options"][tag].items()})
    b = np.minimum(z, ct.g) / r
    ov = b.sum(1) > mech.B
    out["overload_intervals"] = int(ov.sum())
    out["overload_dominant_tenant_counts"] = np.bincount(b[ov].argmax(1), minlength=ct.n).tolist()
    out["rate_hat_p05"] = np.percentile(r, 5, axis=0).tolist()
    path = sio.write_json(sio.ROOT / "results" / "raw" / "diagnostics" / "gbr_sweep.json", out)
    print("wrote", path)


from slicemech.config import PARAMS  # noqa: E402

PARAMS_G = {c: v["gbr_x_mu"] for c, v in PARAMS["classes"].items()}

if __name__ == "__main__":
    main()

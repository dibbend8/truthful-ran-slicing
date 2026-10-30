"""Welfare-headroom regime diagnostic (round-1 review, finding 4; gate G4 of experiment_design.md §9).

Run from ``code/``:

    PYTHONDONTWRITEBYTECODE=1 /home/darpan/anaconda3/bin/python experiments/diag_headroom.py [--workers 12]
        [--train-windows 36] [--dev-windows 0 (= all 126)] [--pop 64 --iters 8]

1. Fits the static priority vector on all 180 TRAIN windows (``control.static.fit_static``, simplex grid).
2. On a seeded random subset of TRAIN windows and on DEV windows (all by default), computes per window
   (``eval.headroom.window_headroom``): equal, TRAIN-static, in-sample best-of-grid (66 vectors), direct and the
   lookahead oracle (base policy = the TRAIN static weights), all realized welfare with truthful reports.
3. Writes ``results/raw/diagnostics/headroom.json`` (summary per split + every window row) and ``headroom.csv``.

TEST is never touched. Nothing in ``spec/`` or the calibrations is changed; the numbers inform the open regime
decision (STATUS.md) and record the expected oracle-direct gap before any PPO run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from slicemech import io as sio  # noqa: E402

CAL = "n3_default"


def _static_fit():
    from slicemech.config import Mechanism
    from slicemech.control.static import fit_static
    from slicemech.data.calibrate import load_calibration, train_window_keys
    cal = load_calibration(CAL)
    wins = [cal.window("train", d, o) for d, o in train_window_keys()]
    fit = fit_static(wins, Mechanism(), steps=20)
    return fit.w, fit.welfare, fit.seconds


def _job(args):
    split, date, off, static_w, oracle_kw = args
    from slicemech.config import Mechanism
    from slicemech.data.calibrate import load_calibration
    from slicemech.eval.headroom import window_headroom
    cal = load_calibration(CAL)
    w = cal.window(split, date, off)
    t0 = time.perf_counter()
    row = window_headroom(w, cal.shape, Mechanism(), static_w=np.asarray(static_w), oracle_kw=oracle_kw)
    row.update(offset_s=off, seconds=time.perf_counter() - t0)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--train-windows", type=int, default=36, help="random TRAIN windows (0 = all 180)")
    ap.add_argument("--dev-windows", type=int, default=0, help="random DEV windows (0 = all 126)")
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261019)
    a = ap.parse_args(argv)
    sio.require_free_gb(3)
    from slicemech.data import compose, puffer
    from slicemech.eval.headroom import summarize

    t0 = time.perf_counter()
    static_w, static_W, static_s = _static_fit()
    print(f"TRAIN static fit: w = {np.round(static_w, 4).tolist()}, TRAIN welfare {static_W:.4f} "
          f"({static_s:.0f} s)", flush=True)
    rng = np.random.default_rng(a.seed)
    jobs = []
    for split, k in (("train", a.train_windows), ("dev", a.dev_windows)):
        keys = [(d, o) for d in puffer.split_dates(split) for o in compose.window_offsets(split)]
        if k and k < len(keys):
            keys = [keys[i] for i in sorted(rng.choice(len(keys), k, replace=False))]
        jobs += [(split, d, o, static_w.tolist(), dict(pop=a.pop, iters=a.iters, seed=0)) for d, o in keys]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_job, jobs, chunksize=1))
    summary = {s: summarize([r for r in rows if r["split"] == s]) for s in ("train", "dev")}
    out = dict(calibration=CAL, static_w=static_w.tolist(), static_train_welfare=static_W,
               oracle=dict(pop=a.pop, iters=a.iters, base="TRAIN static weights", horizon="auto"),
               summary=summary, rows=rows, seconds=time.perf_counter() - t0)
    path = sio.ROOT / "results" / "raw" / "diagnostics" / "headroom.json"
    sio.write_json(path, out)
    import pandas as pd
    pd.DataFrame(rows).to_csv(sio.confine(path.with_suffix(".csv")), index=False)
    print(json.dumps(summary, indent=1))
    print(path)


if __name__ == "__main__":
    main()

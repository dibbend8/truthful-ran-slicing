"""End-to-end smoke run of ``sim.episode.run_window`` on TRAIN data (integration gate; not an experiment).

Run from ``code/``:

    PYTHONDONTWRITEBYTECODE=1 /home/darpan/anaconda3/bin/python experiments/smoke_episode.py [--workers 12]

Two TRAIN windows at the default configuration (``data/calibration/n3_default.json``: n = 3, rho = 0.9, video lead
4.004 s, report lead 50 ms, K = 200 s, warm-up 40 s; measurement delay d = 2 intervals) are played by

* ``equal_full``          equal weights, full-decision price;
* ``protected_full``      protected map with random (a, kappa) per interval (a in [-4, 4], kappa in [0, 2]), full price;
* ``protected_fixed``     the same map billed with the fixed-priority price (manipulable when the premium is > 0);
* ``unconstrained_fixed`` randomly initialised unconstrained score network (CallableMap, not monotone), fixed price;
  its ``p`` column is the grid envelope formula (not truthful, uncertified);
* ``direct_full``         direct declared-welfare optimiser, full price;

each with truthful tenants and with tenant 1 (URLLC) replaced by an ex-post best responder to the billed price
(truthful opponents). Outputs (``results/raw/smoke/``): one parquet per run (episode table, ``record=True``),
``summary.csv`` (one row per run), ``pairs.csv`` (best responder vs truthful on the same window and arm; victim
harm from ``eval.metrics.victim_effects``: honest tenants' on-time Mbit, deadline miss and realized value, with
total realized welfare as a descriptive, tariff-weighted column) and ``checks.json`` (invariants: conservation, PRB
budget, certificates, full-price best-response gain).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from slicemech import io as sio  # noqa: E402

OUT = sio.ROOT / "results" / "raw" / "smoke"
WINDOWS = (("2024-01-01", 60), ("2024-01-14", 21660))      # two of the DATA_REPORT TRAIN report windows
ARMS = ("equal_full", "protected_full", "protected_fixed", "unconstrained_fixed", "direct_full")
PROFILES = ("truthful", "br_urllc")
BR_TENANT = 1
D = 2                                                         # measurement delay (intervals), params.yaml default


def _arm(name: str):
    from slicemech.control import nets
    from slicemech.sim import episode as E
    if name == "equal_full":
        return E.FixedWeights(), E.PriceRule("full")
    if name == "protected_full":
        return E.RandomProtected(seed=1), E.PriceRule("full")
    if name == "protected_fixed":
        return E.RandomProtected(seed=1), E.PriceRule("fixed")
    if name == "unconstrained_fixed":
        return nets.make_net("unconstrained", seed=0), E.PriceRule("fixed", nonmonotone_full=True)
    if name == "direct_full":
        return E.DirectController(), E.PriceRule("full")
    raise ValueError(name)


def _job(args):
    """One (window, arm, profile) run in a worker process; returns the summary row."""
    date, offset, arm, profile = args
    import torch
    torch.set_num_threads(1)
    from slicemech.config import Mechanism
    from slicemech.data.calibrate import load_calibration
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import BestResponder, Truthful

    cal = load_calibration("n3_default")
    mech = Mechanism()
    w = cal.window("train", date, offset)
    ctrl, pr = _arm(arm)
    pols = [Truthful() for _ in range(w.n)]
    if profile == "br_urllc":
        pols[BR_TENANT] = BestResponder(pr.charge)
    # the TRAIN-fitted operator forecaster (round-1 fix: no silent default for context-reading controllers)
    df = E.run_window(w, ctrl, pr, pols, D, mech, cal.shape, forecaster=cal.operator_forecaster(D))
    tag = f"{arm}__{profile}__{date}_o{offset:05d}"
    df.insert(0, "profile", profile)
    df.insert(0, "arm", arm)
    df.insert(0, "window", w.id)
    attrs = dict(df.attrs)
    df.attrs = {}
    df.to_parquet(OUT / f"{tag}.parquet", index=False, compression="zstd")
    chk = E.episode_checks(df, w.contract, mech)
    sc = df[df["scored"]]
    per_k = sc.groupby("k")
    row = dict(window=w.id, date=date, arm=arm, profile=profile, tag=tag, warmup=int(w.warmup),
               seconds=attrs["seconds"],
               W_realized=float(per_k["realized_value"].sum().mean()),
               W_declared=float(per_k["V_declared"].sum().mean()),
               revenue=float(per_k["price"].sum().mean()),
               idle_prb=float(mech.B - per_k["quota"].sum().mean()))
    for i in range(w.n):
        t = sc[sc["tenant"] == i]
        on, dr = float(t["on_time_Mbit"].sum()), float(t["dropped_Mbit"].sum())
        row.update({f"on_time_{i}": float(t["on_time_Mbit"].mean()), f"miss_pct_{i}": 100 * dr / (on + dr),
                    f"quota_{i}": float(t["quota"].mean()), f"price_{i}": float(t["price"].mean()),
                    f"T_{i}": float(t["T"].mean()), f"U_decl_{i}": float(t["utility_declared"].mean())})
    b = sc[sc["tenant"] == BR_TENANT]
    if profile == "br_urllc":
        g = b["br_gain"].to_numpy()
        row.update(br_gain_mean=float(g.mean()), br_gain_max=float(g.max()),
                   br_frac_misreport=float((b["report"] != b["theta_adm"]).mean()),
                   br_report_over_type=float((b["report"] / b["theta_adm"]).mean()))
    row.update({f"chk_{k}": v for k, v in chk.items()})
    row.update({f"t_{k}": v for k, v in attrs["timing_s"].items()})
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    sio.require_free_gb(3)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(d, o, a, p) for d, o in WINDOWS for a in ARMS for p in PROFILES]
    t0 = time.perf_counter()
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_job, jobs))
    summ = pd.DataFrame(rows).sort_values(["date", "arm", "profile"]).reset_index(drop=True)
    summ.to_csv(OUT / "summary.csv", index=False)
    # best responder vs truthful on the same window and arm. Harm is measured on the honest tenants' own outcomes
    # (on-time Mbit, deadline miss, realized value); total realized welfare is tariff-weighted and descriptive only
    # (round-2 review: URLLC carries pi = 3, so moving PRBs to it can raise total welfare while victims lose).
    from slicemech.eval.metrics import victim_effects
    pairs = []
    for (win, arm), g in summ.groupby(["window", "arm"]):
        tr, br = g[g["profile"] == "truthful"].iloc[0], g[g["profile"] == "br_urllc"].iloc[0]
        ve = victim_effects(pd.read_parquet(OUT / f"{tr['tag']}.parquet"),
                            pd.read_parquet(OUT / f"{br['tag']}.parquet"), BR_TENANT, int(tr["warmup"]))
        pairs.append(dict(window=win, arm=arm, dU_decl_br=br[f"U_decl_{BR_TENANT}"] - tr[f"U_decl_{BR_TENANT}"],
                          br_gain_mean=br["br_gain_mean"], br_gain_max=br["br_gain_max"],
                          br_frac_misreport=br["br_frac_misreport"],
                          d_on_time_br=br[f"on_time_{BR_TENANT}"] - tr[f"on_time_{BR_TENANT}"],
                          d_on_time_victims=ve["d_on_time_victims"], d_miss_pct_victims=ve["d_miss_pct_victims"],
                          d_value_victims=ve["d_value_victims"], d_value_br=ve["d_value_deviator"],
                          dW_realized=br["W_realized"] - tr["W_realized"],
                          dW_realized_pct=100 * (br["W_realized"] - tr["W_realized"]) / tr["W_realized"]))
    pd.DataFrame(pairs).to_csv(OUT / "pairs.csv", index=False)
    chk_cols = [c for c in summ.columns if c.startswith("chk_")]
    worst = {c[4:]: float(np.nanmax(summ[c].to_numpy(dtype=float))) for c in chk_cols}
    sio.write_json(OUT / "checks.json", dict(worst=worst, runs=len(summ), wall_s=time.perf_counter() - t0,
                                             windows=[f"{d}+{o}" for d, o in WINDOWS], d=D))
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(summ[["date", "arm", "profile", "seconds", "W_realized", "revenue", "idle_prb"]])
        print(pd.DataFrame(pairs))
        print(json.dumps(worst, indent=1))


if __name__ == "__main__":
    main()

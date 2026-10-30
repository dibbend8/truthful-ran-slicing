"""Pricing benchmark (not a test): wall time of the envelope prices of ALL tenants for one interval.

Run from ``code/``:

    PYTHONDONTWRITEBYTECODE=1 /home/darpan/anaconda3/bin/python tests/bench_pricing.py [--intervals 200] [--seed 0]

Instances mimic the deployed setting: contracts from ``spec/params.yaml`` (classes cycled video/urllc/mmtc, so
n = 3, 6, 9 hold 1, 2, 3 tenants per class), the default mechanism (B = 106 PRB, gamma = 0.2, eta = 2, integer
floors), Raca-like per-PRB rates ``r = SE * 132 * 1000 * 0.9 / 1e6`` Mbit/s per PRB with SE uniform on the
TS 38.214 64QAM table range, offered load 0.9 of the mean cell capacity split 0.6/0.3/0.1 over the classes,
64-atom quantile burst shapes, a random protected map (a in [-4, 4], kappa in [0, 2]) and log-normal reports
around the mean demand.  Per interval the script times (median and 95th percentile over intervals, ms):

* ``full_prices``      - full-decision prices of all n tenants in one batched threshold pass;
* ``n x full_price``   - the same as n separate calls (the plain per-tenant interface);
* ``fixed_prices``     - fixed-priority prices of all n tenants (one batched pass), needed for the premium T;
* ``BR full / fixed``  - one exact best response (tenant 0, type = its report) under the full and the
  fixed-priority price (the cost of a ``BestResponder`` tenant per interval).

It also reports the mean number of own-quota levels priced per tenant and the worst certificate width
``err`` relative to ``pi Delta C`` (bracket tolerance 2**-40 C).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slicemech.config import PARAMS, Mechanism, Shape, contract_from_params  # noqa: E402
from slicemech.model.allocation import QuotaRule  # noqa: E402
from slicemech.model.scores import ProtectedMap  # noqa: E402
from slicemech.model.shapes import fit_shape  # noqa: E402
from slicemech.pricing.envelope import fixed_priority_prices, full_price, full_prices  # noqa: E402
from slicemech.pricing.response import best_response  # noqa: E402

CLASSES = ("video", "urllc", "mmtc")
SE_RANGE = (0.1523, 5.5547)                     # TS 38.214 Table 5.2.2.1-2 (64QAM), CQI 1 .. 15 (bit/s/Hz)
_BURST_SHAPE = {"video": 1.5, "urllc": 0.6, "mmtc": 0.3}   # gamma shape of the synthetic sub-window rates


def _instance(rng: np.random.Generator, n: int, mech: Mechanism, shapes: dict) -> tuple:
    """One interval: contract, rates (Mbit/s per PRB), shape, rule and reports (Mbit/s)."""
    cls = [CLASSES[k % 3] for k in range(n)]
    se = rng.uniform(*SE_RANGE, n)
    r = se * PARAMS["network"]["re_per_prb_slot"] * 1000 * (1 - PARAMS["network"]["bler"]) / 1e6
    shares = PARAMS["load"]["shares"]
    per_class = {c: cls.count(c) for c in CLASSES}
    capacity = mech.B * float(np.mean(r))                                       # Mbit/s
    mu = np.array([PARAMS["load"]["rho"] * capacity * shares[c] / per_class[c] for c in cls])
    ct = contract_from_params(cls, mu)
    shape = Shape(beta=tuple(shapes[c][0] for c in cls), rho=tuple(shapes[c][1] for c in cls))
    pm = ProtectedMap(rng.uniform(-4, 4, n), rng.uniform(0, 2, n), mu)
    rule = QuotaRule(ct, mech, r, pm)
    z = np.minimum(mu * rng.lognormal(0.0, 0.5, n), ct.C)
    return ct, r, shape, rule, z


def _ms(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return 1e3 * (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--intervals", type=int, default=200, help="random intervals per n")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, nargs="+", default=[3, 6, 9])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mech = Mechanism()
    shapes = {c: fit_shape(rng.gamma(k, 1.0 / k, 20_000), n_atoms=64) for c, k in _BURST_SHAPE.items()}

    rows = []
    for n in args.n:
        t_batch, t_loop, t_fix, t_brf, t_brx, levels, rel_err = [], [], [], [], [], [], []
        for k in range(args.intervals + 3):
            ct, r, shape, rule, z = _instance(rng, n, mech, shapes)
            out = {}
            tb = _ms(lambda: out.__setitem__("full", full_prices(z, rule, ct, shape, mech, r)))
            tl = _ms(lambda: [full_price(i, z, rule, ct, shape, mech, r) for i in range(n)])
            tf = _ms(lambda: fixed_priority_prices(z, rule, ct, shape, mech, r))
            kw = dict(ct=ct, shape=shape, mech=mech, r=r)
            tbf = _ms(lambda: best_response(0, z[0], z[1:], rule, "full", **kw))
            tbx = _ms(lambda: best_response(0, z[0], z[1:], rule, "fixed", **kw))
            if k < 3:                                                          # warm-up (imports, caches)
                continue
            t_batch.append(tb)
            t_loop.append(tl)
            t_fix.append(tf)
            t_brf.append(tbf)
            t_brx.append(tbx)
            levels.append(np.mean([p.n_levels for p in out["full"]]))
            rel_err.append(max(p.err / (ct.pi[i] * mech.Delta * ct.C[i]) for i, p in enumerate(out["full"])))
        rows.append((n, t_batch, t_loop, t_fix, t_brf, t_brx, float(np.mean(levels)), float(np.max(rel_err))))

    q = lambda a, p: float(np.percentile(a, p))                                # noqa: E731
    print(f"envelope pricing per interval, all tenants ({args.intervals} intervals per n, B = {mech.B} PRB, "
          f"integer quotas, tol = 2^-40 C, 64 atoms)")
    head = (f"{'n':>3} | {'full_prices ms':>17} | {'n x full_price ms':>19} | {'fixed_prices ms':>17} | "
            f"{'levels/tenant':>13} | {'max err/(pi D C)':>16}")
    print(head)
    print(f"{'':>3} | {'median':>8} {'p95':>8} | {'median':>9} {'p95':>9} | {'median':>8} {'p95':>8} | "
          f"{'':>13} | {'':>16}")
    print("-" * len(head))
    for n, tb, tl, tf, _, _, lev, err in rows:
        print(f"{n:>3} | {q(tb, 50):8.2f} {q(tb, 95):8.2f} | {q(tl, 50):9.2f} {q(tl, 95):9.2f} | "
              f"{q(tf, 50):8.2f} {q(tf, 95):8.2f} | {lev:13.1f} | {err:16.2e}")
    print()
    print("exact best response of one tenant per interval (ms)")
    print(f"{'n':>3} | {'BR full median':>14} {'p95':>8} | {'BR fixed median':>15} {'p95':>8}")
    for n, _, _, _, bf, bx, _, _ in rows:
        print(f"{n:>3} | {q(bf, 50):14.2f} {q(bf, 95):8.2f} | {q(bx, 50):15.2f} {q(bx, 95):8.2f}")


if __name__ == "__main__":
    main()

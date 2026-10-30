"""Welfare-headroom regime diagnostic (round-1 review, finding 4; experiment_design.md §9 gate G4).

The five regime targets of ``data.calibrate.regime_diagnostics`` count *integer-quota* changes; they can all pass
while priority weights barely move realized welfare. This module measures the quantity gate G4 is about, in credits
per scored interval, with truthful reports ``z = min(theta, C)`` and the stale rate ``rate_hat`` for every arm:

* ``W_equal``   : equal weights;
* ``W_static``  : the TRAIN-fitted static weights (``control.static.fit_static`` on TRAIN windows only);
* ``W_grid``    : the best of a floored simplex grid of fixed weights *on this window* (in-sample, an optimistic
                  upper reference for static priorities; n = 3 only);
* ``W_direct``  : the direct optimiser of declared welfare (``model.direct.DirectRule``);
* ``W_oracle``  : the lookahead hindsight oracle (``control.oracle.oracle_window``, base policy = the static
                  weights when given, else equal weights);

plus the window's offered load ``sum_i mean m_i / (B mean_k mean_i rate_true)``. The G4 statistic is
``(W_oracle - max(W_static, W_direct)) / W_equal``; :func:`summarize` adds the share of windows on which the best
grid vector gains less than 0.5 % over equal weights and the correlation of that gain with the load.

All welfare values are realized: ``sum_i pi_i on_time_Mbit_i`` from the fluid replay with TRUE arrivals and rates.
"""
from __future__ import annotations

import numpy as np

from ..config import Mechanism, Shape

__all__ = ["offered_load", "window_load", "window_headroom", "summarize"]


def offered_load(window, r_bar: float, mech: Mechanism | None = None, scored_only: bool = False) -> float:
    """D2 offered load of a window: mean over intervals of the summed measured demand / (B x r_bar_TRAIN)
    (dimensionless; exogenous). The default averages over ALL K intervals, the quantity every driver writes as
    ``offered_load`` and the analysis uses as the D3 subgroup variable; ``scored_only`` gives ``offered_load_scored``
    (round-3 review: drivers used one name for both averaging periods)."""
    mech = mech or Mechanism()
    k0 = int(window.warmup) if scored_only else 0
    m = np.asarray(window.measured, float)[k0:]
    return float(m.mean(0).sum() / (mech.B * float(r_bar)))       # E0/E1 order of operations


def window_load(window, mech: Mechanism | None = None, scored_only: bool = True) -> float:
    """Offered load of a window: mean total measured demand / (B x mean true per-PRB rate) (dimensionless).

    An exogenous, outcome-independent window property, so it can define a pre-registered subgroup.
    """
    mech = mech or Mechanism()
    k0 = int(window.warmup) if scored_only else 0
    m = np.asarray(window.measured, float)[k0:]
    r = np.asarray(window.rate_true, float)[k0:]
    return float(m.sum(1).mean() / (mech.B * r.mean()))


def _welfare(window, Q, mech) -> float:
    from ..sim import fluid
    ct = window.contract
    out, _, _ = fluid.replay(fluid.cumulative_arrivals(window.arrivals_ms), np.asarray(Q, float),
                             np.asarray(window.rate_true, float), ct.delay_ms,
                             slots_per_interval=int(round(mech.Delta * fluid.SLOTS_PER_S)))
    return float((out.on_time_Mbit[int(window.warmup):] @ ct.pi).mean())


def window_headroom(window, shape: Shape, mech: Mechanism | None = None, *, static_w=None, grid_steps: int = 10,
                    grid_floor: float = 0.05, oracle_kw: dict | None = None, horizon: int | None = None) -> dict:
    """Welfare of the reference arms on one window (credits per scored interval; see the module docstring).

    Args:
        window: composed ``data.compose.Window`` of any split (the diagnostic only reads it).
        shape: public burst shapes (direct rule).
        static_w: (n,) TRAIN-fitted static weights, or None (then ``W_static`` is NaN and the oracle continues with
            equal weights).
        grid_steps, grid_floor: in-sample simplex grid (n = 3); ``grid_steps=0`` skips it.
        oracle_kw: extra arguments of ``control.oracle.oracle_window`` (population, iterations, seed).
        horizon: oracle lookahead (None: auto).
    """
    from ..control import oracle as _oracle
    from ..control import ppo as _ppo
    from ..control import static as _static
    from ..model import allocation
    from ..model.direct import DirectRule
    mech = mech or Mechanism()
    ct = window.contract
    n = ct.n
    z = np.minimum(np.asarray(window.theta, float), ct.C)
    r = np.asarray(window.rate_hat, float)
    q_fn = allocation.integer if mech.integer else allocation.fractional
    out = dict(window_id=getattr(window, "id", None), split=getattr(window, "split", None),
               date=getattr(window, "date", None), load=window_load(window, mech))
    out["W_equal"] = _welfare(window, q_fn(z, np.ones(n), r, ct, mech), mech)
    out["W_static"] = (float("nan") if static_w is None
                       else _welfare(window, q_fn(z, np.asarray(static_w, float), r, ct, mech), mech))
    if grid_steps and n == 3:
        G = _static.simplex_grid(3, int(grid_steps), float(grid_floor))
        data = _ppo.prepare([window], mech=mech)
        S = _static.rollout_fixed(data, G, mech=mech, env_factory=_ppo.fluid_env_factory,
                                  quota_fn=_ppo.default_quota_fn)
        out["W_grid"] = float(S.max())
    else:
        out["W_grid"] = float("nan")
    Qd = np.stack([DirectRule(ct, mech, r[k], shape).quotas(z[k][None, :])[0] for k in range(len(z))])
    out["W_direct"] = _welfare(window, Qd, mech)
    res = _oracle.oracle_window(window, mech, horizon=horizon, base_weights=static_w, **(oracle_kw or {}))
    out["W_oracle"] = res.mean_welfare
    out["oracle_horizon"] = res.horizon
    best_ref = float(np.nanmax([out["W_static"], out["W_direct"]]))
    out["G4_gap"] = float((out["W_oracle"] - best_ref) / out["W_equal"])
    out["oracle_gain"] = out["W_oracle"] / out["W_equal"] - 1.0
    out["direct_gain"] = out["W_direct"] / out["W_equal"] - 1.0
    out["grid_gain"] = out["W_grid"] / out["W_equal"] - 1.0
    out["static_gain"] = out["W_static"] / out["W_equal"] - 1.0
    return out


def summarize(rows, small_gain: float = 0.005) -> dict:
    """Pooled summary of :func:`window_headroom` rows (one split): means of the gains (fractions of ``W_equal``),
    the mean G4 gap, the share of windows whose in-sample grid gain is below ``small_gain`` and corr(load, gain)."""
    rows = list(rows)
    if not rows:
        raise ValueError("no rows")
    f = {k: np.array([r[k] for r in rows], float) for k in ("load", "G4_gap", "oracle_gain", "direct_gain",
                                                              "grid_gain", "static_gain")}
    grid = f["grid_gain"]
    ok = np.isfinite(grid)
    return dict(windows=len(rows), mean_load=float(f["load"].mean()), median_load=float(np.median(f["load"])),
                mean_G4_gap=float(f["G4_gap"].mean()), mean_oracle_gain=float(f["oracle_gain"].mean()),
                mean_direct_gain=float(f["direct_gain"].mean()),
                mean_static_gain=float(np.nanmean(f["static_gain"])) if np.isfinite(f["static_gain"]).any()
                else float("nan"),
                mean_grid_gain=float(grid[ok].mean()) if ok.any() else float("nan"),
                median_grid_gain=float(np.median(grid[ok])) if ok.any() else float("nan"),
                share_grid_gain_below=float((grid[ok] < small_gain).mean()) if ok.any() else float("nan"),
                small_gain=small_gain,
                corr_load_grid_gain=(float(np.corrcoef(f["load"][ok], grid[ok])[0, 1]) if ok.sum() > 2
                                     else float("nan")))

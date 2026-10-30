"""Physical-unit metrics per interval, tenant and window (INTERFACES.md §10).

Input is the episode table of ``sim.episode.run_window`` (INTERFACES.md §11): one row per (interval k, tenant) with at
least the columns in ``EPISODE_COLUMNS``. Units:

* on_time_Mbit, dropped_Mbit: Mbit in the interval (served before the deadline / dropped at the deadline);
* quota: PRB (integer quota held for the whole interval);
* p, p_fix, T, V_declared, realized_value, utility_*: credits per interval;
* realized_value = pi_i * on_time_Mbit (credits per interval); realized welfare W_k = sum_i realized_value.

Deadline-miss percentage of an interval: 100 * dropped / (on_time + dropped), the share of the bits whose service
resolved in the interval (served or expired) that missed their deadline; undefined (NaN) when nothing resolved.
Window aggregates weight it by volume: 100 * sum dropped / sum(on_time + dropped) over the scored intervals.

Scored intervals are k >= warmup (the window's warm-up is excluded from every aggregate).

Manipulation harm (:func:`victim_effects`; round-2 review): when one tenant deviates, the harm to the honest tenants
is measured on their own outcomes - on-time Mbit, deadline-miss % and their realized value
``sum_{j != i} pi_j on_time_j`` - and not on total realized welfare. Tariffs differ by class (URLLC pi = 3 against
1 for video and 0.5 for mMTC), so PRBs moved from honest low-tariff tenants to a high-tariff manipulator can raise
total realized welfare while the honest tenants lose service; total welfare is reported as a descriptive column.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

EPISODE_COLUMNS = ("k", "tenant", "cls", "theta", "report", "quota", "p", "V_declared", "on_time_Mbit",
                   "dropped_Mbit", "realized_value", "utility_declared", "utility_realized")
INTERVAL_COLUMNS = ("k", "tenant", "cls", "theta", "report", "quota", "price", "V_declared", "on_time_Mbit",
                    "dropped_Mbit", "deadline_miss_pct", "realized_value", "utility_declared", "utility_realized")


def _price_col(df: pd.DataFrame, price_col: str | None) -> str:
    """Resolve the charged-price column: explicit name, else ``'price'`` (the billed price written by
    ``sim.episode.run_window``, consistent with its utility columns) if present, else ``'p'``."""
    if price_col is not None:
        return price_col
    return "price" if "price" in df.columns else "p"


def _require(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"episode table lacks columns {missing}")


def deadline_miss_pct(on_time, dropped) -> np.ndarray:
    """100 * dropped / (on_time + dropped) elementwise (Mbit in, percent out); NaN where nothing resolved."""
    on_time = np.asarray(on_time, dtype=float)
    dropped = np.asarray(dropped, dtype=float)
    if np.any(on_time < 0) or np.any(dropped < 0):
        raise ValueError("on-time and dropped volumes must be >= 0 (Mbit)")
    tot = on_time + dropped
    out = np.full(tot.shape, np.nan)
    np.divide(100.0 * dropped, tot, out=out, where=tot > 0)
    return out


def interval_metrics(df: pd.DataFrame, price_col: str | None = None,
                     pi: Mapping[int, float] | Sequence[float] | None = None,
                     rtol: float = 1e-9) -> pd.DataFrame:
    """Per (interval, tenant) metrics table (columns ``INTERVAL_COLUMNS``).

    Args:
        df: episode table (one row per interval and tenant).
        price_col: the charged price column ('p' full price, 'p_fix' fixed-priority price, or a usage-price column);
            default: 'price' (the billed price of an episode table) if present, else 'p'.
        pi: optional tariffs per tenant index (credits/Mbit); if given, realized_value is checked to equal
            pi * on_time_Mbit within ``rtol`` (raises otherwise).

    Returns:
        DataFrame sorted by (k, tenant).
    """
    _require(df, EPISODE_COLUMNS)
    price_col = _price_col(df, price_col)
    _require(df, (price_col,))
    out = df.loc[:, [c for c in EPISODE_COLUMNS]].copy()
    out["price"] = df[price_col].to_numpy(dtype=float)
    out["deadline_miss_pct"] = deadline_miss_pct(df["on_time_Mbit"], df["dropped_Mbit"])
    if pi is not None:
        tenants = out["tenant"].to_numpy(dtype=int)
        if isinstance(pi, Mapping):
            missing = sorted(set(tenants.tolist()) - set(pi))
            if missing:
                raise KeyError(f"pi has no tariff for tenant(s) {missing}")
            pi_t = np.array([float(pi[t]) for t in tenants])
        else:
            pi_arr = np.asarray(pi, dtype=float)
            if pi_arr.ndim != 1 or tenants.max() >= pi_arr.size:
                raise ValueError(f"pi must be a 1-D array covering tenants 0..{tenants.max()}")
            pi_t = pi_arr[tenants]
        expect = pi_t * out["on_time_Mbit"].to_numpy(dtype=float)
        got = out["realized_value"].to_numpy(dtype=float)
        if not np.allclose(got, expect, rtol=rtol, atol=1e-12):
            bad = int(np.argmax(np.abs(got - expect)))
            raise ValueError(f"realized_value != pi * on_time_Mbit (row {bad}: {got[bad]} vs {expect[bad]})")
    return out.loc[:, list(INTERVAL_COLUMNS)].sort_values(["k", "tenant"]).reset_index(drop=True)


def interval_welfare(df: pd.DataFrame, price_col: str | None = None, B: int | None = None) -> pd.DataFrame:
    """Per-interval totals over tenants.

    Returns DataFrame indexed by k with W_realized (sum pi * on-time Mbit, credits), W_declared (sum V_declared,
    credits), revenue (sum of the charged price, credits), on_time_Mbit, dropped_Mbit, quota_total (PRB) and,
    if B is given, idle_prb = B - quota_total (PRB; must be >= 0).
    """
    _require(df, EPISODE_COLUMNS)
    price_col = _price_col(df, price_col)
    _require(df, (price_col,))
    g = df.groupby("k", sort=True)
    out = pd.DataFrame({
        "W_realized": g["realized_value"].sum(),
        "W_declared": g["V_declared"].sum(),
        "revenue": g[price_col].sum(),
        "on_time_Mbit": g["on_time_Mbit"].sum(),
        "dropped_Mbit": g["dropped_Mbit"].sum(),
        "quota_total": g["quota"].sum(),
    })
    if B is not None:
        out["idle_prb"] = B - out["quota_total"]
        if np.any(out["idle_prb"] < 0):
            raise ValueError("quotas exceed the PRB budget B in some interval")
    return out


def _aggregate(block: pd.DataFrame, n_intervals: int) -> dict:
    """Window aggregates of one tenant (or of the per-interval tenant sums) over the scored intervals."""
    on, dr = float(block["on_time_Mbit"].sum()), float(block["dropped_Mbit"].sum())
    per_k_drop = block.groupby("k")["dropped_Mbit"].sum()
    return dict(
        n_intervals=n_intervals,
        on_time_Mbit=on / n_intervals,                 # mean Mbit per interval
        dropped_Mbit=dr / n_intervals,
        deadline_miss_pct=(100.0 * dr / (on + dr)) if on + dr > 0 else np.nan,
        intervals_with_drop_pct=100.0 * float((per_k_drop > 0).mean()),
        quota=float(block["quota"].sum()) / n_intervals,
        price=float(block["price"].sum()) / n_intervals,
        V_declared=float(block["V_declared"].sum()) / n_intervals,
        realized_value=float(block["realized_value"].sum()) / n_intervals,
        utility_declared=float(block["utility_declared"].sum()) / n_intervals,
        utility_realized=float(block["utility_realized"].sum()) / n_intervals,
        utility_declared_total=float(block["utility_declared"].sum()),
        utility_realized_total=float(block["utility_realized"].sum()),
    )


def window_summary(df: pd.DataFrame, warmup: int, price_col: str | None = None, keys: Mapping[str, object] | None = None
                   ) -> pd.DataFrame:
    """Window-level summary over the scored intervals k >= warmup.

    One row per tenant plus a row with tenant = -1 and cls = 'all' (per-interval sums over tenants; its
    ``realized_value`` is the mean realized welfare per interval and ``V_declared`` the mean declared welfare).
    Means are per interval (Mbit per interval, PRB, credits per interval); ``*_total`` columns sum over the window.

    Args:
        df: episode table of one window and arm.
        warmup: number of leading warm-up intervals to exclude.
        price_col: charged price column (default as in :func:`interval_metrics`).
        keys: constant identifier columns to prepend (e.g. dict(window=..., date=..., arm=...)).
    """
    m = interval_metrics(df, price_col=price_col)
    m = m[m["k"] >= warmup]
    ks = m["k"].unique()
    if ks.size == 0:
        raise ValueError(f"no scored intervals (warmup = {warmup})")
    rows = []
    for (t, c), block in m.groupby(["tenant", "cls"], sort=True):
        if block["k"].nunique() != ks.size:
            raise ValueError(f"tenant {t} is missing scored intervals")
        rows.append(dict(tenant=int(t), cls=c, **_aggregate(block, ks.size)))
    rows.append(dict(tenant=-1, cls="all", **_aggregate(m, ks.size)))
    out = pd.DataFrame(rows)
    for j, (name, val) in enumerate((keys or {}).items()):
        out.insert(j, name, val)
    return out


def paired_table(summaries: pd.DataFrame, metric: str, arm_a: str, arm_b: str, tenant: int = -1,
                 arm_col: str = "arm", unit_cols: Sequence[str] = ("window", "date")) -> pd.DataFrame:
    """Per-window paired comparison of two arms.

    Args:
        summaries: concatenated ``window_summary`` tables with identifier columns ``unit_cols`` and ``arm_col``.
        metric: summary column to compare (e.g. 'realized_value' with tenant=-1 for realized welfare).
        arm_a, arm_b: arms; the difference is a - b.
        tenant: tenant index (-1 = all tenants).

    Returns:
        DataFrame with unit_cols, a, b, diff, pct_of_b (= 100 * diff / b; NaN where b == 0). Raises if the arms are
        not observed on the same windows (the comparison must be paired).
    """
    _require(summaries, (*unit_cols, arm_col, "tenant", metric))
    s = summaries[summaries["tenant"] == tenant]
    a = s[s[arm_col] == arm_a].set_index(list(unit_cols))[metric]
    b = s[s[arm_col] == arm_b].set_index(list(unit_cols))[metric]
    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("duplicate windows within an arm")
    if set(a.index) != set(b.index):
        raise ValueError(f"arms {arm_a!r} and {arm_b!r} are not observed on the same windows")
    b = b.loc[a.index]
    out = pd.DataFrame({"a": a.to_numpy(float), "b": b.to_numpy(float)}, index=a.index).reset_index()
    out["diff"] = out["a"] - out["b"]
    out["pct_of_b"] = percent_of(out["diff"], out["b"], allow_zero=True)
    return out


def victim_effects(truth: pd.DataFrame, deviation: pd.DataFrame, deviator: int, warmup: int) -> dict:
    """Paired effect of one tenant's deviation on the honest tenants (same window and arm; scored intervals).

    Args:
        truth, deviation: episode tables of the truthful run and of the run in which tenant ``deviator`` deviates
            (every other tenant truthful); both must cover the same (k, tenant) rows.
        deviator: index of the deviating tenant; warmup: leading intervals excluded (k >= warmup are scored).

    Returns:
        dict of per-interval means (deviation minus truth): ``d_on_time_victims`` and ``d_dropped_victims`` (Mbit
        per interval, summed over honest tenants), ``miss_pct_victims_truth`` / ``_deviation`` and
        ``d_miss_pct_victims`` (volume-weighted deadline-miss %, percentage points), ``d_value_victims``
        (honest tenants' realized value ``sum_{j != i} pi_j on_time_j``, credits per interval; the harm measure),
        ``d_value_deviator`` and ``dW_realized`` (= the sum of the two; descriptive, tariff-weighted), and
        ``n_intervals``.
    """
    cols = ("k", "tenant", "on_time_Mbit", "dropped_Mbit", "realized_value")
    _require(truth, cols)
    _require(deviation, cols)
    a = truth[truth["k"] >= warmup].set_index(["k", "tenant"]).sort_index()
    b = deviation[deviation["k"] >= warmup].set_index(["k", "tenant"]).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("truthful and deviation runs must cover the same (k, tenant) rows")
    tenants = a.index.get_level_values("tenant")
    if deviator not in set(tenants):
        raise ValueError(f"deviator {deviator} is not a tenant of the table")
    n_k = int(a.index.get_level_values("k").nunique())
    if n_k == 0:
        raise ValueError(f"no scored intervals (warmup = {warmup})")
    vic = tenants != deviator

    def tot(df, col, mask):
        return float(df[col].to_numpy(float)[mask].sum())

    out = dict(n_intervals=n_k)
    for name, col in (("on_time", "on_time_Mbit"), ("dropped", "dropped_Mbit")):
        out[f"d_{name}_victims"] = (tot(b, col, vic) - tot(a, col, vic)) / n_k
    for tag, df in (("truth", a), ("deviation", b)):
        on, dr = tot(df, "on_time_Mbit", vic), tot(df, "dropped_Mbit", vic)
        out[f"miss_pct_victims_{tag}"] = 100.0 * dr / (on + dr) if on + dr > 0 else np.nan
    out["d_miss_pct_victims"] = out["miss_pct_victims_deviation"] - out["miss_pct_victims_truth"]
    out["d_value_victims"] = (tot(b, "realized_value", vic) - tot(a, "realized_value", vic)) / n_k
    out["d_value_deviator"] = (tot(b, "realized_value", ~vic) - tot(a, "realized_value", ~vic)) / n_k
    out["dW_realized"] = out["d_value_victims"] + out["d_value_deviator"]
    return out


def percent_of(num, den, allow_zero: bool = False) -> np.ndarray:
    """100 * num / den with a named denominator; raises on den == 0 unless allow_zero (then NaN)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    if np.any(den == 0) and not allow_zero:
        raise ZeroDivisionError("denominator is zero")
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(100.0 * num, den, out=out, where=den != 0)
    return out


def from_arrays(on_time, dropped, quota, price, V_declared, pi, theta=None, report=None, cls=None, k0: int = 0
                ) -> pd.DataFrame:
    """Build an episode-format table from (K, n) arrays (Mbit, PRB, credits) and tariffs pi (n,) credits/Mbit.

    realized_value = pi * on_time; utility_declared = V_declared - price; utility_realized = realized_value - price.
    """
    on_time = np.asarray(on_time, dtype=float)
    K, n = on_time.shape
    arrs = dict(dropped=dropped, quota=quota, price=price, V_declared=V_declared)
    arrs = {k: np.asarray(v, dtype=float) for k, v in arrs.items()}
    for name, a in arrs.items():
        if a.shape != (K, n):
            raise ValueError(f"{name} must have shape {(K, n)}, got {a.shape}")
    pi = np.asarray(pi, dtype=float)
    if pi.shape != (n,):
        raise ValueError(f"pi must have shape ({n},)")
    theta = np.full((K, n), np.nan) if theta is None else np.asarray(theta, dtype=float)
    report = theta if report is None else np.asarray(report, dtype=float)
    cls = tuple(cls) if cls is not None else tuple(f"t{j}" for j in range(n))
    realized = pi[None, :] * on_time
    k = np.repeat(np.arange(k0, k0 + K), n)
    return pd.DataFrame({
        "k": k, "tenant": np.tile(np.arange(n), K), "cls": np.tile(np.array(cls, dtype=object), K),
        "theta": theta.ravel(), "report": report.ravel(), "quota": arrs["quota"].ravel(),
        "p": arrs["price"].ravel(), "V_declared": arrs["V_declared"].ravel(), "on_time_Mbit": on_time.ravel(),
        "dropped_Mbit": arrs["dropped"].ravel(), "realized_value": realized.ravel(),
        "utility_declared": (arrs["V_declared"] - arrs["price"]).ravel(),
        "utility_realized": (realized - arrs["price"]).ravel(),
    })

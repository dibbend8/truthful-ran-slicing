"""Statistics for paired window-level comparisons (INTERFACES.md §10; experiment_design.md §7).

Unit of analysis: the window (paired across arms on identical compositions). Cluster: the date.

* ``cluster_bootstrap``: two-stage cluster bootstrap. Each replicate draws G dates with replacement, then, inside every
  drawn date, as many windows as that date has, with replacement; the replicate statistic is computed from the pooled
  resampled windows. 10,000 replicates, percentile confidence interval (default 95%).
  ``cluster_bootstrap_multi`` evaluates several statistics on the same resamples (ratio-of-sums tables, the power
  simulation's rule and p-value).
* ``sign_counts``: per-date sign of the date-mean effect.
* ``holm``: Holm step-down adjustment across the statistical primaries (H1a, H1c, H4a, H5b).
* ``crossed_bootstrap`` + ``window_memberships``: sensitivity analysis for the reuse of exogenous sources across
  dates (round-1 review, finding 1). Every TEST date reuses the same few TRACTOR captures and Raca runs, so dates are
  not independent; the crossed (pigeonhole, Owen 2007) bootstrap resamples the levels of every factor (date, URLLC
  capture, mMTC capture, CQI run) independently and weights each window by the product over factors of its
  membership-weighted level multiplicities. Report it next to the date-cluster bootstrap.

Everything is vectorised over replicates (processed in chunks to bound memory) and deterministic given ``seed``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

ALTERNATIVES = ("greater", "less", "two-sided")


# --------------------------------------------------------------------------- grouping
def _group(values, dates) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort windows by date. Returns (v (W, m) sorted, unique dates (G,), counts (G,), offsets (G,), order (W,))."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    if v.ndim != 2 or v.shape[0] == 0:
        raise ValueError(f"values must have shape (W,) or (W, m) with W >= 1, got {np.shape(values)}")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite (drop or impute missing windows explicitly)")
    d = np.asarray(dates)
    if d.shape != (v.shape[0],):
        raise ValueError(f"dates must have shape ({v.shape[0]},), got {d.shape}")
    uniq, inv = np.unique(d, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=len(uniq))
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return v[order], uniq, counts, offsets, order


def date_means(values, dates) -> pd.DataFrame:
    """Per-date mean of ``values`` (W,) with the number of windows: columns date, mean, n_windows."""
    v, uniq, counts, offsets, _ = _group(values, dates)
    if v.shape[1] != 1:
        raise ValueError("date_means / sign_counts need one value per window (reduce multi-column values first)")
    sums = np.add.reduceat(v[:, 0], offsets)
    return pd.DataFrame({"date": uniq, "mean": sums / counts, "n_windows": counts})


# --------------------------------------------------------------------------- bootstrap
@dataclass(frozen=True)
class BootResult:
    """Result of a cluster bootstrap.

    Attributes:
        estimate: statistic on the observed windows; lo, hi: percentile CI at ``level``;
        reps (R,): bootstrap replicates; n_dates, n_windows: sample sizes.
    """

    estimate: float
    lo: float
    hi: float
    level: float
    reps: np.ndarray
    n_dates: int
    n_windows: int

    def p_value(self, null: float = 0.0, alternative: str = "greater") -> float:
        """Percentile-bootstrap p-value (see ``bootstrap_pvalue``)."""
        return bootstrap_pvalue(self.reps, null, alternative)

    def upper(self, level: float = 0.95) -> float:
        """One-sided upper confidence bound (the ``level`` quantile of the replicates)."""
        return float(np.quantile(self.reps, level))

    def lower(self, level: float = 0.95) -> float:
        """One-sided lower confidence bound (the 1 - ``level`` quantile of the replicates)."""
        return float(np.quantile(self.reps, 1.0 - level))

    def as_dict(self) -> dict:
        return dict(estimate=self.estimate, lo=self.lo, hi=self.hi, level=self.level, n_dates=self.n_dates,
                    n_windows=self.n_windows, n_boot=int(self.reps.size))


def bootstrap_pvalue(reps, null: float = 0.0, alternative: str = "greater") -> float:
    """p-value from bootstrap replicates by inverting the percentile interval.

    'greater' (H1: effect > null): (1 + #{rep <= null}) / (R + 1); 'less': (1 + #{rep >= null}) / (R + 1);
    'two-sided': min(1, 2 * min of the two one-sided values).
    """
    reps = np.asarray(reps, dtype=float)
    R = reps.size
    if R == 0:
        raise ValueError("no replicates")
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}")
    pg = (1 + np.sum(reps <= null)) / (R + 1)
    pl = (1 + np.sum(reps >= null)) / (R + 1)
    if alternative == "greater":
        return float(pg)
    if alternative == "less":
        return float(pl)
    return float(min(1.0, 2 * min(pg, pl)))


def _identity_stat(means: np.ndarray) -> np.ndarray:
    if means.shape[-1] != 1:
        raise ValueError("multi-column values need an explicit stat function")
    return means[..., 0]


def cluster_bootstrap(values, dates, n_boot: int = 10_000, level: float = 0.95, seed: int = 0,
                      stat: Callable[[np.ndarray], np.ndarray] | None = None, weighting: str = "window",
                      max_chunk_elems: int = 4_000_000) -> BootResult:
    """Two-stage cluster bootstrap with a percentile CI.

    Args:
        values: per-window values, shape (W,) (e.g. a paired difference) or (W, m) (several quantities per window
            for ratio or slope statistics).
        dates: cluster label per window, shape (W,).
        n_boot: number of replicates R (default 10,000).
        level: two-sided CI level (default 0.95).
        seed: seed of the replicate stream.
        stat: maps column means of shape (..., m) to the statistic (...,); default: the mean of a 1-column input.
            See ``ratio_stat`` and ``slope_stat``.
        weighting: 'window' pools the resampled windows (each date weighted by its window count; the point estimate
            is the plain window mean); 'date' averages date means (each drawn date weighted equally).
        max_chunk_elems: memory bound for one chunk of replicates (resampled window values, all columns).

    Returns:
        BootResult.
    """
    res = cluster_bootstrap_multi(values, dates, {"stat": stat or _identity_stat}, n_boot=n_boot, level=level,
                                  seed=seed, weighting=weighting, max_chunk_elems=max_chunk_elems)
    return res["stat"]


def _check_boot_args(n_boot: int, level: float, weighting: str) -> None:
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if weighting not in ("window", "date"):
        raise ValueError("weighting must be 'window' or 'date'")


def cluster_bootstrap_multi(values, dates, stats: Mapping[str, Callable[[np.ndarray], np.ndarray]],
                            n_boot: int = 10_000, level: float = 0.95, seed: int = 0, weighting: str = "window",
                            max_chunk_elems: int = 4_000_000) -> dict[str, BootResult]:
    """Several statistics of the SAME two-stage resamples (pre-freeze audit HC-4/PV-4: ratio-of-sums statistics
    that must share resamples, and the power simulation's rule + p-value pairs).

    ``stats`` maps a name to a statistic of the column means (as ``stat`` in :func:`cluster_bootstrap`); returns
    ``{name: BootResult}``. With one statistic the replicates are bit-identical to :func:`cluster_bootstrap` with the
    same arguments (that function is this one with a single statistic).
    """
    _check_boot_args(n_boot, level, weighting)
    if not stats:
        raise ValueError("need at least one statistic")
    v, uniq, counts, offsets, _ = _group(values, dates)
    G, n_max = len(uniq), int(counts.max())

    if weighting == "window":
        est_means = v.mean(0)
    else:
        est_means = (np.add.reduceat(v, offsets, axis=0) / counts[:, None]).mean(0)
    estimates = {k: float(f(est_means[None, :])[0]) for k, f in stats.items()}

    rng = np.random.default_rng(seed)
    chunk = max(1, min(n_boot, max_chunk_elems // max(1, G * n_max * v.shape[1])))
    reps = {k: np.empty(n_boot) for k in stats}
    slots = np.arange(n_max)
    for start in range(0, n_boot, chunk):
        R = min(chunk, n_boot - start)
        D = rng.integers(G, size=(R, G))                                   # stage 1: dates
        n_sel = counts[D]                                                  # (R, G)
        u = rng.random((R, G, n_max))                                      # stage 2: windows within dates
        idx = np.minimum((u * n_sel[..., None]).astype(np.int64), n_sel[..., None] - 1)
        valid = slots[None, None, :] < n_sel[..., None]
        flat = np.where(valid, offsets[D][..., None] + idx, 0)
        x = v[flat] * valid[..., None]                                     # (R, G, n_max, m)
        if weighting == "window":
            means = x.sum((1, 2)) / n_sel.sum(1)[:, None]
        else:
            means = (x.sum(2) / n_sel[..., None]).mean(1)
        for k, f in stats.items():
            reps[k][start:start + R] = f(means)
    a = (1.0 - level) / 2.0
    out = {}
    for k in stats:
        lo, hi = np.quantile(reps[k], [a, 1.0 - a])
        out[k] = BootResult(estimate=estimates[k], lo=float(lo), hi=float(hi), level=level, reps=reps[k], n_dates=G,
                            n_windows=int(v.shape[0]))
    return out


def iid_bootstrap(values, n_boot: int = 10_000, level: float = 0.95, seed: int = 0) -> BootResult:
    """Naive window-level bootstrap that ignores clustering (reference only: it under-covers clustered data)."""
    v = np.asarray(values, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    reps = v[rng.integers(v.size, size=(n_boot, v.size))].mean(1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [a, 1.0 - a])
    return BootResult(float(v.mean()), float(lo), float(hi), level, reps, n_dates=v.size, n_windows=v.size)


def ratio_stat(minus_one: bool = True, percent: bool = True) -> Callable[[np.ndarray], np.ndarray]:
    """Statistic for 2-column values (a, b): mean(a) / mean(b) [- 1] [x 100], e.g. '% of measurement-only welfare'."""
    def f(means: np.ndarray) -> np.ndarray:
        if means.shape[-1] != 2:
            raise ValueError("ratio_stat needs 2-column values (numerator, denominator)")
        r = means[..., 0] / means[..., 1]
        r = r - 1.0 if minus_one else r
        return 100.0 * r if percent else r
    return f


def slope_stat(levels: Sequence[float]) -> Callable[[np.ndarray], np.ndarray]:
    """Statistic for (W, m) values observed at m sweep levels: OLS slope of the column means on ``levels`` (H1b)."""
    x = np.asarray(levels, dtype=float)
    xc = x - x.mean()
    den = float(xc @ xc)
    if den == 0:
        raise ValueError("levels must not all be equal")

    def f(means: np.ndarray) -> np.ndarray:
        if means.shape[-1] != x.size:
            raise ValueError(f"slope_stat expects {x.size} columns")
        return (means - means.mean(-1, keepdims=True)) @ xc / den
    return f


# --------------------------------------------------------------------------- per-date signs and Holm
def sign_counts(values, dates, tol: float = 0.0) -> dict:
    """Per-date sign of the date-mean effect.

    Returns dict(n_dates, positive, negative, zero, table) where a date counts as zero when |mean| <= tol and
    ``table`` is the per-date DataFrame (date, mean, n_windows, sign).
    """
    t = date_means(values, dates)
    t["sign"] = np.where(t["mean"] > tol, 1, np.where(t["mean"] < -tol, -1, 0))
    return dict(n_dates=len(t), positive=int((t["sign"] > 0).sum()), negative=int((t["sign"] < 0).sum()),
                zero=int((t["sign"] == 0).sum()), table=t)


def holm(pvalues: Mapping[str, float] | Sequence[float], alpha: float = 0.05) -> pd.DataFrame:
    """Holm (1979) step-down adjustment.

    Adjusted p of the j-th smallest raw p (1-based): max_{l <= j} min(1, (m - l + 1) * p_(l)).

    Returns a DataFrame in the input order with columns name, p, p_holm, reject (p_holm <= alpha).
    """
    if isinstance(pvalues, Mapping):
        names, p = list(pvalues.keys()), np.asarray(list(pvalues.values()), dtype=float)
    else:
        p = np.asarray(pvalues, dtype=float)
        names = [f"H{j + 1}" for j in range(p.size)]
    if p.ndim != 1 or p.size == 0 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a non-empty 1-D array in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return pd.DataFrame({"name": names, "p": p, "p_holm": adj, "reject": adj <= alpha})


def paired_summary(values, dates, n_boot: int = 10_000, level: float = 0.95, seed: int = 0,
                   null: float = 0.0, alternative: str = "greater", tol: float = 0.0) -> dict:
    """One-call summary of a paired per-window difference: bootstrap CI, p-value and per-date sign counts."""
    b = cluster_bootstrap(values, dates, n_boot=n_boot, level=level, seed=seed)
    s = sign_counts(values, dates, tol=tol)
    return dict(**b.as_dict(), p=b.p_value(null, alternative), alternative=alternative,
                dates_positive=s["positive"], dates_negative=s["negative"], dates_zero=s["zero"])


# --------------------------------------------------------------------------- crossed (multi-factor) bootstrap
def crossed_bootstrap(values, memberships: Mapping[str, np.ndarray], n_boot: int = 10_000, level: float = 0.95,
                      seed: int = 0, stat: Callable[[np.ndarray], np.ndarray] | None = None,
                      max_chunk_elems: int = 8_000_000) -> BootResult:
    """Pigeonhole bootstrap for crossed (and mixed-membership) clustering factors.

    Each replicate draws, independently for every factor ``f`` with ``L_f`` levels, the multiplicities ``N_f`` of
    a resample of ``L_f`` levels with replacement (multinomial, mean 1 per level). Window ``w`` gets the weight
    ``prod_f sum_l A_f[w, l] N_f[l]`` where ``A_f`` (W, L_f) is its membership (rows sum to 1: a window built from
    several captures belongs to each in proportion to its share), and the statistic is computed from the weighted
    column means. With one one-hot factor this is the one-stage cluster bootstrap.

    Args:
        values: per-window values (W,) or (W, m) (as in :func:`cluster_bootstrap`).
        memberships: factor name -> (W, L_f) nonnegative membership matrix with rows summing to 1.
        n_boot, level, seed, stat: as in :func:`cluster_bootstrap`.

    Returns:
        BootResult with ``n_dates`` = the number of levels of the first factor. Replicates whose weights are all
        zero (every window lost a factor level) are redrawn.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    if v.ndim != 2 or v.shape[0] == 0 or not np.all(np.isfinite(v)):
        raise ValueError("values must be finite with shape (W,) or (W, m), W >= 1")
    W = v.shape[0]
    if not memberships:
        raise ValueError("need at least one clustering factor")
    if n_boot < 1 or not 0 < level < 1:
        raise ValueError("need n_boot >= 1 and level in (0, 1)")
    mats = []
    for name, A in memberships.items():
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != W or A.shape[1] < 1:
            raise ValueError(f"membership {name!r} must have shape ({W}, L), got {A.shape}")
        if np.any(A < 0) or not np.allclose(A.sum(1), 1.0, atol=1e-9):
            raise ValueError(f"membership {name!r} must be nonnegative with rows summing to 1")
        mats.append(A)
    stat = stat or _identity_stat
    estimate = float(stat(v.mean(0)[None, :])[0])
    rng = np.random.default_rng(seed)
    L_tot = sum(A.shape[1] for A in mats)
    chunk = max(1, min(n_boot, max_chunk_elems // max(1, W + L_tot)))
    reps = np.empty(n_boot)
    done = 0
    while done < n_boot:
        R = min(chunk, n_boot - done)
        wts = np.ones((R, W))
        for A in mats:
            L = A.shape[1]
            N = rng.multinomial(L, np.full(L, 1.0 / L), size=R).astype(float)    # (R, L) level multiplicities
            wts *= N @ A.T
        tot = wts.sum(1)
        ok = tot > 0
        means = (wts[ok] @ v) / tot[ok, None]
        k = int(ok.sum())
        reps[done:done + k] = stat(means)
        done += k
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [a, 1.0 - a])
    return BootResult(estimate=estimate, lo=float(lo), hi=float(hi), level=level, reps=reps,
                      n_dates=int(mats[0].shape[1]), n_windows=int(W))


def window_memberships(metas: Sequence[Mapping]) -> dict:
    """Membership matrices of composed windows (``Window.meta``) for :func:`crossed_bootstrap`.

    Factors: ``'date'`` (one-hot), one ``'tractor_<cls>'`` factor per TRACTOR class (share of the class's user
    windows drawn from each capture; pool parts ``<stem>#A``/``#B`` count as their capture) and ``'raca_run'``
    (share of the window's tenants whose CQI comes from each run). Returns ``{name: (labels, A (W, L))}``.
    """
    metas = list(metas)
    if not metas:
        raise ValueError("no windows")
    rows: dict[str, list[dict]] = {"date": []}
    for m in metas:
        rows["date"].append({str(m["date"]): 1.0})
        per_cls: dict[str, dict] = {}
        runs: dict = {}
        tenants = m.get("tenants")
        if not tenants:
            raise KeyError("window meta has no 'tenants' composition record")
        for t in tenants:
            if "users" in t:
                cnt = per_cls.setdefault(t["cls"], {})
                for stem, _off in t["users"]:
                    key = str(stem).split("#", 1)[0]
                    cnt[key] = cnt.get(key, 0.0) + 1.0
            run = t.get("cqi_run", {}).get("run")
            if run is not None:
                runs[run] = runs.get(run, 0.0) + 1.0
        for c, cnt in per_cls.items():
            rows.setdefault(f"tractor_{c}", [])
        for name in [k for k in rows if k.startswith("tractor_")]:
            cnt = per_cls.get(name[len("tractor_"):], {})
            rows[name].append(cnt)
        rows.setdefault("raca_run", []).append(runs)
    out = {}
    for name, lst in rows.items():
        if len(lst) != len(metas):
            raise ValueError(f"factor {name!r} is missing for some windows (mixed compositions)")
        labels = sorted({k for d in lst for k in d}, key=str)
        idx = {k: j for j, k in enumerate(labels)}
        A = np.zeros((len(lst), len(labels)))
        for i, d in enumerate(lst):
            tot = sum(d.values())
            if tot <= 0:
                raise ValueError(f"window {i} has no members in factor {name!r}")
            for k, c in d.items():
                A[i, idx[k]] = c / tot
        out[name] = (labels, A)
    return out

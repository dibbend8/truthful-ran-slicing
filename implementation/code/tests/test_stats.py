"""Tests for slicemech.eval.stats (cluster bootstrap, Holm, sign counts) and slicemech.eval.metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slicemech.eval import metrics as Mx
from slicemech.eval.stats import (
    bootstrap_pvalue,
    cluster_bootstrap,
    date_means,
    holm,
    iid_bootstrap,
    paired_summary,
    ratio_stat,
    sign_counts,
    slope_stat,
)


def _clustered(rng, G=16, n=8, mu=0.3, sd_date=1.0, sd_win=1.0, unequal=False):
    counts = rng.integers(4, 13, G) if unequal else np.full(G, n)
    dates = np.repeat(np.array([f"2024-09-{d:02d}" for d in range(1, G + 1)]), counts)
    eff = rng.normal(0, sd_date, G)
    v = mu + np.repeat(eff, counts) + rng.normal(0, sd_win, counts.sum())
    return v, dates


# --------------------------------------------------------------------------- bootstrap
def test_bootstrap_deterministic_and_estimate_is_window_mean():
    rng = np.random.default_rng(0)
    v, d = _clustered(rng, unequal=True)
    b1 = cluster_bootstrap(v, d, n_boot=10_000, seed=5)
    b2 = cluster_bootstrap(v, d, n_boot=10_000, seed=5)
    np.testing.assert_array_equal(b1.reps, b2.reps)
    assert b1.estimate == pytest.approx(v.mean())
    assert b1.lo < b1.estimate < b1.hi and b1.reps.size == 10_000
    assert b1.n_dates == 16 and b1.n_windows == v.size
    shuffled = np.random.default_rng(1).permutation(v.size)          # input order must not matter
    b3 = cluster_bootstrap(v[shuffled], d[shuffled], n_boot=10_000, seed=5)
    assert b3.estimate == pytest.approx(b1.estimate) and abs(b3.hi - b1.hi) < 0.05


def test_bootstrap_degenerate_and_chunking():
    v = np.full(12, 2.5)
    b = cluster_bootstrap(v, np.repeat(["a", "b", "c"], 4), n_boot=500)
    assert b.lo == b.hi == b.estimate == 2.5
    rng = np.random.default_rng(3)
    v, d = _clustered(rng)
    big = cluster_bootstrap(v, d, n_boot=3000, seed=1)
    small = cluster_bootstrap(v, d, n_boot=3000, seed=1, max_chunk_elems=1000)   # many chunks
    assert abs(big.lo - small.lo) < 0.1 and abs(big.hi - small.hi) < 0.1
    wide = np.column_stack([v] * 10)                                    # memory bound counts all columns
    bw = cluster_bootstrap(wide, d, n_boot=500, seed=1, stat=lambda m: m.mean(-1), max_chunk_elems=1280)
    assert bw.estimate == pytest.approx(v.mean()) and bw.reps.size == 500


def test_bootstrap_variance_matches_two_stage_formula():
    """Replicate variance ~ (G-1)/G * var(date means)/G + within-date resampling term."""
    rng = np.random.default_rng(4)
    v, d = _clustered(rng, G=20, n=10)
    b = cluster_bootstrap(v, d, n_boot=10_000, seed=0)
    dm = date_means(v, d)
    G, n = 20, 10
    within = np.mean([np.var(v[d == x]) for x in dm["date"]]) / n / G
    expect = np.var(dm["mean"].to_numpy()) / G + within
    assert b.reps.var() == pytest.approx(expect, rel=0.08)


def test_two_stage_bootstrap_coverage_on_clustered_data():
    """Coverage of the nominal 95% percentile CI on synthetic clustered data (16 dates x 8 windows, ICC = 0.5).

    Expected ~0.93: with 16 clusters the percentile interval inherits the normal-vs-t(15) gap
    (t_{0.975,15} = 2.13 vs 1.96). The naive window bootstrap, which ignores the date clusters, covers ~0.63.
    """
    rng = np.random.default_rng(2026)
    nsim, mu = 400, 0.3
    hit = hit_iid = 0
    for s in range(nsim):
        v, d = _clustered(rng, mu=mu)
        b = cluster_bootstrap(v, d, n_boot=2000, seed=s)
        hit += b.lo <= mu <= b.hi
        bi = iid_bootstrap(v, n_boot=2000, seed=s)
        hit_iid += bi.lo <= mu <= bi.hi
    cov, cov_iid = hit / nsim, hit_iid / nsim
    assert 0.90 <= cov <= 0.98, cov
    assert cov_iid < 0.80, cov_iid
    assert cov - cov_iid > 0.15


def test_bootstrap_coverage_unequal_cluster_sizes_date_weighting():
    rng = np.random.default_rng(7)
    nsim, mu, hit = 200, -0.2, 0
    for s in range(nsim):
        v, d = _clustered(rng, mu=mu, unequal=True)
        b = cluster_bootstrap(v, d, n_boot=2000, seed=s, weighting="date")
        hit += b.lo <= mu <= b.hi
    assert 0.88 <= hit / nsim <= 0.99


def test_ratio_and_slope_statistics():
    rng = np.random.default_rng(5)
    G, n = 16, 8
    d = np.repeat(np.arange(G), n)
    base = 100 + rng.normal(0, 5, G * n)
    two = np.column_stack([1.02 * base, base])
    b = cluster_bootstrap(two, d, n_boot=2000, stat=ratio_stat())
    assert b.estimate == pytest.approx(2.0)                           # +2 % of the denominator arm
    levels = np.array([1.0, 2.0, 5.0, 15.0])
    vals = 0.5 + 0.1 * levels[None, :] + rng.normal(0, 0.01, (G * n, 4))
    s = cluster_bootstrap(vals, d, n_boot=2000, stat=slope_stat(levels))
    assert s.estimate == pytest.approx(0.1, abs=2e-3) and s.lo > 0.09
    with pytest.raises(ValueError):
        cluster_bootstrap(two, d, n_boot=10)                          # multi-column needs a stat
    with pytest.raises(ValueError):
        slope_stat([1.0, 1.0])


def test_pvalues_and_bounds():
    reps = np.linspace(0.1, 1.0, 999)
    assert bootstrap_pvalue(reps, 0.0, "greater") == pytest.approx(1 / 1000)
    assert bootstrap_pvalue(reps, 0.0, "less") == pytest.approx(1.0)
    assert bootstrap_pvalue(reps, 0.0, "two-sided") == pytest.approx(2 / 1000)
    with pytest.raises(ValueError):
        bootstrap_pvalue(reps, 0.0, "bigger")
    rng = np.random.default_rng(6)
    v, d = _clustered(rng, mu=0.0, sd_date=0.1, sd_win=0.1)
    b = cluster_bootstrap(v, d, n_boot=5000)
    assert b.lower(0.95) < b.upper(0.95) and b.upper(0.95) < b.hi     # one-sided bound inside the two-sided CI


def test_input_validation():
    with pytest.raises(ValueError):
        cluster_bootstrap([1.0, np.nan], ["a", "b"])
    with pytest.raises(ValueError):
        cluster_bootstrap([1.0, 2.0], ["a"])
    with pytest.raises(ValueError):
        cluster_bootstrap([], [])
    with pytest.raises(ValueError):
        cluster_bootstrap([1.0, 2.0], ["a", "b"], level=1.5)
    with pytest.raises(ValueError):
        sign_counts(np.ones((4, 2)), ["a", "a", "b", "b"])                 # multi-column needs reducing first


# --------------------------------------------------------------------------- Holm, signs
def test_holm_known_values():
    t = holm({"H1a": 0.01, "H1c": 0.04, "H4a": 0.03, "H5b": 0.005})
    np.testing.assert_allclose(t["p_holm"], [0.03, 0.06, 0.06, 0.02])
    assert t["reject"].tolist() == [True, False, False, True]
    assert t["name"].tolist() == ["H1a", "H1c", "H4a", "H5b"]
    t2 = holm([0.5, 0.9])
    np.testing.assert_allclose(t2["p_holm"], [1.0, 1.0])               # capped at 1, monotone
    with pytest.raises(ValueError):
        holm([0.1, 1.2])
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.2, 7)
    adj = holm(p)["p_holm"].to_numpy()
    assert np.all(adj >= p) and np.all(np.diff(adj[np.argsort(p)]) >= 0)


def test_sign_counts_and_paired_summary():
    v = np.array([1, 2, -1, -2, -3, 0, 0, 5.0])
    d = np.array(["a", "a", "b", "b", "b", "c", "c", "d"])
    s = sign_counts(v, d)
    assert (s["n_dates"], s["positive"], s["negative"], s["zero"]) == (4, 2, 1, 1)
    assert s["table"].set_index("date").loc["b", "mean"] == pytest.approx(-2.0)
    out = paired_summary(v, d, n_boot=1000)
    assert out["dates_positive"] == 2 and 0 < out["p"] <= 1 and out["n_boot"] == 1000


# --------------------------------------------------------------------------- metrics
def _episode(K=6, warm=2, seed=0):
    rng = np.random.default_rng(seed)
    n = 3
    on = rng.uniform(0, 5, (K, n))
    dr = rng.uniform(0, 1, (K, n))
    dr[0, 0] = on[0, 0] = 0.0                                            # nothing resolved -> NaN miss %
    q = rng.integers(4, 40, (K, n)).astype(float)
    p = rng.uniform(0, 1, (K, n))
    V = p + rng.uniform(0, 2, (K, n))
    pi = np.array([1.0, 3.0, 0.5])
    df = Mx.from_arrays(on, dr, q, p, V, pi, theta=on + dr, cls=("video", "urllc", "mmtc"))
    df["p_fix"] = 0.5 * df["p"]
    return df, dict(on=on, dr=dr, q=q, p=p, V=V, pi=pi, K=K, warm=warm)


def test_interval_metrics_and_welfare():
    df, a = _episode()
    m = Mx.interval_metrics(df, pi=a["pi"])
    assert np.isnan(m.loc[0, "deadline_miss_pct"])
    row = m[(m.k == 1) & (m.tenant == 2)].iloc[0]
    assert row.deadline_miss_pct == pytest.approx(100 * a["dr"][1, 2] / (a["on"][1, 2] + a["dr"][1, 2]))
    assert row.realized_value == pytest.approx(0.5 * a["on"][1, 2])
    assert row.utility_realized == pytest.approx(row.realized_value - row.price)
    mf = Mx.interval_metrics(df, price_col="p_fix")
    np.testing.assert_allclose(mf["price"], 0.5 * m["price"])
    w = Mx.interval_welfare(df, B=106)
    np.testing.assert_allclose(w["W_realized"], (a["pi"] * a["on"]).sum(1))
    np.testing.assert_allclose(w["idle_prb"], 106 - a["q"].sum(1))
    bad = df.copy()
    bad.loc[3, "realized_value"] += 1.0
    with pytest.raises(ValueError):
        Mx.interval_metrics(bad, pi=a["pi"])
    with pytest.raises(KeyError):
        Mx.interval_metrics(df.drop(columns="quota"))
    Mx.interval_metrics(df, pi={0: 1.0, 1: 3.0, 2: 0.5})                # tariffs as a mapping
    with pytest.raises(ValueError):
        Mx.interval_metrics(df, pi={0: 1.0, 1: 3.0, 2: 0.6})
    with pytest.raises(KeyError):
        Mx.interval_metrics(df, pi={0: 1.0, 1: 3.0})
    with pytest.raises(ValueError):
        Mx.deadline_miss_pct([-1.0], [0.0])


def test_window_summary_excludes_warmup():
    df, a = _episode()
    s = Mx.window_summary(df, warmup=a["warm"], keys=dict(window="w1", date="d1", arm="protected"))
    assert list(s.columns[:3]) == ["window", "date", "arm"]
    sc = slice(a["warm"], None)
    t1 = s[s.tenant == 1].iloc[0]
    assert t1.n_intervals == a["K"] - a["warm"]
    assert t1.on_time_Mbit == pytest.approx(a["on"][sc, 1].mean())
    assert t1.deadline_miss_pct == pytest.approx(100 * a["dr"][sc, 1].sum() / (a["on"][sc, 1] + a["dr"][sc, 1]).sum())
    assert t1.quota == pytest.approx(a["q"][sc, 1].mean())
    tot = s[s.tenant == -1].iloc[0]
    assert tot.realized_value == pytest.approx((a["pi"] * a["on"])[sc].sum(1).mean())     # mean realized welfare
    assert tot.utility_declared_total == pytest.approx((a["V"] - a["p"])[sc].sum())
    with pytest.raises(ValueError):
        Mx.window_summary(df, warmup=99)


def test_paired_table_and_percent():
    rows = []
    for arm, scale in (("protected", 1.02), ("meas_only", 1.0)):
        for w in range(4):
            df, _ = _episode(seed=w)
            df["realized_value"] *= scale
            rows.append(Mx.window_summary(df, 2, keys=dict(window=f"w{w}", date=f"d{w // 2}", arm=arm)))
    S = pd.concat(rows, ignore_index=True)
    t = Mx.paired_table(S, "realized_value", "protected", "meas_only")
    assert len(t) == 4
    np.testing.assert_allclose(t["pct_of_b"], 2.0)
    with pytest.raises(ValueError):
        Mx.paired_table(S[~((S.arm == "protected") & (S.window == "w0"))], "realized_value", "protected", "meas_only")
    with pytest.raises(ZeroDivisionError):
        Mx.percent_of([1.0], [0.0])

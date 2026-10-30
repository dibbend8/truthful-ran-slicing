"""TRAIN-only calibration: tenant counts, contracts, tenant forecasters, burst shapes, regime diagnostics.

Port of ``ICC_rebuild_2026-10-10/src/rebuild/data.fit_calibration`` (L556-585) for any n (experiment_design.md
§2 "Congestion calibration" and §3 "Burst-shape fit"). Nothing here reads DEV or TEST data.

1. ``r_bar``: pooled mean per-PRB rate of the TRAIN Raca runs long enough for a window (Mbit/s per PRB).
2. Unit rates (Mbit/s): one video viewer (channel uniform over the tenant's group, format mix, averaged over the
   TRAIN windows used); one URLLC user / mMTC device (pooled TRAIN trace mean = the expectation under
   ``tractor.draw_user``).
3. Counts. Tenant i targets ``target_i = rho * B * r_bar * share[cls_i] / (number of tenants of cls_i)`` Mbit/s
   (shares 0.6 / 0.3 / 0.1 for video / URLLC-like / mMTC-like; with n = 6 or 9 a class share is split equally among
   the tenants of the class). Three load rules (``Calibration.load_rule``):

   * ``'per_date'`` (default; orchestrator decision **D2, offered-load normalization per date**, pre-registered):
     for every date of every split (TRAIN, DEV, TEST) and every tenant, the integer count is chosen by
     :func:`counts_for_date` so that the date's mean offered traffic of the tenant, over the split's pre-declared
     windows of that date, is closest to ``target_i``. ``r_bar`` is always the TRAIN mean per-PRB rate, so no DEV
     or TEST channel statistic enters. The rule reads only offered traffic (Puffer chunk sizes, TRACTOR captures;
     ``compose.unit_offered_bytes``), never a CQI run, a report or a controller outcome, and is identical for every
     compared controller. TEST counts are computed only behind the freeze gate
     (``puffer_fetch.require_test_read_gate``) and are never stored. Composition draws are nested, so load sweeps
     (another ``rho``) share every unit of the lower load. ``Calibration.counts`` then holds the reference counts of
     the expectation rule below (reporting only); windows use :meth:`Calibration.counts_for`.
   * ``'expectation'`` (the rule before D2): ``count_i = max(1, round(target_i / unit_i))`` on every date.
   * ``'fixed'``: given counts on every date (``counts=``).
4. Compose the TRAIN windows (with the rule's counts); ``mu`` = mean measured demand (Mbit/s) over all their
   intervals.
5. Tenant forecasters: per-tenant AR(6), horizon 1, target = measured demand, input = own zero-delay history.
   Operator forecasters: the direct-horizon AR(6) on released counters (``control.features.fit_operator_ar``),
   one per measurement delay ``d`` of ``spec/params.yaml: information.measurement_delay_sweep_s`` (1, 2, 5, 15 s),
   stored in ``Calibration.operator_ar`` (round-1 review, finding 5: contexts, PPO and the measurement-only arm
   must use the fitted forecaster, never the untrained mean of the last 6 counters).
6. Contracts: :func:`slicemech.config.contract_from_params` (``spec/params.yaml`` multipliers of mu). Orchestrator
   decision **D1** (TRAIN only, the pre-declared first adjustment step): the GBR fractions are 0.8 x the design
   values, g / mu = 0.40 (video), 0.64 (URLLC-like), 0.16 (mMTC-like); see ``spec/params.yaml``.
7. Burst shapes per class: ``u = A(sub-window) / (w_i * theta_k)`` for non-overlapping sub-windows of length
   ``w_i = min(delay_i, Delta)`` aligned to the interval start (``floor(Delta / w_i)`` per interval; the remainder
   of the interval is not sampled), scored intervals with ``theta_k > 0`` only, pooled over the class's tenants and
   compressed by :func:`slicemech.model.shapes.fit_shape` (mean 1, 64 quantile atoms). With exact advance
   information and ``w = Delta`` the video shape is degenerate at 1.
8. Regime diagnostics with truthful reports ``z = min(theta, C)``, ``r = rate_hat`` and equal weights, compared with
   ``spec/params.yaml: regime_targets``. Reported only; nothing is tuned here.

TEST data (round-1 review, finding 2): no statistic of a TEST source (TRACTOR TEST captures, Raca TEST runs, Puffer
TEST dates) is computed before the pre-registration freeze. :func:`data_report` withholds TEST rows unless it is
given the frozen pre-registration file (the same gate as ``data.puffer_fetch``). Every count, rho, GBR or other
regime decision is taken on TRAIN; DEV is descriptive.
"""
from __future__ import annotations

import dataclasses
import functools
import itertools
import pathlib
import types
from dataclasses import dataclass, field

import numpy as np

from .. import io as sio
from ..config import PARAMS, Contract, Mechanism, Shape, contract_from_params
from ..control.features import OperatorAR, fit_operator_ar
from ..forecast import ARForecast, fit_ar
from . import compose, puffer, raca, tractor

CALIB_DIR = sio.DATA_DIR / "calibration"
TILT = 2.0   # weight ratio of the single-tilted priority vectors in the regime diagnostic
#: Measurement delays (intervals of Delta = 1 s) with a fitted operator forecaster (spec/params.yaml sweep).
OPERATOR_DELAYS = tuple(int(x) for x in PARAMS["information"]["measurement_delay_sweep_s"])


def default_shares() -> dict:
    """Class shares of offered load (spec/params.yaml load.shares)."""
    return {k: float(v) for k, v in PARAMS["load"]["shares"].items()}


def unit_rates(classes, video_groups, windows, K: int, lead_s: float) -> np.ndarray:
    """Mean offered rate (Mbit/s) of one viewer / user / device per tenant over the TRAIN windows.

    Video: expectation over the given TRAIN (date, offset) windows (:func:`puffer.viewer_window_mbps`), so the
    realised TRAIN load matches the target up to integer rounding and viewer sampling; URLLC/mMTC: pooled TRAIN
    trace mean (the exact expectation under ``tractor.draw_user``, independent of the window times).
    """
    out = []
    for c, g in zip(classes, video_groups):
        if c == "video":
            out.append(puffer.viewer_window_mbps(windows, K, round(lead_s * 1e9), tuple(g)))
        else:
            out.append(tractor.pool_mean_mbps("train", c))
    return np.asarray(out)


def tenant_counts(classes, unit: np.ndarray, rho: float, shares: dict, B: int, r_bar: float) -> tuple[int, ...]:
    """Integer viewer/user/device counts giving tenant mean offered rates ``rho * B * r_bar * share / n_cls``."""
    if rho <= 0 or r_bar <= 0 or np.any(unit <= 0):
        raise ValueError("need rho > 0, r_bar > 0 and positive unit rates")
    if abs(sum(shares[c] for c in set(classes)) - 1) > 1e-9:
        raise ValueError("shares of the classes present must sum to 1")
    n_cls = {c: sum(1 for x in classes if x == c) for c in set(classes)}
    target = np.array([rho * B * r_bar * shares[c] / n_cls[c] for c in classes])
    return tuple(int(max(1, round(t / u))) for t, u in zip(target, unit))


# ----------------------------------------------------------------------------- D2: offered-load normalization
#: Load rules (module docstring, step 3).
LOAD_RULES = ("per_date", "expectation", "fixed")
#: Units drawn first when searching a date's count; doubled until the target is bracketed (at most MAX_UNITS).
FIRST_UNITS = 32
MAX_UNITS = 1 << 16


def tenant_targets(classes, rho: float, shares: dict, B: int, r_bar: float) -> np.ndarray:
    """Per-tenant offered-rate targets (Mbit/s): ``rho * B * r_bar * share[cls] / (tenants of cls)``."""
    classes = tuple(classes)
    if rho <= 0 or r_bar <= 0:
        raise ValueError("need rho > 0 and r_bar > 0")
    if abs(sum(shares[c] for c in set(classes)) - 1) > 1e-9:
        raise ValueError("shares of the classes present must sum to 1")
    n_cls = {c: sum(1 for x in classes if x == c) for c in set(classes)}
    return np.array([rho * B * r_bar * shares[c] / n_cls[c] for c in classes], float)


@dataclass(frozen=True)
class DateLoad:
    """Result of the per-date count rule for one (split, date) (Mbit/s; loads are dimensionless).

    ``window_offered_mbps[w, i]`` is tenant i's mean offered rate over all K intervals of window ``offsets[w]`` at
    the chosen counts; ``offered_mbps`` its mean over the date's windows (the quantity matched to ``target_mbps``).
    Loads divide by ``B * r_bar`` with ``r_bar`` the TRAIN mean per-PRB rate.
    """

    split: str
    date: str
    rho: float
    counts: tuple
    target_mbps: np.ndarray
    offered_mbps: np.ndarray
    window_offered_mbps: np.ndarray
    offsets: tuple
    r_bar: float
    B: int

    @property
    def load(self) -> float:
        """Date offered load ``sum_i offered_i / (B r_bar)``."""
        return float(self.offered_mbps.sum() / (self.B * self.r_bar))

    @property
    def rel_error(self) -> np.ndarray:
        """Per-tenant ``offered / target - 1``."""
        return self.offered_mbps / self.target_mbps - 1.0

    @property
    def window_loads(self) -> np.ndarray:
        """Offered load of each of the date's windows (all K intervals), ``(W,)``."""
        return self.window_offered_mbps.sum(1) / (self.B * self.r_bar)


def _layout(classes, video_groups):
    if isinstance(classes, (int, np.integer)) and not isinstance(classes, bool):
        cl, vg = compose.tenant_layout(int(classes))
        return cl, tuple(None if g is None else tuple(g) for g in (vg if video_groups is None else video_groups))
    classes = tuple(classes)
    if video_groups is None:
        video_groups = tuple(puffer.CHANNELS if c == "video" else None for c in classes)
    if len(video_groups) != len(classes):
        raise ValueError("one video group (or None) per tenant")
    return classes, tuple(None if g is None else tuple(g) for g in video_groups)


def counts_for_date(split: str, date: str, rho: float | None = None, classes=3, video_groups=None, *,
                    r_bar: float | None = None, shares: dict | None = None, B: int | None = None,
                    K: int = compose.K_DEFAULT, seed: int = compose.DEFAULT_SEED,
                    lead_s: float = compose.LEAD_DEFAULT, offsets=None, test_freeze=None, info: bool = False):
    """Per-date counts of orchestrator decision D2 (offered-load normalization; pre-registered).

    ``offered_i(U)`` is the mean offered rate (Mbit/s) of the first U units of tenant i over all K intervals of the
    date's windows (``offsets``; default the split's pre-declared offsets ``compose.window_offsets(split)``), and
    ``target_i = rho * B * r_bar * share[cls_i] / (tenants of cls_i)``. ``offered_i(U)`` is nondecreasing in U
    (nested draws). Rule, per class c with tenants T_c:

    1. each tenant's two bracketing counts ``lo_i`` = the largest U with ``offered_i(U) <= target_i`` and
       ``lo_i + 1`` (only ``1`` when one unit already exceeds the target; counts are >= 1);
    2. the class's counts are the combination of bracketing counts whose class mean offered traffic
       ``sum_{i in T_c} offered_i`` is closest to the class target ``share_c * rho * B * r_bar`` (ties: fewer units,
       then the lexicographically smaller count vector).

    With one tenant per class (n = 3) this is the integer count closest to the target. With n = 6 or 9 the class
    share is split equally (each tenant within one unit of its equal share) while the class total stays within half
    of the largest single-unit step of its tenants of the class target (the bracketing sums form a chain whose gaps
    are single-unit steps), so the date's load stays close to rho. Only offered traffic of that date is read
    (``compose.offered_units``); ``r_bar`` defaults to the TRAIN mean per-PRB rate ``raca.mean_rate('train', K+1)``
    and must be a TRAIN statistic.

    Args:
        split, date: the Puffer date and its split ('train' | 'dev' | 'test').
        rho: offered load (default ``spec/params.yaml`` load.rho); ``classes``: an int n (standard layout
            ``compose.tenant_layout(n)``) or a class tuple (``video_groups`` default: all channels).
        lead_s: video lead time of the composed traffic (the default calibration's 4.004 s, also for lead sweeps,
            so a lead sweep keeps the counts).
        test_freeze: frozen pre-registration for split 'test' (default: the one recorded at the TEST download).
            TEST counts raise PermissionError before the freeze (``puffer_fetch.require_test_read_gate``), before
            any TEST source is opened.
        info: return the :class:`DateLoad` instead of the counts tuple.
    """
    if split == "test":
        from .puffer_fetch import require_test_read_gate   # local import: no network module at load time
        require_test_read_gate(test_freeze)
    if split not in compose.SPLIT_CODE:
        raise ValueError(f"unknown split {split!r}")
    if puffer.split_of_date(date) != split:
        raise ValueError(f"Puffer date {date} is not in split {split}")
    classes, video_groups = _layout(classes, video_groups)
    rho = float(PARAMS["load"]["rho"] if rho is None else rho)
    shares = default_shares() if shares is None else dict(shares)
    B = int(Mechanism().B if B is None else B)
    r_bar = float(raca.mean_rate("train", int(K) + 1) if r_bar is None else r_bar)
    offsets = tuple(int(o) for o in (compose.window_offsets(split) if offsets is None else offsets))
    if not offsets:
        raise ValueError("need at least one window offset")
    dl = _date_load(split, date, rho, classes, video_groups, r_bar, tuple(sorted(shares.items())), B, int(K),
                    int(seed), float(lead_s), offsets)
    return dl if info else dl.counts


@functools.lru_cache(maxsize=1024)
def _date_load(split, date, rho, classes, video_groups, r_bar, shares_items, B, K, seed, lead_s, offsets) -> DateLoad:
    target = tenant_targets(classes, rho, dict(shares_items), B, r_bar)
    W, n = len(offsets), len(classes)
    units, cum, opts = {}, {}, {}
    for i, c in enumerate(classes):
        goal = target[i] * W * K / 8e-6                      # bytes over the date's windows at the target rate
        U = FIRST_UNITS
        while True:
            units[i] = np.stack([compose.offered_units(split, date, o, i, c, U, video_groups[i], lead_s=lead_s, K=K,
                                                       seed=seed) for o in offsets])       # (W, U) bytes
            cum[i] = np.cumsum(units[i].sum(0))             # bytes over the date's windows with the first u+1 units
            if cum[i][-1] > goal:
                break
            if U >= MAX_UNITS:
                raise ValueError(f"{split} {date} tenant {i} ({c}): {MAX_UNITS} units do not reach the target")
            U *= 2
        lo = int(np.searchsorted(cum[i], goal, side="right"))   # largest count whose traffic is <= the target
        opts[i] = (lo, lo + 1) if lo >= 1 else (1,)              # the two counts bracketing the target (>= 1)
    counts = [0] * n
    for c in dict.fromkeys(classes):        # apportion each class: its date mean offered traffic closest to its share
        idx = [i for i in range(n) if classes[i] == c]
        goal_c = sum(target[i] for i in idx) * W * K / 8e-6
        best = min(itertools.product(*(opts[i] for i in idx)),
                   key=lambda us: (abs(sum(float(cum[i][u - 1]) for i, u in zip(idx, us)) - goal_c), sum(us), us))
        for i, u in zip(idx, best):
            counts[i] = int(u)
    per_window = np.stack([units[i][:, :counts[i]].sum(1) for i in range(n)], axis=1) * 8e-6 / K
    per_window.flags.writeable = False
    offered = per_window.mean(0)
    offered.flags.writeable = False
    target.flags.writeable = False
    return DateLoad(split, date, rho, tuple(counts), target, offered, per_window, offsets, r_bar, B)


def subwindow_volumes(bytes_ms: np.ndarray, delay_ms) -> list[np.ndarray]:
    """Per tenant, Mbit offered in each sub-window ``(K, floor(1000 / w_i))`` with ``w_i = min(delay_i, 1000)`` ms,
    sub-windows aligned to the interval start (the remainder of each interval is not sampled)."""
    T_ms, n = bytes_ms.shape
    K = T_ms // 1000
    out = []
    for i in range(n):
        w_ms = int(min(delay_ms[i], 1000))
        n_sub = 1000 // w_ms
        out.append(bytes_ms[:, i].reshape(K, 1000)[:, : n_sub * w_ms].reshape(K, n_sub, w_ms).sum(2) * 8e-6)
    return out


def burst_samples(subvols: list[np.ndarray], theta: np.ndarray, delay_ms, warmup: int) -> list[np.ndarray]:
    """Per tenant, normalised sub-window rates ``u = A_sub / (w * theta_k)`` (dimensionless) over scored
    intervals with ``theta_k > 0``."""
    out = []
    for i, sub in enumerate(subvols):
        w_s = min(float(delay_ms[i]), 1000.0) / 1000.0
        th = theta[warmup:, i]
        ok = th > 0
        out.append((sub[warmup:][ok] / (w_s * th[ok, None])).ravel())
    return out


def regime_diagnostics(z: np.ndarray, r: np.ndarray, ct: Contract, mech: Mechanism, tilt: float = TILT) -> dict:
    """Regime diagnostics over intervals (rows of ``z``, ``r``: (N, n) reports in [0, C] and per-PRB rates).

    * ``frac_sumH_gt_B``: share of intervals with ``sum_i min(eta z_i, M_i)/r_i > B``;
    * ``frac_all_caps_fit``: share with ``sum_i (M_i / r_i) 1{z_i > 0} <= B`` (every MBR cap fits);
    * ``frac_weights_change_quota``: share where some single-tilted priority vector (one weight ``tilt`` x the
      others) gives integer quotas different from equal weights;
    * ``mean_weighted_share``: mean of (stage total - sum of fill floors L) / B, the PRBs placed by weighted filling;
    * ``gbr_outage``: share of intervals in the overload stage (``sum_i min(z_i, g_i)/r_i > B``);
    plus stage shares, mean ``sum H / B`` and the mean idle PRBs left by flooring (equal weights).
    """
    from ..model import allocation   # model builder's module (INTERFACES §2)
    N, n = z.shape
    B = float(mech.B)
    st = allocation.stage(z, r, ct, mech)
    sumH = (np.minimum(mech.eta * z, ct.M) / r).sum(-1)          # need caps H_i (PRB), as in theory_design §1.5
    caps = np.where(z > 0, ct.M / r, 0.0).sum(-1)
    w_eq = np.ones(n)
    x_eq = allocation.integer(z, w_eq, r, ct, mech)
    changed = np.zeros(N, bool)
    for i in range(n):
        w = np.ones(n); w[i] = tilt
        changed |= np.any(allocation.integer(z, w, r, ct, mech) != x_eq, axis=-1)
    weighted = (st["total"] - st["L"].sum(-1)) / B
    need = ~st["overload"] & ~st["leftover"]
    return dict(frac_sumH_gt_B=float(np.mean(sumH > B)), frac_all_caps_fit=float(np.mean(caps <= B)),
                frac_weights_change_quota=float(np.mean(changed)), mean_weighted_share=float(np.mean(weighted)),
                gbr_outage=float(np.mean(st["overload"])), frac_need_stage=float(np.mean(need)),
                frac_leftover_stage=float(np.mean(st["leftover"])), mean_sumH_over_B=float(np.mean(sumH / B)),
                mean_idle_prb_floor=float(np.mean(st["total"] - x_eq.sum(-1))), intervals=int(N), tilt=tilt)


def check_targets(diag: dict) -> dict:
    """Compare diagnostics with ``spec/params.yaml: regime_targets``; returns {target: (value, bound, pass)}."""
    t = PARAMS["regime_targets"]
    rows = {"frac_sumH_gt_B": ("min", t["frac_sumH_gt_B_min"]),
            "frac_all_caps_fit": ("max", t["frac_all_caps_fit_max"]),
            "frac_weights_change_quota": ("min", t["frac_weights_change_quota_min"]),
            "mean_weighted_share": ("min", t["mean_weighted_share_min"]),
            "gbr_outage": ("max", t["gbr_outage_max"])}
    out = {}
    for k, (kind, bound) in rows.items():
        v = diag[k]
        out[k] = dict(value=v, bound=float(bound), kind=kind, ok=bool(v >= bound if kind == "min" else v <= bound))
    return out


@dataclass(frozen=True)
class Calibration:
    """Everything fitted on TRAIN for one composition (classes, load, lead time). See module docstring.

    ``counts``: under ``load_rule='per_date'`` (D2) the reference counts of the expectation rule (reporting only);
    windows use :meth:`counts_for`. ``date_counts``: ``{split: {date: counts}}`` of the per-date rule at this
    calibration's ``rho``, stored for TRAIN and DEV (never TEST; TEST counts are computed after the freeze).
    """

    classes: tuple
    counts: tuple
    video_groups: tuple
    contract: Contract
    shape: Shape
    forecaster: ARForecast
    unit_mbps: np.ndarray
    r_bar: float
    rho: float
    shares: dict
    lead_s: float
    report_lead_s: float
    K: int
    warmup: int
    seed: int
    diagnostics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    operator_ar: dict = field(default_factory=dict)     # {d (intervals): control.features.OperatorAR}, TRAIN fit
    load_rule: str = "fixed"                            # 'per_date' (D2) | 'expectation' | 'fixed'
    date_counts: dict = field(default_factory=dict)     # {split: {date: counts}} of the per-date rule at self.rho

    @property
    def n(self) -> int:
        return len(self.classes)

    @property
    def count_lead_s(self) -> float:
        """Video lead time whose traffic sets the per-date counts (the default 4.004 s, also in lead sweeps)."""
        return float(self.meta.get("count_lead_s", compose.LEAD_DEFAULT))

    def counts_for(self, split: str, date: str, rho: float | None = None, *, test_freeze=None) -> tuple:
        """Viewer/user/device counts of every window of ``date`` (module docstring, step 3).

        ``'per_date'``: the stored counts at this calibration's rho, else :func:`counts_for_date` with the TRAIN
        ``r_bar``, shares, K, seed and count lead of this calibration (TEST: only after the freeze). Other rules:
        ``self.counts`` on every date. ``rho`` (per-date rule only) gives the nested counts of a load sweep.
        """
        if self.load_rule != "per_date":
            if rho is not None and float(rho) != self.rho:
                raise ValueError(f"load rule {self.load_rule!r} has no counts for rho={rho}; recalibrate")
            return tuple(int(c) for c in self.counts)
        r = self.rho if rho is None else float(rho)
        if split != "test" and r == self.rho:
            hit = self.date_counts.get(split, {}).get(date)
            if hit is not None:
                return tuple(int(c) for c in hit)
        return counts_for_date(split, date, r, self.classes, self.video_groups, r_bar=self.r_bar,
                               shares=self.shares, B=int(self.meta.get("mech", {}).get("B", Mechanism().B)),
                               K=self.K, seed=self.seed, lead_s=self.count_lead_s, test_freeze=test_freeze)

    def operator_forecaster(self, d: int) -> OperatorAR:
        """The TRAIN-fitted operator AR for measurement delay ``d`` (intervals); raises if it was not fitted."""
        if d not in self.operator_ar:
            raise ValueError(f"no operator forecaster for d={d} in this calibration (fitted: "
                             f"{sorted(self.operator_ar)}); run attach_operator_ar(cal, ds) on TRAIN")
        return self.operator_ar[d]

    @property
    def mu(self) -> np.ndarray:
        """TRAIN-mean demand per tenant (Mbit/s)."""
        return self.contract.mu

    def window(self, split: str, date: str, offset_s: int, *, rho: float | None = None, test_freeze=None,
               **overrides) -> compose.Window:
        """Compose one window of any split with this TRAIN calibration (contract, tenant forecaster) and the
        date's counts (:meth:`counts_for`; ``counts=`` overrides them, ``rho=`` gives a load-sweep composition).

        The per-date counts follow the calibration's K, seed and count lead: overriding ``K``, ``seed`` or
        ``lead_s`` here keeps the date's counts (so a short test window or a lead sweep keeps the traffic volume).
        """
        counts = overrides.pop("counts", None)
        if counts is None:
            counts = self.counts_for(split, date, rho, test_freeze=test_freeze)
        kw = dict(video_groups=self.video_groups, lead_s=self.lead_s, report_lead_s=self.report_lead_s, K=self.K,
                  warmup=self.warmup, seed=self.seed)
        kw.update(overrides)
        return compose.compose_window(split, date, offset_s, self.classes, counts, self.contract,
                                      self.forecaster, **kw)

    def windows(self, split: str, dates=None, offsets=None, **overrides):
        """Iterate over the split's windows (all pre-declared dates x offsets unless given)."""
        for d in (puffer.split_dates(split) if dates is None else dates):
            for o in (compose.window_offsets(split) if offsets is None else offsets):
                yield self.window(split, d, o, **overrides)

    def to_dict(self) -> dict:
        ct = self.contract
        return dict(classes=list(self.classes), counts=list(self.counts),
                    video_groups=[None if g is None else list(g) for g in self.video_groups],
                    contract=dict(cls=list(ct.cls), g=ct.g, M=ct.M, C=ct.C, pi=ct.pi, delay_ms=ct.delay_ms, mu=ct.mu),
                    shape=dict(beta=[b.tolist() for b in self.shape.beta], rho=[r.tolist() for r in self.shape.rho]),
                    forecaster=self.forecaster.to_dict(), unit_mbps=self.unit_mbps, r_bar=self.r_bar, rho=self.rho,
                    shares=self.shares, lead_s=self.lead_s, report_lead_s=self.report_lead_s, K=self.K,
                    warmup=self.warmup, seed=self.seed, diagnostics=self.diagnostics, meta=self.meta,
                    operator_ar={str(k): v.to_dict() for k, v in sorted(self.operator_ar.items())},
                    load_rule=self.load_rule,
                    date_counts={s: {d: [int(c) for c in v] for d, v in sorted(dc.items())}
                                 for s, dc in sorted(self.date_counts.items())})

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        c = d["contract"]
        ct = Contract(cls=tuple(c["cls"]), g=c["g"], M=c["M"], C=c["C"], pi=c["pi"], delay_ms=c["delay_ms"], mu=c["mu"])
        shape = Shape(beta=tuple(np.asarray(b) for b in d["shape"]["beta"]),
                      rho=tuple(np.asarray(r) for r in d["shape"]["rho"]))
        return cls(tuple(d["classes"]), tuple(d["counts"]),
                   tuple(None if g is None else tuple(g) for g in d["video_groups"]), ct, shape,
                   ARForecast.from_dict(d["forecaster"]), np.asarray(d["unit_mbps"]), float(d["r_bar"]),
                   float(d["rho"]), dict(d["shares"]), float(d["lead_s"]), float(d["report_lead_s"]), int(d["K"]),
                   int(d["warmup"]), int(d["seed"]), dict(d["diagnostics"]), dict(d["meta"]),
                   {int(k): OperatorAR.from_dict(v) for k, v in d.get("operator_ar", {}).items()},
                   d.get("load_rule") or ("expectation" if d["meta"].get("counts_rule") == "load" else "fixed"),
                   {s: {dd: tuple(int(c) for c in v) for dd, v in dc.items()}
                    for s, dc in d.get("date_counts", {}).items()})

    def save(self, name: str) -> pathlib.Path:
        """Write ``data/calibration/<name>.json``."""
        return sio.write_json(CALIB_DIR / f"{name}.json", self.to_dict())


def load_calibration(name: str) -> Calibration:
    """Read ``data/calibration/<name>.json``."""
    return Calibration.from_dict(sio.read_json(CALIB_DIR / f"{name}.json"))


def train_window_keys() -> list[tuple[str, int]]:
    """All pre-declared TRAIN (date, offset) pairs (15 dates x 12 offsets)."""
    return [(d, o) for d in puffer.TRAIN_DATES for o in compose.window_offsets("train")]


def fit_operator_forecasters(measured, contract: Contract, ds=OPERATOR_DELAYS, *, keys) -> dict:
    """Operator AR(6) per delay ``d`` from TRAIN measured-demand series (list of (K, n) arrays, Mbit/s).

    ``keys``: the (date, offset_s) pair of every series (required, keyword-only). Each view carries the split of
    its Puffer date, so ``control.features.fit_operator_ar`` refuses any non-TRAIN series (round-2 review: the
    views used to be labelled 'train' whatever data they held).
    """
    measured, keys = list(measured), [tuple(k) for k in keys]
    if len(keys) != len(measured):
        raise ValueError(f"need one (date, offset) key per measured series ({len(measured)}), got {len(keys)}")
    views = [types.SimpleNamespace(split=puffer.split_of_date(str(d)), measured=np.asarray(m, float),
                                   contract=contract, cls=tuple(contract.cls)) for m, (d, _) in zip(measured, keys)]
    return {int(d): fit_operator_ar(views, int(d)) for d in ds}


def train_measured(cal: "Calibration", keys=None) -> list:
    """Measured demand (K, n) of the calibration's TRAIN windows (recomposed; contract-free parts only)."""
    keys = [tuple(k) for k in (cal.meta["train_windows"] if keys is None else keys)]
    if any(puffer.split_of_date(d) != "train" for d, _ in keys):
        raise ValueError("operator forecasters are fitted on TRAIN windows only")
    return [compose.compose_parts("train", d, int(o), cal.classes, cal.counts_for("train", d), cal.video_groups,
                                  lead_s=cal.lead_s, report_lead_s=cal.report_lead_s, K=cal.K,
                                  seed=cal.seed)["measured"]
            for d, o in keys]


def attach_operator_ar(cal: "Calibration", ds=OPERATOR_DELAYS) -> "Calibration":
    """Return ``cal`` with operator forecasters fitted on its own TRAIN windows for every delay in ``ds``
    (for calibrations written before ``operator_ar`` existed; every other field is unchanged)."""
    keys = [tuple(k) for k in cal.meta["train_windows"]]
    return dataclasses.replace(cal, operator_ar=fit_operator_forecasters(train_measured(cal, keys), cal.contract, ds,
                                                                        keys=keys))


def calibrate(n: int = 3, classes=None, video_groups=None, *, rho: float | None = None, shares: dict | None = None,
              lead_s: float = compose.LEAD_DEFAULT, report_lead_s: float = compose.REPORT_LEAD_DEFAULT,
              K: int = compose.K_DEFAULT, warmup: int = compose.WARMUP_DEFAULT, seed: int = compose.DEFAULT_SEED,
              mech: Mechanism | None = None, train_windows=None, diag_windows=None, n_atoms: int = 64,
              history: int = 6, counts=None, load_rule: str | None = None,
              count_lead_s: float = compose.LEAD_DEFAULT) -> Calibration:
    """Fit a :class:`Calibration` on TRAIN windows (see module docstring for the steps).

    Args:
        n / classes / video_groups: standard layout ``compose.tenant_layout(n)`` unless classes are given.
        rho, shares: offered load and class shares (defaults from spec/params.yaml).
        train_windows: list of TRAIN (date, offset_s) pairs used for mu, forecasters and shapes (default: all 180).
        diag_windows: TRAIN (date, offset_s) pairs for the regime diagnostics (default: same as train_windows).
        counts: fixed viewer/user/device counts per tenant on every date (load rule 'fixed').
        load_rule: 'per_date' (orchestrator decision D2; the default, ``spec/params.yaml: load.rule``),
            'expectation' (the rule before D2) or 'fixed' (implied by ``counts``).
        count_lead_s: video lead time whose traffic sets the per-date counts. Keep the default 4.004 s in a
            lead-time sweep, so the sweep has the default counts and changes only the information.
    """
    if classes is None:
        classes, video_groups = compose.tenant_layout(n)
    classes = tuple(classes)
    if video_groups is None:
        video_groups = tuple(puffer.CHANNELS if c == "video" else None for c in classes)
    video_groups = tuple(None if g is None else tuple(g) for g in video_groups)
    mech = mech or Mechanism()
    rho = float(PARAMS["load"]["rho"] if rho is None else rho)
    shares = default_shares() if shares is None else dict(shares)
    keys = train_window_keys() if train_windows is None else list(train_windows)
    if any(puffer.split_of_date(d) != "train" for d, _ in keys):
        raise ValueError("calibration uses TRAIN dates only")
    for d in puffer.TRAIN_DATES:
        puffer.prepare_date(d)
    r_bar = raca.mean_rate("train", K + 1)
    unit = unit_rates(classes, video_groups, keys, K, lead_s)
    if counts is not None:
        if load_rule not in (None, "fixed"):
            raise ValueError("counts= implies load_rule='fixed'")
        load_rule = "fixed"
        counts = tuple(int(c) for c in counts)
        if len(counts) != len(classes) or any(c < 1 for c in counts):
            raise ValueError("need one positive count per tenant")
    else:
        load_rule = str(PARAMS["load"].get("rule", "per_date")) if load_rule is None else load_rule
        if load_rule not in ("per_date", "expectation"):
            raise ValueError(f"load_rule must be one of {LOAD_RULES} ('fixed' needs counts=)")
        counts = tenant_counts(classes, unit, rho, shares, mech.B, r_bar)    # reference counts (expectation rule)
    dkeys = [tuple(k) for k in (keys if diag_windows is None else diag_windows)]
    if any(puffer.split_of_date(d) != "train" for d, _ in dkeys):
        raise ValueError("diagnostics use TRAIN windows only")
    date_loads = {}
    if load_rule == "per_date":     # D2: per-date counts from each TRAIN date's offered traffic and the TRAIN r_bar
        for d in sorted({d for d, _ in keys} | {d for d, _ in dkeys}):
            date_loads[d] = counts_for_date("train", d, rho, classes, video_groups, r_bar=r_bar, shares=shares,
                                            B=mech.B, K=K, seed=seed, lead_s=count_lead_s, info=True)

    def counts_of(day):
        return date_loads[day].counts if load_rule == "per_date" else counts

    delay_ms = np.array([PARAMS["classes"][c]["delay_budget_ms"] for c in classes], float)
    parts = []
    for d, o in keys:   # keep compact summaries only (per-ms bytes of 180 windows would not fit comfortably)
        p = compose.compose_parts("train", d, o, classes, counts_of(d), video_groups, lead_s=lead_s,
                                  report_lead_s=report_lead_s, K=K, seed=seed)
        p["subvols"] = subwindow_volumes(p.pop("bytes_ms"), delay_ms)
        parts.append(p)
    measured = [p["measured"] for p in parts]
    mu = np.concatenate(measured).mean(0)
    if np.any(mu <= 0):
        raise ValueError("zero TRAIN mean demand: contract cannot be calibrated")
    forecaster = fit_ar(measured, [p["own_history"] for p in parts], history=history, horizon=1,
                        meta=dict(kind="tenant own-history AR", used_by=[i for i, c in enumerate(classes)
                                                                         if c != "video"],
                                  train_windows=len(keys)),
                        splits=[puffer.split_of_date(d) for d, _ in keys])
    contract = contract_from_params(list(classes), mu)
    thetas = [compose.truthful_theta(p, classes, forecaster) for p in parts]

    pooled = {c: [] for c in set(classes)}
    for p, th in zip(parts, thetas):
        for i, u in enumerate(burst_samples(p["subvols"], th, contract.delay_ms, warmup)):
            pooled[classes[i]].append(u)
    from ..model.shapes import fit_shape   # model builder's module (INTERFACES §3)
    class_shape = {c: fit_shape(np.concatenate(v), n_atoms=n_atoms) for c, v in pooled.items()}
    shape = Shape(beta=tuple(class_shape[c][0] for c in classes), rho=tuple(class_shape[c][1] for c in classes))

    idx = {tuple(k): j for j, k in enumerate(keys)}
    zs, rs = [], []
    for k in dkeys:
        if k in idx:
            th, rh = thetas[idx[k]], parts[idx[k]]["rate_hat"]
        else:
            p = compose.compose_parts("train", k[0], k[1], classes, counts_of(k[0]), video_groups, lead_s=lead_s,
                                      report_lead_s=report_lead_s, K=K, seed=seed)
            th, rh = compose.truthful_theta(p, classes, forecaster), p["rate_hat"]
        zs.append(np.minimum(th, contract.C)[warmup:]); rs.append(rh[warmup:])
    diag = regime_diagnostics(np.concatenate(zs), np.concatenate(rs), contract, mech)
    diag["targets"] = check_targets(diag)
    diag["windows"] = [list(k) for k in dkeys]
    th_all = np.concatenate([t[warmup:] for t in thetas])
    m_all = np.concatenate([p["measured"][warmup:] for p in parts])
    # realized per-window loads of the calibration windows: offered (all K intervals, the D2 rule's quantity, over
    # B * TRAIN r_bar) and capacity load (scored intervals over B * the window's mean true per-PRB rate, as
    # eval.headroom.window_load)
    window_offered = {f"{d}|{int(o)}": p["measured"].mean(0).tolist() for (d, o), p in zip(keys, parts)}
    window_capacity = {f"{d}|{int(o)}": float(p["measured"][warmup:].sum(1).mean()
                                              / (mech.B * p["rate_true"][warmup:].mean()))
                       for (d, o), p in zip(keys, parts)}
    meta = dict(train_windows=[list(k) for k in keys], counts_rule=load_rule, count_lead_s=float(count_lead_s),
                target_mbps=tenant_targets(classes, rho, shares, mech.B, r_bar).tolist(),
                offered_load=float(mu.sum() / (mech.B * r_bar)),
                frac_theta_above_C=(th_all > contract.C).mean(0).tolist(),
                report_rel_mae=(np.abs(th_all - m_all).mean(0) / mu).tolist(),
                burst_samples={c: int(sum(len(u) for u in v)) for c, v in pooled.items()},
                mech=dict(B=mech.B, gamma=mech.gamma, eta=mech.eta, Delta=mech.Delta, integer=mech.integer),
                gbr_x_mu={c: float(PARAMS["classes"][c]["gbr_x_mu"]) for c in sorted(set(classes))},
                window_offered_mbps={"train": window_offered}, window_capacity_load={"train": window_capacity},
                date_offered_mbps={"train": {d: v.offered_mbps.tolist() for d, v in sorted(date_loads.items())}})
    date_counts = {"train": {d: v.counts for d, v in sorted(date_loads.items())}} if date_loads else {}
    operator_ar = fit_operator_forecasters(measured, contract, keys=keys)
    return Calibration(classes, counts, video_groups, contract, shape, forecaster, unit, float(r_bar), rho, shares,
                       float(lead_s), float(report_lead_s), int(K), int(warmup), int(seed), diag, meta, operator_ar,
                       load_rule, date_counts)


def attach_date_counts(cal: Calibration, splits=("dev",)) -> Calibration:
    """Return ``cal`` with the per-date counts (D2) of every date of ``splits`` stored in ``date_counts``, and their
    per-window and per-date offered rates in ``meta`` (``window_offered_mbps`` / ``date_offered_mbps``).

    Nothing is fitted: these are the counts :meth:`Calibration.counts_for` would compute on demand, stored so every
    worker composes identical windows and the frozen configuration records them. TEST is refused (its counts are
    computed only after the freeze, behind the gate, and never stored).
    """
    if cal.load_rule != "per_date":
        return cal
    if "test" in splits:
        raise ValueError("TEST counts are never stored: Calibration.counts_for computes them after the freeze")
    dc = {s: dict(v) for s, v in cal.date_counts.items()}
    meta = dict(cal.meta)
    wo = {s: dict(v) for s, v in meta.get("window_offered_mbps", {}).items()}
    do = {s: dict(v) for s, v in meta.get("date_offered_mbps", {}).items()}
    B = int(meta.get("mech", {}).get("B", Mechanism().B))
    for s in splits:
        for d in puffer.split_dates(s):
            dl = counts_for_date(s, d, cal.rho, cal.classes, cal.video_groups, r_bar=cal.r_bar, shares=cal.shares,
                                 B=B, K=cal.K, seed=cal.seed, lead_s=cal.count_lead_s, info=True)
            dc.setdefault(s, {})[d] = dl.counts
            do.setdefault(s, {})[d] = dl.offered_mbps.tolist()
            for o, row in zip(dl.offsets, dl.window_offered_mbps):
                wo.setdefault(s, {})[f"{d}|{o}"] = row.tolist()
    meta.update(window_offered_mbps=wo, date_offered_mbps=do)
    return dataclasses.replace(cal, date_counts=dc, meta=meta)


# ----------------------------------------------------------------------------- data report
def diag_windows_8() -> list[tuple[str, int]]:
    """The 8 TRAIN windows of the data report: every second TRAIN date, offsets spread over the day."""
    offs = compose.window_offsets("train")
    return [(puffer.TRAIN_DATES[2 * i], offs[(3 * i) % len(offs)]) for i in range(8)]


def _fmt(x, nd=3):
    return f"{x:.{nd}f}"


def _pct(x, q):
    return float(np.percentile(np.asarray(x, float), q))


def _dist_cells(x, nd=3) -> str:
    """``mean | sd | min | p5 | p50 | p95 | max`` cells of a sample."""
    x = np.asarray(x, float)
    vals = (x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0, x.min(), _pct(x, 5), _pct(x, 50), _pct(x, 95), x.max())
    return " | ".join(_fmt(v, nd) for v in vals)


def _window_summary(w, mech: Mechanism) -> dict:
    """Compact per-window quantities for the report (the window itself is dropped afterwards)."""
    ct = w.contract
    return dict(date=w.date, offset=int(w.meta["offset_s"]), counts=tuple(w.meta["counts"]),
                z=np.minimum(w.theta, ct.C)[w.warmup:], r=w.rate_hat[w.warmup:],
                offered=w.measured.mean(0), scored=w.measured[w.warmup:].mean(0),
                capacity_load=float(w.measured[w.warmup:].sum(1).mean() / (mech.B * w.rate_true[w.warmup:].mean())),
                wrapped=sum(t.get("wrapped_users", 0) for t in w.meta["tenants"]),
                user_windows=sum(t["count"] for t in w.meta["tenants"] if t["cls"] != "video"))


def _count_range(cal: Calibration, split: str) -> list[str]:
    """Per tenant ``min-max (median)`` of the stored per-date counts of a split, or '-' when none is stored."""
    dc = cal.date_counts.get(split, {})
    if not dc:
        return ["-"] * cal.n
    a = np.array(list(dc.values()))
    return [f"{a[:, i].min()}-{a[:, i].max()} ({np.median(a[:, i]):g})" for i in range(cal.n)]


def data_report(cal: Calibration, path=None, extra: dict | None = None, test_freeze=None, *,
                dev_all: bool = False) -> pathlib.Path:
    """Write ``data/DATA_REPORT.md``: per-split counts, mean rates per class, mean per-PRB rate, the per-date
    counts and realized loads of the D2 load rule, and regime diagnostics at the calibration's load (8 TRAIN windows,
    all TRAIN calibration windows; DEV descriptive: one window per date, and all DEV windows with ``dev_all``).

    TEST statistics (Raca TEST runs, TRACTOR TEST captures) are withheld unless ``test_freeze`` is the frozen
    pre-registration file under ``prereg/`` (``data.puffer_fetch.require_freeze_for_test_reads``: the freeze block
    with code, config and checkpoint hashes; after the TEST download, the very freeze file recorded then): no TEST
    number may inform a pre-registration decision (round-1 review, finding 2; gate strengthened in round 2).
    """
    path = sio.DATA_DIR / "DATA_REPORT.md" if path is None else path
    show_test = False
    if test_freeze is not None:
        from .puffer_fetch import require_freeze_for_test_reads
        require_freeze_for_test_reads(test_freeze)
        show_test = True
    shown = ("train", "dev", "test") if show_test else ("train", "dev")
    mech = Mechanism(**{k: v for k, v in cal.meta["mech"].items()})
    L = []
    L.append("# Data report (builder \"data\")\n")
    L.append("Generated by `python -m slicemech.data.calibrate --report` from local files only (no download). "
             "Units: Mbit/s for rates and demands, Mbit/s per PRB for per-PRB rates, PRB for quotas.\n")
    # --- splits
    L.append("## 1. Splits and source counts\n")
    L.append("| Split | Puffer dates | windows/date | windows | dates with a missing channel | Raca runs (>= %d s) | Raca seconds | mean r (all) | mean r (runs >= %d s) |" % (cal.K + 1, cal.K + 1))
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in shown:
        dates = puffer.split_dates(s)
        local = [d for d in dates if _has_local(d)]
        missing = [f"{d} ({', '.join(puffer.date_info(d)['channels_missing'])})" for d in local
                   if puffer.date_info(d)["channels_missing"]]
        per = len(compose.window_offsets(s))
        runs_all, runs_ok = raca.pool(s), raca.pool(s, cal.K + 1)
        L.append(f"| {s.upper()} | {len(dates)} ({len(local)} local) | {per} | {len(dates) * per} | "
                 f"{'; '.join(missing) or 'none'} | {len(runs_all)} ({len(runs_ok)}) | {sum(len(r) for r in runs_all)} | "
                 f"{_fmt(raca.mean_rate(s))} | {_fmt(raca.mean_rate(s, cal.K + 1))} |")
    if not show_test:
        L.append(f"| TEST | {len(puffer.split_dates('test'))} (fresh, not downloaded) | "
                 f"{len(compose.window_offsets('test'))} | "
                 f"{len(puffer.split_dates('test')) * len(compose.window_offsets('test'))} | withheld | withheld | "
                 "withheld | withheld | withheld |")
    local_all = [d for s in ("train", "dev") for d in puffer.split_dates(s) if _has_local(d)]
    verified = [d for d in local_all if puffer.date_info(d)["verified_hashes"]]
    extra_ch = sorted({c for d in local_all for c in puffer.date_info(d)["extra_channels"]})
    L.append("\nPuffer TEST dates are fresh and not on disk (downloaded only after the pre-registration freeze by "
             f"`data/puffer_fetch.py`). {len(verified)} of {len(local_all)} local TRAIN/DEV objects were hash-checked "
             "against their recorded GCS md5 (or the download sha256 for the 2024-01-01 pilot object) in the same "
             "single gzip pass that wrote `data/puffer/<date>.parquet` (4 formats on every channel: "
             f"{', '.join(puffer.FORMATS)}). Channels outside the fixed set ({', '.join(extra_ch) or 'none'}; "
             "it replaces abc on the dates listed) are kept in the chunk tables but never assigned to viewers; viewers "
             "of a missing channel are spread uniformly over the channels of their group that are present.\n")
    L.append("| Split | Class | TRACTOR traces | seconds | per-user mean (Mbit/s) | trace means (Mbit/s) |")
    L.append("|---|---|---|---|---|---|")
    for s in shown:
        for c in tractor.CLASSES:
            ts = tractor.pool(s, c)
            L.append(f"| {s.upper()} | {c} | {', '.join(t.stem for t in ts)} | {sum(len(t) for t in ts) // 1000} | "
                     f"{_fmt(tractor.pool_mean_mbps(s, c), 4)} | {', '.join(_fmt(t.mean_mbps, 4) for t in ts)} |")
    if show_test:
        L.append("\nTEST rows are shown after the pre-registration freeze (descriptive only).")
    else:
        L.append("\nTEST TRACTOR/Raca statistics are withheld until the pre-registration freeze: every count, rho, "
                 "GBR or regime decision is taken on TRAIN, DEV is descriptive, and no TEST statistic informs a "
                 "decision (those sources were inspected in earlier work, which does not permit tuning on them now).")
    L.append("Idle trim: only `urll` (first campaign, URLLC) loses %d s of all-zero runs > %d s.\n"
             % (tractor.load_trace("urll").info["trimmed_seconds"], tractor.GAP_S))
    # --- calibration
    ct = cal.contract
    B_rbar = mech.B * cal.r_bar
    L.append(f"## 2. Default calibration (TRAIN only; n = {cal.n}, rho = {cal.rho}, shares {cal.shares}, "
             f"video lead {cal.lead_s} s, report lead {cal.report_lead_s * 1000:.0f} ms)\n")
    gx = cal.meta.get("gbr_x_mu", {})
    L.append("Orchestrator decisions applied (TRAIN only, pre-registered):\n")
    L.append(f"- **D1, GBR fractions:** g / mu = {', '.join(f'{c} {v:g}' for c, v in sorted(gx.items()))} "
             "(0.8 x the design values 0.5 / 0.8 / 0.2; the pre-declared first adjustment step; TRAIN evidence in "
             "`code/STATUS.md` §5.2).")
    if cal.load_rule == "per_date":
        L.append("- **D2, offered-load normalization per date** (load rule `per_date`): for every date of every "
                 "split, each tenant's integer count of viewers / users / devices is the one whose mean offered "
                 "traffic over the date's pre-declared windows is closest to its target rho x B x r_bar_TRAIN x "
                 "share / (tenants of the class). The rule reads only offered traffic, never a CQI run or a "
                 "controller outcome, and r_bar is the TRAIN rate on every split. TEST counts are computed only "
                 "after the pre-registration freeze. The reference count below is the expectation rule used before "
                 "D2 (TRAIN unit rate); windows use the per-date counts.\n")
    else:
        L.append(f"- Load rule `{cal.load_rule}`: the same counts on every date.\n")
    L.append(f"Mean per-PRB rate r_bar = {_fmt(cal.r_bar, 4)} Mbit/s per PRB (B r_bar = {_fmt(B_rbar, 2)} "
             f"Mbit/s). Realised TRAIN offered load sum(mu) / (B r_bar) = {_fmt(cal.meta['offered_load'])} "
             f"over {len(cal.meta['train_windows'])} TRAIN windows.\n")
    L.append("| Tenant | class | unit | unit rate (Mbit/s) | reference count | TRAIN date counts: min-max (median) | "
             "DEV date counts: min-max (median) | target (Mbit/s) | mu (Mbit/s) | g | M | C | pi | delay (ms) | "
             "burst atoms: min / max beta | report MAE / mu |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    unit_name = {"video": "viewer", "urllc": "user", "mmtc": "device"}
    tr_rng, dv_rng = _count_range(cal, "train"), _count_range(cal, "dev")
    for i, c in enumerate(cal.classes):
        b = cal.shape.beta[i]
        L.append(f"| {i} | {c} | {unit_name[c]} | {_fmt(cal.unit_mbps[i], 4)} | {cal.counts[i]} | {tr_rng[i]} | "
                 f"{dv_rng[i]} | {_fmt(cal.meta['target_mbps'][i], 2)} | {_fmt(ct.mu[i], 3)} | {_fmt(ct.g[i], 2)} | "
                 f"{_fmt(ct.M[i], 2)} | {_fmt(ct.C[i], 2)} | {ct.pi[i]:g} | {ct.delay_ms[i]:g} | "
                 f"{_fmt(b.min())} / {_fmt(b.max())} | {_fmt(cal.meta['report_rel_mae'][i])} |")
    L.append("\nReport MAE / mu: mean |theta - m| over scored TRAIN intervals divided by mu (video: 0 = exact advance "
             "information; URLLC/mMTC: the tenant's AR(6) on its own zero-delay history).\n")

    # --- compose the report windows (compact summaries only)
    w8 = [cal.window("train", d, o) for d, o in diag_windows_8()]
    offs = compose.window_offsets("dev")
    dev = [_window_summary(cal.window("dev", d, offs[j % len(offs)]), mech) for j, d in enumerate(puffer.DEV_DATES)]
    dev_full = ([_window_summary(cal.window("dev", d, o), mech) for d in puffer.DEV_DATES for o in offs]
                if dev_all else None)

    # --- D2: per-date counts and realized loads
    if cal.load_rule == "per_date":
        L.append("## 3. Offered-load normalization (D2): per-date counts and realized loads\n")
        L.append("Offered load = mean offered traffic (all K intervals) / (B r_bar_TRAIN), the quantity the rule "
                 "matches; target = rho = %g. Tenant error = date mean offered / target - 1. Counts are integers: "
                 "with one tenant per class each tenant's date mean misses its target by at most half the traffic of "
                 "one unit; with several tenants per class the class total does (each tenant within one unit).\n"
                 % cal.rho)
        L.append("| Split | date | counts | offered (Mbit/s) | date offered load | max abs tenant error (%) |")
        L.append("|---|---|---|---|---|---|")
        tgt = np.asarray(cal.meta["target_mbps"], float)
        per_split = {}
        for s in ("train", "dev"):
            do = cal.meta.get("date_offered_mbps", {}).get(s, {})
            for d in sorted(do):
                off = np.asarray(do[d], float)
                load, err = off.sum() / B_rbar, np.abs(off / tgt - 1).max() * 100
                per_split.setdefault(s, []).append((load, err, off / tgt - 1))
                L.append(f"| {s.upper()} | {d} | {list(cal.date_counts[s][d])} | "
                         f"{', '.join(_fmt(x, 2) for x in off)} | {_fmt(load)} | {_fmt(err, 2)} |")
        L.append("")
        for s, rows in per_split.items():
            loads = np.array([r[0] for r in rows])
            rel = np.array([r[2] for r in rows])
            cls_err = {c: np.abs(rel[:, [i for i, x in enumerate(cal.classes) if x == c]]).max() * 100
                       for c in dict.fromkeys(cal.classes)}
            L.append(f"- {s.upper()} ({len(rows)} dates): date offered load {_fmt(loads.min())} to "
                     f"{_fmt(loads.max())} (max |load / rho - 1| = {_fmt(np.abs(loads / cal.rho - 1).max() * 100, 2)} "
                     "%); max |tenant error| by class: "
                     + ", ".join(f"{c} {_fmt(v, 2)} %" for c, v in cls_err.items()) + ".")
        L.append("\n**Realized per-window loads.** Offered load as above, per window. Capacity load = mean total "
                 "measured demand over the scored intervals / (B x the window's mean true per-PRB rate), i.e. "
                 "`eval.headroom.window_load` (it varies with the window's CQI runs, which the rule never reads).\n")
        L.append("| Windows | N | offered load: mean | sd | min | p5 | p50 | p95 | max | capacity load: mean | sd | "
                 "min | p5 | p50 | p95 | max |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        wo_tr = cal.meta.get("window_offered_mbps", {}).get("train", {})
        cap_tr = cal.meta.get("window_capacity_load", {}).get("train", {})
        if wo_tr:
            off_l = [np.sum(v) / B_rbar for v in wo_tr.values()]
            L.append(f"| TRAIN, all calibration windows | {len(off_l)} | {_dist_cells(off_l)} | "
                     f"{_dist_cells(list(cap_tr.values()))} |")
        wo_dv = cal.meta.get("window_offered_mbps", {}).get("dev", {})
        check = None
        if dev_full is not None:
            L.append(f"| DEV, all windows | {len(dev_full)} | "
                     f"{_dist_cells([x['offered'].sum() / B_rbar for x in dev_full])} | "
                     f"{_dist_cells([x['capacity_load'] for x in dev_full])} |")
            if wo_dv:
                gap = max(float(np.abs(x["offered"] - np.asarray(wo_dv[f"{x['date']}|{x['offset']}"])).max())
                          for x in dev_full)
                check = (f"Check: the composed DEV windows reproduce the rule's per-window offered rates to "
                         f"{gap:.2g} Mbit/s (exact integer bytes; float rounding only).")
        elif wo_dv:
            off_l = [np.sum(v) / B_rbar for v in wo_dv.values()]
            L.append(f"| DEV, all windows (offered only) | {len(off_l)} | {_dist_cells(off_l)} | - | - | - | - | - "
                     "| - | - |")
        L.append(f"| DEV, one window per date (as in sections 4-5) | {len(dev)} | "
                 f"{_dist_cells([x['offered'].sum() / B_rbar for x in dev])} | "
                 f"{_dist_cells([x['capacity_load'] for x in dev])} |")
        L.append("")
        if check:
            L.append(check + "\n")
        sec = 4
    else:
        sec = 3

    # --- mean rates per class per split
    L.append(f"## {sec}. Mean demand per tenant (Mbit/s, all intervals) with the calibration's counts\n")
    L.append("| Windows | " + " | ".join(f"{c} (tenant {i})" for i, c in enumerate(cal.classes))
             + " | sum / (B r_bar_TRAIN) | sum / (B r_bar_split) |")
    L.append("|---|" + "---|" * (cal.n + 2))
    rdev = raca.mean_rate("dev", cal.K + 1)
    rows = {"TRAIN, all calibration windows": (ct.mu, cal.r_bar)}
    rows["TRAIN, 8 report windows"] = (np.mean([w.measured.mean(0) for w in w8], 0), cal.r_bar)
    rows[f"DEV, {len(dev)} windows (one per date)"] = (np.mean([x["offered"] for x in dev], 0), rdev)
    if dev_full is not None:
        rows[f"DEV, all {len(dev_full)} windows"] = (np.mean([x["offered"] for x in dev_full], 0), rdev)
    for name, (m, rb) in rows.items():
        L.append(f"| {name} | " + " | ".join(_fmt(x, 3) for x in m) + f" | {_fmt(m.sum() / B_rbar)} | "
                 f"{_fmt(m.sum() / (mech.B * rb))} |")
    wrap_src = dev_full if dev_full is not None else dev
    L.append("\nTEST demand is not computed (Puffer TEST dates are not downloaded). DEV wraps "
             f"{sum(x['wrapped'] for x in wrap_src)} of {sum(x['user_windows'] for x in wrap_src)} TRACTOR user "
             f"windows circularly over the {len(wrap_src)} DEV windows (the DEV URLLC captures are only 202-210 s "
             "long).\n")
    sec += 1

    # --- regime diagnostics
    z = np.concatenate([np.minimum(w.theta, ct.C)[w.warmup:] for w in w8])
    r = np.concatenate([w.rate_hat[w.warmup:] for w in w8])
    d8 = regime_diagnostics(z, r, ct, mech)
    t8 = check_targets(d8)
    tall = cal.diagnostics["targets"]
    tdev = check_targets(regime_diagnostics(np.concatenate([x["z"] for x in dev]),
                                            np.concatenate([x["r"] for x in dev]), ct, mech))
    tdev_all = (check_targets(regime_diagnostics(np.concatenate([x["z"] for x in dev_full]),
                                                 np.concatenate([x["r"] for x in dev_full]), ct, mech))
                if dev_full is not None else None)
    L.append(f"## {sec}. Regime diagnostics (truthful reports z = min(theta, C), r = rate_hat, equal weights; "
             "scored intervals)\n")
    L.append(f"8 TRAIN windows: {', '.join(f'{d} +{o} s' for d, o in diag_windows_8())} ({d8['intervals']} intervals). "
             f"Single tilt = one weight x{TILT:g}. The DEV columns (TRAIN calibration, DEV per-date counts) are "
             "descriptive only; nothing is tuned on them.\n")
    head = "| Diagnostic | target | 8 TRAIN windows | pass | all calibration windows | pass | DEV, one per date | pass |"
    if tdev_all is not None:
        head += " DEV, all windows | pass |"
    L.append(head)
    L.append("|---|---|---|---|---|---|---|---|" + ("---|---|" if tdev_all is not None else ""))
    for k, v in t8.items():
        op = ">=" if v["kind"] == "min" else "<="
        row = (f"| {k} | {op} {v['bound']:g} | {_fmt(v['value'])} | {'yes' if v['ok'] else 'NO'} | "
               f"{_fmt(tall[k]['value'])} | {'yes' if tall[k]['ok'] else 'NO'} | "
               f"{_fmt(tdev[k]['value'])} | {'yes' if tdev[k]['ok'] else 'NO'} |")
        if tdev_all is not None:
            row += f" {_fmt(tdev_all[k]['value'])} | {'yes' if tdev_all[k]['ok'] else 'NO'} |"
        L.append(row)
    L.append(f"\nStage shares (8 windows): need {_fmt(d8['frac_need_stage'])}, leftover {_fmt(d8['frac_leftover_stage'])}, "
             f"overload {_fmt(d8['gbr_outage'])}; mean sum H / B = {_fmt(d8['mean_sumH_over_B'])}; mean idle PRBs "
             f"from flooring = {_fmt(d8['mean_idle_prb_floor'])}.\n")
    iv = [i for i, c in enumerate(cal.classes) if c == "video"]
    gap = max(float(np.abs(w.theta[:, iv] - w.measured[:, iv]).max()) for w in w8)
    vol = max(float(np.abs(w.arrivals_ms.astype(float).reshape(w.K, 1000, w.n).sum(1) - w.measured).max()) for w in w8)
    L.append(f"Causality checks on the 8 windows (lead {cal.lead_s} s >= 1.05 s): max |theta - m| of the video tenant = "
             f"{gap:.3g} Mbit/s (exact to the byte); max |sum of per-ms arrivals - m| = {vol:.2g} Mbit/s (float32 "
             "rounding of the per-ms series only).\n")
    failed = [k for k, v in t8.items() if not v["ok"]] + [k for k, v in tall.items() if not v["ok"]]   # TRAIN only
    if failed:
        L.append(f"**TRAIN targets not met: {', '.join(sorted(set(failed)))}.** Reported only; nothing is tuned "
                 "here.\n")
    else:
        L.append("All five regime targets pass on TRAIN (8 report windows and all calibration windows).\n")
    sec += 1
    if extra:
        L.append(f"## {sec}. Scaling variants (TRAIN calibration, same B, rho and load rule)\n")
        L.append("Rows: TRAIN = all calibration windows (the targets); DEV = all DEV windows with the DEV per-date "
                 "counts (descriptive only)" + ("" if dev_all else "; DEV rows need `dev_all`") + ". Offered load = "
                 "mean offered traffic / (B r_bar_TRAIN) over the windows; date offered load = its range over the "
                 "split's dates.\n")
        L.append("| n | windows | date counts: min-max per tenant | mu (Mbit/s) | offered load | date offered load: "
                 "min-max | " + " | ".join(tall) + " |")
        L.append("|---|---|---|---|---|---|" + "---|" * len(tall))
        for n_, c_ in extra.items():
            bb = int(c_.meta["mech"]["B"]) * c_.r_bar
            rows_ = [("train", c_.diagnostics["targets"], c_.meta["offered_load"])]
            if dev_all:
                summ = [_window_summary(c_.window("dev", d, o), mech) for d in puffer.DEV_DATES
                        for o in compose.window_offsets("dev")]
                tdv = check_targets(regime_diagnostics(np.concatenate([x["z"] for x in summ]),
                                                       np.concatenate([x["r"] for x in summ]), c_.contract, mech))
                rows_.append(("dev", tdv, float(np.mean([x["offered"].sum() for x in summ]) / bb)))
            for s, tg, load in rows_:
                dl = [np.sum(v) / bb for v in c_.meta.get("date_offered_mbps", {}).get(s, {}).values()]
                dc = c_.date_counts.get(s)
                cr = (", ".join(f"{a.min()}-{a.max()}" for a in np.array(list(dc.values())).T) if dc
                      else str(list(c_.counts)))
                mu_s = ", ".join(_fmt(x, 2) for x in c_.mu) if s == "train" else "-"
                L.append(f"| {n_} | {s.upper()}{'' if s == 'train' else ' (descriptive)'} | {cr} | {mu_s} | "
                         f"{_fmt(load)} | {_fmt(min(dl)) + '-' + _fmt(max(dl)) if dl else '-'} | " +
                         " | ".join(f"{_fmt(tg[k]['value'])}{'' if tg[k]['ok'] else ' (NO)'}" for k in tall) + " |")
        L.append("")
    out = sio.confine(path)
    out.write_text("\n".join(L) + "\n")
    return out


def _has_local(day: str) -> bool:
    try:
        puffer.local_source(day)
        return True
    except FileNotFoundError:
        return False


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="TRAIN calibration and data report")
    ap.add_argument("--report", action="store_true", help="write data/DATA_REPORT.md")
    ap.add_argument("--scaling", action="store_true", help="also calibrate n = 6 and 9")
    ap.add_argument("--test-freeze", default=None,
                    help="frozen pre-registration file (prereg/PREREG.md); only then are TEST rows reported")
    ap.add_argument("--operator-ar-only", action="store_true",
                    help="only fit the operator forecasters into the existing n3/n6/n9 calibration files")
    ap.add_argument("--workers", type=int, default=3, help="processes for the n = 3, 6, 9 calibrations")
    ap.add_argument("--dev-one", action="store_true",
                    help="report DEV regime diagnostics on one window per date only (default: all DEV windows)")
    a = ap.parse_args(argv)
    if a.operator_ar_only:
        for name in ("n3_default", "n6_default", "n9_default"):
            if (CALIB_DIR / f"{name}.json").exists():
                print(attach_operator_ar(load_calibration(name)).save(name))
        return
    for d in puffer.TRAIN_DATES + puffer.DEV_DATES:
        puffer.prepare_date(d)
    ns = (3, 6, 9) if a.scaling else (3,)
    if a.workers > 1 and len(ns) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(a.workers, len(ns)), mp_context=mp.get_context("spawn")) as ex:
            names = list(ex.map(_calibrate_and_save, ns))
    else:
        names = [_calibrate_and_save(n) for n in ns]
    cals = {n: load_calibration(name) for n, name in zip(ns, names)}
    for n, name in zip(ns, names):
        print(f"n = {n}: {CALIB_DIR / (name + '.json')}")
    if a.report:
        extra = {n: c for n, c in cals.items() if n != 3}
        print(data_report(cals[3], extra=extra or None, test_freeze=a.test_freeze, dev_all=not a.dev_one))


def _calibrate_and_save(n: int) -> str:
    """Default calibration for n tenants (TRAIN fit, D2 per-date counts for TRAIN and DEV), saved as
    ``n<n>_default``; returns the name."""
    sio.require_free_gb(3)
    name = f"n{n}_default"
    attach_date_counts(calibrate(n=n), ("dev",)).save(name)
    return name


if __name__ == "__main__":
    main()

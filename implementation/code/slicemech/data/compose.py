"""Compose replay windows: per-ms arrivals, truthful reports, measured demand and per-PRB rates for n tenants.

Port of ``ICC_rebuild_2026-10-10/src/rebuild/data._window/_compose`` (L493-553) for any n, with Puffer video
tenants (experiment_design.md §2).

Timing (interval k = ``[k, k+1)`` s from the window start, Delta = 1 s, ``ell`` = report lead = 50 ms):

* ``arrivals_ms[t, i]``: Mbit offered by tenant i in ms ``t`` (float32; exact integer bytes * 8e-6 before the cast).
* ``measured[k, i]`` = m_ik: interval-k ingress (Mbit/s) from the exact integer bytes. The operator may use interval
  j at the start of interval k only if ``j <= k - d`` (:meth:`Window.released`).
* ``theta[k, i]``: the tenant's truthful information, formed at ``k - ell``.
  Video: bytes of interval k from chunks reported by ``k - ell`` (exact next-interval ingress when
  lead >= 1.05 s; plus a persistence forecast otherwise; :func:`slicemech.data.puffer.video_tenant`).
  URLLC/mMTC: the tenant's own AR(6) forecast (horizon 1) on its zero-delay own history
  ``own_history[j, i]`` = ingress over ``[j - ell, j + 1 - ell)`` (Mbit/s, known at ``k - ell`` for every
  ``j <= k - 1``; ``own_history[0]`` is NaN because it needs 50 ms before the window start).
  ``theta`` is NOT clipped to the report ceiling C; strategies clip reports to [0, C] (fraction above C in meta).
* ``rate_true[k, i]``: per-PRB rate (Mbit/s per PRB) of the CQI in second k of the tenant's own Raca run;
  ``rate_hat[k, i] = rate_true[k - 1, i]`` (the run is drawn one second longer so ``rate_hat[0]`` is measured too).

Composition randomness is nested: every viewer, user and CQI draw has its own generator seeded by
``(seed, split, date, offset, tenant index, kind, unit index)``, so the first V viewers / U users of a window are
identical whatever the counts (load sweeps share random numbers).

Offered traffic per unit (orchestrator decision D2, offered-load normalization per date):
:func:`unit_offered_bytes` returns the exact integer bytes that each viewer / user / device of a tenant offers over
a whole window, drawn with the same generators as :func:`compose_parts` (so the bytes of the first U units sum to
the tenant's composed bytes with count U). It reads offered traffic only (Puffer chunk sizes, TRACTOR captures),
never a CQI run, a report or any controller outcome. ``data.calibrate.counts_for_date`` chooses the per-date
counts from it; :func:`offered_units` caches the nested per-unit series so load sweeps reuse them.

Pool parts (round-1 review, finding 1; revised in round 2): ``pool_part`` ('A' | 'B' | None) is the E5 *phase*:
E5 composes the retraining windows of a date with ``pool_part='A'`` and the evaluation windows with ``'B'``. The
phase selects one of two disjoint halves of the exogenous pools (:func:`physical_part`): the interleaved 2-s block
halves of every TRACTOR capture (``tractor.pool``) and the alternating halves of the split's Raca runs
(``raca.pool``), so the "later" evaluation never replays exogenous URLLC/mMTC traffic or CQI the retraining saw.
Which half serves which phase alternates with the date (the halves swap on dates with an odd ordinal; consecutive
TEST Sundays and Mondays therefore get opposite assignments): any residual difference between the halves enters
the retraining-vs-evaluation contrast with opposite signs on half the dates instead of the same sign on every date.
The window id carries the phase; ``meta['pool_half']`` records the half used.

``Window.meta`` keys: ``split, date, offset_s, start_ns, tod_utc_s / tod_start_s`` (UTC seconds of day of the window
start), ``lead_s, report_lead_s, seed, classes, counts, pool_part, pool_half, part_block_s``, ``tenants`` (per
tenant: viewers (channel, format, phase_ns), channel group and the group's channels present (``group_fallback``
when none was present and the viewers use every present channel of ``puffer.CHANNELS``) or users (trace stem,
offset ms), wrapped users, CQI run), ``own_history`` (K, n), ``cqi`` / ``cqi_age`` (K+1, n; row k+1 is interval k),
``rate_age_s`` (K, n; age of the sample behind ``rate_hat``: 1 s + hold age) and ``frac_theta_above_C`` (n,).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np

from .. import io as sio
from ..config import Contract, PARAMS
from ..forecast import ARForecast
from . import puffer, raca, tractor

CLASSES = ("video", "urllc", "mmtc")
SPLIT_CODE = {"train": 0, "dev": 1, "test": 2}
_KIND = {"viewer": 0, "user": 1, "cqi": 2}
DEFAULT_SEED = 20261018
K_DEFAULT = int(PARAMS["windows"]["length_s"])
WARMUP_DEFAULT = int(PARAMS["windows"]["warmup_s"])
LEAD_DEFAULT = float(PARAMS["information"]["video_lead_time_s"])
REPORT_LEAD_DEFAULT = float(PARAMS["information"]["report_lead_ms"]) / 1000.0

#: Video channel groups of the scaling variants (experiment_design.md E6).
VIDEO_GROUPS = {
    3: (puffer.CHANNELS,),
    6: (("abc", "cbs", "cw"), ("fox", "nbc", "pbs", "univision")),
    9: (("abc", "cbs"), ("cw", "fox"), ("nbc", "pbs", "univision")),
}
#: Window start offsets: (first offset s after 11:00 UTC, spacing s, windows per date).
OFFSETS = {"train": (60, 7200, 12), "dev": (60, 14400, 6), "test": (60, 10800, 8)}


POOL_PARTS = ("A", "B")


def part_flip(date: str) -> bool:
    """True when the E5 phases swap pool halves on ``date`` (odd date ordinal; see the module docstring)."""
    return _dt.date.fromisoformat(date).toordinal() % 2 == 1


def physical_part(date: str, pool_part: str | None) -> str | None:
    """Pool half ('A'|'B', as in ``tractor.pool`` / ``raca.pool``) that the E5 phase ``pool_part`` uses on ``date``."""
    if pool_part is None:
        return None
    if pool_part not in POOL_PARTS:
        raise ValueError(f"pool_part must be one of {POOL_PARTS} or None, got {pool_part!r}")
    return POOL_PARTS[1 - POOL_PARTS.index(pool_part)] if part_flip(date) else pool_part


def tenant_layout(n: int = 3) -> tuple[tuple[str, ...], tuple]:
    """Classes and video channel groups of the standard n = 3, 6, 9 compositions (n/3 tenants per class)."""
    if n not in VIDEO_GROUPS:
        raise ValueError("standard layouts exist for n in {3, 6, 9}")
    m = n // 3
    classes = tuple(c for c in CLASSES for _ in range(m))
    groups = tuple(list(VIDEO_GROUPS[n]) + [None] * (2 * m))
    return classes, groups


def window_offsets(split: str) -> tuple[int, ...]:
    """Pre-declared window start offsets (s after 11:00 UTC) of a split's dates."""
    a, step, count = OFFSETS[split]
    return tuple(a + step * i for i in range(count))


@dataclass(frozen=True)
class Window:
    """One composed replay window (INTERFACES §8). Arrays have the tenant axis last and are read-only."""

    id: str
    split: str
    date: str
    n: int
    cls: tuple
    arrivals_ms: np.ndarray      # (T_ms, n) float32, Mbit per ms
    theta: np.ndarray            # (K, n) Mbit/s, truthful information (not clipped to C)
    measured: np.ndarray         # (K, n) Mbit/s, measured demand m
    rate_true: np.ndarray        # (K, n) Mbit/s per PRB, CQI of interval k
    rate_hat: np.ndarray         # (K, n) Mbit/s per PRB, rate_true shifted by one interval
    warmup: int
    contract: Contract
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        K = self.theta.shape[0]
        n = self.n
        if len(self.cls) != n or self.contract.n != n or tuple(self.contract.cls) != tuple(self.cls):
            raise ValueError("class list, n and contract disagree")
        if self.arrivals_ms.shape != (K * 1000, n) or self.arrivals_ms.dtype != np.float32:
            raise ValueError("arrivals_ms must be float32 of shape (K*1000, n)")
        for name in ("theta", "measured", "rate_true", "rate_hat"):
            a = getattr(self, name)
            if a.shape != (K, n) or not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError(f"{name} must be finite, nonnegative, shape ({K}, {n})")
        if np.any(self.rate_true <= 0) or np.any(self.rate_hat <= 0):
            raise ValueError("per-PRB rates must be positive")
        if not 0 <= self.warmup < K:
            raise ValueError("warm-up must leave scored intervals")
        for name in ("arrivals_ms", "theta", "measured", "rate_true", "rate_hat"):
            getattr(self, name).flags.writeable = False

    @property
    def K(self) -> int:
        """Number of intervals."""
        return self.theta.shape[0]

    @property
    def T_ms(self) -> int:
        """Number of 1-ms slots."""
        return self.arrivals_ms.shape[0]

    @property
    def scored(self) -> slice:
        """Slice of scored intervals (after the warm-up)."""
        return slice(self.warmup, self.K)

    def released(self, k: int, d: int) -> np.ndarray:
        """Measured demand the operator may use at the start of interval k with delay d: ``measured[: k - d + 1]``.

        ``d`` is an integer number of intervals >= 1 (as in ``control.features`` and ``sim.episode``): with d = 0 the
        counter of interval k itself would be returned, which nobody has measured at the start of k (round-2 review).
        """
        if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1:
            raise ValueError(f"measurement delay d must be an integer >= 1 interval, got {d!r}")
        if not 0 <= k < self.K:
            raise ValueError("need 0 <= k < K")
        return self.measured[: max(0, k - d + 1)]


def _rng(seed: int, split: str, date: str, offset_s: int, tenant: int, kind: str, unit: int) -> np.random.Generator:
    return np.random.default_rng([seed, SPLIT_CODE[split], int(date.replace("-", "")), offset_s, tenant,
                                  _KIND[kind], unit])


def _shifted_history(bytes_ms: np.ndarray, K: int, ell_ms: int) -> np.ndarray:
    """Own history (Mbit/s) over [j - ell, j + 1 - ell) for j = 1..K-1 from per-ms bytes; NaN at j = 0."""
    c = np.r_[0, np.cumsum(bytes_ms)]
    j = np.arange(1, K)
    h = np.full(K, np.nan)
    h[1:] = (c[1000 * (j + 1) - ell_ms] - c[1000 * j - ell_ms]) * 8e-6
    if ell_ms == 0:
        h[0] = c[1000] * 8e-6
    return h


def _video_group(date: str, group) -> tuple[tuple[str, ...], bool]:
    """Channels the viewers of a video tenant with channel ``group`` use on ``date``, and whether it fell back.

    Pre-registered handling: viewers of a missing channel are reassigned uniformly to the group's channels present;
    if the whole group is missing, to every channel of the fixed universe that is present. Channels outside
    ``puffer.CHANNELS`` (e.g. 'kron' replacing 'abc' on some dates) are never used.
    """
    present = puffer.channels_present(date)
    g = tuple(ch for ch in group if ch in present)
    fallback = not g
    if fallback:
        g = tuple(ch for ch in puffer.CHANNELS if ch in present)
    if not g:
        raise ValueError(f"{date}: none of the channels {puffer.CHANNELS} is present")
    return g, fallback


def _check_split(split: str, date: str) -> None:
    if split not in SPLIT_CODE:
        raise ValueError(f"unknown split {split!r}")
    if puffer.split_of_date(date) != split:
        raise ValueError(f"Puffer date {date} is not in split {split}")


_CUMSUM: dict = {}


def _trace_cumsum(trace) -> np.ndarray:
    """``[0, cumsum(bytes_ms)]`` (int64) of a TRACTOR trace or pool half, cached by stem."""
    c = _CUMSUM.get(trace.stem)
    if c is None or len(c) != len(trace) + 1:
        c = np.concatenate([[0], np.cumsum(trace.bytes_ms, dtype=np.int64)])
        c.flags.writeable = False
        _CUMSUM[trace.stem] = c
    return c


def _circular_sum(c: np.ndarray, offset_ms: int, length_ms: int) -> int:
    """Bytes of ``length_ms`` bins from ``offset_ms`` with circular wrap (as ``tractor.user_slice``), from the
    cumulative sum ``c`` (length L + 1)."""
    L = len(c) - 1
    q, r = divmod(int(length_ms), L)
    end = int(offset_ms) + r
    part = int(c[end] - c[offset_ms]) if end <= L else int(c[L] - c[offset_ms] + c[end - L])
    return q * int(c[L]) + part


def unit_offered_bytes(split: str, date: str, offset_s: int, tenant: int, cls: str, start: int, stop: int,
                       video_group=None, *, lead_s: float = LEAD_DEFAULT, K: int = K_DEFAULT,
                       seed: int = DEFAULT_SEED, pool_part: str | None = None) -> np.ndarray:
    """Exact bytes (int64, shape ``(stop - start,)``) offered over the whole window ``[0, K)`` s by the units
    ``start .. stop-1`` (viewers, users or devices) of tenant ``tenant`` of class ``cls``.

    The draws are those of :func:`compose_parts` (same generators, channel-group rule and pool half), so for every
    count U: ``unit_offered_bytes(..., 0, U).sum() == compose_parts(..., counts with U at tenant)['bytes_ms'][:,
    tenant].sum()``. Only offered traffic is read (Puffer chunk tables, TRACTOR captures); no CQI run is drawn.
    Video bytes are the chunk bytes paced into the window (``puffer.bytes_released``, identical to the per-ms sum
    of ``puffer.video_tenant``); user bytes are the circular window sums of ``tractor.user_slice``.
    """
    _check_split(split, date)
    if cls not in CLASSES:
        raise ValueError(f"classes must be in {CLASSES}")
    if isinstance(start, bool) or not 0 <= int(start) <= int(stop):
        raise ValueError("need 0 <= start <= stop")
    start, stop = int(start), int(stop)
    out = np.zeros(stop - start, np.int64)
    if stop == start:
        return out
    T_ms = int(K) * 1000
    d0, d1 = puffer.day_bounds_ns(date)
    start_ns = d0 + int(offset_s) * puffer.NS
    if cls == "video":
        group, _ = _video_group(date, puffer.CHANNELS if video_group is None else tuple(video_group))
        lead_ns = round(lead_s * 1e9)
        end_ns = start_ns + int(K) * puffer.NS
        if start_ns - lead_ns - 2 * puffer.PACING_NS < d0 or end_ns > d1:
            raise ValueError("window (with pacing pre-roll) must lie inside the Puffer date")
        streams = [_rng(seed, split, date, offset_s, tenant, "viewer", v) for v in range(start, stop)]
        viewers = puffer.viewer_draws(stop - start, group, streams)
        chunks = puffer.load_chunks(date)
        for j, v in enumerate(viewers):
            t, size = chunks[(v["channel"], v["format"])]
            shift = lead_ns + v["phase_ns"]
            lo = int(np.searchsorted(t, start_ns - puffer.PACING_NS - shift, side="right"))
            hi = int(np.searchsorted(t, end_ns - shift, side="left"))
            out[j] = int(puffer.bytes_released(t[lo:hi] + shift, size[lo:hi], start_ns, end_ns).sum())
    else:
        pool = tractor.pool(split, cls, part=physical_part(date, pool_part))
        for j, u in enumerate(range(start, stop)):
            ti, off = tractor.draw_user(pool, _rng(seed, split, date, offset_s, tenant, "user", u))
            out[j] = _circular_sum(_trace_cumsum(pool[ti]), off, T_ms)
    return out


_UNITS: dict = {}


def offered_units(split: str, date: str, offset_s: int, tenant: int, cls: str, n_units: int, video_group=None, *,
                  lead_s: float = LEAD_DEFAULT, K: int = K_DEFAULT, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Bytes offered by the first ``n_units`` units of a tenant over one window (:func:`unit_offered_bytes` with
    ``pool_part=None``), cached and extended on demand: the draws are nested, so a load sweep reuses them.
    Read-only int64 array of shape ``(n_units,)``."""
    g = None if video_group is None else tuple(video_group)
    key = (split, date, int(offset_s), int(tenant), cls, g, round(lead_s * 1e9), int(K), int(seed))
    have = _UNITS.get(key, np.zeros(0, np.int64))
    if len(have) < n_units:
        more = unit_offered_bytes(split, date, offset_s, tenant, cls, len(have), int(n_units), g, lead_s=lead_s,
                                  K=K, seed=seed)
        have = np.concatenate([have, more])
        have.flags.writeable = False
        _UNITS[key] = have
    return have[: int(n_units)]


def compose_parts(split: str, date: str, offset_s: int, classes, counts, video_groups=None, *,
                  lead_s: float = LEAD_DEFAULT, report_lead_s: float = REPORT_LEAD_DEFAULT, K: int = K_DEFAULT,
                  seed: int = DEFAULT_SEED, pool_part: str | None = None) -> dict:
    """Contract-free parts of a window (used by calibration and :func:`compose_window`).

    Returns dict: ``bytes_ms`` (T_ms, n) int64, ``measured`` (K, n) Mbit/s, ``own_history`` (K, n) Mbit/s,
    ``video_report`` (K, n) Mbit/s (NaN for non-video tenants), ``rate_true``/``rate_hat``/``cqi``/``cqi_age``
    (K, n), and ``meta`` with every draw.

    Raises ValueError when the Puffer ``date`` is not in ``split``: the split selects the TRACTOR captures and Raca
    runs, so a mismatched date would mix splits (e.g. TEST traces on a TRAIN date before the freeze; round-2 review).
    """
    classes = tuple(classes)
    n = len(classes)
    counts = tuple(int(c) for c in counts)
    if split not in SPLIT_CODE or len(counts) != n or any(c < 1 for c in counts):
        raise ValueError("need a known split and one positive count per tenant")
    if puffer.split_of_date(date) != split:
        raise ValueError(f"Puffer date {date} is not in split {split}")
    if any(c not in CLASSES for c in classes):
        raise ValueError(f"classes must be in {CLASSES}")
    if video_groups is None:
        video_groups = tuple(puffer.CHANNELS if c == "video" else None for c in classes)
    if len(video_groups) != n:
        raise ValueError("one video group (or None) per tenant")
    ell_ms = round(report_lead_s * 1000)
    if abs(ell_ms - report_lead_s * 1000) > 1e-9 or not 0 <= ell_ms < 1000:
        raise ValueError("report lead must be a whole number of ms in [0, 1000)")
    lead_ns = round(lead_s * 1e9)
    T_ms = K * 1000
    d0, _ = puffer.day_bounds_ns(date)
    start_ns = d0 + int(offset_s) * puffer.NS
    bytes_ms = np.zeros((T_ms, n), np.int64)
    video_report = np.full((K, n), np.nan)
    own = np.zeros((K, n))
    rate_true = np.zeros((K, n)); rate_hat = np.zeros((K, n))
    cqi = np.zeros((K + 1, n), np.int64); cqi_age = np.zeros((K + 1, n), np.int64)
    half = physical_part(date, pool_part)
    runs = raca.pool(split, K + 1, part=half)
    tenants = []
    for i, (c, cnt) in enumerate(zip(classes, counts)):
        info = dict(cls=c, count=cnt)
        if c == "video":
            group, group_fallback = _video_group(date, video_groups[i])
            streams = [_rng(seed, split, date, offset_s, i, "viewer", v) for v in range(cnt)]
            viewers = puffer.viewer_draws(cnt, group, streams)
            vt = puffer.video_tenant(date, start_ns, K, viewers, lead_ns, ell_ms * puffer.MS)
            bytes_ms[:, i] = vt["arrivals_bytes"]
            video_report[:, i] = vt["report_bytes"] * 8e-6
            info.update(viewers=viewers, group=list(video_groups[i]), group_present=list(group),
                        group_fallback=group_fallback, exact_reports=vt["exact"], chunks_used=vt["chunks_used"])
        else:
            pool = tractor.pool(split, c, part=half)
            draws = [tractor.draw_user(pool, _rng(seed, split, date, offset_s, i, "user", u)) for u in range(cnt)]
            total, wrapped = tractor.users_sum(pool, draws, T_ms)
            bytes_ms[:, i] = total
            info.update(users=[(pool[j].stem, off) for j, off in draws], wrapped_users=wrapped)
        run, off = raca.draw_window(runs, K + 1, _rng(seed, split, date, offset_s, i, "cqi", 0))
        cqi[:, i] = run.cqi[off: off + K + 1]
        cqi_age[:, i] = run.age[off: off + K + 1]
        r = run.rate[off: off + K + 1]
        rate_hat[:, i], rate_true[:, i] = r[:-1], r[1:]
        info.update(cqi_run=dict(run=run.run_id, file=run.file, date=run.date, offset_s=off))
        own[:, i] = _shifted_history(bytes_ms[:, i], K, ell_ms)
        tenants.append(info)
    measured = bytes_ms.reshape(K, 1000, n).sum(1) * 8e-6
    meta = dict(split=split, date=date, offset_s=int(offset_s), start_ns=int(start_ns),
                tod_utc_s=int((start_ns // puffer.NS) % 86400), lead_s=lead_s, report_lead_s=report_lead_s,
                seed=seed, classes=list(classes), counts=list(counts), pool_part=pool_part, pool_half=half,
                part_block_s=None if half is None else tractor.PART_BLOCK_S, tenants=tenants)
    return dict(bytes_ms=bytes_ms, measured=measured, own_history=own, video_report=video_report,
                rate_true=rate_true, rate_hat=rate_hat, cqi=cqi, cqi_age=cqi_age, meta=meta)


def truthful_theta(parts: dict, classes, forecaster: ARForecast | None) -> np.ndarray:
    """Truthful information (K, n): video reports for video tenants, the tenant AR forecast (horizon 1) otherwise."""
    theta = parts["video_report"].copy()
    tract = [i for i, c in enumerate(classes) if c != "video"]
    if tract:
        if forecaster is None or forecaster.horizon != 1 or forecaster.n != len(classes):
            raise ValueError("TRACTOR tenants need a horizon-1 per-tenant forecaster over all n tenants")
        f = forecaster.forecast_series(parts["own_history"])
        theta[:, tract] = f[:, tract]
    if np.isnan(theta).any():
        raise AssertionError("missing truthful information")
    return theta


def window_id(split: str, date: str, offset_s: int, classes, counts, lead_s, seed, pool_part: str | None = None) -> str:
    """Stable window identifier: split, date, offset, the pool part (if any) and a short digest of the composition."""
    tag = sio.digest(dict(classes=list(classes), counts=list(counts), lead=lead_s, seed=seed))[:8]
    part = "" if pool_part is None else f"-p{pool_part}"
    return f"{split}-{date}-o{int(offset_s):05d}{part}-{tag}"


def compose_window(split: str, date: str, offset_s: int, classes, counts, contract: Contract,
                   forecaster: ARForecast | None, *, video_groups=None, lead_s: float = LEAD_DEFAULT,
                   report_lead_s: float = REPORT_LEAD_DEFAULT, K: int = K_DEFAULT, warmup: int = WARMUP_DEFAULT,
                   seed: int = DEFAULT_SEED, pool_part: str | None = None) -> Window:
    """Build a :class:`Window` for a split, Puffer date and start offset (s after 11:00 UTC of that date);
    ``pool_part`` ('A'|'B', the E5 phase) draws TRACTOR users and Raca runs from one disjoint half of the split's
    pools (:func:`physical_part`)."""
    parts = compose_parts(split, date, offset_s, classes, counts, video_groups, lead_s=lead_s,
                          report_lead_s=report_lead_s, K=K, seed=seed, pool_part=pool_part)
    theta = truthful_theta(parts, tuple(classes), forecaster)
    # rate_hat[k] is the CQI sample of second k-1: one interval old at the decision, plus any hold age
    rate_age_s = 1.0 + parts["cqi_age"][:K].astype(float)
    meta = dict(parts["meta"], own_history=parts["own_history"], cqi=parts["cqi"], cqi_age=parts["cqi_age"],
                rate_age_s=rate_age_s, tod_start_s=float(parts["meta"]["tod_utc_s"]),
                frac_theta_above_C=(theta > contract.C).mean(0).tolist())
    return Window(id=window_id(split, date, offset_s, classes, counts, lead_s, seed, pool_part), split=split, date=date,
                  n=len(classes), cls=tuple(classes), arrivals_ms=(parts["bytes_ms"] * 8e-6).astype(np.float32),
                  theta=theta, measured=parts["measured"], rate_true=parts["rate_true"],
                  rate_hat=parts["rate_hat"], warmup=warmup, contract=contract, meta=meta)

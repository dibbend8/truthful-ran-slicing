"""Input files of the 5G-LENA open-loop quota replay (INTERFACES.md §10; experiment_design.md E7).

Formats are those read by ``ICC/code/ns3/vslice-nr-5g.cc`` (header comment and ``ReadCsv``):

* ``--scheduleFile``: CSV, one row per bin of ``--binMs`` ms (default 1 ms), header ``bin,c0,...,c{cols-1}``, then
  ``<bin index>,<IP bytes of column 0>,...``. Columns are the 3 slices in ns-3 order (aggregate mode; ns-3 cuts each
  bin's bytes into packets and deals them round-robin to the slice's UEs) or 3 x U columns slice-major (per-UE mode,
  ``--uePerSlice=U``). Lines end with LF only (a CRLF schedule made an older harness abort).
* ``--budgetFile``: CSV without header, row k = ``n0,n1,n2`` = PRB quotas of epoch k in ns-3 slice order (the last row
  repeats if the run is longer).

ns-3 slice order is fixed: 0 eMBB (our 'video' tenant), 1 URLLC ('urllc'), 2 mMTC ('mmtc'). The scenario has exactly
three slices, so the NR replay needs n = 3 with one tenant of each class; tenant arrays are re-ordered here.

Minimum quota: the scenario floors every slice at ``--minPrb`` PRBs (a smaller mask can deadlock 5G-LENA, see SetMask
in the .cc). Feeding it smaller quotas would silently exceed the cell budget, so ``raise_min_prb`` raises every quota
below ``min_prb`` (default 4) to ``min_prb`` by moving PRBs one at a time from the currently largest quota (the cell
total is unchanged). Only if no other slice has more than ``min_prb`` PRBs are idle PRBs (B - sum of quotas) used; if
neither suffices a ``ValueError`` is raised. ``prefer_idle=True`` uses idle PRBs first instead (other quotas
untouched). Every move is logged (``logging`` and the returned table), and the same
adjusted quotas must feed the fluid comparison.

Units: arrivals in Mbit per ms (``Window.arrivals_ms``), schedule in IP bytes per bin (1 Mbit = 125,000 bytes),
quotas in PRB, intervals (= ns-3 epochs) of 1 s = 1000 bins of 1 ms.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("slicemech.nr.schedule")

NR_SLICES = ("eMBB", "URLLC", "mMTC")
NR_SLICE_CLASS = ("video", "urllc", "mmtc")        # tenant class carried by each ns-3 slice
BYTES_PER_MBIT = 1e6 / 8.0
MS_PER_EPOCH = 1000
MIN_PRB = 4


def slice_order(cls: Sequence[str]) -> np.ndarray:
    """Tenant index carried by each ns-3 slice (eMBB, URLLC, mMTC); requires one tenant of each class (n = 3)."""
    cls = tuple(cls)
    if len(cls) != 3 or sorted(cls) != sorted(NR_SLICE_CLASS):
        raise ValueError(f"the NR scenario has exactly 3 slices (video, urllc, mmtc); got classes {cls}")
    return np.array([cls.index(c) for c in NR_SLICE_CLASS], dtype=np.int64)


def _cumulative_round(bytes_per_bin: np.ndarray) -> np.ndarray:
    """Integer bytes per bin whose running sums equal round(running sums of the exact bytes).

    Rounding each bin separately would drift by up to 0.5 B per bin (100 kB over a 200 s window); rounding the
    cumulative curve keeps every prefix (hence the window total) within 0.5 B and every epoch total within 1 B of the
    exact volume.
    """
    cum = np.rint(np.cumsum(bytes_per_bin, axis=0))
    out = np.diff(cum, axis=0, prepend=np.zeros((1,) + cum.shape[1:]))
    if np.any(out < 0):
        raise ValueError("negative bytes in a bin (arrivals must be >= 0)")
    return out.astype(np.int64)


def build_schedule(arrivals_ms, cls: Sequence[str], ue_arrivals_ms=None, n_ue: int = 4,
                   window_id: str | None = None) -> dict:
    """Per-ms IP-byte schedule of one window in ns-3 slice order.

    Args:
        arrivals_ms: tenant arrivals, shape (T_ms, 3), Mbit per ms (``Window.arrivals_ms``); T_ms a multiple of 1000.
        cls: tenant classes (n = 3, one of each of video/urllc/mmtc).
        ue_arrivals_ms: optional per-UE split, shape (T_ms, 3, n_ue), Mbit per ms, summing over UEs to
            ``arrivals_ms``; if given the schedule is per-UE (3 * n_ue columns, slice-major), else aggregate.
        n_ue: UEs per slice (``--uePerSlice``).
        window_id: carried into the result for bookkeeping.

    Returns:
        dict(bytes=int64 (T_ms, cols), per_ue, n_ue, K (epochs), L=1000 (bins per epoch), order (3,) tenant index per
        slice, cls, window_id, interval_mbit (K, 3) Mbit per epoch per slice of the written integer bytes).
    """
    a = np.asarray(arrivals_ms, dtype=np.float64)
    order = slice_order(cls)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"arrivals_ms must have shape (T_ms, 3), got {a.shape}")
    if a.shape[0] == 0 or a.shape[0] % MS_PER_EPOCH:
        raise ValueError(f"T_ms = {a.shape[0]} must be a positive multiple of {MS_PER_EPOCH}")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("arrivals must be finite and >= 0 (Mbit per ms)")
    a = a[:, order]
    if ue_arrivals_ms is None:
        b = _cumulative_round(a * BYTES_PER_MBIT)
        per_ue = False
    else:
        u = np.asarray(ue_arrivals_ms, dtype=np.float64)
        if u.shape != (a.shape[0], 3, n_ue):
            raise ValueError(f"ue_arrivals_ms must have shape {(a.shape[0], 3, n_ue)}, got {u.shape}")
        u = u[:, order, :]
        if not np.allclose(u.sum(2), a, rtol=1e-6, atol=1e-12):
            raise ValueError("per-UE arrivals do not sum to the tenant arrivals")
        b = _cumulative_round(u.reshape(a.shape[0], 3 * n_ue) * BYTES_PER_MBIT)
        per_ue = n_ue > 1
    K = a.shape[0] // MS_PER_EPOCH
    interval_mbit = b.reshape(K, MS_PER_EPOCH, 3, -1).sum((1, 3)) / BYTES_PER_MBIT
    return dict(bytes=b, per_ue=per_ue, n_ue=int(n_ue), K=int(K), L=MS_PER_EPOCH, order=order, cls=tuple(cls),
                window_id=window_id, interval_mbit=interval_mbit)


def build_schedule_from_window(window, ue_arrivals_ms=None, n_ue: int = 4) -> dict:
    """``build_schedule`` for a ``data.Window`` (uses ``arrivals_ms``, ``cls`` and ``id``)."""
    return build_schedule(window.arrivals_ms, window.cls, ue_arrivals_ms=ue_arrivals_ms, n_ue=n_ue,
                          window_id=getattr(window, "id", None))


def write_schedule(sched: dict, path: str | Path, bin_ms: int = 1) -> Path:
    """Write the ``--scheduleFile`` CSV (header ``bin,c0,...``; integer bytes; LF line ends).

    ``bin_ms > 1`` sums consecutive 1-ms rows (then pass ``--binMs=bin_ms`` to ns-3); it must divide 1000.
    """
    B = np.asarray(sched["bytes"])
    if B.dtype.kind not in "iu":
        raise TypeError("schedule bytes must be integers")
    if bin_ms < 1 or MS_PER_EPOCH % bin_ms:
        raise ValueError("bin_ms must divide 1000")
    if bin_ms > 1:
        B = B.reshape(-1, bin_ms, B.shape[1]).sum(1)
    path = Path(path)
    with open(path, "w", newline="\n") as f:
        f.write("bin," + ",".join(f"c{j}" for j in range(B.shape[1])) + "\n")
        np.savetxt(f, np.column_stack([np.arange(len(B), dtype=np.int64), B]), fmt="%d", delimiter=",",
                   newline="\n")
    return path


def read_schedule(path: str | Path) -> np.ndarray:
    """Read a schedule CSV back as int64 (bins, cols) (verification helper; skips the header and '#' lines)."""
    df = pd.read_csv(path, comment="#")
    if list(df.columns[:1]) != ["bin"]:
        raise ValueError("schedule must start with a 'bin' column")
    if not np.array_equal(df["bin"].to_numpy(), np.arange(len(df))):
        raise ValueError("schedule bins must be 0, 1, 2, ...")
    return df.drop(columns="bin").to_numpy(dtype=np.int64)


# --------------------------------------------------------------------------- PRB budgets
def _as_quotas(quotas, B: int) -> np.ndarray:
    q = np.asarray(quotas)
    if q.ndim != 2 or q.shape[0] == 0:
        raise ValueError(f"quotas must have shape (K, n), got {q.shape}")
    if q.dtype.kind == "f":
        if not np.all(np.isfinite(q)) or np.any(q != np.rint(q)):
            raise ValueError("quotas must be integers (PRB); floor fractional quotas before writing a budget")
    elif q.dtype.kind not in "iu":
        raise TypeError("quotas must be numeric")
    q = q.astype(np.int64)
    if np.any(q < 0):
        raise ValueError("quotas must be >= 0")
    if np.any(q.sum(1) > B):
        raise ValueError(f"quotas exceed the cell budget B = {B} in {int((q.sum(1) > B).sum())} interval(s)")
    return q


def raise_min_prb(quotas, B: int = 106, min_prb: int = MIN_PRB, names: Sequence[str] | None = None,
                  prefer_idle: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Raise every quota below ``min_prb`` to ``min_prb`` (see module docstring for the rule).

    Args:
        quotas: integer quotas, shape (K, n), PRB, row sums <= B.
        B: PRBs per slot.
        min_prb: per-slice floor (``--minPrb``).
        names: labels of the n columns for the log (default 'c0', 'c1', ...).
        prefer_idle: False (INTERFACES.md §10 rule): take from the largest quota first, idle PRBs only as a fallback;
            the cell total is unchanged. True: use idle PRBs (B - sum of quotas) first, which leaves the other
            tenants' quotas untouched and raises the total (still <= B); the largest quota donates only the rest.

    Returns:
        (adjusted (K, n) int64, log) where log has one row per (interval, recipient, source) with columns
        k, recipient, from_quota, to_quota, donor ('idle' for idle PRBs), prb_moved.

    Raises:
        ValueError: if an interval cannot give every slice ``min_prb`` PRBs within B.
    """
    q = _as_quotas(quotas, B)
    K, n = q.shape
    names = list(names) if names is not None else [f"c{j}" for j in range(n)]
    if len(names) != n:
        raise ValueError("names must have one label per column")
    if n * min_prb > B:
        raise ValueError(f"{n} slices x min_prb {min_prb} exceeds B = {B}")
    out = q.copy()
    rows = []
    for k in np.flatnonzero((q < min_prb).any(1)):
        x = out[k]
        idle = B - int(x.sum())
        for j in np.flatnonzero(x < min_prb):
            start = int(x[j])
            moved: dict[str, int] = {}
            while x[j] < min_prb:
                donors = [d for d in range(n) if d != j and x[d] > min_prb]
                if prefer_idle and idle > 0:
                    idle -= 1
                    src = "idle"
                elif donors:
                    d = max(donors, key=lambda c: (x[c], -c))          # largest quota, lowest index on ties
                    x[d] -= 1
                    src = names[d]
                elif idle > 0:
                    idle -= 1
                    src = "idle"
                else:
                    raise ValueError(f"interval {k}: cannot give every slice {min_prb} PRB within B = {B} "
                                     f"(quotas {q[k].tolist()})")
                x[j] += 1
                moved[src] = moved.get(src, 0) + 1
            for src, cnt in moved.items():
                rows.append(dict(k=int(k), recipient=names[j], from_quota=start, to_quota=int(x[j]), donor=src,
                                 prb_moved=cnt))
                log.info("min-PRB adjustment: interval %d, %s %d -> %d PRB, %d PRB from %s", k, names[j], start,
                         int(x[j]), cnt, src)
    table = pd.DataFrame(rows, columns=["k", "recipient", "from_quota", "to_quota", "donor", "prb_moved"])
    if not table.empty:
        log.info("min-PRB adjustment changed %d of %d intervals", table["k"].nunique(), K)
    return out, table


def write_budget(quotas, cls: Sequence[str], path: str | Path, B: int = 106, min_prb: int = MIN_PRB,
                 prefer_idle: bool = False) -> tuple[Path, np.ndarray, pd.DataFrame]:
    """Adjust tenant-order quotas for ``--minPrb`` and write the ``--budgetFile`` in ns-3 slice order.

    Args:
        quotas: integer quotas in tenant order, shape (K, 3), PRB.
        cls: tenant classes (defines the slice mapping).
        path: output CSV.
        B, min_prb, prefer_idle: see ``raise_min_prb``.

    Returns:
        (path, adjusted quotas in TENANT order (K, 3) for the fluid comparison, adjustment log with slice names).
    """
    order = slice_order(cls)
    q = _as_quotas(quotas, B)
    if q.shape[1] != 3:
        raise ValueError("NR budgets need n = 3 tenants")
    adj_slice, table = raise_min_prb(q[:, order], B=B, min_prb=min_prb, names=NR_SLICES, prefer_idle=prefer_idle)
    path = Path(path)
    with open(path, "w", newline="\n") as f:
        np.savetxt(f, adj_slice, fmt="%d", delimiter=",", newline="\n")
    adj_tenant = np.empty_like(adj_slice)
    adj_tenant[:, order] = adj_slice
    return path, adj_tenant, table


def read_budget(path: str | Path) -> np.ndarray:
    """Read a budget CSV (ns-3 slice order) as int64 (K, 3)."""
    return np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)

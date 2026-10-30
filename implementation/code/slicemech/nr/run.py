"""Open-loop quota replay in ns-3.46 / 5G-LENA (INTERFACES.md §10; experiment_design.md E7).

Port of ``ICC/code/ns3/vslice_ns3.py`` (``_argv``, ``_env``, ``kpi_frame``, ``run_open_loop``) for the scenario binary
``~/ns-3-dev/build/scratch/ns3.46-vslice-nr-5g-default`` (source identical to ``ICC/code/ns3/vslice-nr-5g.cc``).

Offline mode (``--port=0``): the per-epoch PRB budget is read from ``--budgetFile`` and per-epoch per-slice KPIs are
written to ``--kpiFile``. Arrivals are open-loop (C4), so the replay equals a closed loop on the same schedule.

Default flags for E7: ``--port=0 --budgetFile --scheduleFile --kpiFile --epochs=K --uePerSlice=4
--pdb=<delay budgets in slice order> --rlcBuf=4194304,1048576,1048576 --mcsTable=1 --minPrb=4`` (+ ``--rngRun``,
``--binMs=1`` and, after calibration, ``--discRadius``).

KPI CSV columns (header written by the scenario; see ``KPI_COLUMNS``): bytes are IP bytes per epoch; ``prb_used_mean``
is the mean number of PRBs used per slot; delays (ms, from the source send time) are -1 when no packet arrived in the
epoch, and ``mcs_mean`` is -1 without a transport block. ``read_kpi`` converts bytes to Mbit, the -1 sentinels to NaN,
adds the per-PRB rate in Mbit/s per PRB and maps slices back to tenant indices.

The binary is run with ``cwd`` = the run directory (inside the rewrite tree), so nothing it might write lands
elsewhere; the old harness used ``cwd=~/ns-3-dev``.
"""
from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from slicemech.config import ROOT
from slicemech.io import require_free_gb

from .schedule import (
    BYTES_PER_MBIT,
    MIN_PRB,
    MS_PER_EPOCH,
    NR_SLICES,
    build_schedule,
    slice_order,
    write_budget,
    write_schedule,
)

# Private patched copy of ns-3.46 + nr-4.1.1 (user decision 2026-10-20): identical to ~/ns-3-dev except the statistics-only
# symbol-union fix in contrib/nr/model/nr-gnb-phy.cc (nr/patches/nr-gnb-phy-symbol-union.patch; see nr/NR_ISSUE_symbol_count.md).
NS3_DIR = Path(os.environ.get("NS3_DIR", "/media/darpan/New Volume/ICC_rewrite_nr/ns-3-dev-icc"))
NS3_BIN = NS3_DIR / "build" / "scratch" / "ns3.46-vslice-nr-5g-default"
NS3_LIB = NS3_DIR / "build" / "lib"
RUNS_DIR = ROOT / "nr" / "runs"

KPI_COLUMNS = ("epoch", "slice", "prb_req", "prb_quota", "prb_used_mean", "sched_bytes", "src_bytes",
               "pdcp_sdu_bytes", "delivered_bytes", "delivered_pdb_bytes", "rlc_drop_bytes", "src_pkts", "rx_pkts",
               "delay_mean_ms", "delay_p50_ms", "delay_p95_ms", "delay_p99_ms", "delay_max_ms", "within_pdb_frac",
               "lost_prev_pkts", "sent_prev_pkts", "tb_new_bytes", "mcs_mean")
BYTE_COLUMNS = ("sched_bytes", "src_bytes", "pdcp_sdu_bytes", "delivered_bytes", "delivered_pdb_bytes",
                "rlc_drop_bytes", "tb_new_bytes")
SENTINEL_COLUMNS = ("delay_mean_ms", "delay_p50_ms", "delay_p95_ms", "delay_p99_ms", "delay_max_ms",
                    "within_pdb_frac", "mcs_mean")
E7_FLAGS = dict(ue_per_slice=4, rlc_buf=(4194304, 1048576, 1048576), mcs_table=1, min_prb=MIN_PRB)


def _fmt(v: float) -> str:
    """Number formatting for ns-3 flags: integers without a trailing '.0'."""
    v = float(v)
    return str(int(v)) if v == int(v) else repr(v)


def _triple(name: str, vals: Sequence[float]) -> str:
    vals = list(vals)
    if len(vals) != 3:
        raise ValueError(f"--{name} needs exactly 3 per-slice values (eMBB, URLLC, mMTC), got {len(vals)}")
    return f"--{name}=" + ",".join(_fmt(v) for v in vals)


def build_command(*, budget_file: str | Path, kpi_file: str | Path, schedule_file: str | Path | None, epochs: int,
                  pdb_ms: Sequence[float], ue_per_slice: int = 4,
                  rlc_buf: Sequence[float] = (4194304, 1048576, 1048576), mcs_table: int = 1,
                  min_prb: int = MIN_PRB, rng_run: int = 1, bin_ms: int = 1, disc_radius: float | None = None,
                  extra: Sequence[str] = (), binary: str | Path = NS3_BIN) -> list[str]:
    """Command line of one offline (``--port=0``) run.

    Args:
        budget_file, kpi_file, schedule_file: paths of the ``--budgetFile``, ``--kpiFile`` and ``--scheduleFile``.
        epochs: number of 1-s epochs (= window intervals K).
        pdb_ms: per-slice packet delay budget for the in-PDB KPI, ns-3 slice order (ms).
        ue_per_slice, rlc_buf (bytes per RLC entity, slice order), mcs_table (1 = 64QAM), min_prb, rng_run, bin_ms:
            scenario flags (see the .cc).
        disc_radius: UE drop radius (m), chosen on TRAIN by the rate calibration; omitted -> scenario default.
        extra: further ``--flag=value`` strings appended verbatim.
        binary: scenario executable.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if any(not str(e).startswith("--") for e in extra):
        raise ValueError("extra flags must look like --name=value")
    a = [str(binary), "--port=0", f"--epochs={int(epochs)}", f"--rngRun={int(rng_run)}",
         f"--uePerSlice={int(ue_per_slice)}", f"--binMs={int(bin_ms)}"]
    if schedule_file is not None:
        a.append(f"--scheduleFile={schedule_file}")
    a += [f"--budgetFile={budget_file}", f"--kpiFile={kpi_file}", _triple("pdb", pdb_ms), _triple("rlcBuf", rlc_buf),
          f"--mcsTable={int(mcs_table)}", f"--minPrb={int(min_prb)}"]
    if disc_radius is not None:
        a.append(f"--discRadius={_fmt(disc_radius)}")
    return a + [str(e) for e in extra]


def ns3_env() -> dict:
    """Environment with the ns-3 shared libraries on LD_LIBRARY_PATH."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{NS3_LIB}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    return env


# --------------------------------------------------------------------------- KPIs
def _postprocess(df: pd.DataFrame, cls: Sequence[str] | None, epoch_s: float) -> pd.DataFrame:
    """Shared post-processing of the per-epoch per-slice KPI table (port of ``kpi_frame``)."""
    if set(df["slice"]) - set(NR_SLICES):
        raise ValueError(f"unknown slice names {sorted(set(df['slice']) - set(NR_SLICES))}")
    counts = df.groupby("epoch")["slice"].agg(lambda s: tuple(sorted(s)))
    if any(c != tuple(sorted(NR_SLICES)) for c in counts):
        raise ValueError("every epoch must have exactly one row per slice")
    df = df.copy()
    for key in BYTE_COLUMNS:
        df[key.replace("_bytes", "_mbit")] = df[key].astype(float) / BYTES_PER_MBIT
    for key in SENTINEL_COLUMNS:
        df[key] = df[key].astype(float).where(df[key] >= 0)          # -1 = no packet / no TB in the epoch
    df["measured_mbps"] = df["pdcp_sdu_mbit"] / epoch_s             # m: PDCP SDU volume per second (Mbit/s)
    used = df["prb_used_mean"].astype(float) * epoch_s              # PRB-seconds actually used
    df["rate_mbps_per_prb"] = (df["tb_new_mbit"] / used).where(used > 0)   # new-TB Mbit per used PRB-second
    df["loss_prev"] = (df["lost_prev_pkts"] / df["sent_prev_pkts"]).where(df["sent_prev_pkts"] > 0)
    df["quota_used_frac"] = (df["prb_used_mean"] / df["prb_quota"]).where(df["prb_quota"] > 0)
    df["slice_index"] = df["slice"].map({s: j for j, s in enumerate(NR_SLICES)}).astype(int)
    if cls is not None:
        order = slice_order(cls)
        df["tenant"] = order[df["slice_index"].to_numpy()]
        df["cls"] = [tuple(cls)[t] for t in df["tenant"]]
    return df.sort_values(["epoch", "slice_index"]).reset_index(drop=True)


def read_kpi(path: str | Path, cls: Sequence[str] | None = None, epoch_s: float = 1.0) -> pd.DataFrame:
    """Parse a ``--kpiFile`` CSV (exact scenario header required) into a per-epoch per-slice DataFrame.

    Added columns: ``*_mbit`` (Mbit per epoch), ``measured_mbps`` (Mbit/s), ``rate_mbps_per_prb`` (Mbit/s per used
    PRB; NaN if no PRB was used), ``loss_prev`` (packet loss of the previous epoch), ``quota_used_frac``,
    ``slice_index`` and, with ``cls``, the tenant index and class of each slice.
    """
    df = pd.read_csv(path)
    if tuple(df.columns) != KPI_COLUMNS:
        raise ValueError(f"unexpected KPI header in {path}: {list(df.columns)}")
    return _postprocess(df, cls, epoch_s)


def kpi_frame(kpis: list[dict], cls: Sequence[str] | None = None, epoch_s: float = 1.0) -> pd.DataFrame:
    """Same table from the scenario's NDJSON ``kpi`` messages (closed-loop mode; arrays in slice order)."""
    rows = []
    for k in kpis:
        for s, name in enumerate(NR_SLICES):
            r = {"epoch": int(k["epoch"]), "slice": name}
            for key in KPI_COLUMNS[2:]:
                if key not in k:
                    raise KeyError(f"KPI message of epoch {k.get('epoch')} lacks {key!r}")
                r[key] = float(k[key][s])
            rows.append(r)
    return _postprocess(pd.DataFrame(rows, columns=list(KPI_COLUMNS)), cls, epoch_s)


def kpi_matrix(df: pd.DataFrame, column: str) -> np.ndarray:
    """(K, 3) array of one KPI column in TENANT order (requires the ``tenant`` column, i.e. ``read_kpi(..., cls)``)."""
    if "tenant" not in df.columns:
        raise KeyError("KPI table has no tenant column; parse it with cls")
    piv = df.pivot(index="epoch", columns="tenant", values=column).sort_index()
    if list(piv.columns) != [0, 1, 2]:
        raise ValueError("KPI table must cover tenants 0, 1, 2")
    return piv.to_numpy(dtype=float)


def check_applied_quotas(df: pd.DataFrame, budget_tenant) -> None:
    """Raise if the quota ns-3 applied (``prb_quota``) differs from the budget written (tenant order, (K, 3))."""
    applied = kpi_matrix(df, "prb_quota")
    b = np.asarray(budget_tenant, dtype=float)
    if applied.shape != b.shape:
        raise ValueError(f"KPI has {applied.shape[0]} epochs, budget has {b.shape[0]}")
    if not np.array_equal(applied, b):
        k, j = np.argwhere(applied != b)[0]
        raise ValueError(f"ns-3 applied {applied[k, j]} PRB to tenant {j} in epoch {k}, budget says {b[k, j]}")


# --------------------------------------------------------------------------- runner
def _inside_root(path: Path) -> Path:
    path = Path(path).resolve()
    if ROOT.resolve() not in (path, *path.parents):
        raise ValueError(f"NR run directory {path} is outside the project tree {ROOT}")
    return path


def run_open_loop(arrivals_ms, quotas, cls: Sequence[str], pdb_ms: Sequence[float], workdir: str | Path,
                  rng_run: int = 1, B: int = 106, ue_per_slice: int = 4, ue_arrivals_ms=None,
                  rlc_buf: Sequence[float] = (4194304, 1048576, 1048576), mcs_table: int = 1,
                  min_prb: int = MIN_PRB, prefer_idle: bool = False, disc_radius: float | None = None,
                  extra: Sequence[str] = (), binary: str | Path = NS3_BIN,
                  runner: Callable[..., subprocess.CompletedProcess] = subprocess.run, timeout: float = 4 * 3600.0,
                  dry_run: bool = False) -> tuple[pd.DataFrame | None, dict]:
    """Replay one window's quotas in 5G-LENA and parse the KPIs.

    Args:
        arrivals_ms: tenant arrivals (T_ms, 3), Mbit per ms (``Window.arrivals_ms``); T_ms = 1000 * K.
        quotas: integer quotas in tenant order, (K, 3), PRB (the mechanism's deployed quotas).
        cls: tenant classes (one video, one urllc, one mmtc).
        pdb_ms: per-tenant delay budgets in TENANT order (ms; ``Contract.delay_ms``), re-ordered to slices here.
        workdir: run directory inside the project tree (created).
        rng_run: ns-3 RngRun (channel, mobility, placement; common random numbers across arms).
        B: PRBs per slot; ue_per_slice, ue_arrivals_ms, rlc_buf, mcs_table, min_prb, disc_radius, extra, binary: see
            ``build_command`` / ``build_schedule``; prefer_idle: see ``schedule.raise_min_prb``.
        runner: ``subprocess.run``-compatible callable (tests inject a fake that writes a KPI file).
        timeout: seconds before the run is killed.
        dry_run: write the inputs and return (None, meta) without running ns-3.

    Returns:
        (kpi DataFrame in tenant terms, meta dict with cmd, paths, adjusted quotas (tenant order), adjustment log,
        schedule interval volumes and wall time).
    """
    workdir = _inside_root(Path(workdir))
    workdir.mkdir(parents=True, exist_ok=True)
    require_free_gb(3.0, workdir)                                   # raises RuntimeError below 3 GB
    order = slice_order(cls)
    q = np.asarray(quotas)
    a = np.asarray(arrivals_ms)
    K = q.shape[0] if q.ndim == 2 else -1
    if q.ndim != 2 or q.shape[1] != 3 or a.shape[0] != K * MS_PER_EPOCH:
        raise ValueError(f"need quotas (K, 3) and arrivals (1000 K, 3); got {q.shape} and {a.shape}")
    pdb = np.asarray(pdb_ms, dtype=float)
    if pdb.shape != (3,) or np.any(pdb <= 0):
        raise ValueError("pdb_ms must be 3 positive delay budgets (ms) in tenant order")

    sched = build_schedule(a, cls, ue_arrivals_ms=ue_arrivals_ms, n_ue=ue_per_slice)
    sp = write_schedule(sched, workdir / f"schedule_run{rng_run}.csv")
    bp, q_adj, adj_log = write_budget(q, cls, workdir / f"budget_run{rng_run}.csv", B=B, min_prb=min_prb,
                                      prefer_idle=prefer_idle)
    adj_log.to_csv(workdir / f"budget_adjustments_run{rng_run}.csv", index=False)
    kp = workdir / f"kpi_run{rng_run}.csv"
    cmd = build_command(budget_file=bp, kpi_file=kp, schedule_file=sp, epochs=K, pdb_ms=pdb[order],
                        ue_per_slice=ue_per_slice, rlc_buf=rlc_buf, mcs_table=mcs_table, min_prb=min_prb,
                        rng_run=rng_run, disc_radius=disc_radius, extra=extra, binary=binary)
    meta = dict(cmd=cmd, workdir=str(workdir), schedule=str(sp), budget=str(bp), kpi=str(kp), quotas_adjusted=q_adj,
                adjustments=adj_log, n_adjusted_intervals=int(adj_log["k"].nunique()) if len(adj_log) else 0,
                schedule_interval_mbit=sched["interval_mbit"], wall_s=0.0)
    (workdir / f"cmd_run{rng_run}.txt").write_text(" ".join(cmd) + "\n")
    if dry_run:
        return None, meta
    if not Path(binary).exists():
        raise FileNotFoundError(f"ns-3 scenario binary not found: {binary}")
    t0 = time.time()
    log_path = workdir / f"ns3_run{rng_run}.log"
    try:
        p = runner(cmd, cwd=str(workdir), env=ns3_env(), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        meta["wall_s"] = time.time() - t0
        partial = (e.stdout or b"") + (e.stderr or b"")
        log_path.write_bytes(partial)
        raise RuntimeError(f"ns-3 timed out after {timeout:.0f} s (log: {log_path}):\n"
                           f"{partial.decode(errors='replace')[-3000:]}") from e
    meta["wall_s"] = time.time() - t0
    log_path.write_bytes((p.stdout or b"") + (p.stderr or b""))
    if p.returncode != 0:
        tail = ((p.stdout or b"") + (p.stderr or b"")).decode(errors="replace")[-3000:]
        raise RuntimeError(f"ns-3 exited with rc={p.returncode} (log: {log_path}):\n{tail}")
    df = read_kpi(kp, cls=cls)
    if df["epoch"].nunique() != K:
        raise RuntimeError(f"KPI file has {df['epoch'].nunique()} epochs, expected {K}")
    check_applied_quotas(df, q_adj)
    meta["pdcp_matches_schedule"] = bool(np.allclose(kpi_matrix(df, "pdcp_sdu_mbit"),
                                                     sched["interval_mbit"][:, np.argsort(order)],
                                                     atol=3 * ue_per_slice * 40 / BYTES_PER_MBIT))
    return df, meta


def run_window(window, quotas, workdir: str | Path, **kw) -> tuple[pd.DataFrame | None, dict]:
    """``run_open_loop`` for a ``data.Window`` (arrivals, classes and delay budgets taken from the window)."""
    return run_open_loop(window.arrivals_ms, quotas, window.cls, window.contract.delay_ms, workdir, **kw)


def nr_rates(df: pd.DataFrame) -> np.ndarray:
    """NR-measured per-interval per-PRB rate in tenant order, (K, 3) Mbit/s per PRB (NaN where no PRB was used);
    input of the 'fluid with NR-measured rate' comparison."""
    return kpi_matrix(df, "rate_mbps_per_prb")

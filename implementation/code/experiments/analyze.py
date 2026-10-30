"""Analysis of experiments E0-E7: driver outputs -> ``results/summary/<exp>.json`` (paper macros + tables).

Run from ``code/`` (``PYTHONDONTWRITEBYTECODE=1 /home/darpan/anaconda3/bin/python``)::

    experiments/analyze.py --split dev [--workers 8]           # DEV -> results/summary/dev/
    experiments/analyze.py --split train --smoke               # driver smoke outputs -> results/summary/smoke/train/
    experiments/analyze.py --split test                        # refused until the pre-registration freeze

``--split test`` is refused before anything is read unless the frozen pre-registration passes
``data.puffer_fetch.require_test_read_gate`` (``--test-freeze`` names it; default: the freeze recorded at the TEST
download). TEST summaries go to ``results/summary/`` (read by ``tools/make_numbers.py`` and the figure scripts), DEV
summaries to ``results/summary/dev/``, TRAIN summaries to ``results/summary/train/``; ``--smoke`` writes to
``results/summary/smoke/<split>/`` and uses 500 bootstrap replicates; ``--source e2=results/raw/dev/e2_pilot``
redirects one experiment's inputs (tagged driver runs). Resumable: an experiment whose recorded input
digest (raw files, this file, settings) is unchanged is skipped unless ``--force``. ``--workers`` analyses the
experiments in parallel processes and composes the crossed-bootstrap membership records in parallel.

Orchestrator decisions (binding, 2026-10-19; ``code/STATUS.md`` section 0) as implemented here
--------------------------------------------------------------------------------------------
* D1 (GBR x 0.8) and D2 (per-date offered-load normalization) are driver inputs. The analysis reports the realized
  per-window offered load (``offered_load`` = sum_i mean_k m_ik / (B rbar_TRAIN), all K intervals, written by E0/E1)
  and uses it as the exogenous subgroup variable.
* D3 Gate G4 dropped: H1c (protected vs static) and H1d (protected vs direct) are SECONDARY two-sided comparisons with
  95 % CIs and the largest advantage / disadvantage the data rule out; H1a, H1c and H1d are also reported within
  offered-load subgroups and as the slope of the paired difference on offered load. Subgroup specification (to be
  copied into PREREG.md §7; round-3 review): variable = per-window ``offered_load`` (summed measured demand over
  B x rbar_TRAIN, averaged over ALL K = 200 intervals including the 40-s warm-up; exogenous); cut points
  ``LOAD_BINS`` = 0.8 and 1.0 (bins < 0.8, [0.8, 1.0), >= 1.0); within each bin the same paired contrast and
  two-stage date bootstrap as the full analysis; slope = OLS slope of the per-window paired difference on the
  offered load (date-cluster bootstrap CI); a bin or slope with fewer than ``MIN_DATES`` dates gets no CI.
* D4 Harm is measured on the honest (non-deviating) tenants' own outcomes: H2c primary = change in their on-time Mbit
  (deadline-miss percentage points and their realized value next to it); H4b compares the honest tenants' realized
  value; total tariff-weighted welfare is descriptive only.
* D5 E5: later gains are measured against the alpha = 1 arm of the same date (the E5 driver contrasts each pipeline
  with its own alpha = 1 arm and checks that the measurement-pipeline alpha = 1 net equals the stored-report alpha = 1
  net, ``checks.json: measurement_equals_stored_ref_all``); 4 retraining + 4 evaluation windows per date.
* D6 short-horizon RL (h = 3): no analysis consequence.
* D7 E4b: welfare-versus-honest-payment frontier of the protected map with kappa scaled by s in {0, 0.5, 1, 2}
  (E4 driver arms ``kappa_x<s>``), with equal, static and direct as references.

Statistics (``prereg/PREREG_DRAFT.md`` sections 6-7)
-------------------------------------------------
Unit = window, paired across arms on identical compositions (pairing key = date + window offset); training seeds are
averaged within a window first (seed spread reported separately); two-stage cluster bootstrap over dates then windows
(``eval.stats.cluster_bootstrap``, 10,000 replicates, percentile 95 % CI, seed ``BOOT_SEED``); per-date sign counts;
Holm across ``HOLM_FAMILY`` = (H1a, H2c, H4a, H5b) of PREREG_DRAFT section 6 (a missing member enters with p = 1,
which keeps the adjustment conservative); the crossed bootstrap over date x URLLC capture x mMTC capture x CQI run
(``eval.stats.crossed_bootstrap``) as the sensitivity analysis of every primary (H1a, H2c, H4a on its
non-inferiority percentage, H5b on the E5 evaluation windows' pool-half compositions). Physical units first; every
percentage names its denominator.

Round-3 review rules (binding for the summaries):

* **Minimum dates.** No CI, p-value or decision is computed from fewer than ``MIN_DATES`` = 5 dates (a date
  bootstrap of 1-4 clusters is degenerate); such results carry the point estimate only, ``inference`` = False, and
  their decisions are None.
* **Tail convention.** CI rules use the two-sided 95 % percentile interval as PREREG_DRAFT §6 words them (H1a: lower
  bound > 0 and effect >= 0.5 %; H2c: upper bound < 0; each bound is a one-sided 97.5 % bound), H4a the one-sided 95 %
  upper bound of the percentage. The p-values entering Holm are one-sided in the pre-registered direction (H1a
  'greater', H2c 'less', H4a non-inferiority against the 1 % margin) and two-sided for H5b.
* **Final verdicts.** A primary is SUPPORTED only when its pre-registered rule holds AND Holm rejects it at
  family-wise alpha = 0.05 (H5b also needs the pre-registered sign ``H5B_SIGN``; a significant effect of the other
  sign is reported as contrary to it) AND, on TEST, its inputs are complete and of one provenance. The per-experiment
  files carry ``rule_met`` only; the verdicts and the ``*Supported`` macros are in ``holm.json``, which is always
  computed from every family member's newest summary in the output directory (not only the experiments of this run).
* **TEST completeness and provenance.** The TEST analysis lists, per experiment, the pre-declared windows (all
  dates x offsets of the split, the E2 replay windows, the E5 dates x seeds x arms, the E6 n x windows, the E7
  cases) that are missing, and whether the driver outputs share one set of models and one code fingerprint
  (E2-E4 provenance sidecars, E5/E6 job records, E1 checkpoint hashes).
* **DEV estimation half.** On DEV each primary is also estimated on the dates of the estimation half only
  (``registry.dev_role``; never used for a checkpoint, cell or static choice), with the number of TEST dates needed
  for 80 % power; these are the inputs of the D9 power analysis.

Pre-freeze audit (review/prefreeze_audit.md, 2026-10-20) as implemented here
-------------------------------------------------------------------------
* PV-2: on TEST each primary is gated only on its own inputs (``PRIMARY_INPUTS``: arm-level completeness and
  scoped provenance); every other completeness/provenance record is information in ``holm.json``.
* PV-4/PV-1/PV-5: ``power_inputs`` gives the power of the registered test (two-stage bootstrap, rule, Holm alpha/4)
  analytically and by simulation at 16 and 24 TEST dates, H1a's 0.5 % part with its status, and signed H5b power.
* PV-3: ``H2c_all`` entries other than the primary cell carry ``role`` = secondary. PV-8: protocol-changing options
  are refused on TEST. PV-9: on TEST, Holm never reuses a summary of other inputs or code.
* HC-4: the G6 radio-efficiency table of every E1 arm (``g6_table``; EfourPrbUsed, EfourPrbUsedFree). HC-2:
  EtwoVictimLoss = -EtwoVictim. HC-6: H3c from the independent engine. HC-9: all-NaN sums stay NaN; no
  mixed-seed H4b fallback on TEST. HC-13: no '[nan, nan]' CI strings.
* NUM-04 (EthreeErr = e, EthreeCert = 2e; EthreeTol is gone), NUM-05 (per-class EtwoGain CIs), NUM-07 (E5 victims,
  repay, EfiveDiffMeas/Stored from kept episodes), NUM-08 (EfiveCost positive = cost, EfiveAttackChange, EfiveNet),
  NUM-09 (labels from spec/controllers.yaml), NUM-11 (TrainMinutesNet), NUM-14/15 (bare-number percent macros,
  LaTeX number text ``tex_num``), NUM-16 (RateMin/RateMax precision), NUM-17 (EsevenCorrMin, EsevenWindows,
  per-n EsixTime), E5-1 (EfivePrimary from ``H5B_PRIMARY_ALPHA``), E5-8 (``h5b_by_seed``).
* Evidence check (2026-10-20 evening): H3a's verdict counts gain + certificate > eps_i (PREREG §6; ``h3a``), and
  descriptive macros only (section "descriptive additions": per-class value of reports, kappa = 0 contrasts, load
  slope, rho = 1.1, per-class miss changes; H2c-cell PRB transfer and ceiling shares; per-seed H4a gap, learning H4b;
  repay minutes, alpha = 2 later gain; H7b; H3c count).

Inputs (the drivers' own layouts; ``sources()``)
----------------------------------------------
E0 ``results/raw/<split>/e0/<config>/windows.parquet`` (or ``parts/*__win.parquet``) and ``vcal.parquet``;
E1 ``results/raw/<split>/e1/<config>/windows/*.parquet`` (config -> cell: default, d<k>, rho<x>, lead<x>, n<k>);
E2 ``results/raw/<split>/e2/{stage,replay,nonmono}/<arm>/s<seed>/*.parquet``, ``bandit/<design>/s<seed>/*.npz``;
E3 ``results/raw/<split>/e3/{cert,ident,h3d}/<arm>/s<seed>/*.parquet``; E4 ``results/raw/<split>/e4/**`` (window
summaries or episode tables with an ``arm`` column; ``kappa_x<s>`` arms = D7 frontier); E5 ``results/raw/<split>/e5/
jobs/*.{pairs,nets,windows}.parquet`` (+ ``checks.json``); E6 ``results/raw/<split>/e6/jobs/*.{runs,tenants,br,
timing}.parquet``; E7 ``results/raw/e7/cases_<split>/{cases.csv,h7.json}``. Smoke locations: ``results/raw/smoke/
{e0,e1}``, ``results/raw/train/{e2_smoke,e3_smoke,e4_smoke,e5/smoke,e6/smoke}``, ``results/raw/e7/smoke_train``.
Arm names are mapped to ``spec/controllers.yaml`` keys (``ARM_MAP``). Rows dated in another split are dropped; a
TEST date in a non-TEST analysis raises PermissionError.

Output per experiment (``e0`` ... ``e7``), plus ``holm.json``, ``example.json`` (closed-form example E3 of the
theory) and ``settings.json`` (setting macros of ``spec/params.yaml: experiment_settings``); the last two are
split-independent and are also written to ``results/summary/`` by non-smoke runs. Layout: ``{"experiment", "split",
"settings", "inputs", "macros": {Name: {value, fmt, unit, source}}, "tables": {name: [records]}, "hypotheses": {...},
"notes": [...]}``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from slicemech import io as sio  # noqa: E402
from slicemech.eval import stats as S  # noqa: E402

ROOT = sio.ROOT
RAW = ROOT / "results" / "raw"
SUMMARY = ROOT / "results" / "summary"
SPEC = ROOT / "spec"

# ----------------------------------------------------------------------------------------------- pre-declared settings
N_BOOT = 10_000
LEVEL = 0.95
BOOT_SEED = 20261019
SMOKE_N_BOOT = 500
MAIN = dict(n=3, d=2.0, rho=0.9, lead_s=4.004)          # default cell (spec/params.yaml)
DELAY_SWEEP = (1.0, 2.0, 5.0, 15.0)                      # s
LOAD_SWEEP = (0.5, 0.7, 0.9, 1.1)
LEAD_SWEEP = (0.0, 1.05, 4.004)                          # s
N_SWEEP = (3, 6, 9)
KAPPA_SCALES = (0.0, 0.5, 1.0, 2.0)                      # D7
LOAD_BINS = (0.8, 1.0)          # D3 subgroups of window offered load: < 0.8, [0.8, 1.0), >= 1.0 (fixed cut points)
HOLM_FAMILY = ("H1a", "H2c", "H4a", "H5b")               # PREREG_DRAFT section 6 (H1c secondary by D3)
ALPHA_FWER = 0.05
H1A_MIN_PCT = 0.5               # H1a fails if the effect is below 0.5 % of measurement-only welfare
H1A_MEI_PCT = 1.0               # minimum effect of interest
H4A_MARGIN_PCT = 1.0            # non-inferiority margin, % of W(protected)
H6B_MARGIN_PCT = 2.0
H6A_TIME_MS = 100.0
H3D_MARGIN_PCT = 1.0
H7A_SHARE = 0.75
H7C_CORR = 0.8
#: Primary H5b cell, fixed on DEV (orchestrator decision D8, 2026-10-20): stored reports, alpha = 0.5, protected arm,
#: video and URLLC-like targets pooled. The DEV pilot (both halves, every date) showed that inflation during the
#: retraining windows LOWERS the later quota (alpha = 2: later gain -3.44 credits per interval on the estimation half)
#: while under-reporting RAISES it (alpha = 0.5: +0.93, 10/10 dates; alpha = 0.7: +0.77, 10/10); the hypothesis is
#: about a later GAIN, so the cell is the under-reporting one. The draft value was alpha = 2.
H5B_PRIMARY_ALPHA = 0.5
H5B_PRIMARY_ARM = "protected"
#: <DEV> pre-registered sign of the H5b later gain (+1 or -1), fixed from the DEV pilot's estimation half before the
#: freeze (round-3 review: H5b used to count an effect of either sign as support). None until then; a TEST analysis
#: refuses to run without it. The frozen code hash covers it.
H5B_SIGN: int | None = 1      # D8: sign of the DEV estimation-half estimate at alpha = 0.5 (+0.93 credits per interval)
#: Pre-declared minimum number of dates for any bootstrap inference (round-3 review).
MIN_DATES = 5
#: One-sided level used for the D9 power inputs (Holm's first step over the 4 primaries, worst case) and the power.
POWER_ALPHA = ALPHA_FWER / 4
POWER = 0.80
#: D9 power simulation (pre-freeze audit PV-4 (b)): estimation-half dates (centred residuals + effect) resampled to
#: each candidate number of TEST dates; every simulated experiment is analysed with the registered two-stage
#: bootstrap (``POWER_SIM_BOOT`` replicates) and the rule + Holm alpha/4 decision; the rejection share is the power.
POWER_DATES = (16, 24)
POWER_SIMS = 400
POWER_SIM_BOOT = 2000
POWER_SIM_SEED = BOOT_SEED + 1
#: H2c primary cell (pre-freeze audit PV-3/HC-1): the Unconstrained net with the fixed-priority price (E2 arm
#: unc_fixed) on the E2 replay windows, ``REPLAY_PER_DATE`` per date (``e2_manipulation.replay_keys``; the E2 and E4
#: ``--replay-per-date`` default); the honest on-time change is summed over the non-deviating tenants and averaged
#: over the deviator roles of a window. The other arms and companion metrics in ``H2c_all`` are secondary.
H2C_PRIMARY_ARM = "unprotected"  # unconstrained net + fixed-priority price
REPLAY_PER_DATE = 2
H4B_ARMS = ("unprotected", "fixed_price", "unprotected_usage")
MANIP_ARMS = ("fixed_price", "unprotected", "unprotected_usage", "unprotected_envelope")
NONMONOTONE_ARMS = ("unprotected", "unprotected_usage", "unprotected_envelope")
CLASS_ORDER = ("video", "urllc", "mmtc")
CLASS_NAME = {"video": "Video", "urllc": "URLLC-like", "mmtc": "mMTC-like"}
CLASS_MACRO = {"video": "Video", "urllc": "Urllc", "mmtc": "Mmtc"}
EXPERIMENTS = ("e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7")
#: driver arm name -> (spec/controllers.yaml key, billed price, kappa scale)
ARM_MAP = {
    "prot_full": ("protected", "full", 1.0), "prot_fixed": ("fixed_price", "fixed", 1.0),
    "unc_fixed": ("unprotected", "fixed", None), "unc_usage": ("unprotected_usage", "usage", None),
    "unc_env": ("unprotected_envelope", "full", None), "unc_truth": ("unprotected", "fixed", None),
    "kappa0_full": ("kappa0", "full", None), "static_full": ("static", "full", None),
    "equal_full": ("equal", "full", None), "direct_full": ("direct", "full", None),
    "directtrue_full": ("direct_truerates", "full", None),
    "protected": ("protected", "full", 1.0), "fixed_price": ("fixed_price", "fixed", 1.0),
    "unprotected": ("unprotected", "fixed", None), "unconstrained": ("unprotected", "fixed", None),
    "unprotected_usage": ("unprotected_usage", "usage", None),
    "unprotected_envelope": ("unprotected_envelope", "full", None), "meas_only": ("meas_only", "usage", None),
    "kappa0": ("kappa0", "full", None), "static": ("static", "full", None), "equal": ("equal", "full", None),
    "direct": ("direct", "full", None), "direct_truerates": ("direct_truerates", "full", None),
    "oracle": ("oracle", "none", None), "meas_only_strong": ("meas_only_strong", "usage", None),
}
WSUM_COLS = ("on_time_Mbit", "dropped_Mbit", "quota", "price", "V_declared", "realized_value", "utility_declared",
             "utility_realized")
NONMONO_GRID = 513


class MissingInput(Exception):
    """An analysis block lacks its inputs: recorded as a note and skipped."""


# ================================================================================================ small helpers
_CTRL: dict = {}


def _controllers() -> dict:
    if not _CTRL:
        _CTRL.update({c["key"]: c for c in yaml.safe_load(open(SPEC / "controllers.yaml"))})
    return _CTRL


def label(arm: str) -> str:
    return _controllers().get(arm, {}).get("label", arm)


def map_arm(name) -> tuple[str, str, float | None]:
    """(controllers.yaml key, billed price, kappa scale) of a driver arm name (``kappa_x<s>`` = D7 scaled map)."""
    s = str(name)
    if s.startswith("kappa_x"):
        return "protected", "full", float(s[len("kappa_x"):])
    if s in ARM_MAP:
        return ARM_MAP[s]
    return s, "", None


def _clean(x):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python, tuples -> lists."""
    if isinstance(x, Mapping):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_clean(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (pd.Timestamp, Path)):
        return str(x)
    return x


def records(df: pd.DataFrame | None) -> list:
    if df is None or len(df) == 0:
        return []
    return _clean(df.to_dict(orient="records"))


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


#: LaTeX minus that works in text and in math mode (pre-freeze audit NUM-15: no hyphen minus signs in the paper).
TEX_MINUS = "\\ensuremath{-}"


def tex_num(value, fmt: str = "{:.2f}") -> str:
    """LaTeX rendering of a number under a Python format (pre-freeze audit NUM-04/NUM-15): Python e-notation
    becomes ``\\ensuremath{m\\times10^{e}}`` and a leading minus ``\\ensuremath{-}`` (safe in text and math mode).

    Convention for result macros (NUM-14): a macro is a bare number in the unit its ``unit`` field names; a
    percentage carries no '%' and no 'percent' (the paper writes the word), and its unit names the denominator."""
    t = fmt.format(value)
    m = re.fullmatch(r"(-?)(\d+(?:\.\d*)?)[eE]([+-]?\d+)", t)
    if m:
        sign, mant, ex = m.groups()
        return f"\\ensuremath{{{'-' if sign else ''}{mant}\\times10^{{{int(ex)}}}}}"
    return TEX_MINUS + t[1:] if t.startswith("-") else t


def ci_text(lo, hi, nd: int = 2) -> str:
    """'[lo, hi]' with LaTeX minus signs (string macro)."""
    return f"[{tex_num(lo, '{:.%df}' % nd)}, {tex_num(hi, '{:.%df}' % nd)}]"


class Macros:
    """Collects ``{Name: {value, fmt, unit, source, text}}``; names are letters only (``tools/make_numbers.py``).
    ``text`` is the LaTeX rendering (:func:`tex_num`) that ``make_numbers`` writes; ``value`` stays numeric."""

    def __init__(self, exp: str):
        self.exp, self.d = exp, {}

    def add(self, name: str, value, fmt: str = "{:.2f}", unit: str = "", source: str = ""):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name must be letters only: {name}")
        if value is None:
            return
        if isinstance(value, (bool, np.bool_)):
            value, fmt = ("yes" if value else "no"), "{}"
        if not isinstance(value, str):
            v = float(value)
            if not math.isfinite(v):
                return
            value = int(round(v)) if fmt in ("{:d}", "{:,d}") else v
        text = value if isinstance(value, str) else tex_num(value, fmt)
        self.d[name] = dict(value=value, fmt=fmt, unit=unit, source=f"{self.exp}: {source}", text=text)

    def ci(self, stem: str, res: Mapping | None, unit: str, source: str, nd: int = 2, pct: bool = True):
        """Estimate, CI bounds and CI string, and the percentage with its CI when present (a CI string only when
        both bounds are finite: pre-freeze audit HC-13, no '[nan, nan]')."""
        if not res or res.get("estimate") is None:
            return
        fmt = "{:.%df}" % nd
        self.add(stem, res["estimate"], fmt, unit, source)
        self.add(stem + "Lo", res.get("lo"), fmt, unit, source + " (95% CI lower)")
        self.add(stem + "Hi", res.get("hi"), fmt, unit, source + " (95% CI upper)")
        if _ok(res.get("lo"), res.get("hi")):
            self.add(stem + "CI", ci_text(res["lo"], res["hi"], nd), unit=unit, source=source + " (95% CI)")
        if pct and res.get("pct") is not None:
            self.add(stem + "Pct", res["pct"], "{:.2f}", "percent of " + res.get("denominator", "?"), source)
            if _ok(res.get("pct_lo"), res.get("pct_hi")):
                self.add(stem + "PctCI", ci_text(res["pct_lo"], res["pct_hi"], 2), unit="percent",
                         source=source + " (95% CI of the percentage)")

    def ci_negated(self, stem: str, res: Mapping | None, unit: str, source: str, nd: int = 2):
        """:meth:`ci` of minus the estimate (bounds swapped): a signed 'change' reported as a positive 'loss'
        (pre-freeze audit HC-2: EtwoVictimLoss = -EtwoVictim)."""
        if not res or res.get("estimate") is None:
            return
        neg = dict(estimate=-_f(res["estimate"]),
                   lo=-_f(res["hi"]) if res.get("hi") is not None else None,
                   hi=-_f(res["lo"]) if res.get("lo") is not None else None)
        self.ci(stem, neg, unit, source, nd, pct=False)


# ================================================================================================ input adapters
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_OFF_RE = re.compile(r"-o(\d+)(?:-|$)")


def _parse_date(s) -> str | None:
    m = _DATE_RE.search(str(s))
    return m.group(1) if m else None


def _parse_offset(s) -> float:
    m = _OFF_RE.search(str(s))
    return float(m.group(1)) if m else float("nan")


def config_cell(name: str) -> dict:
    """Cell of an E0/E1 configuration name (``e0_setup.CONFIGS``): default, d<k>, rho<x>, lead<x>, n<k>."""
    cell = dict(MAIN)
    m = re.fullmatch(r"(d|rho|lead|n)([0-9.]+)", str(name))
    if str(name) == "default" or m is None:
        return cell
    key = {"d": "d", "rho": "rho", "lead": "lead_s", "n": "n"}[m.group(1)]
    cell[key] = int(m.group(2)) if key == "n" else float(m.group(2))
    return cell


def sources(split: str, smoke: bool) -> dict:
    """Input directories per experiment (module docstring)."""
    if smoke:
        t = RAW / split
        return dict(e0=RAW / "smoke" / "e0", e1=RAW / "smoke" / "e1", e2=t / "e2_smoke", e3=t / "e3_smoke",
                    e4=t / "e4_smoke", e5=t / "e5" / "smoke", e6=t / "e6" / "smoke", e7=RAW / "e7" / f"smoke_{split}")
    b = RAW / split
    return dict(e0=b / "e0", e1=b / "e1", e2=b / "e2", e3=b / "e3", e4=b / "e4", e5=b / "e5", e6=b / "e6",
                e7=RAW / "e7" / f"cases_{split}")


def input_files(src: Mapping[str, Path]) -> list[Path]:
    out = []
    for e, p in src.items():
        p = Path(p)
        if not p.exists():
            continue
        if e in ("e5", "e6"):        # their smoke runs live below the full-run directory
            out += sorted(x for x in p.glob("*") if x.is_file())
            out += sorted(x for x in (p / "jobs").glob("*") if x.is_file())
        else:
            out += sorted(x for x in p.rglob("*") if x.is_file() and x.suffix in (".parquet", ".csv", ".npz",
                                                                                     ".json"))
    return out


def input_digest(files: Sequence[Path], extra: Mapping) -> str:
    h = hashlib.sha256()
    h.update(sio.file_sha256(__file__).encode())
    h.update(sio.file_sha256(S.__file__).encode())        # pre-freeze audit F7: the statistics module too
    h.update(json.dumps(_clean(extra), sort_keys=True).encode())
    for p in files:
        st = p.stat()
        h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _cat(paths) -> pd.DataFrame:
    paths = [p for p in paths if Path(p).is_file()]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True, sort=False)


def _units(df: pd.DataFrame, offset_col: str | None = None, window_col: str = "window") -> pd.DataFrame:
    """Add ``window`` (str), ``date``, ``offset`` and the pairing key ``unit`` = '<date>|o<offset>'."""
    df = df.copy()
    if window_col in df.columns and window_col != "window":
        df["window"] = df[window_col]
    if "window" in df.columns:
        df["window"] = df["window"].astype(str)
    if "date" not in df.columns:
        df["date"] = df["window"].map(_parse_date)
    df["date"] = df["date"].astype(str)
    if offset_col and offset_col in df.columns:
        df["offset"] = df[offset_col].astype(float)
    elif "offset" not in df.columns:
        df["offset"] = df["window"].map(_parse_offset) if "window" in df.columns else np.nan
    off = df["offset"].astype(float)
    key = df["window"] if "window" in df.columns else df["date"]
    df["unit"] = np.where(np.isfinite(off), df["date"] + "|o" + off.fillna(-1).astype(int).astype(str), key)
    return df


def _cell_cols(df: pd.DataFrame, cell: Mapping) -> pd.DataFrame:
    for c, v in cell.items():
        if c not in df.columns:
            df[c] = v
    df["n"] = df["n"].astype(int)
    for c in ("d", "rho", "lead_s"):
        df[c] = df[c].astype(float).round(4)
    return df


def _std_windows(df: pd.DataFrame, exp: str) -> pd.DataFrame:
    """Canonical window-summary rows: arm key, kappa scale, policy, deviator, totals (tenant -1)."""
    df = df.copy()
    mapped = df["arm"].map(map_arm)
    df["arm_raw"] = df["arm"].astype(str)
    df["arm"] = mapped.map(lambda t: t[0])
    ks = mapped.map(lambda t: t[2]).astype(float)
    df["kappa_scale"] = ks if "kappa_scale" not in df.columns else df["kappa_scale"].astype(float).fillna(ks)
    for c, v in (("policy", "truthful"), ("deviator", -1), ("seed", -1)):
        if c not in df.columns:
            df[c] = v
    df["seed"] = df["seed"].fillna(-1).astype(int)
    df["deviator"] = df["deviator"].fillna(-1).astype(int)
    if "utility_realized" not in df.columns:
        df["utility_realized"] = df["realized_value"] - df["price"]
    df["exp"] = exp
    return _add_totals(df)


def _add_totals(W: pd.DataFrame) -> pd.DataFrame:
    """Rebuild tenant = -1 rows (per-interval sums over tenants) for runs that lack them."""
    keys = [c for c in ("exp", "date", "unit", "window", "offset", "arm", "arm_raw", "seed", "n", "d", "rho", "lead_s",
                        "kappa_scale", "policy", "deviator") if c in W.columns]
    ten = W[W["tenant"] >= 0]
    if len(ten) == 0:
        return W
    tot = ten.groupby(keys, dropna=False)[[c for c in WSUM_COLS + ("p", "p_fix", "T") if c in ten.columns]].sum(
        min_count=1).reset_index()
    first = ten.groupby(keys, dropna=False)[[c for c in ("offered_load", "capacity_load") if c in ten.columns]].first()
    if first.shape[1]:
        tot = tot.merge(first.reset_index(), on=keys, how="left")
    tot["tenant"], tot["cls"] = -1, "all"
    on, dr = tot["on_time_Mbit"], tot["dropped_Mbit"]
    tot["deadline_miss_pct"] = (100 * dr / (on + dr)).where(on + dr > 0)
    have = W[W["tenant"] == -1][keys].drop_duplicates()
    if len(have):
        m = tot.merge(have.assign(_h=1), on=keys, how="left")["_h"].fillna(0).to_numpy() == 1
        tot = tot[~m]
    return pd.concat([W, tot], ignore_index=True, sort=False)


def _episode_window_rows(ep: pd.DataFrame, warmup: int, cls_of: Sequence[str]) -> pd.DataFrame:
    """Per-tenant per-interval means over the scored intervals of one episode-like table."""
    sc = ep[ep["k"] >= warmup]
    g = sc.groupby("tenant")
    out = g[[c for c in WSUM_COLS if c in sc.columns]].mean().reset_index()
    out["cls"] = [cls_of[int(t)] if int(t) < len(cls_of) else f"t{t}" for t in out["tenant"]]
    on, dr = out["on_time_Mbit"], out["dropped_Mbit"]
    out["deadline_miss_pct"] = (100 * dr / (on + dr)).where(on + dr > 0)
    out["n_intervals"] = int(sc["k"].nunique())
    return out


def _classes(n: int) -> tuple:
    try:
        from slicemech.data.calibrate import load_calibration
        return tuple(load_calibration(f"n{int(n)}_default").classes)
    except Exception:                                                  # noqa: BLE001
        return CLASS_ORDER if n == 3 else tuple(f"t{i}" for i in range(n))


def load_e0(p: Path) -> dict:
    out: dict = {}
    for cfg in (sorted(x for x in p.iterdir() if x.is_dir()) if p.exists() else []):
        win = _cat([cfg / "windows.parquet"]) if (cfg / "windows.parquet").exists() else \
            _cat(sorted((cfg / "parts").glob("*__win.parquet")))
        vc = _cat([cfg / "vcal.parquet"]) if (cfg / "vcal.parquet").exists() else \
            _cat(sorted((cfg / "parts").glob("*__vcal.parquet")))
        if len(win):
            out.setdefault("e0_windows", []).append(_units(win.assign(config=cfg.name), "offset_s", "window_id"))
        if len(vc):
            out.setdefault("e0_vcal", []).append(_units(vc.assign(config=cfg.name), "offset_s", "window_id"))
        wj, wc = cfg / "worked_interval.json", cfg / "worked_interval.parquet"
        if cfg.name == "default" and wj.is_file() and wc.is_file():
            for k, v in _worked_tables(json.loads(wj.read_text())["meta"], pd.read_parquet(wc),
                                       cfg / "worked_staircase.parquet").items():
                out.setdefault(k, []).append(v)
    return out


WORKED_MAPS = {"protected": ("protected", "full"), "unconstrained": ("unprotected", "fixed")}


def _worked_tables(meta: Mapping, curves: pd.DataFrame, stairs_path: Path) -> dict:
    """E0's worked interval (``e0_setup.worked_interval``; default configuration only) as canonical tables:
    ``e0_worked_meta`` (one row per map: the interval, tenant, truth and the map's quantities at truth; carries
    ``date`` so the split guard applies), ``e0_worked`` (own-report curves; ``gain_billed`` = declared utility at the
    true type under the map's billed price minus its value at truth: full-decision price for Protected,
    fixed-priority price for the Unconstrained net) and ``e0_worked_stairs`` (exact own-quota staircases)."""
    rows = []
    for m, (key, billed) in WORKED_MAPS.items():
        q = dict(meta.get(m) or {})
        rows.append(dict(map=m, arm=key, billed=billed, date=str(meta["date"]), offset_s=float(meta["offset_s"]),
                         window_id=str(meta.get("window_id", "")), k=int(meta["k"]), tenant=int(meta["tenant"]),
                         cls=str(meta["cls"]), theta=float(meta["theta_adm"]), C=float(meta["C"]),
                         pi=float(meta["pi"]), r_hat=float(meta["r_hat"]), n_grid=int(meta.get("n_grid", 0)),
                         **{k: (float(v) if isinstance(v, (int, float)) else None) for k, v in q.items()
                            if not isinstance(v, (list, dict))}))
    cur = curves.copy()
    cur["arm"] = cur["map"].map(lambda m: WORKED_MAPS.get(m, (m, ""))[0])
    gains = []
    for m, g in cur.groupby("map", sort=False):
        col = "U_full" if WORKED_MAPS.get(m, ("", "full"))[1] == "full" else "U_fix"
        t = g[g["is_truth"].astype(bool)]
        u0 = float(t[col].iloc[0]) if len(t) else float("nan")
        gains.append((g[col] - u0).rename("gain_billed"))
    cur["gain_billed"] = pd.concat(gains).reindex(cur.index)
    out = dict(e0_worked_meta=pd.DataFrame(rows), e0_worked=cur.sort_values(["map", "u"]).reset_index(drop=True))
    if stairs_path.is_file():
        st = pd.read_parquet(stairs_path)
        st["arm"] = st["map"].map(lambda m: WORKED_MAPS.get(m, (m, ""))[0])
        out["e0_worked_stairs"] = st
    return out


def load_e1(p: Path) -> dict:
    parts, mans = [], []
    for cfg in (sorted(x for x in p.iterdir() if x.is_dir()) if p.exists() else []):
        files = sorted((cfg / "windows").glob("*.parquet"))
        df = _cat(files) if files else _cat([cfg / "windows.parquet"])
        if len(df) == 0:
            continue
        df = _units(df.assign(config=cfg.name), "offset_s", "window_id")
        parts.append(_cell_cols(df, config_cell(cfg.name)))
        mp = cfg / "manifest.json"
        if mp.is_file():                      # the run's own configuration record (round-3 review: EoneDelay)
            m = json.loads(mp.read_text())
            c = m.get("config") or {}
            mans.append(dict(config=cfg.name, d=_f(c.get("d")), rho=_f(c.get("rho")), lead_s=_f(c.get("lead_s")),
                             n=_f(c.get("n")), analysis_code_sha256=m.get("analysis_code_sha256"),
                             expected_seeds=json.dumps(m.get("expected_seeds"))))
    if not parts:
        return {}
    out = dict(windows=[_std_windows(pd.concat(parts, ignore_index=True, sort=False), "e1")])
    if mans:
        out["e1_manifest"] = [pd.DataFrame(mans)]
    return out


def _mapped_br(df: pd.DataFrame, exp: str) -> pd.DataFrame:
    df = _units(df)
    mapped = df["arm"].map(map_arm)
    df["arm_raw"] = df["arm"]
    df["arm"] = mapped.map(lambda t: t[0])
    df["price"] = mapped.map(lambda t: t[1])
    df["exp"] = exp
    return _cell_cols(df, MAIN)


def load_e2(p: Path) -> dict:
    out: dict = {}
    if not p.exists():
        return out
    st = _cat(sorted(p.glob("stage/*/*/*.parquet")))
    if len(st):
        out["br"] = [_mapped_br(st, "e2")]
    cls3 = _classes(3)
    rows = []
    for f in sorted(p.glob("replay/*/*/*.parquet")):
        rep = pd.read_parquet(f)
        warm = int(rep["warmup"].iloc[0]) if "warmup" in rep.columns else 40
        for prof, g in rep.groupby("profile"):
            r = _episode_window_rows(g, warm, cls3)
            dev = -1 if prof == "truthful" else int(str(prof)[2:])
            rows.append(r.assign(window=str(rep["window"].iloc[0]), date=str(rep["date"].iloc[0]),
                                 arm=rep["arm"].iloc[0], seed=int(rep["seed"].iloc[0]),
                                 policy="truthful" if dev < 0 else "br", deviator=dev))
    if rows:
        out["windows"] = [_std_windows(_cell_cols(_units(pd.concat(rows, ignore_index=True)), MAIN), "e2")]
    nm = _cat(sorted(p.glob("nonmono/*/*/*.parquet")))
    if len(nm):
        nm = _units(nm)
        nm["arm"] = nm["net"].map({"unconstrained": "unprotected", "protected": "protected"}).fillna(nm["net"])
        nm = nm.rename(columns={"n_decreases": "n_decrease", "max_drop": "max_drop_prb"})
        nm["n_points"] = NONMONO_GRID
        out["monotone"] = [nm.assign(exp="e2")]
    brows = []
    for f in sorted(p.glob("bandit/*/*/*.npz")):
        if ".tmp" in f.name:                                # pre-freeze audit HC-10: never a driver's temp file
            continue
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        m, gm, gain = z["m"], z["greedy_m"], z["gain"]
        E, n = m.shape
        cls = _classes(n)
        arm, price, _ = map_arm(meta["design"])
        ep, ten = np.meshgrid(np.arange(E), np.arange(n), indexing="ij")
        brows.append(pd.DataFrame(dict(arm=arm, arm_raw=meta["design"], price=price, algo=meta["algo"],
                                       seed=int(meta["seed"]), run=int(meta["run"]), episode=ep.ravel(),
                                       tenant=ten.ravel(), cls=[cls[t] for t in ten.ravel()], m=m.ravel(),
                                       greedy_m=gm.ravel(), gain=gain.ravel())))
    if brows:
        out["bandit"] = [pd.concat(brows, ignore_index=True)]
    return out


def load_e3(p: Path) -> dict:
    out: dict = {}
    if not p.exists():
        return out
    cert = _cat(sorted(p.glob("cert/*/*/*.parquet")))
    if len(cert):
        cert = _units(cert)
        mapped = cert["arm"].map(map_arm)
        cert["arm_raw"] = cert["arm"]
        cert["arm"] = mapped.map(lambda t: t[0])
        sc = cert["scored"].astype(bool)
        br = cert[(cert["profile"] == "all_br") & sc & cert["br_gain"].notna()]
        br = br.drop(columns=[c for c in ("report", "quota", "price") if c in br.columns])
        br = br.rename(columns={"theta_adm": "theta", "br_z_star": "z_star", "br_gain": "gain", "br_cert": "cert",
                                "br_exact": "exact"})
        out["br"] = [_cell_cols(br.assign(price="full", exp="e3"), MAIN)]
        tr = cert[sc & ((cert["profile"] == "truthful") | ~cert["deviated"].astype(bool))]
        out["premium"] = [tr.assign(exp="e3")]
    idn = _cat(sorted(p.glob("ident/*/*/*.parquet")))
    if len(idn):
        idn = _units(idn)
        idn["arm"] = idn["arm"].map(lambda a: map_arm(a)[0])
        out["ident"] = [idn.assign(exp="e3")]
    h = _cat(sorted(p.glob("h3d/*/*/*.parquet")))
    if len(h):
        h = _units(h)
        h["arm"] = h["arm"].map(lambda a: map_arm(a)[0])
        out["h3d"] = [h.assign(exp="e3")]
    return out


def load_e4(p: Path) -> dict:
    """E4 driver: ``pay/<arm>/s<seed>/<window>.parquet`` (scored (k, tenant) rows of truthful runs with every price
    column; ``kappa_x<s>`` = D7 frontier arms), ``h4b_br.parquet`` (honest value, Protected truthful minus the arm with
    a best responder, same checkpoint seed), ``learn/<design>/s<seed>/<profile>.parquet`` (bandit end states replayed)
    and ``bench.parquet`` (single-core pricing time at n = 3)."""
    if not p.exists():
        return {}
    out: dict = {}
    pay = _cat(sorted(p.glob("pay/*/*/*.parquet")))
    if len(pay):
        cols = [c for c in WSUM_COLS + ("p", "p_fix", "p_usage", "T") if c in pay.columns]
        w = pay.groupby(["window", "date", "arm", "seed", "tenant", "cls"])[cols].mean().reset_index()
        on, dr = w["on_time_Mbit"], w["dropped_Mbit"]
        w["deadline_miss_pct"] = (100 * dr / (on + dr)).where(on + dr > 0)
        w["n_intervals"] = pay.groupby(["window", "date", "arm", "seed", "tenant"])["k"].nunique().to_numpy()
        out["windows"] = [_std_windows(_cell_cols(_units(w), MAIN), "e4")]
    if (p / "h4b_br.parquet").is_file():
        h = _units(pd.read_parquet(p / "h4b_br.parquet"))
        h["arm_raw"] = h["arm"].astype(str)
        h["arm"] = h["arm"].map(lambda a: map_arm(a)[0])
        out["e4_h4b"] = [h]
    learn = _cat(sorted(p.glob("learn/*/*/*.parquet")))
    if len(learn):
        learn = _units(learn)
        learn["arm"] = learn["design"].map(lambda a: map_arm(a)[0])
        out["e4_learn"] = [learn]
    if (p / "bench.parquet").is_file():
        b = pd.read_parquet(p / "bench.parquet").rename(columns={"full_ms": "t_price_ms", "fixed_ms": "t_fixed_ms"})
        out["timing"] = [_units(b).assign(n=3, arm="protected", exp="e4")]
    return out


def _jobs_or_agg(p: Path, name: str) -> pd.DataFrame:
    """Finished job files ``jobs/<job>.<name>.parquet`` (with ``.done.json``), else the aggregate ``<name>.parquet``."""
    jd = p / "jobs"
    if jd.is_dir():
        done = [x.name[: -len(".done.json")] for x in jd.glob("*.done.json")]
        files = [jd / f"{j}.{name}.parquet" for j in sorted(done)]
        files = [f for f in files if f.is_file()]
        if files:
            return _cat(files)
    return _cat([p / f"{name}.parquet"])


def _e5_wdiff(jd: Path) -> pd.DataFrame:
    """Pre-freeze audit NUM-07 (only when E5 ran with ``--keep-episodes``): per finished job, later window, pipeline
    and distorted profile, the number of scored later intervals whose priority vector ``w`` differs (any tenant,
    |dw| > 1e-12) from the alpha = 1 run of the same pipeline. One job file at a time (the episode tables are large)."""
    rows = []
    if not jd.is_dir():
        return pd.DataFrame()
    done = {x.name[: -len(".done.json")] for x in jd.glob("*.done.json")}
    cols = ["phase", "pipeline", "target", "alpha", "window", "k", "tenant", "w", "scored", "date", "seed", "arm"]
    for f in sorted(jd.glob("*.episodes.parquet")):
        if f.name[: -len(".episodes.parquet")] not in done:
            continue
        try:
            ep = pd.read_parquet(f, columns=cols)
        except Exception:                                              # noqa: BLE001  (older layouts: no w)
            continue
        ep = ep[(ep["phase"] == "later") & ep["scored"].astype(bool) & ep["pipeline"].isin(("measurement",
                                                                                             "stored"))]
        if len(ep) == 0:
            continue
        ep = ep.assign(alpha=ep["alpha"].astype(float))
        for (pipe, win), g in ep.groupby(["pipeline", "window"]):
            ref = g[np.isclose(g["alpha"], 1.0)]
            if len(ref) == 0:
                continue
            rw = ref.drop_duplicates(["k", "tenant"]).set_index(["k", "tenant"])["w"].astype(float).sort_index()
            for (tgt, a), h in g[~np.isclose(g["alpha"], 1.0)].groupby(["target", "alpha"]):
                hw = h.set_index(["k", "tenant"])["w"].astype(float).sort_index()
                dk = (hw - rw).abs().groupby(level="k").max()
                rows.append(dict(date=str(h["date"].iloc[0]), seed=int(h["seed"].iloc[0]), arm=str(h["arm"].iloc[0]),
                                 pipeline=pipe, target=tgt, alpha=float(a), window=str(win), intervals=int(len(dk)),
                                 differ=int((dk.fillna(np.inf) > 1e-12).sum())))
    return pd.DataFrame(rows)


def load_e5(p: Path) -> dict:
    if not p.exists():
        return {}
    out = {}
    for name in ("pairs", "nets", "windows"):
        df = _jobs_or_agg(p, name)
        if len(df):
            out[f"e5_{name}"] = [_units(df) if "window" in df.columns else df]
    if (p / "checks.json").is_file():
        out["e5_checks"] = [pd.DataFrame([json.loads((p / "checks.json").read_text())])]
    wd = _e5_wdiff(p / "jobs")
    if len(wd):
        out["e5_wdiff"] = [wd]
    return out


def load_e6(p: Path) -> dict:
    if not p.exists():
        return {}
    out = {}
    ten = _jobs_or_agg(p, "tenants")
    if len(ten):
        out["windows"] = [_std_windows(_cell_cols(_units(ten, "offset_s"), MAIN), "e6")]
    br = _jobs_or_agg(p, "br")
    if len(br):
        br = _cell_cols(_units(br, "offset_s"), MAIN).rename(columns={"allowance": "eps"})
        br["arm"] = br["arm"].map(lambda a: map_arm(a)[0])
        out["br"] = [br.assign(price="full", exp="e6")]
    tm = _jobs_or_agg(p, "timing")
    if len(tm):
        out["timing"] = [_units(tm, "offset_s").rename(columns={"t_full_ms": "t_price_ms"}).assign(exp="e6")]
    runs = _jobs_or_agg(p, "runs")
    if len(runs):
        out["e6_runs"] = [_units(runs, "offset_s")]
    return out


def load_e7(p: Path) -> dict:
    out = {}
    if (p / "cases.csv").is_file():
        out["nr"] = [pd.read_csv(p / "cases.csv")]
    return out


LOADERS = dict(e0=load_e0, e1=load_e1, e2=load_e2, e3=load_e3, e4=load_e4, e5=load_e5, e6=load_e6, e7=load_e7)


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def load_all(split: str, smoke: bool, notes: list, src: Mapping[str, Path] | None = None) -> dict[str, pd.DataFrame]:
    """Every driver's outputs as canonical tables; split guard (other splits dropped, TEST outside TEST refused)."""
    from slicemech.data import puffer
    src = src or sources(split, smoke)
    parts: dict[str, list] = {}
    for exp, fn in LOADERS.items():
        p = Path(src.get(exp, "/nonexistent"))
        if not p.exists():
            notes.append(f"{exp}: no driver output at {_rel(p)}")
            continue
        for k, v in fn(p).items():
            parts.setdefault(k, []).extend(v)
    T = {k: pd.concat(v, ignore_index=True, sort=False) for k, v in parts.items()}
    for kind, df in list(T.items()):
        if len(df) == 0 or "date" not in df.columns:
            continue
        sp = df["date"].map(lambda d: puffer.split_of_date(str(d)))
        if split != "test" and (sp == "test").any():
            raise PermissionError(f"table {kind!r} contains TEST dates but the analysis split is {split!r}")
        bad = sp.notna() & (sp != split)
        if bad.any():
            notes.append(f"{kind}: dropped {int(bad.sum())} rows of other splits {sorted(set(sp[bad]))}")
            T[kind] = df[~bad].reset_index(drop=True)
    if "windows" in T:
        T["windows"] = _attach_loads(T["windows"], T)
    return T


def window_loads(T: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Exogenous per-window loads of the main cell (unit -> offered_load, capacity_load): E0 default, then E1."""
    frames = []
    e0 = T.get("e0_windows")
    if e0 is not None and len(e0):
        frames.append(e0[e0["config"] == "default"][["unit", "date", "offered_load", "capacity_load"]])
    W = T.get("windows")
    if W is not None and len(W) and "offered_load" in W.columns:
        x = W[(W["exp"] == "e1") & _cell_mask(W, MAIN)]
        frames.append(x[["unit", "date", "offered_load", "capacity_load"]].dropna(subset=["offered_load"]))
    if not frames:
        return pd.DataFrame(columns=["unit", "date", "offered_load", "capacity_load"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("unit").reset_index(drop=True)


def _attach_loads(W: pd.DataFrame, T: Mapping) -> pd.DataFrame:
    for c in ("offered_load", "capacity_load"):
        if c not in W.columns:
            W[c] = np.nan
    L = window_loads(dict(T, windows=W))
    if len(L) == 0:
        return W
    main = _cell_mask(W, MAIN).to_numpy()
    m = W.loc[main, ["unit"]].merge(L[["unit", "offered_load", "capacity_load"]], on="unit", how="left")
    for c in ("offered_load", "capacity_load"):
        W.loc[main, c] = W.loc[main, c].fillna(pd.Series(m[c].to_numpy(), index=W.index[main]))
    return W


# ================================================================================================ crossed memberships
def _membership_job(args):
    split, date, offset, test_freeze, pool_part = args
    import torch                                                        # noqa: F401  (spawned worker hygiene)
    from slicemech.data.calibrate import load_calibration
    cal = load_calibration("n3_default")
    kw = dict(test_freeze=test_freeze) if split == "test" else {}
    if pool_part:                                     # E5 windows are composed from one pool half (D5)
        kw["pool_part"] = pool_part
    w = cal.window(split, date, int(offset), **kw)
    mem = S.window_memberships([w.meta])
    return f"{date}|o{int(offset)}", {f: {str(lv): float(A[0, j]) for j, lv in enumerate(labels) if A[0, j] > 0}
                                      for f, (labels, A) in mem.items()}


def memberships(units: Sequence[str], split: str, cache: Path, workers: int = 1, test_freeze=None,
                confine: bool = True, pool_part: str | None = None) -> dict:
    """{unit: {factor: {level: weight}}} of main-cell windows (composition metadata only, no outcome is read),
    composed with the default calibration and cached in ``cache``; ``pool_part`` composes E5 windows from that
    pool phase (round-3 review: H5b had no crossed bootstrap although E5 reuses TRACTOR captures most)."""
    have = json.loads(cache.read_text()) if cache.is_file() else {}
    todo = []
    for u in sorted(set(units)):
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\|o(\d+)", u)
        if u not in have and m:
            todo.append((split, m.group(1), int(m.group(2)), test_freeze, pool_part))
    if todo:
        if workers > 1 and len(todo) > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=mp.get_context("spawn")) as ex:
                res = list(ex.map(_membership_job, todo))
        else:
            res = [_membership_job(j) for j in todo]
        have.update(dict(res))
        write_json(cache, have, confine)
    return {u: have[u] for u in units if u in have}


def _membership_mats(p: pd.DataFrame, mem: Mapping) -> dict:
    units = p["unit"].tolist()
    if not mem or not all(u in mem for u in units):
        return {}
    factors = sorted(set.intersection(*[set(mem[u]) for u in units]))
    mats = {}
    for f in factors:
        labels = sorted({lv for u in units for lv in mem[u][f]})
        idx = {lv: j for j, lv in enumerate(labels)}
        A = np.zeros((len(units), len(labels)))
        for i, u in enumerate(units):
            row = mem[u][f]
            tot = sum(row.values())
            for lv, w in row.items():
                A[i, idx[lv]] = w / tot
        mats[f] = A
    return mats


# ================================================================================================ statistics
def _cell_mask(df: pd.DataFrame, cell: Mapping) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for c, v in cell.items():
        if c in df.columns:
            m &= np.isclose(df[c].astype(float), float(v), atol=1e-6)
    return m


def unit_values(W: pd.DataFrame, cell: Mapping, arm: str, metric: str = "realized_value", tenant: int = -1,
                policy: str = "truthful", kappa_scale: float | None = None, exp: str | Sequence[str] | None = None
                ) -> pd.DataFrame:
    """Seed-averaged per-window values (columns date, unit, value, n_seeds, offered_load)."""
    m = _cell_mask(W, cell) & (W["arm"] == arm) & (W["policy"] == policy) & (W["tenant"] == tenant)
    if exp is not None:
        m &= W["exp"].isin([exp] if isinstance(exp, str) else list(exp))
    if kappa_scale is not None:
        m &= np.isclose(W["kappa_scale"].astype(float), kappa_scale)
    elif arm == "protected":
        m &= np.isclose(W["kappa_scale"].astype(float).fillna(1.0), 1.0)
    x = W[m]
    if len(x) == 0 or metric not in x.columns:
        return pd.DataFrame(columns=["date", "unit", "value", "n_seeds", "offered_load"])
    g = x.groupby(["date", "unit"])
    out = pd.DataFrame({"value": g[metric].mean(), "n_seeds": g["seed"].nunique(),
                        "offered_load": g["offered_load"].first()})
    return out.reset_index()


class NoInference:
    """Point estimate without inference: fewer than ``MIN_DATES`` dates (round-3 review). Same interface as
    ``eval.stats.BootResult``; every bound and p-value is NaN (None in the JSON)."""

    def __init__(self, estimate: float, n_dates: int, n_windows: int):
        self.estimate, self.n_dates, self.n_windows = float(estimate), int(n_dates), int(n_windows)
        self.lo = self.hi = float("nan")
        self.level = LEVEL
        self.reps = np.empty(0)

    def p_value(self, null: float = 0.0, alternative: str = "greater") -> float:
        return float("nan")

    def upper(self, level: float = 0.95) -> float:
        return float("nan")

    def lower(self, level: float = 0.95) -> float:
        return float("nan")

    def as_dict(self) -> dict:
        return dict(estimate=self.estimate, lo=self.lo, hi=self.hi, level=self.level, n_dates=self.n_dates,
                    n_windows=self.n_windows, n_boot=0,
                    inference=f"none: {self.n_dates} date(s) < MIN_DATES = {MIN_DATES}")


def boot(values, dates, *, n_boot: int, seed: int = BOOT_SEED, stat=None):
    """Two-stage cluster bootstrap, or :class:`NoInference` with fewer than ``MIN_DATES`` dates."""
    d = np.asarray(dates)
    nd = int(len(np.unique(d)))
    if nd < MIN_DATES:
        v = np.asarray(values, float)
        v = v[:, None] if v.ndim == 1 else v
        f = stat or (lambda m: m[..., 0])
        return NoInference(float(np.asarray(f(v.mean(0)[None, :]))[0]), nd, v.shape[0])
    return S.cluster_bootstrap(values, dates, n_boot=n_boot, level=LEVEL, seed=seed, stat=stat)


def _ok(*xs) -> bool:
    """All values finite numbers (a decision on a bound needs the bound)."""
    return all(x is not None and math.isfinite(_f(x)) for x in xs)


def _rule(cond, *xs):
    """``cond`` (a bool) when every value in ``xs`` is finite, else None (no inference)."""
    return bool(cond) if _ok(*xs) else None


def pair(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    j = a.merge(b, on=["date", "unit"], suffixes=("_a", "_b"), how="inner")
    return pd.DataFrame(dict(date=j["date"], unit=j["unit"], a=j["value_a"].astype(float),
                             b=j["value_b"].astype(float),
                             offered_load=j["offered_load_a"].astype(float).fillna(j["offered_load_b"].astype(float))))


def contrast(pairs: pd.DataFrame, *, n_boot: int, alternative: str = "two-sided", null: float = 0.0,
             denominator: str = "", mem: Mapping | None = None, pct_null: float | None = None,
             pct_alternative: str | None = None, pct: bool = True) -> dict:
    """Paired per-window contrast a - b (columns date, unit, a, b): estimate, 95 % CI, p, one-sided bounds, per-date
    signs, the difference as a percentage of b (ratio of means, same resamples) and, with ``mem``, the crossed
    bootstrap."""
    p = pairs.dropna(subset=["a", "b"])
    if len(p) == 0:
        raise MissingInput("no paired windows")
    diff = (p["a"] - p["b"]).to_numpy(float)
    V = np.column_stack([diff, p["b"].to_numpy(float)])
    dates = p["date"].to_numpy()
    rd = boot(V, dates, n_boot=n_boot, stat=lambda m: m[..., 0])
    out = dict(rd.as_dict(), p=rd.p_value(null, alternative), alternative=alternative, null=null,
               lower_one_sided=rd.lower(LEVEL), upper_one_sided=rd.upper(LEVEL), mean_a=float(p["a"].mean()),
               mean_b=float(p["b"].mean()), denominator=denominator, inference=not isinstance(rd, NoInference))
    sc = S.sign_counts(diff, dates)
    out.update(dates_positive=sc["positive"], dates_negative=sc["negative"], dates_zero=sc["zero"],
               per_date=records(sc["table"]))
    if pct and p["b"].mean() != 0:
        rp = boot(V, dates, n_boot=n_boot, stat=S.ratio_stat(minus_one=False, percent=True))
        out.update(pct=rp.estimate, pct_lo=rp.lo, pct_hi=rp.hi, pct_lower_one_sided=rp.lower(LEVEL),
                   pct_upper_one_sided=rp.upper(LEVEL))
        if pct_null is not None:
            out["p_pct"] = rp.p_value(pct_null, pct_alternative or alternative)
    if mem and out["inference"]:
        A = _membership_mats(p, mem)
        if A:
            cb = S.crossed_bootstrap(diff, A, n_boot=n_boot, level=LEVEL, seed=BOOT_SEED)
            out["crossed"] = dict(lo=cb.lo, hi=cb.hi, p=cb.p_value(null, alternative), factors=sorted(A),
                                  levels={f: int(a.shape[1]) for f, a in A.items()})
            if pct and pct_null is not None and p["b"].mean() != 0:
                # round-3 review: the crossed sensitivity analysis of a percentage hypothesis (H4a) tests that
                # hypothesis, on (difference, denominator) with the ratio statistic against the margin
                cp = S.crossed_bootstrap(V, A, n_boot=n_boot, level=LEVEL, seed=BOOT_SEED,
                                         stat=S.ratio_stat(minus_one=False, percent=True))
                out["crossed"].update(pct_lo=cp.lo, pct_hi=cp.hi, pct_lower_one_sided=cp.lower(LEVEL),
                                      pct_upper_one_sided=cp.upper(LEVEL), pct_null=pct_null,
                                      p_pct=cp.p_value(pct_null, pct_alternative or alternative))
    return out


def one_sample(df: pd.DataFrame, col: str, *, n_boot: int, alternative: str = "two-sided", mem=None) -> dict:
    """Cluster-bootstrap summary of a per-window quantity (columns date, unit, col) against 0."""
    x = df.dropna(subset=[col])
    if len(x) == 0:
        raise MissingInput(f"no windows with {col}")
    return contrast(pd.DataFrame(dict(date=x["date"], unit=x["unit"], a=x[col].astype(float), b=0.0)),
                    n_boot=n_boot, alternative=alternative, mem=mem, pct=False)


def mean_ci(df: pd.DataFrame, col: str = "value", *, n_boot: int) -> dict:
    d = df.dropna(subset=[col])
    if len(d) == 0:
        raise MissingInput("no windows")
    return boot(d[col].to_numpy(float), d["date"].to_numpy(), n_boot=n_boot).as_dict()


def load_subgroups(pairs: pd.DataFrame, *, n_boot: int, alternative: str) -> dict:
    """D3: the paired difference within offered-load bins (``LOAD_BINS``) and its slope on the offered load."""
    p = pairs.dropna(subset=["a", "b", "offered_load"])
    if len(p) == 0:
        return dict(status="missing offered_load", bins=[])
    edges = (-np.inf, *LOAD_BINS, np.inf)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        q = p[(p["offered_load"] >= lo) & (p["offered_load"] < hi)]
        row = dict(load_lo=lo if np.isfinite(lo) else None, load_hi=hi if np.isfinite(hi) else None,
                   n_windows=len(q), n_dates=int(q["date"].nunique()),
                   mean_load=float(q["offered_load"].mean()) if len(q) else None)
        if q["date"].nunique() >= MIN_DATES:
            r = contrast(q, n_boot=n_boot, alternative=alternative)
            row.update({k: r.get(k) for k in ("estimate", "lo", "hi", "p", "pct", "pct_lo", "pct_hi",
                                              "dates_positive", "dates_negative")})
        rows.append(row)
    slope = None
    x = p["offered_load"].to_numpy(float)
    if len(p) >= 3 and np.ptp(x) > 0 and p["date"].nunique() >= MIN_DATES:
        y = (p["a"] - p["b"]).to_numpy(float)
        V = np.column_stack([x, y, x * y, x * x])
        r = boot(V, p["date"].to_numpy(), n_boot=n_boot,
                 stat=lambda m: (m[..., 2] - m[..., 0] * m[..., 1]) / (m[..., 3] - m[..., 0] ** 2))
        slope = dict(estimate=r.estimate, lo=r.lo, hi=r.hi, unit="credits per interval per unit of offered load")
    return dict(status="ok", cut_points=list(LOAD_BINS), variable="offered_load (all K intervals, D2 quantity)",
                min_dates=MIN_DATES, bins=rows, slope=slope)


def slope_over(W: pd.DataFrame, var: str, levels: Sequence[float], arm_a: str, arm_b: str, *, n_boot: int,
               exp: str = "e1") -> dict | None:
    """H1b: OLS slope of the per-window paired difference over sweep levels (windows present at every level)."""
    cols = []
    for lv in levels:
        pr = pair(unit_values(W, dict(MAIN, **{var: lv}), arm_a, exp=exp),
                  unit_values(W, dict(MAIN, **{var: lv}), arm_b, exp=exp))
        if len(pr) == 0:
            return None
        cols.append(pr.assign(v=pr["a"] - pr["b"])[["date", "unit", "v"]].rename(columns={"v": f"l{lv}"}))
    j = cols[0]
    for c in cols[1:]:
        j = j.merge(c.drop(columns="date"), on="unit", how="inner")
    if len(j) < 2 or j["date"].nunique() < MIN_DATES:
        return None
    V = j[[f"l{lv}" for lv in levels]].to_numpy(float)
    r = boot(V, j["date"].to_numpy(), n_boot=n_boot, stat=S.slope_stat(levels))
    return dict(r.as_dict(), levels=list(levels), var=var, p=r.p_value(0.0, "greater"),
                nondecreasing_supported=_rule(r.lo > 0, r.lo), unit=f"credits per interval per unit of {var}")


# ================================================================================================ helpers per block
def _run(notes: list, name: str, fn: Callable, *a, **kw):
    try:
        return fn(*a, **kw)
    except MissingInput as e:
        notes.append(f"{name}: skipped ({e})")
        return None


def _need(T: Mapping, key: str):
    df = T.get(key)
    if df is None or len(df) == 0:
        raise MissingInput(f"no {key} rows")
    return df


def arm_contrast(W, cell, arm_a, arm_b, *, n_boot, alternative, denominator, exp="e1", mem=None,
                 metric="realized_value", pct_null=None, pct_alternative=None, kappa_a=None):
    a = unit_values(W, cell, arm_a, metric, kappa_scale=kappa_a, exp=exp)
    b = unit_values(W, cell, arm_b, metric, exp=exp)
    if len(a) == 0 or len(b) == 0:
        raise MissingInput(f"arm {arm_a if len(a) == 0 else arm_b} missing in cell {dict(cell)} ({exp})")
    pr = pair(a, b)
    r = contrast(pr, n_boot=n_boot, alternative=alternative, denominator=denominator, mem=mem, pct_null=pct_null,
                 pct_alternative=pct_alternative)
    r.update(arm_a=arm_a, arm_b=arm_b, unit="credits per interval", unpaired=int(len(a) + len(b) - 2 * len(pr)),
             seeds_a=int(a["n_seeds"].max()), seeds_b=int(b["n_seeds"].max()))
    return r, pr


def arm_table(W: pd.DataFrame, cell: Mapping, *, n_boot: int, exp: str = "e1") -> pd.DataFrame:
    """Per arm: mean realized welfare (CI), per-class on-time Mbit, miss %, PRBs and billed price."""
    x = W[_cell_mask(W, cell) & (W["policy"] == "truthful") & (W["exp"] == exp)]
    rows = []
    for arm in sorted(x["arm"].unique()):
        uv = unit_values(W, cell, arm, exp=exp)
        if len(uv) == 0:
            continue
        g = x[x["arm"] == arm]
        if arm == "protected":
            g = g[np.isclose(g["kappa_scale"].astype(float).fillna(1.0), 1.0)]
        r = mean_ci(uv, n_boot=n_boot)
        row = dict(arm=arm, label=label(arm), welfare=r["estimate"], welfare_lo=r["lo"], welfare_hi=r["hi"],
                   n_windows=r["n_windows"], n_dates=r["n_dates"], seeds=int(uv["n_seeds"].max()))
        gs = g.groupby(["date", "unit", "tenant", "cls"])[["on_time_Mbit", "dropped_Mbit", "quota", "price"]].mean()
        for c, gc in gs.reset_index().groupby("cls"):
            on, dr = gc["on_time_Mbit"].mean(), gc["dropped_Mbit"].mean()
            row.update({f"on_time_{c}": on, f"miss_pct_{c}": 100 * dr / (on + dr) if on + dr > 0 else None,
                        f"prb_{c}": gc["quota"].mean(), f"price_{c}": gc["price"].mean()})
        rows.append(row)
    return pd.DataFrame(rows)


def seed_spread(W: pd.DataFrame, cell: Mapping, arms: Sequence[str], exp: str = "e1") -> pd.DataFrame:
    """Per learned arm and seed: mean realized welfare over the windows (credits per interval)."""
    x = W[_cell_mask(W, cell) & (W["policy"] == "truthful") & (W["tenant"] == -1) & W["arm"].isin(arms)
          & (W["exp"] == exp) & (W["seed"] >= 0)]
    x = x[(x["arm"] != "protected") | np.isclose(x["kappa_scale"].astype(float).fillna(1.0), 1.0)]
    if len(x) == 0:
        return pd.DataFrame()
    g = x.groupby(["arm", "seed"])["realized_value"].mean().reset_index()
    s = g.groupby("arm")["realized_value"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    return g.merge(s, on="arm").rename(columns={"realized_value": "welfare", "count": "n_seeds"})


def window_mean(df: pd.DataFrame, by: Sequence[str], col: str) -> pd.DataFrame:
    """Per-window mean of a tenant-interval column, averaged over seeds within the window."""
    x = df if "seed" in df.columns else df.assign(seed=-1)
    g = x.groupby(list(by) + ["date", "unit", "seed"])[col].mean().reset_index()
    return g.groupby(list(by) + ["date", "unit"])[col].mean().reset_index()


# ---------------------------------------------------------------------------------------------------------- E0
def analyze_e0(T: dict, cfg: dict) -> dict:
    nb, mac, tables, notes, hyp = cfg["n_boot"], Macros("e0"), {}, [], {}
    win = _need(T, "e0_windows")
    w = win[win["config"] == "default"]
    if len(w) == 0:
        raise MissingInput("no default-configuration E0 windows")
    targets = yaml.safe_load(open(SPEC / "params.yaml"))["regime_targets"]
    wts = w["rg_intervals"].to_numpy(float)
    rows = []
    for c in sorted(x for x in w.columns if x.startswith("rg_") and x != "rg_intervals"):
        name = c[3:]
        v = float(np.average(w[c].to_numpy(float), weights=wts))
        tmin, tmax = targets.get(f"{name}_min"), targets.get(f"{name}_max")
        rows.append(dict(diagnostic=name, value=v, target_min=tmin, target_max=tmax,
                         meets=None if tmin is None and tmax is None else bool(
                             (tmin is None or v >= tmin) and (tmax is None or v <= tmax))))
    reg = pd.DataFrame(rows)
    tables["regime"] = records(reg)
    hyp["regime_targets_met"] = bool(reg["meets"].dropna().all()) if reg["meets"].notna().any() else None
    r = reg.set_index("diagnostic")["value"]
    if "frac_sumH_gt_B" in r:
        mac.add("EzeroNeedShare", 100 * r["frac_sumH_gt_B"], "{:.0f}", "percent of scored intervals",
                "need caps exceed B (truthful reports, equal weights)")
    if "frac_weights_change_quota" in r:
        mac.add("EzeroWeightShare", 100 * r["frac_weights_change_quota"], "{:.0f}", "percent of scored intervals",
                "weights change integer quotas")
    for name, stem in (("mean_weighted_share", "EzeroWeightedPRB"), ("gbr_outage", "EzeroOutage"),
                       ("frac_all_caps_fit", "EzeroAllFit")):
        if name in r:
            mac.add(stem, 100 * r[name], "{:.1f}", "percent", name)
    loads = []
    for c in ("offered_load", "capacity_load"):
        v = w[c].to_numpy(float)
        loads.append(dict(measure=c, windows=v.size, mean=v.mean(), sd=v.std(ddof=1) if v.size > 1 else 0.0,
                          p5=np.quantile(v, 0.05), p50=np.median(v), p95=np.quantile(v, 0.95), min=v.min(),
                          max=v.max()))
        stem = "EzeroOffered" if c == "offered_load" else "EzeroCapacity"
        mac.add(stem + "Mean", v.mean(), "{:.2f}", "fraction of cell capacity", f"mean per-window {c}")
        mac.add(stem + "Min", v.min(), "{:.2f}", "fraction of cell capacity", f"smallest per-window {c}")
        mac.add(stem + "Max", v.max(), "{:.2f}", "fraction of cell capacity", f"largest per-window {c}")
    tables["loads"] = records(pd.DataFrame(loads))
    by_date = w.groupby("date")["offered_load"].mean()
    tables["date_offered_load"] = records(by_date.reset_index())
    mac.add("EzeroDateLoadDev", 100 * float((by_date / MAIN["rho"] - 1).abs().max()), "{:.1f}", "percent of rho",
            "largest deviation of a date's offered load from rho (D2)")
    vc = T.get("e0_vcal")
    if vc is not None and len(vc):
        vc = vc[vc["config"] == "default"]
        Sc = ["N", "Sx", "Sy", "Sxx", "Syy", "Sxy"]
        vrows = []
        for c, g in vc.groupby("cls"):
            pw = g.groupby(["date", "unit"])[Sc].sum().astype(float).reset_index()
            row = dict(cls=c, **_ols(*pw[Sc].sum().to_numpy()))
            if pw["date"].nunique() >= 2:
                for which in ("slope", "r2"):
                    b = boot(pw[Sc].to_numpy(), pw["date"].to_numpy(), n_boot=nb, stat=_ols_stat(which))
                    row[f"{which}_lo"], row[f"{which}_hi"] = b.lo, b.hi
            row["flag_r2_below_0p5"] = bool(row["r2"] < 0.5)
            vrows.append(row)
            if c in CLASS_MACRO:
                mac.add("EzeroRsq" + CLASS_MACRO[c], row["r2"], "{:.2f}", "R squared",
                        f"on-time Mbit regressed on V/pi, {c}")
        tables["vcal"] = records(pd.DataFrame(vrows))
    else:
        notes.append("vcal: skipped (no E0 V-calibration rows)")
    wm = T.get("e0_worked_meta")
    if wm is not None and len(wm):
        _worked_summary(T, wm, mac, tables, hyp)
    else:
        notes.append("worked interval: skipped (no E0 worked-interval files for the default configuration)")
    return dict(macros=mac.d, tables=tables, hypotheses=hyp, notes=notes)


def _worked_summary(T: Mapping, wm: pd.DataFrame, mac: Macros, tables: dict, hyp: dict) -> None:
    """Tables and macros of E0's worked interval (fig:worked). Gains are grid gains of the declared utility at the
    true type over the 257-point own-report grid plus truth (credits per interval; positive = the misreport pays);
    the drop is the largest own-quota decrease on that grid (PRB)."""
    tables["worked_meta"] = records(wm)
    cur = T.get("e0_worked")
    if cur is not None and len(cur):
        tables["worked_curves"] = records(cur[[c for c in ("map", "arm", "u", "quota", "p_full", "p_fix", "U_full",
                                                         "U_fix", "gain_billed", "is_truth") if c in cur.columns]])
    st = T.get("e0_worked_stairs")
    if st is not None and len(st):
        tables["worked_staircase"] = records(st)
    m = wm.set_index("map")
    cls = str(m["cls"].iloc[0])
    mac.add("EzeroWorkedClass", CLASS_NAME.get(cls, cls), "{}", "", "worked interval: deviating tenant's class")
    mac.add("EzeroWorkedTheta", m["theta"].iloc[0], "{:.1f}", "Mbit/s", "worked interval: true demand (report)")
    for mp, stem in (("protected", "Prot"), ("unconstrained", "Unc")):
        if mp not in m.index:
            continue
        r = m.loc[mp]
        mac.add(f"EzeroWorkedQuota{stem}", r.get("quota_truth"), "{:d}", "PRB", f"worked interval: {mp} quota at truth")
        mac.add(f"EzeroWorkedPrice{stem}", r.get("p_full_truth"), "{:.2f}", "credits per interval",
                f"worked interval: {mp} full-decision price at truth" + (" (envelope formula, uncertified)"
                                                                         if mp == "unconstrained" else ""))
        mac.add(f"EzeroWorkedFixPrice{stem}", r.get("p_fix_truth"), "{:.2f}", "credits per interval",
                f"worked interval: {mp} fixed-priority price at truth")
        mac.add(f"EzeroWorkedFixGain{stem}", r.get("fixed_price_grid_gain"), "{:.2f}", "credits per interval",
                f"worked interval: {mp} best grid gain under the fixed-priority price")
        mac.add(f"EzeroWorkedDrop{stem}", r.get("max_drop_grid"), "{:d}", "PRB",
                f"worked interval: {mp} largest own-quota drop on the grid")
    if "protected" in m.index:
        r = m.loc["protected"]
        mac.add("EzeroWorkedPremium", r.get("premium_truth"), "{:.3f}", "credits per interval",
                "worked interval: protected premium T(theta) = p - p_fix at truth")
        mac.add("EzeroWorkedFullGain", r.get("full_price_grid_gain"), "{:.1e}", "credits per interval",
                "worked interval: protected best grid gain under the full-decision price")
        hyp["worked_protected_monotone"] = bool(_f(r.get("max_drop_grid")) <= 0)


def _ols(N, Sx, Sy, Sxx, Syy, Sxy) -> dict:
    vx, vy, cxy = Sxx - Sx * Sx / N, Syy - Sy * Sy / N, Sxy - Sx * Sy / N
    slope = cxy / vx if vx > 0 else float("nan")
    return dict(N=int(N), slope=slope, intercept=(Sy - slope * Sx) / N if vx > 0 else float("nan"),
                r2=cxy * cxy / (vx * vy) if vx > 0 and vy > 0 else float("nan"))


def _ols_stat(which: str):
    def f(m):
        N, Sx, Sy, Sxx, Syy, Sxy = (m[..., j] for j in range(6))
        vx, vy, cxy = Sxx - Sx * Sx / N, Syy - Sy * Sy / N, Sxy - Sx * Sy / N
        return cxy / vx if which == "slope" else cxy * cxy / (vx * vy)
    return f


# ---------------------------------------------------------------------------------------------------------- E1
def analyze_e1(T: dict, cfg: dict) -> dict:
    nb, mac, tables, notes, hyp = cfg["n_boot"], Macros("e1"), {}, [], {}
    W = _need(T, "windows")
    W = W[(W["exp"] == "e1") & (W["policy"] == "truthful")]
    if len(W) == 0:
        raise MissingInput("no E1 windows")
    mem = cfg.get("memberships") or {}
    # pre-freeze audit NUM-09: every controller name comes from spec/controllers.yaml (meas_only = 'No reports')
    lp, lm = label("protected"), label("meas_only")
    den = f"welfare of {lm} on the same windows"

    def h1(cell, crossed=False):
        return arm_contrast(W, cell, "protected", "meas_only", n_boot=nb, alternative="greater", denominator=den,
                            mem=mem if crossed else None)

    res = _run(notes, "H1a", h1, MAIN, True)
    if res:
        r, pr = res
        lo_pos = _rule(_f(r["lo"]) > 0, r["lo"])
        big = _rule(_f(r.get("pct")) >= H1A_MIN_PCT, r.get("pct"))
        r["decision"] = dict(ci_lower_positive=lo_pos, pct_at_least_min=big,
                             rule_met=None if lo_pos is None or big is None else bool(lo_pos and big),
                             min_effect_pct=H1A_MIN_PCT, mei_pct=H1A_MEI_PCT,
                             verdict="holm.json (rule and Holm rejection)")
        r["rule_met"] = r["decision"]["rule_met"]
        r["load_subgroups"] = load_subgroups(pr, n_boot=nb, alternative="greater")
        r["role"] = "primary"
        if cfg.get("split") == "dev":
            r["dev_estimation_half"] = power_inputs(pr, "H1a", n_boot=nb)
        hyp["H1a"] = r
        man = T.get("e1_manifest")
        dm = man[man["config"] == "default"] if man is not None and len(man) else pd.DataFrame()
        if len(dm):                          # round-3 review: from the run's own configuration record
            mac.add("EoneDelay", float(dm["d"].iloc[0]), "{:.0f}", "s",
                    "measurement delay of the main cell (E1 manifest of the default configuration)")
        else:
            notes.append("EoneDelay: no E1 manifest for the default configuration")
        mac.ci("EoneGain", r, "credits per interval", f"H1a: W({lp}) - W({lm}), main cell")
        mac.add("EoneGainReports", r["estimate"], "{:.2f}", "credits per interval", "alias of EoneGain")
        mac.add("EonePct", r.get("pct"), "{:.2f}", "percent of " + den, "H1a as a percentage (alias of EoneGainPct)")
        mac.add("EoneDates", r["dates_positive"], "{:d}", "dates", "dates with a positive date-mean H1a difference")
        mac.add("nTestDates", r["n_dates"], "{:d}", "dates", "dates in the analysis")
        mac.add("EoneWindows", r["n_windows"], "{:d}", "windows", "windows in the H1a analysis")
        mac.add("EoneMeasWelfare", r["mean_b"], "{:.2f}", "credits per interval", f"welfare of {lm}")
        mac.add("EoneProtWelfare", r["mean_a"], "{:.2f}", "credits per interval", f"welfare of {lp}")
    for var, levels, tname in (("d", DELAY_SWEEP, "value_by_delay"), ("rho", LOAD_SWEEP, "value_by_load"),
                               ("lead_s", LEAD_SWEEP, "value_by_lead")):
        rows = []
        for lv in levels:
            rr = _run(notes, f"H1 {var}={lv:g}", h1, dict(MAIN, **{var: lv}))
            if rr:
                r = rr[0]
                rows.append({var: lv, **{k: r.get(k) for k in ("estimate", "lo", "hi", "p", "pct", "pct_lo", "pct_hi",
                                                                "n_dates", "n_windows", "dates_positive", "mean_a",
                                                                "mean_b")}})
        tables[tname] = rows
        if len(rows) >= 2 and var in ("d", "lead_s"):
            sl = slope_over(W, var, [r[var] for r in rows], "protected", "meas_only", n_boot=nb)
            if sl:
                hyp.setdefault("H1b", {})[var] = sl
                mac.ci("EoneSlopeDelay" if var == "d" else "EoneSlopeLead", sl, sl["unit"], f"H1b slope over {var}",
                       nd=3, pct=False)
    for r in tables["value_by_delay"]:
        nm = {1.0: "One", 5.0: "Five", 15.0: "Fifteen"}.get(r["d"])
        if nm:
            mac.add("EoneGainDelay" + nm, r["estimate"], "{:.2f}", "credits per interval",
                    f"H1a contrast at d = {r['d']:g} s")
    for hid, arm_b, stem in (("H1c", "static", "EoneStatic"), ("H1d", "direct", "EoneDirect")):
        res = _run(notes, hid, arm_contrast, W, MAIN, "protected", arm_b, n_boot=nb, alternative="two-sided",
                   denominator=f"{label(arm_b)} welfare on the same windows")
        if res:
            r, pr = res
            r.update(role="secondary (D3)", rules_out=dict(
                advantage_above=r["hi"], disadvantage_beyond=r["lo"],
                text=(f"the data rule out an advantage of Protected over {label(arm_b)} above {r['hi']:.3f} and a "
                      f"disadvantage beyond {-r['lo']:.3f} credits per interval") if _ok(r["lo"], r["hi"])
                else f"no inference (fewer than {MIN_DATES} dates)"))
            r["load_subgroups"] = load_subgroups(pr, n_boot=nb, alternative="two-sided")
            hyp[hid] = r
            mac.ci(stem, r, "credits per interval", f"{hid}: W({lp}) - W({label(arm_b)})")
    res = _run(notes, "H1d diagnostic", arm_contrast, W, MAIN, "direct_truerates", "direct", n_boot=nb,
               alternative="two-sided", denominator="Direct optimizer welfare")
    if res:
        hyp.setdefault("H1d", {})["truerates_minus_direct"] = res[0]
        mac.ci("EoneTrueRate", res[0], "credits per interval",
               f"W({label('direct_truerates')}) - W({label('direct')})")
    desc = []
    for arm_b in ("equal", "static", "direct", "kappa0", "unprotected", "oracle", "direct_truerates", "meas_only"):
        res = _run(notes, f"Protected - {arm_b}", arm_contrast, W, MAIN, "protected", arm_b, n_boot=nb,
                   alternative="two-sided", denominator=f"{label(arm_b)} welfare")
        if res:
            desc.append({k: res[0].get(k) for k in ("arm_a", "arm_b", "estimate", "lo", "hi", "pct", "pct_lo",
                                                     "pct_hi", "n_dates", "n_windows", "dates_positive",
                                                     "dates_negative")})
    tables["contrasts_main"] = desc
    tables["arms_main"] = records(arm_table(W, MAIN, n_boot=nb))
    tables["seed_spread"] = records(seed_spread(W, MAIN, ("protected", "meas_only", "unprotected", "kappa0")))
    _e1_extra(W, nb, mac, tables, hyp, notes)
    return dict(macros=mac.d, tables=tables, hypotheses=hyp, notes=notes)


# ---------------------------------------------------------------------------------------------------------- E2
#: H2a arm -> pooled gain macro stem (per-class macros append Video / Urllc / Mmtc)
H2A_GAIN_MACRO = {"unprotected": "EtwoGain", "fixed_price": "EtwoGainFixed", "unprotected_usage": "EtwoGainUsage",
                  "unprotected_envelope": "EtwoGainEnvelope"}


def br_by_class(BR: pd.DataFrame, arms: Sequence[str], *, n_boot: int, alternative: str = "greater") -> pd.DataFrame:
    """Per (arm, n, class): mean best-response gain per tenant-interval (window means; cluster bootstrap)."""
    rows = []
    for (arm, n, c), g in BR[BR["arm"].isin(arms)].groupby(["arm", "n", "cls"]):
        wv = window_mean(g, [], "gain")
        r = boot(wv["gain"].to_numpy(float), wv["date"].to_numpy(), n_boot=n_boot)
        row = dict(arm=arm, label=label(arm), n=int(n), cls=c, mean_gain=r.estimate, lo=r.lo, hi=r.hi,
                   p=r.p_value(0.0, alternative), max_gain=float(g["gain"].max()), n_tenant_intervals=len(g),
                   n_windows=r.n_windows, n_dates=r.n_dates, exact_share=float(g["exact"].astype(bool).mean()))
        if "z_star" in g.columns and g["z_star"].notna().any():
            mis = ~np.isclose(g["z_star"].astype(float), g["theta"].astype(float), rtol=1e-12, atol=0)
            mult = (g["z_star"] / g["theta"]).where(g["theta"] > 0)
            row.update(share_misreport=float(mis.mean()),
                       mean_multiple_when_misreport=float(mult[mis].mean()) if mis.any() else None)
        rows.append(row)
    return pd.DataFrame(rows)


def br_pooled(BR: pd.DataFrame, arm: str, *, n_boot: int, n: int = 3) -> dict:
    x = BR[(BR["arm"] == arm) & (BR["n"] == n)]
    if len(x) == 0:
        raise MissingInput(f"no best responses for {arm}")
    wv = window_mean(x, [], "gain")
    r = boot(wv["gain"].to_numpy(float), wv["date"].to_numpy(), n_boot=n_boot)
    return dict(r.as_dict(), p=r.p_value(0.0, "greater"), max_gain=float(x["gain"].max()), arm=arm)


def victim_table(W: pd.DataFrame, exp: str = "e2") -> pd.DataFrame:
    """Per (window, arm, deviator), seeds averaged: D4 victim effects (deviation run minus truthful run)."""
    x = W[(W["exp"] == exp) & (W["tenant"] >= 0) & _cell_mask(W, MAIN)]
    dev, tr = x[x["policy"] == "br"], x[x["policy"] == "truthful"]
    if len(dev) == 0 or len(tr) == 0:
        return pd.DataFrame()
    keys = ["date", "unit", "arm", "seed", "tenant"]
    cols = ["on_time_Mbit", "dropped_Mbit", "realized_value", "utility_declared", "utility_realized"]
    j = dev.merge(tr[keys + cols], on=keys, suffixes=("", "_t"))

    def miss(on, dr):
        return 100 * dr / (on + dr) if on + dr > 0 else np.nan
    rows = []
    for (date, unit, arm, seed, dv), g in j.groupby(["date", "unit", "arm", "seed", "deviator"]):
        v, d = g[g["tenant"] != dv], g[g["tenant"] == dv]
        on_t, dr_t = v["on_time_Mbit_t"].sum(), v["dropped_Mbit_t"].sum()
        on_d, dr_d = v["on_time_Mbit"].sum(), v["dropped_Mbit"].sum()
        rows.append(dict(date=date, unit=unit, arm=arm, seed=seed, deviator=int(dv),
                         deviator_cls=d["cls"].iloc[0] if len(d) else "none",
                         d_on_time_victims=on_d - on_t, d_dropped_victims=dr_d - dr_t,
                         d_miss_pct_victims=miss(on_d, dr_d) - miss(on_t, dr_t),
                         d_value_victims=v["realized_value"].sum() - v["realized_value_t"].sum(),
                         d_value_deviator=d["realized_value"].sum() - d["realized_value_t"].sum(),
                         d_utility_declared_deviator=d["utility_declared"].sum() - d["utility_declared_t"].sum(),
                         dW_realized=g["realized_value"].sum() - g["realized_value_t"].sum()))
    out = pd.DataFrame(rows)
    return out.groupby(["date", "unit", "arm", "deviator", "deviator_cls"]).mean(numeric_only=True).reset_index() \
        .drop(columns="seed")


def victim_contrast(VT: pd.DataFrame, arm: str, metric: str, *, n_boot: int, alternative: str, mem=None) -> dict:
    """Window-level victim effect of one arm, averaged over the deviator roles played on the window."""
    x = VT[VT["arm"] == arm]
    if len(x) == 0:
        raise MissingInput(f"no victim rows for {arm}")
    wv = x.groupby(["date", "unit"])[metric].mean().reset_index()
    r = one_sample(wv, metric, n_boot=n_boot, alternative=alternative, mem=mem)
    r.update(arm=arm, metric=metric, roles=sorted(x["deviator_cls"].dropna().unique().tolist()))
    return r


def bandit_summary(BD: pd.DataFrame | None, last: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(final multipliers per arm and class, episode-binned mean played multipliers)."""
    if BD is None or len(BD) == 0:
        return pd.DataFrame(), pd.DataFrame()
    fin = []
    for (arm, c), g in BD.groupby(["arm", "cls"]):
        E = int(g["episode"].max()) + 1
        last_ep = g[g["episode"] == E - 1]
        tail = g[g["episode"] >= E - last]
        modal = last_ep["greedy_m"].round(4).mode()
        fin.append(dict(arm=arm, label=label(arm), cls=c, final_modal_mult=float(modal.min()) if len(modal) else None,
                        final_mean_greedy=float(last_ep["greedy_m"].mean()), tail_mean_played=float(tail["m"].mean()),
                        tail_share_truth=float(np.isclose(tail["m"], 1.0).mean()),
                        tail_mean_gain=float(tail["gain"].mean()),
                        runs=int(g.groupby(["algo", "seed", "run"]).ngroups), episodes=E))
    E = int(BD["episode"].max()) + 1
    width = max(1, E // 40)
    x = BD.assign(ep_bin=(BD["episode"] // width) * width)
    traj = x.groupby(["arm", "cls", "ep_bin"]).agg(mean_mult=("m", "mean"), mean_greedy=("greedy_m", "mean"),
                                                    share_truth=("m", lambda s: float(np.isclose(s, 1.0).mean())),
                                                    runs=("run", "nunique")).reset_index()
    traj = traj.rename(columns={"ep_bin": "episode"})
    traj["label"] = traj["arm"].map(label)
    traj["bin_width"] = width
    return pd.DataFrame(fin), traj


def _mult_text(fin: pd.DataFrame, arm: str) -> str | None:
    f = fin[fin["arm"] == arm] if len(fin) else fin
    parts = [f"{CLASS_NAME[c]} {f[f['cls'] == c]['final_modal_mult'].iloc[0]:g}" for c in CLASS_ORDER
             if len(f) and (f["cls"] == c).any()]
    return ", ".join(parts) if parts else None


def analyze_e2(T: dict, cfg: dict) -> dict:
    nb, mac, tables, notes, hyp = cfg["n_boot"], Macros("e2"), {}, [], {}
    mem = cfg.get("memberships") or {}
    BR = T.get("br")
    BR2 = BR[BR["exp"] == "e2"] if BR is not None and len(BR) else pd.DataFrame()
    if len(BR2):
        tb = br_by_class(BR2, MANIP_ARMS + ("protected",), n_boot=nb)
        tables["br_by_class"] = records(tb)
        manip = tb[tb["arm"].isin(("fixed_price", "unprotected", "unprotected_usage"))]
        lo_ok = manip["lo"].astype(float).notna() if len(manip) else pd.Series(dtype=bool)
        hyp["H2a"] = dict(supported=bool((manip["lo"].astype(float) > 0).any()) if len(manip) and lo_ok.any()
                          else None,
                          positive=records(manip[manip["lo"] > 0][["arm", "cls", "mean_gain", "lo"]]),
                          rule="mean gain CI lower bound > 0 for at least one class under the fixed-priority price "
                               "or the usage tariff")
        for arm, stem in H2A_GAIN_MACRO.items():
            r = _run(notes, f"BR {arm}", br_pooled, BR2, arm, n_boot=nb)
            if r:
                hyp["H2a"][arm] = r
                mac.ci(stem, r, "credits per interval", f"mean best-response gain, {label(arm)}, classes pooled",
                       pct=False)
                mac.add(stem + "Max", r["max_gain"], "{:.2f}", "credits per interval", f"largest gain, {label(arm)}")
        # pre-freeze audit NUM-05: the pooled macro pools the classes; the H2a verdict is per class, so every H2a arm
        # also gets per-class estimates with their CIs (<stem><Class>, ...Lo, ...Hi, ...CI)
        for _, r in tb.iterrows():
            stem = H2A_GAIN_MACRO.get(r["arm"])
            if stem and r["arm"] != "unprotected_envelope" and r["cls"] in CLASS_MACRO and int(r["n"]) == 3:
                mac.ci(stem + CLASS_MACRO[r["cls"]], dict(estimate=r["mean_gain"], lo=r["lo"], hi=r["hi"]),
                       "credits per interval",
                       f"mean best-response gain of the {CLASS_NAME[r['cls']]} tenant, {label(r['arm'])}", pct=False)
    else:
        notes.append("H2a: skipped (no E2 best-response rows)")
    MO = T.get("monotone")
    if MO is not None and len(MO):
        rows = []
        for (arm, c), g in MO.groupby(["arm", "cls"]):
            steps = float((g["n_points"] - 1).sum())
            rows.append(dict(arm=arm, label=label(arm), cls=c, states=len(g),
                             share_states_nonmonotone=float((g["n_decrease"] > 0).mean()),
                             pct_gridsteps_decrease=100 * float(g["n_decrease"].sum()) / steps,
                             max_drop_prb=float(g["max_drop_prb"].max())))
        tb = pd.DataFrame(rows)
        tables["nonmonotone_by_class"] = records(tb)
        u = MO[MO["arm"] == "unprotected"]
        env = BR2[BR2["arm"] == "unprotected_envelope"] if len(BR2) else pd.DataFrame()
        expl = None
        if len(u) and len(env):
            j = u[u["n_decrease"] > 0].merge(env[["window", "k", "tenant", "gain", "eps"]],
                                             on=["window", "k", "tenant"])
            expl = float((j["gain"] > j["eps"]).mean()) if len(j) else None
        if len(u):
            hyp["H2b"] = dict(states=len(u), nonmonotone_states=int((u["n_decrease"] > 0).sum()),
                              share_states_nonmonotone=float((u["n_decrease"] > 0).mean()),
                              max_drop_prb=float(u["max_drop_prb"].max()), exploitable_share=expl,
                              protected_negative_control=int((MO[MO["arm"] == "protected"]["n_decrease"] > 0).sum()),
                              supported=bool((u["n_decrease"] > 0).any()))
            mac.add("EtwoNonmono", 100 * float(u["n_decrease"].sum() / (u["n_points"] - 1).sum()), "{:.2f}",
                    "percent of own-report grid steps", "own quota decreases, Unconstrained net, classes pooled")
            mac.add("EtwoNonmonoStates", 100 * hyp["H2b"]["share_states_nonmonotone"], "{:.1f}",
                    "percent of tenant-intervals", "tenant-intervals with a non-monotone own-quota path")
            mac.add("EtwoDrop", hyp["H2b"]["max_drop_prb"], "{:.0f}", "PRB", "largest own-quota drop")
            if expl is not None:
                mac.add("EtwoExploit", 100 * expl, "{:.1f}", "percent of non-monotone tenant-intervals",
                        "non-monotone states in which the envelope-formula best response gains")
            for _, r in tb[tb["arm"] == "unprotected"].iterrows():
                if r["cls"] in CLASS_MACRO:
                    mac.add("EtwoNonmono" + CLASS_MACRO[r["cls"]], r["pct_gridsteps_decrease"], "{:.2f}",
                            "percent of own-report grid steps", r["cls"])
    else:
        notes.append("H2b: skipped (no non-monotonicity rows)")
    W = T.get("windows")
    VT = victim_table(W) if W is not None and len(W) else pd.DataFrame()
    if len(VT):
        tables["victims_by_role"] = records(VT.groupby(["arm", "deviator_cls"]).mean(numeric_only=True).reset_index())
        for arm in sorted(VT["arm"].unique()):
            for metric, alt in (("d_on_time_victims", "less"), ("d_miss_pct_victims", "greater"),
                                ("d_value_victims", "less"), ("dW_realized", "two-sided"),
                                ("d_utility_declared_deviator", "greater")):
                is_primary = arm == H2C_PRIMARY_ARM and metric == "d_on_time_victims"
                r = _run(notes, f"H2c {arm} {metric}", victim_contrast, VT, arm, metric, n_boot=nb, alternative=alt,
                         mem=mem if is_primary else None)
                if r:
                    # pre-freeze audit PV-3: only the primary cell enters Holm; everything else is secondary
                    r["role"] = ("primary (copied to H2c; Holm family)" if is_primary
                                 else "secondary (unadjusted CI, not in the Holm family)")
                    hyp.setdefault("H2c_all", {}).setdefault(arm, {})[metric] = r
        tables["victim_contrasts"] = [dict(arm=arm, label=label(arm), metric=metric,
                                           **{k: r.get(k) for k in ("estimate", "lo", "hi", "p", "n_dates",
                                                                    "n_windows", "role")})
                                      for arm, mm in hyp.get("H2c_all", {}).items() for metric, r in mm.items()]
        prim = hyp.get("H2c_all", {}).get(H2C_PRIMARY_ARM, {})
        if "d_on_time_victims" in prim:
            r = dict(prim["d_on_time_victims"])
            seeds = sorted(int(x) for x in W[(W["exp"] == "e2") & (W["arm"] == H2C_PRIMARY_ARM)]["seed"].unique())
            r.update(role="primary (D4)", unit="Mbit per interval (honest tenants, summed)",
                     rule_met=_rule(_f(r["hi"]) < 0, r["hi"]),
                     rule="CI upper bound of the honest tenants' on-time change < 0",
                     verdict="holm.json (rule and Holm rejection)", arm=H2C_PRIMARY_ARM, checkpoint_seeds=seeds,
                     n_seeds=len(seeds))
            if len(seeds) == 1:
                r["seed_note"] = ("one checkpoint seed: the CI excludes seed variability (pre-registration item, "
                                  "STATUS.md §8.3 item 2)")
            if cfg.get("split") == "dev":
                x = VT[VT["arm"] == H2C_PRIMARY_ARM].groupby(["date", "unit"])["d_on_time_victims"].mean()
                pw = x.reset_index().rename(columns={"d_on_time_victims": "a"}).assign(b=0.0)
                r["dev_estimation_half"] = power_inputs(pw, "H2c", n_boot=nb)
            hyp["H2c"] = r
            mac.ci("EtwoVictim", r, "Mbit per interval",
                   "H2c: change in honest tenants' on-time Mbit (deviation minus truthful; negative = loss)", pct=False)
            # pre-freeze audit HC-2: the paper states a positive loss; EtwoVictimLoss = -EtwoVictim (bounds swapped)
            mac.ci_negated("EtwoVictimLoss", r,
                           "Mbit per interval lost by the honest tenants (summed; positive = loss)",
                           "H2c: on-time Mbit lost by the honest tenants (minus EtwoVictim)")
            for metric, stem, unit in (("d_miss_pct_victims", "EtwoVictimMiss", "percentage points"),
                                       ("d_value_victims", "EtwoVictimValue", "credits per interval"),
                                       ("dW_realized", "EtwoTotalWelfare", "credits per interval")):
                mac.ci(stem, prim.get(metric), unit, f"H2c companion: {metric}", pct=False)
        else:
            notes.append(f"H2c: no truthful/best-response replay pairs for {H2C_PRIMARY_ARM}")
    else:
        notes.append("H2c: skipped (no replay windows)")
    fin, _ = bandit_summary(T.get("bandit"))
    tables["bandit_final"] = records(fin)
    for arm, stem in (("unprotected", "EtwoMult"), ("fixed_price", "EtwoMultFixed")):
        mac.add(stem, _mult_text(fin, arm), unit="multiplier", source=f"final modal multiplier, {label(arm)}")
    _e2_extra(W, BR2, nb, mac, tables, notes)
    return dict(macros=mac.d, tables=tables, hypotheses=hyp, notes=notes)


# ---------------------------------------------------------------------------------------------------------- E3
def h3a(BR: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Deterministic certificate over every scored tenant-interval of the monotone full-price designs.

    Rule (PREREG §6 H3a, evidence check 2026-10-20 item 7): the CERTIFIED maximal deviation gain, i.e. the computed
    gain plus its numerical certificate, must not exceed eps_i = 1e-8 pi_i mu_i Delta: a violation is
    ``gain + cert > eps_i`` (``violations`` = ``violations_strict``; ``holds`` = none). The earlier allowance
    ``gain > max(cert, eps_i)`` is kept as a reported check (``violations_max_rule``). H6a (``analyze_e6``) reads the
    same certificate through this function ('zero certified gain')."""
    x = BR[(BR["price"].astype(str) == "full") & ~BR["arm"].isin(NONMONOTONE_ARMS)]
    if len(x) == 0:
        raise MissingInput("no full-price best responses of monotone designs")
    gain, cert = x["gain"].astype(float), x["cert"].astype(float).fillna(0.0)
    eps = x["eps"].astype(float)
    viol_max = gain > np.maximum(cert, eps)
    strict = gain + cert > eps
    by = []
    # pre-freeze audit NUM-04: the driver's certificate ``cert`` is 2 e (pricing/response.py: cert = 2 max(err)), so
    # the guaranteed error bound of the computed price is e = cert / 2 (max_err)
    for (arm, n, c), g in x.assign(_v=viol_max, _s=strict, _c=cert).groupby(["arm", "n", "cls"]):
        by.append(dict(arm=arm, label=label(arm), n=int(n), cls=c, tenant_intervals=len(g),
                       max_gain=float(g["gain"].max()), max_cert=float(g["_c"].max()),
                       max_err=float(g["_c"].max()) / 2.0, min_eps=float(g["eps"].min()),
                       violations=int(g["_s"].sum()), violations_strict=int(g["_s"].sum()),
                       violations_max_rule=int(g["_v"].sum()),
                       max_certified_gain_over_eps=float(((g["gain"] + g["_c"]) / g["eps"]).max()),
                       exact_share=float(g["exact"].astype(bool).mean())))
    seeds = sorted(int(v) for v in x["seed"].unique()) if "seed" in x.columns else []
    res = dict(tenant_intervals=len(x), max_gain=float(gain.max()), max_cert=float(cert.max()),
               max_err=float((cert / 2.0).max()),
               min_eps=float(eps.min()), violations=int(strict.sum()), violations_strict=int(strict.sum()),
               violations_max_rule=int(viol_max.sum()),
               max_certified_gain_over_eps=float(((gain + cert) / eps).max()),
               all_exact=bool(x["exact"].astype(bool).all()), holds=bool(strict.sum() == 0),
               checkpoint_seeds=[v for v in seeds if v >= 0],
               seed_note=("certified for the listed checkpoint seeds of the learned maps only (pre-registration item, "
                          "STATUS.md §8.3 item 2)"),
               rule="no scored tenant-interval whose certified gain (computed gain + certificate) exceeds "
                    "1e-8 pi_i mu_i Delta (PREREG §6; exact cell enumeration): violations = violations_strict = "
                    "#(gain + cert > eps_i); violations_max_rule = #(gain > max(cert, eps_i)) is a reported check")
    return res, pd.DataFrame(by)


def gain_by_mult(ID: pd.DataFrame, *, n_boot: int) -> pd.DataFrame:
    """fig_e2e3_gain: mean deviation gain at report min(m theta, C) per class, multiple and price rule (protected
    map: full-decision vs fixed-priority price); window means with a cluster-bootstrap CI; m = 1 gives 0."""
    x = ID[ID["tag"].astype(str).str.fullmatch(r"x[0-9.]+")].copy()
    if len(x) == 0:
        return pd.DataFrame()
    x["mult"] = x["tag"].str[1:].astype(float)
    rows = []
    for (c, m), g in x.groupby(["cls", "mult"]):
        for col, arm, price in (("G_full", "protected", "full"), ("G_fix", "fixed_price", "fixed")):
            wv = window_mean(g, [], col)
            r = boot(wv[col].to_numpy(float), wv["date"].to_numpy(), n_boot=n_boot)
            rows.append(dict(cls=c, mult=m, arm=arm, label=label(arm), price=price, mean_gain=r.estimate, lo=r.lo,
                             hi=r.hi, max_gain=float(g[col].max()), n_windows=r.n_windows, n_dates=r.n_dates))
    for c in x["cls"].unique():
        for arm, price in (("protected", "full"), ("fixed_price", "fixed")):
            rows.append(dict(cls=c, mult=1.0, arm=arm, label=label(arm), price=price, mean_gain=0.0, lo=0.0, hi=0.0,
                             max_gain=0.0, n_windows=None, n_dates=None))
    return pd.DataFrame(rows).sort_values(["cls", "arm", "mult"]).reset_index(drop=True)


def h3d_table(H: pd.DataFrame) -> pd.DataFrame:
    """Pooled realized-utility gain of constant multipliers over truth per (arm, tenant, m), percent of the truthful
    realized utility (E3 driver totals over the scored intervals)."""
    rows = []
    for (arm, t), g in H.groupby(["arm", "tenant"]):
        base = g[g["m"] == 1.0].set_index(["unit", "seed"])
        for m, h in g[g["m"] != 1.0].groupby("m"):
            h = h.set_index(["unit", "seed"])
            b = base.loc[h.index]
            dU = h["U_realized"] - b["U_realized"]
            den = b["U_realized"].abs().sum()
            rows.append(dict(arm=arm, tenant=int(t), cls=g["cls"].iloc[0], m=float(m), windows=len(h),
                             pooled_rel_gain_pct=100 * float(dU.sum() / den) if den > 0 else None,
                             mean_declared_gain=float(((h["U_declared"] - b["U_declared"]) / h["n_intervals"]).mean()),
                             d_value_victims=float(((h["value_victims"] - b["value_victims"])
                                                    / h["n_intervals"]).mean())))
    return pd.DataFrame(rows)


def analyze_e3(T: dict, cfg: dict) -> dict:
    nb, mac, tables, notes, hyp = cfg["n_boot"], Macros("e3"), {}, [], {}
    BR = T.get("br")
    B3 = BR[BR["exp"] == "e3"] if BR is not None and len(BR) else pd.DataFrame()
    if len(B3):
        r = _run(notes, "H3a", h3a, B3)
        if r:
            hyp["H3a"], by = r
            tables["h3a_by_design"] = records(by)
            mac.add("EthreeMaxGain", hyp["H3a"]["max_gain"], "{:.2g}", "credits per interval",
                    "largest computed full-price deviation gain, monotone designs")
            # pre-freeze audit NUM-04: EthreeTol (which held the certificate 2e under the name 'error bound') is
            # replaced by EthreeErr = e and EthreeCert = 2e; rendered as LaTeX scientific numbers (tex_num)
            mac.add("EthreeErr", hyp["H3a"]["max_err"], "{:.1e}", "credits per interval",
                    "largest guaranteed error bound e of the computed price (pi_i Delta x summed bracket widths)")
            mac.add("EthreeCert", hyp["H3a"]["max_cert"], "{:.1e}", "credits per interval",
                    "largest certificate 2e, twice the guaranteed error bound (the H3a allowance)")
            mac.add("EthreeEps", hyp["H3a"]["min_eps"], "{:.1e}", "credits per interval",
                    "smallest allowance 1e-8 pi mu Delta")
            mac.add("EthreeViol", hyp["H3a"]["violations"], "{:d}", "tenant-intervals", "H3a violations")
            mac.add("EthreeChecked", hyp["H3a"]["tenant_intervals"], "{:,d}", "tenant-intervals",
                    "scored tenant-intervals checked")
            _e3_extra_h3a(hyp["H3a"], mac)
    else:
        notes.append("H3a: skipped (no E3 certificate rows)")
    fin, traj = bandit_summary(T.get("bandit"))
    tables["bandit_final"] = records(fin)
    tables["bandit_trajectories"] = records(traj)
    if len(fin) and (fin["arm"] == "protected").any():
        f = fin[fin["arm"] == "protected"]
        hyp["H3b"] = dict(final=records(f), holds=bool(np.allclose(f["final_modal_mult"].astype(float), 1.0)))
        mac.add("EthreeMult", _mult_text(fin, "protected"), unit="multiplier",
                source="final modal multiplier of learning tenants, Protected (ours)")
    else:
        notes.append("H3b: skipped (no protected bandit runs)")
    PR = T.get("premium")
    if PR is not None and len(PR) and (PR["arm"] == "protected").any():
        x = PR[PR["arm"] == "protected"]
        rows = []
        for c, g in list(x.groupby("cls")) + [("all", x)]:
            gw = window_mean(g, [], "T").merge(window_mean(g, [], "p"), on=["date", "unit"])
            rT = boot(gw["T"].to_numpy(float), gw["date"].to_numpy(), n_boot=nb)
            pay = float(gw["p"].mean())
            rows.append(dict(cls=c, premium=rT.estimate, premium_lo=rT.lo, premium_hi=rT.hi, payment=pay,
                             premium_pct_of_payment=100 * float(gw["T"].mean()) / pay if pay > 0 else None,
                             share_T_certified_positive=float((g["T_lo"] > 0).mean()) if "T_lo" in g else None))
        tb = pd.DataFrame(rows)
        tables["premium_by_class"] = records(tb)
        tot = tb[tb["cls"] == "all"].iloc[0]
        hyp["H3c"] = _clean(tot.to_dict())
        mac.add("EthreePrem", tot["premium"], "{:.3f}", "credits per tenant-interval",
                "premium T(theta) of an honest tenant (mean over tenants and intervals)")
        mac.add("EthreePremPct", tot["premium_pct_of_payment"], "{:.1f}", "percent of payment",
                "premium as a share of the full-decision payment")
    else:
        notes.append("H3c premium: skipped (no protected certificate rows)")
    ID = T.get("ident")
    if ID is not None and len(ID):
        # pre-freeze audit HC-6: the same-engine residual holds by construction, so the identity is judged on the
        # residual against the INDEPENDENT engine (within its error bound); the same-engine figures stay as a check
        n_ind = int((ID["resid_indep"].abs() > ID["err_bound"] + 1e-12).sum())
        hyp["H3c_identity"] = dict(rows=len(ID), max_abs_resid_same=float(ID["resid_same"].abs().max()),
                                   n_above_1e9=int((ID["resid_same"].abs() > 1e-9).sum()),
                                   max_abs_resid_indep=float(ID["resid_indep"].abs().max()),
                                   n_indep_above_bound=n_ind, holds=bool(n_ind == 0),
                                   holds_same_engine=bool(ID["resid_same"].abs().max() < 1e-9),
                                   rule="no row whose independent-engine residual exceeds its error bound "
                                        "(resid_indep <= err_bound); the same-engine residual < 1e-9 is reported")
        mac.add("EthreeIdentity", hyp["H3c_identity"]["max_abs_resid_same"], "{:.1e}", "credits per interval",
                "largest gain-identity residual (same engine; holds by construction)")
        mac.add("EthreeIdentityIndep", hyp["H3c_identity"]["max_abs_resid_indep"], "{:.1e}", "credits per interval",
                "largest gain-identity residual against the independent engine (the H3c rule: within its bound)")
        mac.add("EthreeIdentityAbove", n_ind, "{:d}", "rows of the identity check",
                "H3c: rows whose independent-engine residual exceeds its error bound (0 = the identity holds)")
        tables["gain_by_mult"] = records(gain_by_mult(ID, n_boot=nb))
    else:
        notes.append("H3c identity and gain by multiple: skipped (no ident rows)")
    H = T.get("h3d")
    if H is not None and len(H):
        tb = h3d_table(H)
        tables["h3d"] = records(tb)
        if len(tb) and tb["pooled_rel_gain_pct"].notna().any():
            mx = tb.loc[tb["pooled_rel_gain_pct"].astype(float).idxmax()]
            hyp["H3d"] = dict(max_pct=float(mx["pooled_rel_gain_pct"]), arm=mx["arm"], cls=mx["cls"],
                              m=float(mx["m"]), holds=bool(mx["pooled_rel_gain_pct"] <= H3D_MARGIN_PCT),
                              margin_pct=H3D_MARGIN_PCT)
            mac.add("EthreeRealGainPct", hyp["H3d"]["max_pct"], "{:.2f}", "percent of truthful realized utility",
                    "H3d: largest realized-utility gain over constant multipliers")
    else:
        notes.append("H3d: skipped (no h3d rows)")
    return dict(macros=mac.d, tables=tables, hypotheses=hyp, notes=notes)


# ---------------------------------------------------------------------------------------------------------- E4
def _h4b_from_e4(H: pd.DataFrame, arm: str, *, n_boot: int) -> dict:
    """H4b from the E4 driver's pairs (same window and checkpoint seed): per window, mean over seeds and deviator
    roles of honest value under Protected (truthful) minus under ``arm`` with the deviator best-responding."""
    x = H[H["arm"] == arm]
    if len(x) == 0:
        raise MissingInput(f"no E4 H4b rows for {arm}")
    g = x.groupby(["date", "unit", "seed"])[["honest_value_protected", "honest_value_arm_br"]].mean() \
        .groupby(["date", "unit"]).mean().reset_index()
    r = contrast(g.rename(columns={"honest_value_protected": "a", "honest_value_arm_br": "b"}), n_boot=n_boot,
                 alternative="greater", denominator=f"honest value under {label(arm)} with a best responder")
    seeds = sorted(int(v) for v in x["seed"].unique()) if "seed" in x.columns else []
    r.update(arm=arm, unit="credits per interval", supported=_rule(_f(r["lo"]) > 0, r["lo"]), source="e4 h4b_br",
             roles=sorted(x["cls"].dropna().unique().tolist()) if "cls" in x else None, checkpoint_seeds=seeds,
             n_seeds=len(seeds))
    return r


def _h4b_from_e1e2(W: pd.DataFrame, arm: str, *, n_boot: int, split: str | None = None) -> dict:
    """Fallback H4b: Protected truthful (E1, seeds averaged) vs E2 best-response replays on the same windows. It mixes
    checkpoint seeds (E1 averages five, E2 replays seed 0), so it is refused on TEST (pre-freeze audit HC-9): there
    H4b needs the E4 driver's ``h4b_br.parquet``."""
    if split == "test":
        raise MissingInput(f"no E4 h4b_br rows for {arm}; the E1/E2 fallback mixes checkpoint seeds and is not used "
                           "on TEST")
    W1 = W[(W["exp"] == "e1") & (W["policy"] == "truthful") & _cell_mask(W, MAIN) & (W["arm"] == "protected")
           & np.isclose(W["kappa_scale"].astype(float).fillna(1.0), 1.0) & (W["tenant"] >= 0)]
    pv = W1.groupby(["date", "unit", "tenant"])["realized_value"].mean()
    dv = W[(W["exp"] == "e2") & (W["policy"] == "br") & (W["tenant"] >= 0) & (W["arm"] == arm)]
    rows = []
    for (date, unit, d), g in dv.groupby(["date", "unit", "deviator"]):
        gd = g.groupby("tenant")["realized_value"].mean()
        honest = [t for t in gd.index if t != d and (date, unit, t) in pv.index]
        if honest:
            rows.append(dict(date=date, unit=unit, a=float(sum(pv[(date, unit, t)] for t in honest)),
                             b=float(gd[honest].sum())))
    if not rows:
        raise MissingInput(f"no E1/E2 pairs for {arm}")
    pr = pd.DataFrame(rows).groupby(["date", "unit"])[["a", "b"]].mean().reset_index()
    r = contrast(pr, n_boot=n_boot, alternative="greater", denominator=f"honest value under {label(arm)}")
    r.update(arm=arm, unit="credits per interval", supported=_rule(_f(r["lo"]) > 0, r["lo"]),
             source="e1 protected vs e2 replays")
    return r


def _h4b_learning(L: pd.DataFrame, W: pd.DataFrame, *, n_boot: int) -> pd.DataFrame:
    """H4b (learning deviators): honest tenants (end multiplier 1) of each bandit end state, value under Protected
    (E4 truthful, same window and seed) minus under the design; per window weighted by the runs ending there."""
    prot = W[(W["exp"] == "e4") & (W["arm"] == "protected") & (W["policy"] == "truthful") & (W["tenant"] >= 0)
             & np.isclose(W["kappa_scale"].astype(float).fillna(1.0), 1.0)]
    pv = prot.set_index(["unit", "seed", "tenant"])["realized_value"]
    rows = []
    for arm, g in L.groupby("arm"):
        hon = g[g["honest"].astype(bool)].copy()
        if len(hon) == 0:
            rows.append(dict(arm=arm, label=label(arm), status="no tenant ends at multiplier 1"))
            continue
        hon["vp"] = [pv.get((u, s, t), np.nan) for u, s, t in zip(hon["unit"], hon["seed"], hon["tenant"])]
        hon["delta"] = hon["vp"] - hon["value"]
        # pre-freeze audit HC-9: an all-NaN group (no Protected run on the window) must stay NaN, not sum to 0
        per = hon.groupby(["date", "unit", "seed", "profile_tag"]).agg(delta=("delta", lambda v: v.sum(min_count=1)),
                                                                      w=("n_runs", "first"))
        per = per.reset_index().dropna(subset=["delta"])
        if len(per) == 0:
            rows.append(dict(arm=arm, label=label(arm), status="no Protected truthful run on these windows"))
            continue
        wv = per.assign(dw=per["delta"] * per["w"]).groupby(["date", "unit"]).agg(dw=("dw", "sum"), w=("w", "sum"))
        wv = (wv["dw"] / wv["w"]).rename("delta").reset_index()
        r = one_sample(wv, "delta", n_boot=n_boot, alternative="greater")
        rows.append(dict(arm=arm, label=label(arm), status="ok", estimate=r["estimate"], lo=r["lo"], hi=r["hi"],
                         p=r["p"], n_windows=r["n_windows"], n_dates=r["n_dates"],
                         profiles=int(g["profile_tag"].nunique())))
    return pd.DataFrame(rows)


def analyze_e4(T: dict, cfg: dict) -> dict:
    nb, mac, tables, notes, hyp = cfg["n_boot"], Macros("e4"), {}, [], {}
    W = _need(T, "windows")
    mem = cfg.get("memberships") or {}
    W1 = W[(W["exp"] == "e1") & (W["policy"] == "truthful")]
    res = _run(notes, "H4a", arm_contrast, W1, MAIN, "unprotected", "protected", n_boot=nb, alternative="greater",
               denominator="Protected welfare on the same windows", mem=mem, pct_null=H4A_MARGIN_PCT,
               pct_alternative="less")
    if res:
        r, pr = res
        r.update(role="primary", margin_pct=H4A_MARGIN_PCT, noninferiority_p=r.get("p_pct"),
                 rule_met=_rule(_f(r.get("pct_upper_one_sided")) < H4A_MARGIN_PCT, r.get("pct_upper_one_sided")),
                 verdict="holm.json (rule and Holm rejection)",
                 rule="one-sided 95% upper bound of [W(Unconstrained net) - W(Protected)] / W(Protected) < 1 %")
        if "crossed" in r and r["crossed"].get("p_pct") is not None:
            r["crossed"]["noninferiority_p"] = r["crossed"]["p_pct"]
        if cfg.get("split") == "dev":
            r["dev_estimation_half"] = power_inputs(pr, "H4a", n_boot=nb)
        hyp["H4a"] = r
        mac.ci("EfourGap", r, "credits per interval", "H4a: W(Unconstrained net) - W(Protected), truthful")
        mac.add("EfourGapUpper", r.get("pct_upper_one_sided"), "{:.2f}", "percent of Protected welfare",
                "H4a one-sided 95% upper bound")
    H = T.get("e4_h4b")
    for arm in H4B_ARMS + ("unprotected_envelope",):
        if H is not None and len(H) and (H["arm"] == arm).any():
            r = _run(notes, f"H4b {arm}", _h4b_from_e4, H, arm, n_boot=nb)
        else:
            r = _run(notes, f"H4b {arm}", _h4b_from_e1e2, W, arm, n_boot=nb, split=cfg.get("split"))
        if r:
            hyp.setdefault("H4b", {})[arm] = r
            if arm == "unprotected":
                mac.ci("EfourHonest", r, "credits per interval",
                       "H4b: honest value, Protected (truthful) minus Unconstrained net with a best responder")
    L = T.get("e4_learn")
    if L is not None and len(L):
        lt = _h4b_learning(L, W, n_boot=nb)
        tables["h4b_learning"] = records(lt)
        hyp.setdefault("H4b", {})["learning"] = records(lt)
        _h4b_learning_macro(lt, mac, notes)
    _e4_seed_spread(W1, nb, mac, tables, notes)
    src = "e4" if ((W["exp"] == "e4") & (W["arm"] == "protected")).any() else "e1"
    prot = W[(W["exp"] == src) & (W["policy"] == "truthful") & _cell_mask(W, MAIN) & (W["arm"] == "protected")
             & np.isclose(W["kappa_scale"].astype(float).fillna(1.0), 1.0)]
    rows = []
    cols = [k for k in ("price", "V_declared", "p", "p_fix", "T") if k in prot.columns]
    for c, g in list(prot[prot["tenant"] >= 0].groupby("cls")) + [("all", prot[prot["tenant"] == -1])]:
        if len(g) == 0:
            continue
        gw = g.groupby(["date", "unit", "seed"])[cols].mean().groupby(["date", "unit"]).mean()
        pay, val = float(gw["price"].mean()), float(gw["V_declared"].mean())
        row = dict(cls=c, source=src, payment=pay, declared_value=val,
                   payment_pct_of_value=100 * pay / val if val > 0 else None, surplus=val - pay)
        if "T" in gw.columns and gw["T"].notna().any():
            row.update(premium=float(gw["T"].mean()), fixed_part=float(gw["p_fix"].mean()),
                       premium_pct_of_payment=100 * float(gw["T"].mean()) / pay if pay > 0 else None)
        rows.append(row)
    pay = pd.DataFrame(rows)
    tables["payments"] = records(pay)
    if len(pay) and (pay["cls"] == "all").any():
        p0 = pay[pay["cls"] == "all"].iloc[0]
        mac.add("EfourPay", p0["payment_pct_of_value"], "{:.1f}", "percent of declared value",
                "honest payments under the full-decision price, all tenants")
        mac.add("EfourPayPct", p0["payment_pct_of_value"], "{:.1f}", "percent of declared value", "alias of EfourPay")
        mac.add("EfourPayCredits", p0["payment"], "{:.2f}", "credits per interval", "honest payment, all tenants")
        if p0.get("premium_pct_of_payment") is not None:
            mac.add("EfourPrem", p0["premium_pct_of_payment"], "{:.1f}", "percent of payment",
                    "premium share of the honest payment")
    TM = T.get("timing")
    if TM is not None and len(TM) and (TM["n"] == 3).any():
        t3 = TM[TM["n"] == 3]["t_price_ms"].dropna()
        tables["pricing_time_n3"] = [dict(median_ms=float(t3.median()), p95_ms=float(t3.quantile(0.95)),
                                          intervals=len(t3))]
        mac.add("EfourTime", float(t3.quantile(0.95)), "{:.1f}", "ms", "p95 pricing time per interval at n = 3")
    tables["welfare_by_arm"] = records(arm_table(W1, MAIN, n_boot=nb))
    g6, g6_notes = g6_table(W1, MAIN, n_boot=nb)
    notes += g6_notes
    tables["g6_radio_efficiency"] = records(g6)
    _g6_macros(g6, mac)
    fr = frontier(W, n_boot=nb)
    tables["frontier"] = records(fr)
    if len(fr) == 0 or not ((fr["arm"] == "protected")
                            & ~np.isclose(fr["kappa_scale"].astype(float).fillna(1.0), 1.0)).any():
        notes.append("E4b: no kappa-scaled protected arms (E4 driver kappa_x<s>) found; the frontier has the "
                     "references only")
    return dict(macros=mac.d, tables=tables, hypotheses=hyp, notes=notes)


# ---------------------------------------------------------------------------------------------------------- G6
#: (statistic, macro stem of Protected, of the report-free controller, digits, unit) of the G6 macros
G6_MACROS = (("pct_reserved_used", "EfourPrbUsed", "EfourPrbUsedFree", 1,
              "percent of the reserved PRB capacity (quota x true rate per PRB x Delta) that carries on-time traffic"),
             ("pct_cell_used", "EfourCellUsed", "EfourCellUsedFree", 1,
              "percent of the cell's B PRBs that carry on-time traffic"),
             ("mbps_per_reserved_prb", "EfourPrbRate", "EfourPrbRateFree", 3,
              "Mbit/s of on-time traffic per reserved PRB"))
G6_COLS = ("cap_Mbit", "prb_busy", "unused_Mbit", "r_true")


def g6_table(W: pd.DataFrame, cell: Mapping, *, n_boot: int, exp: str = "e1") -> tuple[pd.DataFrame, list]:
    """Goal G6, radio efficiency (pre-freeze audit HC-4 / NUM-06; review/decisions.md D6.7), descriptive: the
    truthful runs of ``exp`` (E1) in ``cell`` (the default configuration), every arm, all tenants together
    (``cls`` = 'all') and per class. Per window the tenant rows are summed (``sum(min_count=1)``) and the seeds of a
    learned arm averaged; the three statistics are ratios of sums over the windows, computed inside ONE two-stage
    date bootstrap (shared resamples, ``BOOT_SEED``, percentile 95 % CI, no adjustment):

    * ``pct_reserved_used`` = 100 sum(on_time_Mbit) / sum(cap_Mbit), cap_Mbit = quota x r_true x Delta (reserved
      capacity; the E1 driver's window column);
    * ``pct_cell_used`` = 100 sum(prb_busy) / (B x number of windows), prb_busy = on_time / (r_true Delta) (PRBs that
      carry on-time traffic per interval);
    * ``mbps_per_reserved_prb`` = sum(on_time_Mbit) / (Delta sum(quota)).

    E1 outputs written before the drivers stored ``cap_Mbit``/``prb_busy`` (DEV outputs before the pre-freeze fix)
    give only the third statistic, with a note. When ``unused_Mbit`` is present, the identity on_time + unused = cap
    (drop mode: no late traffic) is checked per window (``identity_max_abs_resid``, Mbit)."""
    from slicemech.config import Mechanism
    mech = Mechanism()
    B, delta = float(mech.B), float(mech.Delta)
    notes: list = []
    x = W[_cell_mask(W, cell) & (W["policy"] == "truthful") & (W["exp"] == exp) & (W["tenant"] >= 0)]
    x = x[(x["arm"] != "protected") | np.isclose(x["kappa_scale"].astype(float).fillna(1.0), 1.0)]
    if len(x) == 0:
        return pd.DataFrame(), [f"G6: no truthful {exp} windows in the cell {dict(cell)}"]
    have = [c for c in G6_COLS if c in x.columns and x[c].notna().any()]
    lacking = [c for c in ("cap_Mbit", "prb_busy") if c not in have]
    if lacking:
        notes.append(f"G6: the {exp} windows lack {lacking} (outputs written before the pre-freeze HC-4 driver "
                      "change); only the on-time Mbit/s per reserved PRB is computed; rerun E1 for the other two")
    cols = ["on_time_Mbit", "quota"] + [c for c in ("cap_Mbit", "prb_busy", "unused_Mbit") if c in have]
    j = {c: i for i, c in enumerate(cols)}
    stats = {"mbps_per_reserved_prb": lambda m: m[..., j["on_time_Mbit"]] / (delta * m[..., j["quota"]])}
    if "cap_Mbit" in j:
        stats["pct_reserved_used"] = lambda m: 100.0 * m[..., j["on_time_Mbit"]] / m[..., j["cap_Mbit"]]
    if "prb_busy" in j:
        stats["pct_cell_used"] = lambda m: 100.0 * m[..., j["prb_busy"]] / B
    rows = []
    for arm, g in x.groupby("arm"):
        for cls, gc in [("all", g)] + list(g.groupby("cls")):
            pw = gc.groupby(["date", "unit", "seed"])[cols].sum(min_count=1).groupby(["date", "unit"]).mean()
            seeds = int(gc["seed"].nunique())
            pw = pw.reset_index().dropna(subset=cols)
            if len(pw) == 0:
                continue
            V, dates = pw[cols].to_numpy(float), pw["date"].to_numpy()
            nd = int(pw["date"].nunique())
            row = dict(arm=arm, label=label(arm), cls=cls, n_windows=len(pw), n_dates=nd, seeds=seeds,
                       on_time_Mbit=float(pw["on_time_Mbit"].mean()), quota_prb=float(pw["quota"].mean()),
                       B=B, Delta_s=delta)
            for c in ("cap_Mbit", "prb_busy", "unused_Mbit"):
                if c in j:
                    row[c] = float(pw[c].mean())
            if nd >= MIN_DATES:
                res = S.cluster_bootstrap_multi(V, dates, stats, n_boot=n_boot, level=LEVEL, seed=BOOT_SEED)
                for k, b in res.items():
                    row.update({k: b.estimate, f"{k}_lo": b.lo, f"{k}_hi": b.hi})
            else:
                for k, f in stats.items():
                    row.update({k: float(f(V.mean(0)[None, :])[0]), f"{k}_lo": None, f"{k}_hi": None})
                row["inference"] = f"none: {nd} date(s) < MIN_DATES = {MIN_DATES}"
            if "unused_Mbit" in j and "cap_Mbit" in j:
                row["identity_max_abs_resid"] = float((pw["on_time_Mbit"] + pw["unused_Mbit"] - pw["cap_Mbit"])
                                                      .abs().max())
            rows.append(row)
    return pd.DataFrame(rows), notes


def _g6_macros(g6: pd.DataFrame, mac: Macros) -> None:
    """EfourPrbUsed (Protected) and EfourPrbUsedFree (the report-free controller), with their CIs, and the two other
    G6 statistics (EfourCellUsed*, EfourPrbRate*); descriptive, E1 truthful runs of the default configuration."""
    if len(g6) == 0:
        return
    for stat, stem_p, stem_m, nd, unit in G6_MACROS:
        if stat not in g6.columns:
            continue
        for arm, stem in (("protected", stem_p), ("meas_only", stem_m)):
            r = g6[(g6["arm"] == arm) & (g6["cls"] == "all")]
            if len(r) and pd.notna(r[stat].iloc[0]):
                r0 = r.iloc[0]
                mac.ci(stem, dict(estimate=r0[stat], lo=r0.get(f"{stat}_lo"), hi=r0.get(f"{stat}_hi")), unit,
                       f"G6 {stat}, {label(arm)}, all tenants (E1 truthful runs, default configuration)", nd=nd,
                       pct=False)


def frontier(W: pd.DataFrame, *, n_boot: int) -> pd.DataFrame:
    """D7: welfare and honest payment (credits per interval) of the kappa-scaled protected maps (E4) and the
    references (E4 rows where present, else E1)."""
    rows = []
    Wt = W[W["policy"] == "truthful"]
    specs = [("protected", s) for s in KAPPA_SCALES] + [(a, None) for a in ("equal", "static", "direct")]
    for arm, s in specs:
        wv, src = pd.DataFrame(), None
        for exp in ("e4", "e1"):
            wv = unit_values(Wt, MAIN, arm, "realized_value", kappa_scale=s, exp=exp)
            if len(wv):
                src = exp
                break
        if len(wv) == 0:
            continue
        pv = unit_values(Wt, MAIN, arm, "price", kappa_scale=s, exp=src)
        rw, rp = mean_ci(wv, n_boot=n_boot), mean_ci(pv, n_boot=n_boot)
        rows.append(dict(arm=arm, label=label(arm), kappa_scale=s, source=src, welfare=rw["estimate"],
                         welfare_lo=rw["lo"], welfare_hi=rw["hi"], payment=rp["estimate"], payment_lo=rp["lo"],
                         payment_hi=rp["hi"], n_windows=rw["n_windows"], n_dates=rw["n_dates"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------- E5
def analyze_e5(T: dict, cfg: dict) -> dict:
    nb, mac, tables, notes, hyp = cfg["n_boot"], Macros("e5"), {}, [], {}
    P = _need(T, "e5_pairs").copy()
    P["alpha"] = P["alpha"].astype(float)
    gcols = ["arm", "pipeline", "target", "alpha"]

    def per_window(df, col):
        g = df.groupby(gcols + ["date", "unit", "seed"])[col].mean().reset_index()
        return g.groupby(gcols + ["date", "unit"])[col].mean().reset_index()

    later, attack = P[P["phase"] == "later"], P[P["phase"] == "attack"]
    lw = per_window(later, "d_utility_declared_target")
    lr = per_window(later, "d_utility_realized_target")
    aw = per_window(attack, "d_utility_declared_target")
    rows = []
    for key, g in lw.groupby(gcols):
        kd = dict(zip(gcols, key))
        r = one_sample(g, "d_utility_declared_target", n_boot=nb)
        sel = (lr["arm"] == kd["arm"]) & (lr["pipeline"] == kd["pipeline"]) & (lr["target"] == kd["target"]) \
            & np.isclose(lr["alpha"], kd["alpha"])
        rr = one_sample(lr[sel], "d_utility_realized_target", n_boot=nb) if sel.any() else {}
        ga = aw[(aw["arm"] == kd["arm"]) & (aw["target"] == kd["target"]) & np.isclose(aw["alpha"], kd["alpha"])]
        ra = one_sample(ga, "d_utility_declared_target", n_boot=nb) if len(ga) else {}
        # pre-freeze audit NUM-08: attack_change is signed (negative = the distortion cost the target utility in the
        # retraining windows); cost = -attack_change (positive = cost)
        ac, ac_lo, ac_hi = _f(ra.get("estimate")), _f(ra.get("lo")), _f(ra.get("hi"))
        rows.append(dict(**kd, label=label(kd["arm"]), later_gain=r["estimate"], lo=r["lo"], hi=r["hi"], p=r["p"],
                         dates_positive=r["dates_positive"], dates_negative=r["dates_negative"],
                         n_dates=r["n_dates"], n_windows=r["n_windows"], later_gain_realized=rr.get("estimate"),
                         later_gain_realized_lo=rr.get("lo"), later_gain_realized_hi=rr.get("hi"),
                         attack_change=ac, attack_change_lo=ac_lo, attack_change_hi=ac_hi,
                         cost=-ac, cost_lo=-ac_hi, cost_hi=-ac_lo))
    tb = pd.DataFrame(rows)
    tables["later_gain"] = records(tb)
    N = T.get("e5_nets")
    if N is not None and len(N):
        m = N[N["pipeline"] == "measurement"]
        grp = m.groupby(["date", "seed", "arm"])["digest"].nunique()
        st = N[N["pipeline"] == "stored"].copy()
        st["alpha"] = st["alpha"].astype(float)
        ref = st[np.isclose(st["alpha"], 1.0)].set_index(["date", "seed", "arm"])["digest"]
        dist = st[~np.isclose(st["alpha"], 1.0)].copy()
        dist["differs"] = [d != ref.get((a, b, c)) for a, b, c, d in zip(dist["date"], dist["seed"], dist["arm"],
                                                                         dist["digest"])]
        hyp["H5a"] = dict(groups=int(grp.size), bit_identical_groups=int((grp == 1).sum()),
                          holds=bool((grp == 1).all()), stored_nets=int(len(dist)),
                          stored_nets_differing_from_alpha1=int(dist["differs"].sum()))
        mac.add("EfiveIdentical", hyp["H5a"]["holds"], source="H5a: measurement-only retraining bit-identical")
        mac.add("EfiveGroups", hyp["H5a"]["groups"], "{:d}", "date x seed x arm groups", "H5a groups checked")
        if len(dist):
            mac.add("EfiveDiffParams", 100 * float(dist["differs"].mean()), "{:.0f}", "percent of stored-report nets",
                    "stored-report retrained parameters differ from the alpha = 1 net")
            tables["param_distance"] = records(dist.groupby(["arm", "target", "alpha"])["l2_to_ref"].mean()
                                               .reset_index())
    C = T.get("e5_checks")
    if C is not None and len(C):
        c = C.iloc[0].to_dict()
        hyp["checks"] = _clean({k: c.get(k) for k in ("h5a_all_bit_identical", "measurement_equals_stored_ref_all",
                                                      "stored_log_equals_plan_all", "attack_ref_truthful_all",
                                                      "protected_attack_max_interval_gain", "jobs_done")})
    meas = tb[tb["pipeline"] == "measurement"] if len(tb) else tb
    if len(meas):
        hyp["H5a_later_gain_max_abs"] = float(meas[["later_gain", "lo", "hi"]].abs().max().max())
        mac.add("EfiveLaterMeasMax", hyp["H5a_later_gain_max_abs"], "{:.2g}", "credits per interval",
                "largest |later gain| or CI bound under measurement-only retraining")
    x = lw[(lw["arm"] == H5B_PRIMARY_ARM) & (lw["pipeline"] == "stored") & np.isclose(lw["alpha"], H5B_PRIMARY_ALPHA)]
    if len(x):
        pw = x.groupby(["date", "unit"])["d_utility_declared_target"].mean().reset_index()
        r = one_sample(pw, "d_utility_declared_target", n_boot=nb, mem=cfg.get("memberships_e5") or None)
        r.update(role="primary", arm=H5B_PRIMARY_ARM, alpha=H5B_PRIMARY_ALPHA, pipeline="stored",
                 targets="pooled (video and URLLC-like target, mean per evaluation window)",
                 reference="alpha = 1 arm of the same date and pipeline (D5)",
                 unit="credits per interval (distorting tenant, evaluation windows)", **h5b_decision(r))
        if cfg.get("split") == "dev":
            r["dev_estimation_half"] = power_inputs(pw.rename(columns={"d_utility_declared_target": "a"})
                                                    .assign(b=0.0), "H5b", n_boot=nb)
        hyp["H5b"] = r
        mac.ci("EfiveLater", r, "credits per interval",
               f"H5b: later gain, stored reports, alpha = {H5B_PRIMARY_ALPHA:g}, targets pooled", pct=False)
        _h5b_by_seed(later, nb, tables, hyp)
        _e5_cost_net(attack, pw, nb, mac, tables, hyp, notes)
    else:
        notes.append(f"H5b: no stored-report later rows at alpha = {H5B_PRIMARY_ALPHA:g} for {H5B_PRIMARY_ARM}")
    _e5_victims(later, nb, mac, tables, notes)
    WD = T.get("e5_wdiff")
    if WD is not None and len(WD):
        _e5_wdiff_summary(WD, mac, tables)
    else:
        notes.append("EfiveDiffMeas/EfiveDiffStored: no E5 episode tables (E5 without --keep-episodes); "
                     "EfiveDiffParams reports the stored-report parameter differences instead")
    _e5_extra(lw, nb, mac, tables, notes)
    return dict(macros=mac.d, tables=tables, hypotheses=hyp, notes=notes)


def _primary_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the H5b primary cell (arm, alpha) of an E5 pairs table (any pipeline)."""
    return df[(df["arm"] == H5B_PRIMARY_ARM) & np.isclose(df["alpha"].astype(float), H5B_PRIMARY_ALPHA)]


def _h5b_by_seed(later: pd.DataFrame, nb: int, tables: dict, hyp: dict) -> None:
    """Pre-freeze audit E5-8: the H5b primary cell per checkpoint seed (targets pooled per evaluation window; the
    registered analysis averages the seeds within a window first)."""
    x = _primary_cell(later[later["pipeline"] == "stored"])
    rows = []
    for seed, g in x.groupby("seed"):
        pw = g.groupby(["date", "unit", "target"])["d_utility_declared_target"].mean() \
            .groupby(["date", "unit"]).mean().reset_index()
        r = one_sample(pw, "d_utility_declared_target", n_boot=nb)
        rows.append(dict(seed=int(seed), estimate=r["estimate"], lo=r["lo"], hi=r["hi"], n_dates=r["n_dates"],
                         n_windows=r["n_windows"], dates_positive=r["dates_positive"],
                         dates_negative=r["dates_negative"]))
    tables["h5b_by_seed"] = records(pd.DataFrame(rows))
    if rows:
        e = np.array([r["estimate"] for r in rows], float)
        hyp["H5b"]["seed_spread"] = dict(seeds=len(rows), min=float(e.min()), max=float(e.max()),
                                         sd=float(e.std(ddof=1)) if len(e) > 1 else None,
                                         note="descriptive; the H5b estimate averages the seeds within a window")


def _e5_cost_net(attack: pd.DataFrame, later_pw: pd.DataFrame, nb: int, mac: Macros, tables: dict, hyp: dict,
                 notes: list) -> None:
    """Pre-freeze audit NUM-08: the attack-phase change of the target's declared utility in the H5b primary cell
    (EfiveAttackChange, signed; EfiveCost = -EfiveAttackChange, positive = cost), the net per date (later gain plus
    the attack change scaled by attack/evaluation windows; EfiveNet, positive = the distortion pays overall) and
    the repay length (EfiveRepay: evaluation periods of the later gain needed to recover the cost)."""
    col = "d_utility_declared_target"
    a = _primary_cell(attack)
    if len(a) == 0:
        notes.append("EfiveCost/EfiveNet: no attack-phase rows of the H5b primary cell")
        return
    apw = a.groupby(["date", "unit", "target", "seed"])[col].mean().groupby(["date", "unit", "target"]).mean() \
        .groupby(["date", "unit"]).mean().reset_index()
    ra = one_sample(apw, col, n_boot=nb)
    hyp["H5b"]["attack_change"] = {k: ra.get(k) for k in ("estimate", "lo", "hi", "n_dates", "n_windows",
                                                           "dates_positive", "dates_negative")}
    mac.ci("EfiveAttackChange", ra, "credits per interval",
           "H5b cell: change of the target's declared utility in the false-report windows (negative = cost)",
           pct=False)
    mac.ci_negated("EfiveCost", ra, "credits per interval (positive = cost)",
                   "H5b cell: cost of the distortion to the target in the false-report windows (minus "
                   "EfiveAttackChange)")
    n_att = apw.groupby("date")["unit"].nunique()
    n_lat = later_pw.groupby("date")["unit"].nunique()
    ld = later_pw.groupby("date")[col].mean()
    ad = apw.groupby("date")[col].mean()
    net = (ld + ad * (n_att / n_lat)).dropna()
    if len(net) == 0:
        return
    rn = boot(net.to_numpy(float), net.index.to_numpy(), n_boot=nb)
    rnd = dict(rn.as_dict(), unit="credits per evaluation interval", dates_positive=int((net > 0).sum()),
               dates_negative=int((net < 0).sum()),
               definition="per date: later gain + attack change x (attack windows / evaluation windows); date "
                          "bootstrap (one value per date)")
    hyp["H5b"]["net"] = rnd
    mac.ci("EfiveNet", rnd, "credits per evaluation interval (positive = the distortion pays overall)",
           "H5b cell: later gain net of the false-report cost", pct=False)
    later, cost = _f(hyp["H5b"].get("estimate")), -_f(ra.get("estimate"))
    ratio = float(n_att.mean() / n_lat.mean())
    repay = cost * ratio / later if later > 0 and cost > 0 else None
    tables["h5b_cost_net"] = [_clean(dict(later_gain=later, attack_change=-cost, cost=cost, net=rn.estimate,
                                          net_lo=rn.lo, net_hi=rn.hi, attack_windows_per_date=float(n_att.mean()),
                                          later_windows_per_date=float(n_lat.mean()), repay_periods=repay,
                                          repay_note="evaluation periods (of the evaluation windows of one date) "
                                                     "of the later gain needed to recover the cost; None when the "
                                                     "later gain is not positive or there is no cost"))]
    if repay is not None:
        mac.add("EfiveRepay", repay, "{:.1f}", "evaluation periods (EfiveEvalWindows windows each)",
                "H5b cell: deployment length after which the later gain repays the distortion cost")
        mac.add("EfiveRepayWindows", repay * float(n_lat.mean()), "{:.0f}", "evaluation windows",
                "H5b cell: evaluation windows of later gain needed to repay the distortion cost")
    else:
        notes.append("EfiveRepay: the later gain does not repay the cost (later gain <= 0 or no cost)")


E5_VICTIM_METRICS = (("d_on_time_victims", "EfiveVictimOnTime", "Mbit per interval (honest tenants, summed)"),
                     ("d_miss_pct_victims", "EfiveVictimMiss", "percentage points"),
                     ("d_value_victims", "EfiveVictimValue", "credits per interval"))


def _e5_victims(later: pd.DataFrame, nb: int, mac: Macros, tables: dict, notes: list) -> None:
    """Pre-freeze audit NUM-07: the honest tenants' outcome in the evaluation windows (later phase; distorted run
    minus the alpha = 1 run of the same pipeline; D4 victim measures), per (arm, pipeline, alpha) with the targets
    pooled per window (``target`` = 'pooled') and per target; seeds averaged within a window; date bootstrap."""
    cols = [m for m, _, _ in E5_VICTIM_METRICS]
    if not all(c in later.columns for c in cols):
        notes.append("E5 victims: the pairs lack the victim columns")
        return
    rows = []
    for (arm, pipe, alpha), g in later.groupby(["arm", "pipeline", "alpha"]):
        wt = g.groupby(["date", "unit", "target", "seed"])[cols].mean().groupby(["date", "unit", "target"]).mean()
        for tgt, wv in [("pooled", wt.groupby(["date", "unit"]).mean().reset_index())] + \
                [(t, h.reset_index()) for t, h in wt.groupby(level="target")]:
            row = dict(arm=arm, label=label(arm), pipeline=pipe, alpha=float(alpha), target=tgt)
            for m in cols:
                try:
                    r = one_sample(wv, m, n_boot=nb)
                except MissingInput:
                    continue
                row.update({m: r["estimate"], f"{m}_lo": r["lo"], f"{m}_hi": r["hi"], "n_dates": r["n_dates"],
                            "n_windows": r["n_windows"]})
                if (arm == H5B_PRIMARY_ARM and pipe == "stored" and np.isclose(float(alpha), H5B_PRIMARY_ALPHA)
                        and tgt == "pooled"):
                    stem, unit = {mm: (st, u) for mm, st, u in E5_VICTIM_METRICS}[m]
                    mac.ci(stem, r, unit, f"H5b cell, evaluation windows: honest tenants' {m} (distorted minus "
                                          "alpha = 1 run; targets pooled)", pct=False)
                    if m == "d_on_time_victims":
                        mac.ci_negated("EfiveVictimLoss", r,
                                       "Mbit per interval lost by the honest tenants (positive = loss)",
                                       "H5b cell, evaluation windows: honest on-time Mbit lost "
                                       "(minus EfiveVictimOnTime)")
            rows.append(row)
    tables["later_victims"] = records(pd.DataFrame(rows))


def _e5_wdiff_summary(WD: pd.DataFrame, mac: Macros, tables: dict) -> None:
    """EfiveDiffMeas / EfiveDiffStored: percent of the scored later intervals in which the priority vector of the
    distorted run differs from the alpha = 1 run (Protected arm, every distortion factor; pre-freeze audit NUM-07)."""
    g = WD.groupby(["arm", "pipeline", "alpha"])[["intervals", "differ"]].sum().reset_index()
    g["pct_differ"] = 100 * g["differ"] / g["intervals"].where(g["intervals"] > 0)
    tables["later_w_differs"] = records(g)
    p = WD[WD["arm"] == H5B_PRIMARY_ARM].groupby("pipeline")[["intervals", "differ"]].sum()
    for pipe, stem in (("measurement", "EfiveDiffMeas"), ("stored", "EfiveDiffStored")):
        if pipe in p.index and p.loc[pipe, "intervals"] > 0:
            mac.add(stem, 100 * float(p.loc[pipe, "differ"] / p.loc[pipe, "intervals"]), "{:.1f}",
                    "percent of later intervals", f"priority vector differs from the alpha = 1 run, {pipe} pipeline, "
                                                  f"{label(H5B_PRIMARY_ARM)}, every distortion factor")


def h5b_decision(r: Mapping, sign: int | None = None) -> dict:
    """H5b rule (round-3 review): the CI must exclude 0 on the side of the pre-registered sign ``H5B_SIGN``
    (``sign`` overrides it, for tests). A CI excluding 0 on the other side is reported as ``contrary`` to the
    pre-registered sign, never as support. Without a pre-registered sign (before the freeze) ``rule_met`` is None
    and ``dev_sign_suggestion`` gives the sign of the estimate (DEV pilot)."""
    sign = H5B_SIGN if sign is None else sign
    lo, hi, est = r.get("lo"), r.get("hi"), _f(r.get("estimate"))
    out = dict(preregistered_sign=sign, test="two-sided p (Holm); support needs the pre-registered side",
               verdict="holm.json (rule, sign and Holm rejection)",
               dev_sign_suggestion=int(np.sign(est)) if math.isfinite(est) and est != 0 else None)
    if sign not in (1, -1):
        out.update(rule_met=None, contrary=None, sign_status="not pre-registered yet (fixed from the DEV pilot)")
        return out
    if not _ok(lo, hi):
        out.update(rule_met=None, contrary=None, sign_status=f"no inference (fewer than {MIN_DATES} dates)")
        return out
    met = (sign > 0 and _f(lo) > 0) or (sign < 0 and _f(hi) < 0)
    contrary = (sign > 0 and _f(hi) < 0) or (sign < 0 and _f(lo) > 0)
    out.update(rule_met=bool(met), contrary=bool(contrary),
               sign_status=("contrary to the pre-registered sign" if contrary else
                            "on the pre-registered side" if met else "CI includes 0"))
    return out


def power_inputs(pairs: pd.DataFrame, hid: str, *, n_boot: int, n_sim: int | None = None,
                 sim_boot: int | None = None) -> dict:
    """D9 power analysis of a primary from the DEV ESTIMATION half only (``registry.dev_role``; round-3 review: the
    selection half chose the checkpoints, cells and static vector, so its estimates carry a winner's curse).

    ``pairs`` (date, unit, a, b) is the primary's paired per-window table. The effect is the distance from the null
    in the pre-registered direction, at the point estimate and at the CI bound nearest the null (PREREG §9.4):
    H1a a - b; H2c -(a - b); H4a the 1 % margin (credits, of the mean of b) minus a - b; H5b ``H5B_SIGN`` (a - b)
    with the bound on the pre-registered side (pre-freeze audit PV-5; None when it is not on that side). Power is
    that of the registered test (pre-freeze audit PV-4): the two-stage date bootstrap, the rule and Holm's first
    step at alpha / 4 (H5b: two-sided, alpha / 8 per side).

    (a) analytic: ``need(delta) = max(MIN_DATES, ceil(((r z_{1-a} + z_{0.8}) sd / delta)^2))`` with sd the SD of
        the date means and r = SE_boot / (sd / sqrt(G_est)) the widening of the two-stage bootstrap over a z-test
        on date means (``BOOT_SEED``, same replicates as the registered analysis);
    (b) simulation (the 16/24 decision): the estimation-half dates (with their windows; residuals centred on the
        estimate, plus the effect) are resampled to G in ``POWER_DATES``; each simulated experiment is analysed with
        ``S.cluster_bootstrap_multi`` (``sim_boot`` replicates) and the rule + Holm alpha / 4; the rejection share
        is the power. ``dates_decision`` = 16 when the share at G = 16 and the CI-bound effect reaches 80 %, else 24.

    H1a (pre-freeze audit PV-1): the rule also needs the effect >= ``H1A_MIN_PCT`` % of b. Its percentage, CI and
    the date SD of the linearised percentage are reported with ``pct_rule_status``; when the DEV percentage is below
    the threshold the percentage part is unattainable (``pct_rule_dates_needed`` None) whatever the number of dates.
    The other fields are the power of the CI/Holm part only."""
    import registry as R
    from scipy.stats import norm
    p = pairs.dropna(subset=["a", "b"]).copy()
    p["role"] = p["date"].map(lambda d: R.dev_role(str(d)))
    q = p[p["role"] == "estimate"].reset_index(drop=True)
    if len(q) == 0:
        return dict(status="no estimation-half dates", dates=0)
    alt = {"H1a": "greater", "H2c": "less", "H4a": "greater", "H5b": "two-sided"}[hid]
    bmean = float(q["b"].mean())
    use_pct = hid in ("H1a", "H4a") and bmean != 0
    r = contrast(q[["date", "unit", "a", "b"]], n_boot=n_boot, alternative=alt, pct=use_pct)
    d, b = (q["a"] - q["b"]).to_numpy(float), q["b"].to_numpy(float)
    dates = q["date"].to_numpy()
    dm_tab = S.date_means(d, dates)
    dm = dm_tab["mean"].to_numpy(float)
    G_est = len(dm)
    sd = float(np.std(dm, ddof=1)) if G_est > 1 else float("nan")
    rd = boot(d, dates, n_boot=n_boot)
    se_boot = float(np.std(rd.reps, ddof=1)) if rd.reps.size > 1 else float("nan")
    r_infl = se_boot / (sd / math.sqrt(G_est)) if _ok(se_boot, sd) and sd > 0 else float("nan")
    est, lo, hi = _f(r["estimate"]), _f(r.get("lo")), _f(r.get("hi"))
    sign = None
    if hid == "H1a":
        point, cons = est, lo
    elif hid == "H2c":
        point, cons = -est, -hi
    elif hid == "H4a":
        m = H4A_MARGIN_PCT / 100.0 * bmean
        point, cons = m - est, m - _f(r.get("upper_one_sided"))
    else:
        sign = H5B_SIGN if H5B_SIGN in (1, -1) else (int(np.sign(est)) if est else 1)
        point = sign * est
        cons = sign * (lo if sign > 0 else hi) if math.isfinite(lo) and math.isfinite(hi) else float("nan")
        if math.isfinite(cons) and cons <= 0:
            cons = float("nan")                        # the CI reaches the null or the other side
    a = POWER_ALPHA / (2 if hid == "H5b" else 1)
    za, zb = float(norm.ppf(1 - a)), float(norm.ppf(POWER))
    rr = r_infl if math.isfinite(r_infl) else 1.0

    def need(delta):
        if not (math.isfinite(delta) and math.isfinite(sd)) or delta <= 0:
            return None
        return max(MIN_DATES, int(math.ceil(((rr * za + zb) * sd / delta) ** 2)))  # never below the inference minimum
    wpd = q.groupby("date")["unit"].nunique()
    out = dict(status="ok", half="estimation", dates=int(G_est), windows=int(len(q)),
               windows_per_date=float(wpd.mean()), estimate=est, lo=r.get("lo"), hi=r.get("hi"),
               sd_date_means=sd, se_boot=se_boot, boot_inflation_r=r_infl, alpha_one_sided=a, power=POWER,
               effect_point=point, effect_conservative=cons if math.isfinite(cons) else None,
               dates_needed_point=need(point), dates_needed_conservative=need(cons),
               method=("(a) analytic: G = ((r z_{1-a} + z_{0.8}) sd / delta)^2, r = SE of the two-stage bootstrap / "
                       "(sd / sqrt(G_est)); (b) simulation: estimation-half dates resampled to G with the registered "
                       "two-stage bootstrap, rule and Holm alpha/4"),
               note=("TEST dates for 80 % power of the registered test (two-stage date bootstrap, rule, Holm alpha/4"
                     + ("; two-sided, alpha/8 per side, pre-registered sign" if hid == "H5b" else "")
                     + f"); DEV has {wpd.mean():g} windows per date, TEST 8 (E1-E4) or the same 4 (E5)"))
    if hid == "H5b":
        out["preregistered_sign"] = sign
    if hid == "H1a":
        pct_rule = _h1a_pct_part(d, b, dates, r, zb)
        out.update(pct_rule)
        dn = out["dates_needed_conservative"]
        out["dates_needed_rule"] = (max(dn, pct_rule["pct_rule_dates_needed"])
                                    if dn is not None and pct_rule["pct_rule_dates_needed"] is not None else None)
        out["power_scope"] = "dates_needed_* and the simulation's 'ci_holm' share: power of the CI/Holm test only"
    deltas = {"point": point, "conservative": cons}
    out["simulation"] = _power_sim(hid, d, b, dates, deltas, n_sim=POWER_SIMS if n_sim is None else n_sim,
                                   sim_boot=min(n_boot, POWER_SIM_BOOT) if sim_boot is None else sim_boot,
                                   sign=sign, est=est)
    by = out["simulation"].get("by_dates") or {}

    no_bound = not _ok(r.get("lo"), r.get("hi"))     # fewer than MIN_DATES estimation dates: no CI, no decision

    def share(G):                     # a CI bound on the wrong side of the null has power <= alpha: counted as 0
        c = (by.get(str(G)) or {}).get("conservative")
        return c["ci_holm"] if c else (None if no_bound or not by else 0.0)
    s_lo, s_hi = share(min(POWER_DATES)), share(max(POWER_DATES))
    out["dates_decision"] = None if s_lo is None else (min(POWER_DATES) if s_lo >= POWER else max(POWER_DATES))
    out["underpowered_at_max_dates"] = None if s_hi is None else bool(s_hi < POWER)
    out["decision_basis"] = (f"simulated power at G = {min(POWER_DATES)} and the CI bound nearest the null "
                             f"(PREREG §9.4); {max(POWER_DATES)} when below {POWER:.0%}"
                             + ("; H1a: CI/Holm part only (see pct_rule_status)" if hid == "H1a" else ""))
    return out


def _h1a_pct_part(d: np.ndarray, b: np.ndarray, dates, r: Mapping, zb: float) -> dict:
    """H1a's percentage part (pre-freeze audit PV-1): the effect must reach ``H1A_MIN_PCT`` % of b. The date SD of
    the linearised percentage 100 (d_j - (dbar / bbar) b_j) / bbar (date means d_j, b_j) and the dates needed for
    P(pct_hat >= H1A_MIN_PCT) >= 80 %; unattainable (None) when the DEV estimation-half percentage is below the
    threshold."""
    bbar, dbar = float(np.mean(b)), float(np.mean(d))
    out = dict(pct=r.get("pct"), pct_lo=r.get("pct_lo"), pct_hi=r.get("pct_hi"), min_effect_pct=H1A_MIN_PCT,
               mei_pct=H1A_MEI_PCT, sd_date_pct=None, pct_rule_dates_needed=None)
    if bbar == 0 or r.get("pct") is None:
        out["pct_rule_status"] = "no percentage (zero denominator)"
        return out
    dmj = S.date_means(d, dates)["mean"].to_numpy(float)
    bmj = S.date_means(b, dates)["mean"].to_numpy(float)
    lin = 100.0 * (dmj - (dbar / bbar) * bmj) / bbar
    sd_pct = float(np.std(lin, ddof=1)) if len(lin) > 1 else float("nan")
    out["sd_date_pct"] = sd_pct
    pct = _f(r["pct"])
    if pct < H1A_MIN_PCT:
        out["pct_rule_status"] = (f"unattainable: DEV estimation-half pct {pct:.3f} % < {H1A_MIN_PCT:g} % "
                                  "(the point estimate must reach the threshold; more dates do not help)")
        return out
    gap = pct - H1A_MIN_PCT            # P(pct_hat >= threshold) >= 80 %: gap >= z_0.8 sd_pct / sqrt(G)
    if gap <= 0:
        out["pct_rule_status"] = "borderline: the DEV percentage equals the threshold (power 50 % at any G)"
    else:
        out["pct_rule_dates_needed"] = (MIN_DATES if not (math.isfinite(sd_pct) and sd_pct > 0)
                                        else max(MIN_DATES, int(math.ceil((zb * sd_pct / gap) ** 2))))
        out["pct_rule_status"] = "attainable"
    return out


def _power_decide(hid: str, dv: np.ndarray, bv: np.ndarray, lab: np.ndarray, n_boot: int, sign: int | None) -> dict:
    """The registered decision on one simulated experiment: {'ci_holm': rule and Holm alpha/4, 'rule': full rule}."""
    if hid in ("H1a", "H4a") and float(np.mean(bv)) != 0:
        V = np.column_stack([dv, bv])
        st = {"diff": lambda m: m[..., 0], "pct": lambda m: 100.0 * m[..., 0] / m[..., 1]}
        res = S.cluster_bootstrap_multi(V, lab, st, n_boot=n_boot, level=LEVEL, seed=BOOT_SEED)
        if hid == "H1a":
            ok = bool(res["diff"].lo > 0 and res["diff"].p_value(0.0, "greater") <= POWER_ALPHA)
            return dict(ci_holm=ok, rule=ok and res["pct"].estimate >= H1A_MIN_PCT)
        ok = bool(res["pct"].upper(LEVEL) < H4A_MARGIN_PCT
                  and res["pct"].p_value(H4A_MARGIN_PCT, "less") <= POWER_ALPHA)
        return dict(ci_holm=ok, rule=ok)
    res = S.cluster_bootstrap(dv, lab, n_boot=n_boot, level=LEVEL, seed=BOOT_SEED)
    if hid in ("H1a", "H4a"):                          # zero denominator: no percentage, so no full rule
        ok = bool(hid == "H1a" and res.lo > 0 and res.p_value(0.0, "greater") <= POWER_ALPHA)
        return dict(ci_holm=ok, rule=False)
    if hid == "H2c":
        ok = bool(res.hi < 0 and res.p_value(0.0, "less") <= POWER_ALPHA)
    else:
        side = (res.lo > 0) if (sign or 1) > 0 else (res.hi < 0)
        ok = bool(side and res.p_value(0.0, "two-sided") <= POWER_ALPHA)
    return dict(ci_holm=ok, rule=ok)


def _power_sim(hid: str, d: np.ndarray, b: np.ndarray, dates, deltas: Mapping, *, n_sim: int, sim_boot: int,
               sign: int | None, est: float) -> dict:
    """Simulated power of the registered test (pre-freeze audit PV-4 (b)): resample the estimation-half dates with
    their windows to G dates, centre the per-window differences on the estimate and add the effect (a - b scale),
    analyse each simulated experiment like the TEST analysis and count rejections."""
    if n_sim < 1 or not math.isfinite(est):
        return dict(status="not simulated")
    uniq, inv = np.unique(np.asarray(dates), return_inverse=True)
    rows_of = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    resid = d - d.mean()
    # the effect in the a - b scale: H1a +delta, H2c -delta, H4a margin - delta (the gap), H5b sign x delta
    m = H4A_MARGIN_PCT / 100.0 * float(np.mean(b))
    to_diff = {"H1a": lambda x: x, "H2c": lambda x: -x, "H4a": lambda x: m - x,
               "H5b": lambda x: (sign or 1) * x}[hid]
    rng = np.random.default_rng(POWER_SIM_SEED)
    by = {}
    for G in POWER_DATES:
        cell = {}
        for name, delta in deltas.items():
            if delta is None or not math.isfinite(_f(delta)) or delta <= 0:
                cell[name] = None
                continue
            shift = to_diff(float(delta))
            k_ci = k_rule = 0
            for _ in range(n_sim):
                pick = rng.integers(len(uniq), size=G)
                idx = np.concatenate([rows_of[g] for g in pick])
                lab = np.repeat(np.arange(G), [len(rows_of[g]) for g in pick])
                dec = _power_decide(hid, resid[idx] + shift, b[idx], lab, sim_boot, sign)
                k_ci += dec["ci_holm"]
                k_rule += dec["rule"]
            cell[name] = dict(effect=float(delta), ci_holm=k_ci / n_sim, rule=k_rule / n_sim)
        by[str(G)] = cell
    return dict(status="ok", n_sim=n_sim, boot_reps=sim_boot, alpha=POWER_ALPHA, seed=POWER_SIM_SEED, by_dates=by,
                note="rejection share of the rule and Holm's first step (alpha/4) among simulated experiments")


# ---------------------------------------------------------------------------------------------------------- E6
N_WORD = {6: "Six", 9: "Nine"}


def analyze_e6(T: dict, cfg: dict) -> dict:
    nb, mac, tables, notes, hyp = cfg["n_boot"], Macros("e6"), {}, [], {}
    TM = T.get("timing")
    if TM is not None and len(TM):
        x = TM[TM["scored"].astype(bool)] if "scored" in TM.columns else TM
        rows = []
        for (n, arm), g in x.groupby(["n", "arm"]):
            t = g["t_price_ms"].dropna()
            if len(t) == 0:
                continue
            rows.append(dict(n=int(n), arm=arm, label=label(arm), intervals=len(t), median_ms=float(t.median()),
                             p95_ms=float(t.quantile(0.95)), max_ms=float(t.max()),
                             fixed_p95_ms=float(g["t_fixed_ms"].quantile(0.95))
                             if "t_fixed_ms" in g and g["t_fixed_ms"].notna().any() else None))
        tt = pd.DataFrame(rows)
        tables["pricing_time"] = records(tt)
        p9 = tt[(tt["n"] == 9) & (tt["arm"] == "protected")] if len(tt) else tt
        if len(p9):
            v = float(p9["p95_ms"].iloc[0])
            hyp["H6a_time"] = dict(p95_ms=v, limit_ms=H6A_TIME_MS, holds=bool(v <= H6A_TIME_MS))
            mac.add("EsixTime", v, "{:.1f}", "ms", "p95 full-price time per interval at n = 9, Protected, one core")
            mac.add("EsixTimeMedian", float(p9["median_ms"].iloc[0]), "{:.1f}", "ms", "median pricing time at n = 9")
        # pre-freeze audit NUM-17: one macro per n (the paper states the time 'at 6 and 9 tenants')
        for n_, word in N_WORD.items():
            pn = tt[(tt["n"] == n_) & (tt["arm"] == "protected")] if len(tt) else tt
            if len(pn):
                mac.add("EsixTime" + word, float(pn["p95_ms"].iloc[0]), "{:.1f}", "ms",
                        f"p95 full-price time per interval at n = {n_}, Protected, one core")
                mac.add("EsixTimeMedian" + word, float(pn["median_ms"].iloc[0]), "{:.1f}", "ms",
                        f"median full-price time per interval at n = {n_}, Protected, one core")
    else:
        notes.append("H6a time: skipped (no timing rows)")
    BR = T.get("br")
    if BR is not None and len(BR) and (BR["exp"] == "e6").any():
        r = _run(notes, "H6a gain", h3a, BR[BR["exp"] == "e6"])
        if r:
            hyp["H6a_gain"], by = r
            tables["max_gain_by_n"] = records(by.groupby("n").agg(max_gain=("max_gain", "max"),
                                                                  max_cert=("max_cert", "max"),
                                                                  max_err=("max_err", "max"),
                                                                  min_eps=("min_eps", "min"),
                                                                  violations=("violations", "sum"),
                                                                  tenant_intervals=("tenant_intervals", "sum"))
                                              .reset_index())
            mac.add("EsixMaxGain", hyp["H6a_gain"]["max_gain"], "{:.2g}", "credits per interval",
                    "largest full-price gain at n = 6 and 9")
            mac.add("EsixViol", hyp["H6a_gain"]["violations"], "{:d}", "tenant-intervals", "H6a violations")
    W = T.get("windows")
    if W is not None and len(W):
        cost = []
        Wt = W[W["policy"] == "truthful"]
        for n in N_SWEEP:
            exp = "e6" if ((Wt["exp"] == "e6") & (Wt["n"] == n)).any() else "e1"
            res = _run(notes, f"cost n={n}", arm_contrast, Wt, dict(MAIN, n=n), "unprotected", "protected",
                       n_boot=nb, alternative="greater", exp=exp, denominator="Protected welfare on the same windows",
                       pct_null=H6B_MARGIN_PCT if n == 9 else None, pct_alternative="less")
            if res:
                r = res[0]
                cost.append(dict(n=n, source=exp, estimate=r["estimate"], lo=r["lo"], hi=r["hi"], pct=r.get("pct"),
                                 pct_lo=r.get("pct_lo"), pct_hi=r.get("pct_hi"), n_dates=r["n_dates"],
                                 n_windows=r["n_windows"], mean_protected=r["mean_b"]))
                if n == 9:
                    hyp["H6b"] = dict(r, margin_pct=H6B_MARGIN_PCT, holds=bool(_f(r.get("pct")) <= H6B_MARGIN_PCT))
                    mac.ci("EsixGap", r, "credits per interval", "W(Unconstrained net) - W(Protected) at n = 9")
        tables["cost_by_n"] = cost
    R = T.get("e6_runs")
    if R is not None and len(R) and "idle_prb" in R.columns:
        tables["rounding_loss"] = records(R.groupby(["n", "arm"])["idle_prb"].mean().reset_index())
    return dict(macros=mac.d, tables=tables, hypotheses=hyp, notes=notes)


# ---------------------------------------------------------------------------------------------------------- E7
#: NR case -> spec/controllers.yaml key (colour, marker and label; round-3 review: E7 runs the truthful
#: unconstrained case with the fixed-priority price, i.e. 'unprotected', and the best-response case is its own variant)
NR_CASES = {"protected_truthful": "protected", "unconstrained_truthful": "unprotected",
            "unconstrained_br": "unprotected_br"}


def analyze_e7(T: dict, cfg: dict) -> dict:
    mac, tables, notes, hyp = Macros("e7"), {}, [], {}
    NR = _need(T, "nr").copy()
    NR["drop_pct"] = 100 * NR["nr_rlc_drop_mbit"] / NR["nr_pdcp_sdu_mbit"].where(NR["nr_pdcp_sdu_mbit"] > 0)
    NR["window"] = NR["date"].astype(str) + "_o" + NR["offset"].astype(int).astype(str)
    keep = ["window", "arm", "tenant", "cls", "br_tenant", "nr_delivered_pdb_mbit", "nr_delivered_mbit",
            "nr_delay_p95_ms_median", "drop_pct", "fluid_raca_on_time", "fluid_cal_on_time", "fluid_nr_on_time",
            "corr_delivered_fluid_cal", "corr_delivered_fluid_nr", "offered_load", "offered_load_scored",
            "binding_share", "adjusted_intervals"]
    tables["per_case"] = records(NR[[c for c in keep if c in NR.columns]])
    agg = NR.groupby(["arm", "cls"]).agg(
        delivered_pdb_mbps=("nr_delivered_pdb_mbit", "mean"), delivered_lo=("nr_delivered_pdb_mbit", "min"),
        delivered_hi=("nr_delivered_pdb_mbit", "max"), delay_p95_ms=("nr_delay_p95_ms_median", "mean"),
        delay_lo=("nr_delay_p95_ms_median", "min"), delay_hi=("nr_delay_p95_ms_median", "max"),
        drop_pct=("drop_pct", "mean"), drop_lo=("drop_pct", "min"), drop_hi=("drop_pct", "max"),
        windows=("window", "nunique")).reset_index()
    agg["key"] = agg["arm"].map(lambda a: NR_CASES.get(a, a))
    agg["label"] = agg["key"].map(label)
    tables["per_slice_mean"] = records(agg)
    rows = []
    for w, g in NR.groupby("window"):
        if not set(NR_CASES) <= set(g["arm"]):
            continue
        br = int(g[g["arm"] == "unconstrained_br"]["br_tenant"].iloc[0])
        h = g[g["tenant"] != br].groupby("arm")[["nr_delivered_pdb_mbit", "nr_delivered_mbit", "fluid_raca_on_time"]] \
            .sum()
        e = h.loc["unconstrained_br"] - h.loc["unconstrained_truthful"]
        nr, fl = float(e["nr_delivered_pdb_mbit"]), float(e["fluid_raca_on_time"])
        rows.append(dict(window=w, deviator=br, nr_effect=nr, fluid_effect=fl,
                         sign_agrees=bool(nr * fl > 0),          # the same NONZERO sign (round-3 review)
                         ratio=nr / fl if fl else None, within_2x=bool(fl and 0.5 <= nr / fl <= 2.0),
                         protected_minus_manipulated=float(h.loc["protected_truthful", "nr_delivered_mbit"]
                                                           - h.loc["unconstrained_br", "nr_delivered_mbit"])))
    vt = pd.DataFrame(rows)
    tables["victims"] = records(vt)
    if len(vt):
        need = math.ceil(H7A_SHARE * len(vt))
        sc, mc = bool(vt["sign_agrees"].sum() >= need), bool(vt["within_2x"].sum() >= need)
        hyp["H7a"] = dict(windows=len(vt), sign_agrees=int(vt["sign_agrees"].sum()),
                          within_2x=int(vt["within_2x"].sum()), sign_criterion=sc, magnitude_criterion=mc,
                          holds=bool(sc and mc),
                          rule=f"same nonzero sign in >= {H7A_SHARE:g} of windows AND NR effect within 2x of the "
                               f"fluid prediction in >= {H7A_SHARE:g} of windows (PREREG_DRAFT §6 H7a)")
        hyp["H7b"] = dict(windows=len(vt), windows_holding=int((vt["protected_minus_manipulated"] > 0).sum()),
                          mean=float(vt["protected_minus_manipulated"].mean()))
        mac.add("EsevenVictim", -float(vt["nr_effect"].mean()), "{:.2f}", "Mbit/s lost (positive = loss)",
                "honest slices' delivered-within-PDB traffic LOST under the manipulated Unconstrained net (NR; "
                "minus EsevenVictimChange)")
        # pre-freeze audit NUM-17: the signed change, the number of windows, the count wording
        mac.add("EsevenVictimChange", float(vt["nr_effect"].mean()), "{:.2f}", "Mbit/s (negative = loss)",
                "change of the honest slices' delivered-within-PDB traffic when one tenant best-responds (NR)")
        mac.add("EsevenWindows", len(vt), "{:d}", "windows", "E7 windows with all three cases (H7a)")
        mac.add("EsevenSign", hyp["H7a"]["sign_agrees"], "{:d}", "windows (of EsevenWindows)",
                "windows in which the NR and fluid victim effects have the same nonzero sign")
        mac.add("EsevenSize", hyp["H7a"]["within_2x"], "{:d}", "windows (of EsevenWindows)",
                "windows in which the NR victim effect is within 2x of the fluid prediction")
        _e7_extra(NR, vt, mac, tables)
    else:
        notes.append("H7a/H7b: skipped (no window with all three cases)")
    if "corr_delivered_fluid_cal" in NR.columns and NR["corr_delivered_fluid_cal"].notna().any():
        c = NR["corr_delivered_fluid_cal"].dropna()
        hyp["H7c"] = dict(variant="fluid_cal (calibration-stage NR rate per class, out of sample)", min=float(c.min()),
                          median=float(c.median()), threshold=H7C_CORR, share_at_least=float((c >= H7C_CORR).mean()),
                          holds=bool(c.min() >= H7C_CORR))
        mac.add("EsevenCorr", float(c.median()), "{:.2f}", "correlation",
                "per-interval delivered traffic, NR vs calibrated fluid (out-of-sample calibration-stage rate; "
                "median over cases and slices)")
        mac.add("EsevenCorrMin", float(c.min()), "{:.2f}", "correlation",
                "smallest per-interval delivered-traffic correlation, NR vs calibrated fluid (the H7c rule)")
    else:
        notes.append("H7c: no out-of-sample calibrated-fluid correlation (fluid_cal needs the NR calibration stage)")
    if "corr_delivered_fluid_nr" in NR.columns and NR["corr_delivered_fluid_nr"].notna().any():
        c = NR["corr_delivered_fluid_nr"].dropna()
        hyp["H7c_in_sample_decomposition"] = dict(
            min=float(c.min()), median=float(c.median()),
            note="fluid_nr uses the same run's measured per-interval rates: a fitted decomposition, not a test of H7c")
    return dict(macros=mac.d, tables=tables, hypotheses=hyp, notes=notes)


# ================================================================================================ descriptive additions
# Pre-freeze descriptive result macros (orchestrator, 2026-10-20 evening; paper evidence check). Every quantity here is
# DESCRIPTIVE: no hypothesis, decision rule, Holm member, existing macro or existing table is changed (the one rule
# change of that batch, H3a's certified-gain rule, is in :func:`h3a`). CIs are the module's two-stage date bootstrap
# (``boot`` / ``S.cluster_bootstrap_multi``: dates, then windows within dates; ``BOOT_SEED``; percentile 95 %; no CI
# below ``MIN_DATES`` dates); ratio statistics are ratios of the resampled window means (= ratio of sums), computed
# inside one bootstrap with shared resamples. Learned arms: training seeds averaged within a window first.
#: relative tolerance of 'the best response reports the ceiling C_i' (z* >= C_i (1 - tol)); the best-response grid
#: ends exactly at C_i (DEV unc_fixed: the share is the same for tolerances 0 ... 1e-6)
CEILING_RTOL = 1e-9
#: (column, statistic) of the E2 PRB-transfer table (per window: deviator roles averaged, as H2c)
E2_TRANSFER_COLS = ("prb_moved", "victim_loss_mbit", "victim_truthful_mbit", "dev_gain_mbit")


def _multi_ci(V: np.ndarray, dates, stats: Mapping[str, Callable], n_boot: int) -> dict:
    """{name: dict(estimate, lo, hi)} of several statistics of the same two-stage resamples (no CI below
    ``MIN_DATES`` dates: the point estimate of the window means only)."""
    dates = np.asarray(dates)
    with np.errstate(divide="ignore", invalid="ignore"):
        if len(np.unique(dates)) >= MIN_DATES:
            res = S.cluster_bootstrap_multi(V, dates, stats, n_boot=n_boot, level=LEVEL, seed=BOOT_SEED)
            return {k: dict(estimate=b.estimate, lo=b.lo, hi=b.hi) for k, b in res.items()}
        m = np.asarray(V, float).mean(0)[None, :]
        return {k: dict(estimate=float(f(m)[0]), lo=None, hi=None) for k, f in stats.items()}


def _class_unit_values(W: pd.DataFrame, cell: Mapping, arm: str, cls: str, metrics: Sequence[str],
                       exp: str = "e1") -> pd.DataFrame:
    """Per window (date, unit): the sum over the tenants of class ``cls`` of each metric (truthful runs of ``exp``),
    training seeds averaged within the window (the per-class analogue of :func:`unit_values`)."""
    m = (_cell_mask(W, cell) & (W["arm"] == arm) & (W["policy"] == "truthful") & (W["tenant"] >= 0)
         & (W["cls"] == cls) & (W["exp"] == exp))
    if arm == "protected":
        m &= np.isclose(W["kappa_scale"].astype(float).fillna(1.0), 1.0)
    x = W[m]
    if len(x) == 0:
        return pd.DataFrame(columns=["date", "unit", *metrics])
    return x.groupby(["date", "unit", "seed"])[list(metrics)].sum(min_count=1).groupby(["date", "unit"]).mean() \
        .reset_index()


def _pairs_of(a: pd.DataFrame, b: pd.DataFrame, col: str) -> pd.DataFrame:
    j = a.merge(b, on=["date", "unit"], suffixes=("_a", "_b"))
    return pd.DataFrame(dict(date=j["date"], unit=j["unit"], a=j[f"{col}_a"].astype(float),
                             b=j[f"{col}_b"].astype(float)))


_CONTRAST_KEYS = ("estimate", "lo", "hi", "pct", "pct_lo", "pct_hi", "mean_a", "mean_b", "n_dates", "n_windows",
                  "dates_positive", "dates_negative")


def _e1_extra(W: pd.DataFrame, nb: int, mac: Macros, tables: dict, hyp: dict, notes: list) -> None:
    """E1 descriptive additions (default configuration, truthful runs, seeds averaged per window):
    (1) the per-class value of reports, on-time value (realized_value = pi_i x on-time Mbit) of the class's tenants
    under Protected minus under the report-free controller (EoneGainVideo/Urllc/Mmtc), and each class's share of the
    total value of reports (EoneGainVideoShare = 100 x video gain / EoneGain, ratio of means in one bootstrap);
    the kappa = 0 contrasts EoneKappaZero = W(Protected) - W(kappa0), EoneKappaZeroVsFree = W(kappa0) - W(No reports);
    (9) EoneLoadSlope (the D3 slope of the H1a difference on offered load, from H1a's load_subgroups) and
    EoneGainRhoHigh (the H1a contrast at rho = 1.1, configuration rho1.1); (10) EoneMissDiffVideo/Urllc/Mmtc: the
    change of the class's deadline-miss share 100 dropped / (on-time + dropped) (seed-averaged per-interval Mbit of
    the window), Protected minus No reports, percentage points."""
    lp, lm = label("protected"), label("meas_only")
    src = "E1 default configuration, truthful runs, seeds averaged per window"
    tot = _pairs_of(unit_values(W, MAIN, "protected", exp="e1"), unit_values(W, MAIN, "meas_only", exp="e1"), "value")
    tot = tot.assign(d_total=tot["a"] - tot["b"])[["date", "unit", "d_total"]]
    rows, mrows = [], []
    for c in CLASS_ORDER:
        a = _class_unit_values(W, MAIN, "protected", c, ("realized_value", "on_time_Mbit", "dropped_Mbit"))
        b = _class_unit_values(W, MAIN, "meas_only", c, ("realized_value", "on_time_Mbit", "dropped_Mbit"))
        if len(a) == 0 or len(b) == 0:
            notes.append(f"EoneGain{CLASS_MACRO[c]}: no {c} rows of {lp} or {lm}")
            continue
        pr = _pairs_of(a, b, "realized_value")
        if len(pr) == 0:
            continue
        r = contrast(pr, n_boot=nb, alternative="two-sided", denominator=f"{CLASS_NAME[c]} on-time value under {lm}")
        row = dict(cls=c, label=CLASS_NAME[c], **{k: r.get(k) for k in _CONTRAST_KEYS})
        sh = pr.assign(d=pr["a"] - pr["b"]).merge(tot, on=["date", "unit"]).dropna(subset=["d", "d_total"])
        if len(sh) and sh["d_total"].mean() != 0:
            rs = boot(sh[["d", "d_total"]].to_numpy(float), sh["date"].to_numpy(), n_boot=nb,
                      stat=S.ratio_stat(minus_one=False, percent=True))
            row.update(share_of_total_pct=rs.estimate, share_of_total_pct_lo=rs.lo, share_of_total_pct_hi=rs.hi)
        rows.append(row)
        mac.ci("EoneGain" + CLASS_MACRO[c], r, "credits per interval",
               f"value of reports, {CLASS_NAME[c]} tenant: on-time value under {lp} minus under {lm} ({src})",
               pct=False)
        if c == "video" and row.get("share_of_total_pct") is not None:
            mac.ci("EoneGainVideoShare", dict(estimate=row["share_of_total_pct"], lo=row["share_of_total_pct_lo"],
                                              hi=row["share_of_total_pct_hi"]),
                   "percent of the total value of reports (EoneGain, all tenants)",
                   f"share of the value of reports that the Video tenant receives ({src}; ratio of means)", nd=1,
                   pct=False)
        for df in (a, b):
            on, dr = df["on_time_Mbit"].astype(float), df["dropped_Mbit"].astype(float)
            df["miss_pct"] = (100 * dr / (on + dr)).where(on + dr > 0)
        pm = _pairs_of(a, b, "miss_pct")
        if len(pm.dropna()):
            rm = contrast(pm, n_boot=nb, alternative="two-sided", pct=False)
            mrows.append(dict(cls=c, label=CLASS_NAME[c], miss_pct_protected=rm["mean_a"],
                              miss_pct_meas_only=rm["mean_b"], **{k: rm.get(k) for k in _CONTRAST_KEYS
                                                                    if not k.startswith("pct")}))
            mac.ci("EoneMissDiff" + CLASS_MACRO[c], rm,
                   "percentage points of the class's traffic that misses its PDB (negative = fewer misses)",
                   f"deadline-miss share of the {CLASS_NAME[c]} tenant, {lp} minus {lm} ({src})", pct=False)
    if rows and len(tot):
        rows.append(dict(cls="all", label="all tenants", estimate=float(tot["d_total"].mean()),
                         class_sum=float(sum(_f(r["estimate"]) for r in rows)), share_of_total_pct=100.0,
                         n_windows=int(len(tot)), n_dates=int(tot["date"].nunique())))
    tables["value_of_reports_by_class"] = rows
    tables["miss_diff_by_class"] = mrows
    krows = []
    for a_arm, b_arm, stem in (("protected", "kappa0", "EoneKappaZero"), ("kappa0", "meas_only", "EoneKappaZeroVsFree")):
        res = _run(notes, stem, arm_contrast, W, MAIN, a_arm, b_arm, n_boot=nb, alternative="two-sided",
                   denominator=f"{label(b_arm)} welfare on the same windows")
        if res:
            r = res[0]
            krows.append(dict(arm_a=a_arm, arm_b=b_arm, **{k: r.get(k) for k in _CONTRAST_KEYS}))
            mac.ci(stem, r, "credits per interval", f"W({label(a_arm)}) - W({label(b_arm)}) ({src})", pct=False)
    tables["kappa0_contrasts"] = krows
    sl = ((hyp.get("H1a") or {}).get("load_subgroups") or {}).get("slope")
    if sl:
        mac.ci("EoneLoadSlope", sl, sl.get("unit", "credits per interval per unit of offered load"),
               f"D3: OLS slope of the per-window H1a difference W({lp}) - W({lm}) on the window's offered load "
               "(date-cluster bootstrap)", pct=False)
    else:
        notes.append("EoneLoadSlope: no D3 slope (H1a missing, no offered_load or fewer than MIN_DATES dates)")
    hi = [r for r in tables.get("value_by_load", []) if math.isclose(_f(r.get("rho")), 1.1)]
    if hi:
        mac.ci("EoneGainRhoHigh", hi[0], "credits per interval",
               f"value of reports W({lp}) - W({lm}) at offered load rho = 1.1 (E1 configuration rho1.1)", pct=False)
    else:
        notes.append("EoneGainRhoHigh: no E1 rho = 1.1 contrast")


def e2_transfer_table(W: pd.DataFrame, exp: str = "e2") -> pd.DataFrame:
    """Per (window, arm, deviator), seeds averaged (as :func:`victim_table`): PRBs moved to the deviator
    (``prb_moved`` = its quota in the deviation run minus in the truthful run, PRB per interval), its on-time gain
    (``dev_gain_mbit``), the honest tenants' on-time loss (``victim_loss_mbit`` = truthful minus deviation, summed)
    and their truthful on-time traffic (``victim_truthful_mbit``), Mbit per interval (scored intervals)."""
    x = W[(W["exp"] == exp) & (W["tenant"] >= 0) & _cell_mask(W, MAIN)]
    dev, tr = x[x["policy"] == "br"], x[x["policy"] == "truthful"]
    if len(dev) == 0 or len(tr) == 0 or "quota" not in x.columns:
        return pd.DataFrame()
    keys = ["date", "unit", "arm", "seed", "tenant"]
    cols = ["on_time_Mbit", "quota"]
    j = dev.merge(tr[keys + cols], on=keys, suffixes=("", "_t"))
    rows = []
    for (date, unit, arm, seed, dv), g in j.groupby(["date", "unit", "arm", "seed", "deviator"]):
        v, d = g[g["tenant"] != dv], g[g["tenant"] == dv]
        rows.append(dict(date=date, unit=unit, arm=arm, seed=seed, deviator=int(dv),
                         deviator_cls=d["cls"].iloc[0] if len(d) else "none",
                         prb_moved=float(d["quota"].sum() - d["quota_t"].sum()),
                         dev_gain_mbit=float(d["on_time_Mbit"].sum() - d["on_time_Mbit_t"].sum()),
                         victim_loss_mbit=float(v["on_time_Mbit_t"].sum() - v["on_time_Mbit"].sum()),
                         victim_truthful_mbit=float(v["on_time_Mbit_t"].sum())))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.groupby(["date", "unit", "arm", "deviator", "deviator_cls"]).mean(numeric_only=True).reset_index() \
        .drop(columns="seed")


E2_TRANSFER_STATS = {
    "prb_moved": lambda m: m[..., 0],
    "victim_loss_pct": lambda m: 100.0 * m[..., 1] / m[..., 2],
    "dev_mbit_per_prb": lambda m: m[..., 3] / m[..., 0],
    "victim_mbit_per_prb": lambda m: m[..., 1] / m[..., 0],
}
#: statistic -> (macro, digits, unit, source) of the H2c-cell transfer macros
E2_TRANSFER_MACROS = {
    "prb_moved": ("EtwoPrbMoved", 2, "PRB per interval moved to the deviator",
                  "deviating tenant's quota, best-response run minus truthful run"),
    "victim_loss_pct": ("EtwoVictimLossPct", 2,
                        "percent of the honest tenants' truthful on-time Mbit (positive = loss)",
                        "honest tenants' on-time loss relative to their truthful on-time traffic (ratio of means)"),
    "dev_mbit_per_prb": ("EtwoDevMbitPerPrb", 3, "on-time Mbit per interval gained by the deviator per PRB moved",
                         "deviator's on-time gain / PRBs moved (ratio of means)"),
    "victim_mbit_per_prb": ("EtwoVictimMbitPerPrb", 3,
                            "on-time Mbit per interval lost by the honest tenants per PRB moved",
                            "honest tenants' on-time loss / PRBs moved (ratio of means)"),
}


def _ceilings(n: int) -> np.ndarray | None:
    """Report ceilings C_i (Mbit/s) of the default calibration with n tenants (fixed per calibration)."""
    try:
        from slicemech.data.calibrate import load_calibration
        return np.asarray(load_calibration(f"n{int(n)}_default").contract.C, float)
    except Exception:                                                  # noqa: BLE001
        return None


def ceiling_share(BR: pd.DataFrame, arm: str, *, n_boot: int, n: int = 3) -> pd.DataFrame:
    """Per class: percent of the E2 stage best responses of ``arm`` (scored tenant-intervals) that report the ceiling
    C_i (z* >= C_i (1 - ``CEILING_RTOL``)); window means (seeds averaged), date bootstrap. C_i from the rows' ``C``
    column when present, else from the default calibration's contract."""
    x = BR[(BR["arm"] == arm) & (BR["n"] == n)].copy()
    if len(x) == 0:
        raise MissingInput(f"no stage best responses for {arm}")
    if "C" in x.columns and x["C"].notna().all():
        C = x["C"].astype(float).to_numpy()
    else:
        cs = _ceilings(n)
        if cs is None:
            raise MissingInput("no report ceilings (calibration unavailable)")
        C = cs[x["tenant"].astype(int).to_numpy()]
    z, th = x["z_star"].astype(float).to_numpy(), x["theta"].astype(float).to_numpy()
    at = z >= C * (1.0 - CEILING_RTOL)
    x = x.assign(at_ceiling=100.0 * at, inflated_to_ceiling=100.0 * (at & (th < C * (1.0 - CEILING_RTOL))),
                 within_1pct=100.0 * (z >= 0.99 * C))
    rows = []
    for c, g in x.groupby("cls"):
        wv = window_mean(g, [], "at_ceiling")
        r = boot(wv["at_ceiling"].to_numpy(float), wv["date"].to_numpy(), n_boot=n_boot)
        rows.append(dict(arm=arm, label=label(arm), cls=c, share_pct=r.estimate, lo=r.lo, hi=r.hi,
                         share_inflated_to_ceiling_pct=float(g["inflated_to_ceiling"].mean()),
                         share_within_1pct_of_ceiling=float(g["within_1pct"].mean()),
                         tenant_intervals=len(g), n_windows=r.n_windows, n_dates=r.n_dates, rtol=CEILING_RTOL))
    return pd.DataFrame(rows)


def _e2_extra(W: pd.DataFrame | None, BR2: pd.DataFrame, nb: int, mac: Macros, tables: dict, notes: list) -> None:
    """E2 descriptive additions. (2) The H2c primary cell (``H2C_PRIMARY_ARM``, replay windows, scored intervals;
    per window the deviator roles averaged, as H2c): EtwoPrbMoved, EtwoVictimLossPct, EtwoDevMbitPerPrb,
    EtwoVictimMbitPerPrb (ratios of means inside one bootstrap). (8) EtwoCeilingShareVideo/Urllc/Mmtc: percent of the
    stage best responses under the Unconstrained net with the fixed-priority price that report the ceiling C_i.
    (6) EtwoAllLoss is not computed: the E2 replays have no all-tenants-at-once profile (profiles truthful, br<i>)."""
    TT = e2_transfer_table(W) if W is not None and len(W) else pd.DataFrame()
    x = TT[TT["arm"] == H2C_PRIMARY_ARM] if len(TT) else TT
    if len(x) == 0:
        notes.append(f"E2 transfer (EtwoPrbMoved ...): no truthful/best-response replay pairs for {H2C_PRIMARY_ARM}")
    else:
        rows = []
        for role, g in [("pooled", x)] + list(x.groupby("deviator_cls")):
            wv = g.groupby(["date", "unit"])[list(E2_TRANSFER_COLS)].mean().reset_index().dropna()
            if len(wv) == 0:
                continue
            res = _multi_ci(wv[list(E2_TRANSFER_COLS)].to_numpy(float), wv["date"].to_numpy(), E2_TRANSFER_STATS, nb)
            row = dict(arm=H2C_PRIMARY_ARM, label=label(H2C_PRIMARY_ARM), deviator_role=role, n_windows=len(wv),
                       n_dates=int(wv["date"].nunique()), **{f"mean_{c}": float(wv[c].mean()) for c in E2_TRANSFER_COLS})
            for k, r in res.items():
                row.update({k: r["estimate"], f"{k}_lo": r["lo"], f"{k}_hi": r["hi"]})
            rows.append(row)
            if role == "pooled":
                for k, (stem, nd, unit, what) in E2_TRANSFER_MACROS.items():
                    mac.ci(stem, res[k], unit, f"H2c cell ({label(H2C_PRIMARY_ARM)}, replay windows, deviator roles "
                                               f"averaged per window): {what}", nd=nd, pct=False)
        tables["h2c_transfer"] = rows
    if BR2 is not None and len(BR2) and (BR2["arm"] == H2C_PRIMARY_ARM).any():
        cs = _run(notes, "EtwoCeilingShare", ceiling_share, BR2, H2C_PRIMARY_ARM, n_boot=nb)
        if cs is not None and len(cs):
            tables["ceiling_share"] = records(cs)
            for _, r in cs.iterrows():
                if r["cls"] in CLASS_MACRO:
                    mac.ci("EtwoCeilingShare" + CLASS_MACRO[r["cls"]], dict(estimate=r["share_pct"], lo=r["lo"],
                                                                           hi=r["hi"]),
                           "percent of the class's best responses (scored stage tenant-intervals)",
                           f"best responses that report the ceiling C_i, {CLASS_NAME[r['cls']]} tenant, "
                           f"{label(H2C_PRIMARY_ARM)} (E2 stage)", nd=1, pct=False)
    else:
        notes.append(f"EtwoCeilingShare: no E2 stage best responses for {H2C_PRIMARY_ARM}")


def _e3_extra_h3a(h: Mapping, mac: Macros) -> None:
    """(7) EthreeViolationsStrict: H3a violations under the PREREG rule gain + certificate > eps_i (= EthreeViol)."""
    mac.add("EthreeViolationsStrict", h.get("violations_strict"), "{:d}", "tenant-intervals",
            "H3a: scored tenant-intervals whose certified gain (gain + certificate) exceeds 1e-8 pi_i mu_i Delta")


def h4a_by_seed(W1: pd.DataFrame, *, n_boot: int) -> pd.DataFrame:
    """(4) The H4a welfare gap W(Unconstrained net) - W(Protected) per training seed s (seed s of each net; E1 default
    configuration, truthful), with its date-bootstrap CI; descriptive (the H4a analysis averages the seeds)."""
    sel = W1[(W1["exp"] == "e1") & (W1["policy"] == "truthful") & (W1["tenant"] == -1) & _cell_mask(W1, MAIN)]
    su = set(sel[(sel["arm"] == "unprotected") & (sel["seed"] >= 0)]["seed"].astype(int))
    sp = set(sel[(sel["arm"] == "protected") & (sel["seed"] >= 0)]["seed"].astype(int))
    rows = []
    for s in sorted(su & sp):
        try:
            res = arm_contrast(W1[W1["seed"] == s], MAIN, "unprotected", "protected", n_boot=n_boot,
                               alternative="two-sided", denominator="Protected welfare (same seed) on the same windows")
        except MissingInput:
            continue
        r = res[0]
        rows.append(dict(seed=int(s), **{k: r.get(k) for k in _CONTRAST_KEYS}))
    return pd.DataFrame(rows)


def _e4_seed_spread(W1: pd.DataFrame, nb: int, mac: Macros, tables: dict, notes: list) -> None:
    try:
        tb = h4a_by_seed(W1, n_boot=nb)
    except MissingInput as e:
        notes.append(f"EfourSeedLo/Hi: skipped ({e})")
        return
    tables["h4a_by_seed"] = records(tb)
    if len(tb) == 0:
        notes.append("EfourSeedLo/Hi: no training seed with both nets")
        return
    e = tb["estimate"].astype(float)
    src = "H4a gap W(Unconstrained net) - W(Protected) per training seed (seed s of each net; E1 default, truthful)"
    mac.add("EfourSeedLo", float(e.min()), "{:.2f}", "credits per interval", f"smallest {src}")
    mac.add("EfourSeedHi", float(e.max()), "{:.2f}", "credits per interval", f"largest {src}")


def _h4b_learning_macro(lt: pd.DataFrame, mac: Macros, notes: list) -> None:
    """(11) EfourHonestLearn: the h4b_learning row of the Unconstrained net (learning-tenant end states; honest value
    under Protected truthful minus under the design, positive = the honest tenants receive more under Protected)."""
    r = lt[(lt["arm"] == "unprotected") & (lt["status"] == "ok")] if len(lt) and "status" in lt else pd.DataFrame()
    if len(r) == 0:
        notes.append("EfourHonestLearn: no learning end state of the Unconstrained net with an honest tenant")
        return
    r0 = r.iloc[0]
    mac.ci("EfourHonestLearn", dict(estimate=r0["estimate"], lo=r0.get("lo"), hi=r0.get("hi")), "credits per interval",
           "H4b (learning tenants): honest tenants' realized value under Protected (truthful) minus under the "
           "Unconstrained net at the bandit end states", pct=False)


def _window_len_s() -> float:
    P = yaml.safe_load(open(SPEC / "params.yaml"))
    v = (P.get("experiment_settings") or {}).get("WindowLen")
    return float(v if v is not None else P["windows"]["length_s"])


def _e5_extra(lw: pd.DataFrame, nb: int, mac: Macros, tables: dict, notes: list) -> None:
    """(13) EfiveLaterInflate: the later gain of the H5b arm and pipeline (Protected, stored reports) at alpha = 2
    (inflation), targets pooled per evaluation window (as EfiveLater). (3) EfiveRepayMinutes = EfiveRepayWindows x
    the window length (spec/params.yaml WindowLen, 200 s incl. the 40 s warm-up) in minutes."""
    rows = []
    x = lw[(lw["arm"] == H5B_PRIMARY_ARM) & (lw["pipeline"] == "stored") & np.isclose(lw["alpha"], 2.0)]
    if len(x):
        pw = x.groupby(["date", "unit"])["d_utility_declared_target"].mean().reset_index()
        r = one_sample(pw, "d_utility_declared_target", n_boot=nb)
        mac.ci("EfiveLaterInflate", r, "credits per interval",
               f"later gain, stored reports, alpha = 2 (inflation), {label(H5B_PRIMARY_ARM)}, targets pooled", pct=False)
        rows.append(dict(quantity="later_gain_alpha2", estimate=r["estimate"], lo=r["lo"], hi=r["hi"],
                         n_dates=r["n_dates"], n_windows=r["n_windows"], unit="credits per interval"))
    else:
        notes.append(f"EfiveLaterInflate: no stored-report later rows at alpha = 2 for {H5B_PRIMARY_ARM}")
    rw = mac.d.get("EfiveRepayWindows")
    if rw is not None:
        wl = _window_len_s()
        minutes = float(rw["value"]) * wl / 60.0
        mac.add("EfiveRepayMinutes", minutes, "{:.0f}", "minutes of deployment (evaluation windows x window length)",
                f"H5b cell: EfiveRepayWindows x the {wl:g}-s window length (incl. warm-up), in minutes")
        rows.append(dict(quantity="repay_minutes", estimate=minutes, repay_windows=float(rw["value"]),
                         window_len_s=wl, unit="minutes"))
    tables["e5_extra"] = rows


def _e7_extra(NR: pd.DataFrame, vt: pd.DataFrame, mac: Macros, tables: dict) -> None:
    """(12) H7b, descriptive: EsevenProtMinusManip = mean over windows of the existing ``protected_minus_manipulated``
    (honest slices' DELIVERED traffic, nr_delivered_mbit, Protected truthful minus Unconstrained with the best
    responder) and EsevenProtBetter = windows where it is positive; the within-PDB variant (nr_delivered_pdb_mbit)
    as EsevenProtMinusManipPdb / EsevenProtBetterPdb."""
    rows = []
    for w, g in NR.groupby("window"):
        if not set(NR_CASES) <= set(g["arm"]):
            continue
        br = int(g[g["arm"] == "unconstrained_br"]["br_tenant"].iloc[0])
        h = g[g["tenant"] != br].groupby("arm")[["nr_delivered_pdb_mbit"]].sum()
        rows.append(dict(window=w, prot_minus_manip_pdb=float(h.loc["protected_truthful", "nr_delivered_pdb_mbit"]
                                                              - h.loc["unconstrained_br", "nr_delivered_pdb_mbit"])))
    pdb = pd.DataFrame(rows)
    t = vt[["window", "protected_minus_manipulated"]].merge(pdb, on="window", how="left") if len(pdb) else \
        vt[["window", "protected_minus_manipulated"]]
    tables["h7b_prot_minus_manip"] = records(t)
    pm = vt["protected_minus_manipulated"].astype(float)
    mac.add("EsevenProtMinusManip", float(pm.mean()), "{:.2f}", "Mbit/s (honest slices, delivered traffic)",
            "H7b: honest slices' delivered traffic (nr_delivered_mbit), Protected truthful minus Unconstrained net "
            "with the best responder, mean over windows (descriptive, no CI)")
    mac.add("EsevenProtBetter", int((pm > 0).sum()), "{:d}", "windows (of EsevenWindows)",
            "H7b: windows in which the honest slices receive more delivered traffic under Protected")
    if len(pdb):
        pp = pdb["prot_minus_manip_pdb"].astype(float)
        mac.add("EsevenProtMinusManipPdb", float(pp.mean()), "{:.2f}",
                "Mbit/s (honest slices, delivered within the PDB)",
                "H7b variant: honest slices' traffic delivered within the PDB (nr_delivered_pdb_mbit), Protected "
                "truthful minus Unconstrained net with the best responder, mean over windows (descriptive)")
        mac.add("EsevenProtBetterPdb", int((pp > 0).sum()), "{:d}", "windows (of EsevenWindows)",
                "H7b variant: windows in which the honest slices receive more within-PDB traffic under Protected")


# ================================================================================================ example (theory E3)
def example_summary(n_grid: int = 801) -> dict:
    """Closed-form example E3 of the theory (``theory/checks/premium_example.py`` check_E3, verified at 50 digits).

    Two tenants, B = 10 PRB, r = 1 Mbit/s per PRB, g = gamma = 0, eta = 3, M = C = 20 Mbit/s, opponent report 10,
    weights w_j = 1 + z_j / 5, own burst multipliers (0, 2) with probability 1/2 each, pi = Delta = 1, true demand
    8 Mbit/s. Own weight share sigma(u) = (5 + u)/(20 + u); own quota x(u) = min(3u, 10 sigma(u)) PRB (priority
    recomputed at every report) and x^v(u) = min(3u, 10 sigma(v)) (priority fixed at report v). Declared value
    V(x, theta) = min(theta, x/2). Fixed-priority price p_fix = 0 at every report; full-decision price
    p(v) = max(0, 5 sigma(v) - u_c), u_c = (5 sqrt(13) - 15)/2. Utility at theta = 8: no price and fixed price
    x(v)/2; full price x(v)/2 - p(v) (= u_c for v >= u_c).
    """
    theta, ztop = 8.0, 20.0
    uc = (5 * math.sqrt(13) - 15) / 2
    sig = lambda u: (5 + u) / (20 + u)                                       # noqa: E731
    u0 = (-50 + math.sqrt(3100)) / 6           # 3u = 10 sigma(u) <=> 3u^2 + 50u - 50 = 0 (cap -> interior switch)
    grid = np.unique(np.concatenate([np.linspace(0.0, ztop, n_grid), [uc, u0, theta, ztop]]))
    x_full = np.minimum(3 * grid, 10 * sig(grid))
    V = np.minimum(theta, x_full / 2)
    p_full = np.maximum(0.0, 5 * sig(grid) - uc)
    p_fix = np.zeros_like(grid)
    T8 = 5 * sig(theta) - uc
    U = lambda v: min(theta, min(3 * v, 10 * sig(v)) / 2)                    # noqa: E731
    gain_fix = U(ztop) - U(theta)
    gain_full = (U(ztop) - max(0.0, 5 * sig(ztop) - uc)) - (U(theta) - T8)
    mac = Macros("example")
    src = "closed-form example E3 (theory/checks/premium_example.py)"
    mac.add("ExFixGain", gain_fix, "{:.2f}", "credits per interval", src + ": gain of reporting 20 under p_fix")
    mac.add("ExPremium", T8, "{:.2f}", "credits per interval", src + ": premium T(8) = full price at the true report")
    mac.add("ExQuotaTrue", 10 * sig(theta), "{:.2f}", "PRB", src + ": own quota at report 8")
    mac.add("ExQuotaLie", 10 * sig(ztop), "{:.2f}", "PRB", src + ": own quota at report 20")
    mac.add("ExValueTrue", U(theta), "{:.2f}", "credits per interval", src + ": declared value at report 8")
    mac.add("ExPremiumPct", 100 * T8 / U(theta), "{:.0f}", "percent of declared value", src + ": premium / value")
    curves = pd.DataFrame(dict(report=grid, quota_full=x_full,
                               quota_fixed_at_lie=np.minimum(3 * grid, 10 * sig(ztop)),
                               quota_fixed_at_truth=np.minimum(3 * grid, 10 * sig(theta)), value=V,
                               price_full=p_full, price_fixed=p_fix, utility_none=V, utility_fixed=V - p_fix,
                               utility_full=V - p_full))
    pts = dict(theta=theta, lie=ztop, uc=uc, u_switch=u0, quota_truth=10 * sig(theta), quota_lie=10 * sig(ztop),
               utility_fixed_truth=U(theta), utility_fixed_lie=U(ztop), utility_full_truth=U(theta) - T8,
               utility_full_lie=U(theta) - T8 + gain_full, gain_fixed=gain_fix, gain_full=gain_full, premium_truth=T8,
               fixed_price_identically_zero=True)
    return dict(macros=mac.d, tables=dict(curves=records(curves), points=[_clean(pts)]), hypotheses={},
                notes=["p_fix = 0 at every report in this example, so the utility under no price equals the utility "
                       "under the fixed-priority price."])


#: Learned arms whose DEV-selected hyper-parameters are published (macro suffix).
PPO_ARM_MACRO = {"protected": "Protected", "unconstrained": "Unconstrained", "meas_only": "MeasOnly",
                 "kappa0": "KappaZero"}


def _sci_tex(x: float) -> str:
    m, e = f"{float(x):.0e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def settings_summary(models_root: Path | None = None) -> dict:
    """Setting macros of Sections VI-VII (split-independent): the design constants of ``spec/params.yaml:
    experiment_settings``, plus the quantities that are measured or selected rather than chosen (round-3 review:
    they used to be typed into params.yaml): ``RateMean`` = r_bar_TRAIN from the default calibration,
    ``RateMin``/``RateMax`` = the per-PRB rates of CQI 1 and 15 (``phy.RATE_TABLE``, the bounds of every trace),
    and the PPO hyper-parameters of each learned arm from the DEV selection (``selection.json`` below
    ``models_root``; ``PpoLr``, ``PpoLrEnd``, ``PpoSamples``, ``PpoSigma`` when every arm selected the same value,
    else a per-arm list; ``PpoLr<Arm>`` etc. per arm)."""
    mac = Macros("settings")
    notes = []
    for k, v in (yaml.safe_load(open(SPEC / "params.yaml")).get("experiment_settings") or {}).items():
        if re.fullmatch(r"[A-Za-z]+", str(k)):
            if isinstance(v, str):
                mac.add(k, v, unit="", source="spec/params.yaml experiment_settings")
            else:
                mac.add(k, v, "{:d}" if float(v) == int(v) and not isinstance(v, float) else "{:g}", "",
                        "spec/params.yaml experiment_settings")
    try:
        from slicemech.data.calibrate import load_calibration
        from slicemech.phy import RATE_TABLE
        cal = load_calibration("n3_default")
        mac.add("RateMean", float(cal.r_bar), "{:.2f}", "Mbit/s per PRB",
                "data/calibration/n3_default.json r_bar (TRAIN mean per-PRB rate)")
        # pre-freeze audit NUM-16: the precision of decisions.md D3.6 (0.0181 and 0.660)
        mac.add("RateMin", float(np.nanmin(RATE_TABLE)), "{:.4f}", "Mbit/s per PRB",
                "slicemech.phy.RATE_TABLE (CQI 1, TS 38.214 Table 5.2.2.1-2)")
        mac.add("RateMax", float(np.nanmax(RATE_TABLE)), "{:.3f}", "Mbit/s per PRB",
                "slicemech.phy.RATE_TABLE (CQI 15)")
    except Exception as e:                                             # noqa: BLE001
        notes.append(f"rate macros: calibration unavailable ({type(e).__name__}: {e})")
    # pre-freeze audit E5-1: the primary H5b cell is emitted from the code constant (it overrides the params.yaml
    # copy, which make_numbers renders first); the sign likewise
    mac.add("EfivePrimary", H5B_PRIMARY_ALPHA, "{:g}", "distortion factor",
            "analyze.H5B_PRIMARY_ALPHA (PREREG D8, section 9 item 3)")
    if H5B_SIGN in (1, -1):
        mac.add("EfiveSign", f"{H5B_SIGN:+d}", unit="", source="analyze.H5B_SIGN (PREREG D8)")
    try:
        import registry as R
        sel = R.load_selection(root=models_root)
    except (FileNotFoundError, ImportError) as e:
        sel = None
        notes.append(f"PPO hyper-parameter macros: no DEV selection yet ({e})")
    try:                                               # pre-freeze audit NUM-11: TrainMinutesNet had no producer
        import registry as R
        idx = Path(models_root or R.MODELS_DIR) / R.INDEX_FILE
        ix = pd.read_csv(idx)
        t = pd.to_numeric(ix[ix["kind"].isin(("tune", "main"))]["train_seconds"], errors="coerce").dropna()
        if len(t):
            mac.add("TrainMinutesNet", float(t.median()) / 60.0, "{:.0f}", "minutes",
                    f"{_rel(idx)}: median train_seconds of the trained networks (kinds tune and main; one core)")
    except (FileNotFoundError, ImportError, KeyError, ValueError) as e:
        notes.append(f"TrainMinutesNet: no model index ({type(e).__name__}: {e})")
    if sel:
        src = f"{_rel(R.selection_path(root=models_root))} (DEV selection)"
        rows = {a: v["params"] for a, v in sel.get("arms", {}).items()}
        for a, pr in rows.items():
            suf = PPO_ARM_MACRO.get(a)
            if suf:
                mac.add("PpoLr" + suf, _sci_tex(pr["lr"]), unit="", source=src)
                mac.add("PpoLrEnd" + suf, _sci_tex(pr["lr"] * R.LR_END_RATIO), unit="", source=src)
                mac.add("PpoSamples" + suf, pr["M"], "{:d}", "", src)
                mac.add("PpoSigma" + suf, pr["sigma0"], "{:g}", "", src)
        for name, key, f in (("PpoLr", "lr", _sci_tex), ("PpoLrEnd", "lr", lambda x: _sci_tex(x * R.LR_END_RATIO)),
                             ("PpoSamples", "M", lambda x: f"{int(x)}"), ("PpoSigma", "sigma0", lambda x: f"{x:g}")):
            vals = {a: pr[key] for a, pr in rows.items() if a in PPO_ARM_MACRO}
            if not vals:
                continue
            if len(set(vals.values())) == 1:
                mac.add(name, f(next(iter(vals.values()))), unit="", source=src)
            else:
                mac.add(name, ", ".join(f"{f(v)} ({label(_ARM_KEY.get(a, a))})" for a, v in vals.items()), unit="",
                        source=src + ", differs by arm")
    return dict(macros=mac.d, tables={}, hypotheses={}, notes=notes)


#: registry arm -> spec/controllers.yaml key (labels of the per-arm PPO macros)
_ARM_KEY = {"protected": "protected", "unconstrained": "unprotected", "meas_only": "meas_only", "kappa0": "kappa0"}


# ================================================================================================ Holm + driver
HOLM_SOURCE = {"H1a": ("e1", "p"), "H2c": ("e2", "p"), "H4a": ("e4", "noninferiority_p"), "H5b": ("e5", "p")}
HOLM_MACRO = {"H1a": "EonePHolm", "H2c": "EtwoPHolm", "H4a": "EfourPHolm", "H5b": "EfivePHolm"}
#: Final-verdict macro of each primary (round-3 review: written by holm.json only, never by the experiment files).
VERDICT_MACRO = {"H1a": "EoneSupported", "H2c": "EtwoSupported", "H4a": "EfourSupported", "H5b": "EfiveSupported"}


def holm_summary(results: Mapping[str, dict], split: str | None = None) -> dict:
    """Holm across HOLM_FAMILY and the FINAL verdict of every primary (module docstring): supported = the
    pre-registered rule (``rule_met`` of its experiment summary; for H5b the pre-registered side) AND the Holm
    rejection AND, on TEST, complete inputs of one provenance (``inputs_check`` of the experiment summary). A
    missing member enters with p = 1 (conservative) and is flagged; a primary whose rule or p cannot be evaluated
    gets the verdict None."""
    pv, missing = {}, []
    for h in HOLM_FAMILY:
        exp, field = HOLM_SOURCE[h]
        p = (results.get(exp) or {}).get("hypotheses", {}).get(h, {}).get(field)
        if p is None or not math.isfinite(_f(p)):
            missing.append(h)
            p = 1.0
        pv[h] = float(p)
    tb = S.holm(pv, alpha=ALPHA_FWER)
    mac = Macros("holm")
    verdicts = []
    for _, r in tb.iterrows():
        h = r["name"]
        exp = HOLM_SOURCE[h][0]
        res = results.get(exp) or {}
        hy = res.get("hypotheses", {}).get(h, {})
        rule = hy.get("rule_met")
        chk = res.get("inputs_check") or {}
        inputs_ok = chk.get("ok") if split == "test" else True
        row = dict(name=h, experiment=exp, rule_met=rule, p=float(r["p"]), p_holm=float(r["p_holm"]),
                   holm_reject=bool(r["reject"]) if h not in missing else None, inputs_ok=inputs_ok,
                   contrary=hy.get("contrary") if h == "H5b" else None)
        if h in missing or rule is None or inputs_ok is None:
            row.update(supported=None, reason=("no p-value" if h in missing else "rule not evaluable"
                                               if rule is None else "inputs not checked"))
        else:
            ok = bool(rule and r["reject"] and inputs_ok)
            if not inputs_ok:                       # pre-freeze audit PV-2 / F5: say why, not the success wording
                why = chk.get("reason") or "inputs incomplete or mixed provenance"
            elif not rule:
                why = "pre-registered rule not met"
            elif not r["reject"]:
                why = "Holm does not reject"
            else:
                why = "rule and Holm" + (" and complete, single-provenance inputs" if split == "test" else "")
            row.update(supported=ok, reason=why)
        verdicts.append(row)
        if h not in missing:
            mac.add(HOLM_MACRO[h], r["p_holm"], "{:.3g}", "Holm-adjusted p", f"{h} in {HOLM_FAMILY}")
        if row["supported"] is not None:
            mac.add(VERDICT_MACRO[h], row["supported"],
                    source=f"{h} final verdict: pre-registered rule, Holm rejection at {ALPHA_FWER:g}"
                           + (", complete single-provenance TEST inputs" if split == "test" else ""))
    notes = [f"members without a p-value entered with p = 1: {missing}"] if missing else []
    power = [dict(name=h, **((results.get(HOLM_SOURCE[h][0]) or {}).get("hypotheses", {}).get(h, {})
                             .get("dev_estimation_half") or {}))
             for h in HOLM_FAMILY]
    power = [r for r in power if len(r) > 1]
    tables = dict(holm=records(tb), verdicts=_clean(verdicts))
    if power:                                   # DEV: the D9 power-analysis inputs (estimation half only)
        tables["dev_power"] = _clean(power)
    return dict(macros=mac.d, tables=tables,
                hypotheses=dict(family=list(HOLM_FAMILY), missing=missing, complete=not missing,
                                verdicts={v["name"]: v["supported"] for v in verdicts}),
                notes=notes)


# ================================================================================================ TEST input checks
def _cover(have, want) -> dict:
    have, want = set(have), set(want)
    miss = sorted(want - have)
    return dict(expected=len(want), present=len(want & have), missing=miss[:12], n_missing=len(miss),
                complete=not miss)


def expected_units(split: str) -> list[str]:
    """Every pre-declared main window of the split as an analysis unit ('<date>|o<offset>')."""
    from slicemech.data import compose, puffer
    return [f"{d}|o{int(o)}" for d in puffer.split_dates(split) for o in compose.window_offsets(split)]


def completeness(T: Mapping, split: str) -> dict:
    """Per experiment, the pre-declared windows / jobs the outputs cover (round-3 review; pre-freeze audit PV-2, F9).
    E0: all windows of the default configuration; E1: all windows for every arm of the default cell with every
    expected seed (keys = spec/controllers.yaml arms), and every sweep configuration (``e1_sweeps``, per config and
    arm; information); E2: stage windows per arm, replay windows (``e2_manipulation.replay_keys``, ``REPLAY_PER_DATE``
    per date) per arm, and the bandit runs (designs x checkpoint seeds x algorithms x runs); E3: every certificate
    arm on all windows; E4: every payment arm on all windows (``pay``), the H4b pairs per arm on the replay windows
    (``h4b``) and the learning-deviator replays (``learn``: every bandit end state of every design); E5: dates x
    seeds x arms (``jobs``) and per arm (``jobs_by_arm``); E6: n x windows; E7: 4 windows x 3 cases. Only the parts
    named in ``PRIMARY_INPUTS`` gate a verdict; the rest is information."""
    import e1_value as E1M
    import e2_manipulation as X
    import e3_protection as E3M
    import e4_cost as E4M
    import e5_training_channel as E5M
    import registry as R
    from slicemech.data import puffer
    units = expected_units(split)

    def by_arm(df, col, want_units, arms):
        got = {a: _cover(g["unit"], want_units) for a, g in df.groupby(col)} if len(df) else {}
        for a in arms:                                       # an expected arm without any row is fully missing
            got.setdefault(a, _cover([], want_units))
        return got

    def e1_cover(w, seeds):
        per = {}
        for arm, g in w.groupby("arm"):
            if arm == "protected":
                g = g[np.isclose(g["kappa_scale"].astype(float).fillna(1.0), 1.0)]
            learned = (g["seed"] >= 0).any()
            ok = g.groupby("unit")["seed"].nunique()
            have = ok[ok >= seeds].index if learned else ok.index
            per[arm] = _cover(have, units)
        for arm in E1M.ARMS:
            per.setdefault(arm, _cover([], units))
        return per
    out = {}
    W = T.get("windows")
    e0 = T.get("e0_windows")
    rk = [f"{d}|o{int(o)}" for d, o in X.replay_keys(X.split_keys(split), REPLAY_PER_DATE)]
    if e0 is not None and len(e0):
        out["e0"] = dict(default=_cover(e0[e0["config"] == "default"]["unit"], units))
    if W is not None and len(W):
        e1 = W[(W["exp"] == "e1") & (W["tenant"] == -1) & (W["policy"] == "truthful")]
        w1 = e1[_cell_mask(e1, MAIN)]
        if "config" in e1.columns:
            w1 = w1[w1["config"].astype(str) == "default"]
        if len(w1):
            out["e1"] = e1_cover(w1, len(R.CONFIGS["default"].seeds))
        if len(e1) and "config" in e1.columns:                # F9: the sweep configurations (information only)
            sw = {}
            for name, rc in R.CONFIGS.items():
                if name == "default":
                    continue
                g = e1[e1["config"].astype(str) == name]
                sw[name] = e1_cover(g, len(rc.seeds)) if len(g) else dict(
                    missing_config=_cover([], [f"{name}|{a}" for a in E1M.ARMS]))
            out["e1_sweeps"] = sw
        w2 = W[(W["exp"] == "e2") & (W["tenant"] == -1)]
        if len(w2):
            out.setdefault("e2", {})["replay"] = by_arm(w2, "arm_raw", rk, X.E2_ARMS)
        w4 = W[(W["exp"] == "e4") & (W["tenant"] == -1)]
        if len(w4):
            out.setdefault("e4", {})["pay"] = by_arm(w4, "arm_raw", units, tuple(E4M.PAY_ARMS) + tuple(E4M.e4b_arms()))
    BR = T.get("br")
    if BR is not None and len(BR):
        for exp, part, arms in (("e2", "stage", X.E2_ARMS), ("e3", "cert", E3M.CERT_ARMS)):
            b = BR[BR["exp"] == exp]
            if len(b):
                out.setdefault(exp, {})[part] = by_arm(b, "arm_raw", units, arms)
    BD = T.get("bandit")
    if BD is not None and len(BD):                            # F9: E2 bandit runs (information only)
        want = [f"{d}|s{sd}|{a}|run{r}" for d in X.BANDIT_DESIGNS for sd in X.CHECKPOINT_SEEDS
                for a in X.BANDIT_ALGOS for r in range(X.BANDIT_RUNS)]
        have = (BD["arm_raw"].astype(str) + "|s" + BD["seed"].astype(int).astype(str) + "|" + BD["algo"].astype(str)
                + "|run" + BD["run"].astype(int).astype(str)).unique()
        out.setdefault("e2", {})["bandit"] = _cover(have, want)
    H4 = T.get("e4_h4b")
    if H4 is not None and len(H4):                            # HC-9: the H4b pairs (information only)
        col = "arm_raw" if "arm_raw" in H4.columns else "arm"
        out.setdefault("e4", {})["h4b"] = by_arm(H4, col, rk, E4M.H4B_ARMS)
    if BD is not None and len(BD):                            # F9: every bandit end state replayed by E4 (learn)
        L = T.get("e4_learn")
        end = BD[BD["episode"] == BD.groupby(["arm_raw", "seed", "algo", "run"])["episode"].transform("max")]
        want = set()
        for (design, sd, _a, _r), g in end.groupby(["arm_raw", "seed", "algo", "run"]):
            if design in E4M.H4B_LEARN_DESIGNS:
                want.add(f"{design}|s{int(sd)}|" + E4M._tag(tuple(g.sort_values("tenant")["greedy_m"].astype(float))))
        have = set()
        if L is not None and len(L) and {"design", "seed", "profile_tag"} <= set(L.columns):
            have = set((L["design"].astype(str) + "|s" + L["seed"].astype(int).astype(str) + "|"
                        + L["profile_tag"].astype(str)).unique())
        if want:
            out.setdefault("e4", {})["learn"] = _cover(have, want)
    P = T.get("e5_pairs")
    if P is not None and len(P):
        dates = puffer.split_dates(split)
        key = P["date"].astype(str) + "|s" + P["seed"].astype(int).astype(str)
        want = [f"{d}|s{s}|{a}" for d in dates for s in E5M.SEEDS_DEFAULT for a in E5M.ARMS]
        out["e5"] = dict(jobs=_cover((key + "|" + P["arm"].astype(str)).unique(), want),
                         jobs_by_arm={a: _cover(key[P["arm"].astype(str) == a].unique(),
                                                [f"{d}|s{s}" for d in dates for s in E5M.SEEDS_DEFAULT])
                                      for a in E5M.ARMS})
    R6 = T.get("e6_runs")
    if R6 is not None and len(R6):
        want = [f"{n}|{u}" for n in N_SWEEP if n != 3 for u in units]
        out["e6"] = dict(windows=_cover((R6["n"].astype(int).astype(str) + "|" + R6["unit"]).unique(), want))
    NR = T.get("nr")
    if NR is not None and len(NR):
        import e7_nr as E7M
        want = [f"{c.date}|o{c.offset}|{c.arm}" for c in E7M.define_cases(split)]
        out["e7"] = dict(cases=_cover((NR["date"].astype(str) + "|o" + NR["offset"].astype(int).astype(str) + "|"
                                       + NR["arm"].astype(str)).unique(), want))
    return out


def _flat_complete(d) -> bool:
    if isinstance(d, Mapping):
        if "complete" in d and "expected" in d:
            return bool(d["complete"])
        return all(_flat_complete(v) for v in d.values())
    return True


def _sidecar_provenance(root: Path, sub: str | None = None) -> dict:
    """E2-E4 provenance of the job outputs below ``root`` (or ``root/sub`` only): every output has a sidecar, one
    code fingerprint, one checkpoint SHA-256 per (asset kind, seed)."""
    base = root / sub if sub else root
    side = sorted(base.rglob("*.prov.json")) if base.is_dir() else []
    outs = [f for f in base.rglob("*") if f.suffix in (".parquet", ".npz") and f.parent != root
            and ".tmp" not in f.name and not f.name.endswith(".prov.json")] if base.is_dir() else []
    no_side = [f for f in outs if not Path(str(f) + ".prov.json").is_file()]
    shas: dict = {}
    codes = set()
    for f in side:
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        codes.add(r.get("code"))
        seed = (r.get("args") or {}).get("seed")
        for k, v in (r.get("assets") or {}).items():
            shas.setdefault(f"{k}|seed{seed}", set()).add(v)
    conflicts = {k: len(v) for k, v in shas.items() if len(v) > 1}
    return dict(scope=sub or "all", sidecars=len(side), outputs=len(outs), outputs_without_sidecar=len(no_side),
                code_versions=len(codes), asset_conflicts=conflicts,
                ok=bool(outs) and not no_side and len(codes) == 1 and not conflicts)


def _job_provenance(jd: Path, arms: Sequence[str] | None = None) -> dict | None:
    """E5/E6 provenance from the job records (``arms``: only the jobs of these arms): one checkpoint SHA-256 per
    model, a provenance fingerprint on every job, and one code version when the records carry one (``code``)."""
    if not jd.is_dir():
        return None
    recs = []
    for f in sorted(jd.glob("*.done.json")):
        try:
            recs.append(json.loads(f.read_text()))
        except ValueError:
            pass
    if arms is not None:
        recs = [r for r in recs if str(r.get("arm")) in set(arms)]
    shas: dict = {}
    for r in recs:
        ck = r.get("ckpt_sha256")
        items = ck.items() if isinstance(ck, Mapping) else [(f"{r.get('arm')}|s{r.get('seed')}", ck)]
        for k, v in items:
            shas.setdefault(k, set()).add(v)
    conflicts = {k: len(v) for k, v in shas.items() if len(v) > 1}
    nofp = sum(1 for r in recs if not r.get("fp"))
    codes = {r.get("code") for r in recs if "code" in r}
    code_ok = not codes or (len(codes) == 1 and None not in codes and all("code" in r for r in recs))
    return dict(scope="all" if arms is None else ",".join(arms), jobs=len(recs), without_fingerprint=nofp,
                ckpt_conflicts=conflicts, code_versions=len(codes) if codes else None,
                ok=bool(recs) and not nofp and not conflicts and code_ok)


def _e1_provenance(W: pd.DataFrame, config: str | None = None, arms: Sequence[str] | None = None) -> dict | None:
    """E1 provenance: one checkpoint SHA-256 per (configuration, arm, seed) and, when the rows carry it, one code
    fingerprint (``code_fp``); restricted to ``config`` and the controllers.yaml ``arms`` when given."""
    if W is None or len(W) == 0 or "ckpt_sha" not in W.columns:
        return None
    w1 = W[W["exp"] == "e1"]
    if config is not None and "config" in w1.columns:
        w1 = w1[w1["config"].astype(str) == config]
    if arms is not None:
        w1 = w1[w1["arm"].isin(arms)]
    lw = w1[w1["ckpt_sha"].notna() & (w1["seed"] >= 0)]
    if len(w1) == 0:
        return None
    n = lw.groupby(["config", "arm_raw", "seed"])["ckpt_sha"].nunique() if len(lw) else pd.Series(dtype=int)
    out = dict(scope=f"{config or 'all'}:{','.join(arms) if arms else 'all'}", runs=int(len(n)),
               conflicts=int((n > 1).sum()), ok=bool((n <= 1).all()))
    if "code_fp" in w1.columns:                  # pre-freeze audit F7: one code version per scope
        k = int(w1["code_fp"].dropna().nunique())
        out.update(code_versions=k, rows_without_code_fp=int(w1["code_fp"].isna().sum()))
        out["ok"] = out["ok"] and k == 1 and out["rows_without_code_fp"] == 0
    return out


def provenance(src: Mapping[str, Path], T: Mapping) -> dict:
    """Do the outputs come from one set of models and one code version? (round-3 review)

    Per experiment (information): E2-E4 the provenance sidecars ``*.prov.json`` of the job outputs (per checkpoint
    seed and asset kind exactly one SHA-256; one code fingerprint; every job output has a sidecar); E5/E6 the job
    records (one checkpoint SHA-256 per model; a provenance fingerprint present; one code version when recorded);
    E1 one checkpoint SHA-256 per (configuration, arm, seed) (and one ``code_fp`` when the rows carry it).
    ``primary`` (pre-freeze audit PV-2): the same checks restricted to what each primary reads
    (``PRIMARY_INPUTS``): E1 default configuration and the primary's arms, E2 ``replay/unc_fixed`` only, E5 the
    protected jobs only. Only ``primary`` gates a verdict."""
    out = {}
    for exp in ("e2", "e3", "e4"):
        root = Path(src.get(exp, "/nonexistent"))
        if root.is_dir():
            out[exp] = _sidecar_provenance(root)
    for exp in ("e5", "e6"):
        pr = _job_provenance(Path(src.get(exp, "/nonexistent")) / "jobs")
        if pr is not None:
            out[exp] = pr
    W = T.get("windows")
    e1 = _e1_provenance(W)
    if e1 is not None:
        out["e1"] = e1
    prim: dict = {}
    for summ, need in PRIMARY_INPUTS.items():
        for exp, spec in need.items():
            if exp == "e1":
                pv = _e1_provenance(W, "default", tuple(spec))
            elif exp == "e2":
                root = Path(src.get("e2", "/nonexistent"))
                subs = [f"{part}/{a}" for part, arms in spec.items() for a in arms]
                pv = {sub: _sidecar_provenance(root, sub) for sub in subs} if root.is_dir() else None
                if pv is not None:
                    pv = dict(parts=pv, ok=all(v["ok"] for v in pv.values()))
            elif exp == "e5":
                arms = [a for arms_ in spec.values() for a in arms_]
                pv = _job_provenance(Path(src.get("e5", "/nonexistent")) / "jobs", arms)
            else:
                pv = out.get(exp)
            prim.setdefault(summ, {})[exp] = pv
    out["primary"] = prim
    return out


#: What each primary reads on TEST (pre-freeze audit PV-2; summary experiment -> input experiment -> the arms or
#: parts of ``completeness()`` that gate it): H1a (e1.json) Protected and No reports in the E1 default cell; H4a
#: (e4.json) Protected and the Unconstrained net in the E1 default cell; H2c (e2.json) the E2 replays of unc_fixed;
#: H5b (e5.json) the protected E5 jobs. Everything else in completeness/provenance is information only.
PRIMARY_INPUTS = {"e1": {"e1": ("protected", "meas_only")},
                  "e4": {"e1": ("protected", "unprotected")},
                  "e2": {"e2": {"replay": ("unc_fixed",)}},
                  "e5": {"e5": {"jobs_by_arm": ("protected",)}}}


def _select(d, spec):
    """The parts of a completeness record that ``spec`` names (a tuple of keys, or a mapping of key -> spec);
    a missing part is None (not complete)."""
    if not isinstance(d, Mapping):
        return None
    if isinstance(spec, Mapping):
        return {k: _select(d.get(k), v) for k, v in spec.items()}
    return {k: d.get(k) for k in spec}


def _complete(v) -> bool:
    if v is None:
        return False
    if isinstance(v, Mapping) and not ("complete" in v and "expected" in v):
        return all(_complete(x) for x in v.values()) if v else False
    return _flat_complete(v)


def primary_dates(T: Mapping) -> dict:
    """Distinct dates in each primary's inputs (``PRIMARY_INPUTS`` scope; summary -> input experiment -> arm or
    part/arm -> number of dates): E1 default-cell truthful windows per arm, E2 replay windows per E2 arm, E5 pairs
    per arm (pre-freeze audit F5: on TEST each must equal the frozen ``test_dates``)."""
    W, P = T.get("windows"), T.get("e5_pairs")
    out: dict = {}
    for summ, need in PRIMARY_INPUTS.items():
        for exp, spec in need.items():
            counts = {}
            if exp == "e1":
                w = W[(W["exp"] == "e1") & (W["tenant"] == -1) & (W["policy"] == "truthful") & _cell_mask(W, MAIN)] \
                    if W is not None and len(W) else pd.DataFrame(columns=["arm", "date"])
                if "config" in w.columns:
                    w = w[w["config"].astype(str) == "default"]
                counts = {a: int(w[w["arm"] == a]["date"].nunique()) for a in spec}
            elif exp == "e2":
                w = W[(W["exp"] == "e2") & (W["tenant"] == -1)] if W is not None and len(W) \
                    else pd.DataFrame(columns=["arm_raw", "date"])
                counts = {f"{part}/{a}": int(w[w["arm_raw"].astype(str) == a]["date"].nunique())
                          for part, arms in spec.items() for a in arms}
            elif exp == "e5":
                q = P if P is not None and len(P) else pd.DataFrame(columns=["arm", "date"])
                counts = {f"{part}/{a}": int(q[q["arm"].astype(str) == a]["date"].nunique())
                          for part, arms in spec.items() for a in arms}
            out.setdefault(summ, {})[exp] = counts
    return out


def inputs_check(exp: str, comp: Mapping, prov: Mapping, dates: Mapping | None = None,
                 frozen_test_dates: int | None = None) -> dict:
    """Completeness and provenance of exactly the inputs a summary's primary reads (``PRIMARY_INPUTS``; pre-freeze
    audit PV-2: an incomplete arm the primary does not read no longer fails it), and, on TEST with the freeze
    record, the number of distinct TEST dates in each of those inputs against the frozen ``test_dates`` (F5).
    Experiments without a primary get ``ok`` = True and the information of their own experiment. ``reason`` says
    why ``ok`` is False."""
    need = PRIMARY_INPUTS.get(exp)
    if need is None:
        return dict(ok=True, primary=False, completeness={exp: comp.get(exp)}, provenance={exp: prov.get(exp)})
    c = {e: _select(comp.get(e), spec) for e, spec in need.items()}
    pp = (prov.get("primary") or {}).get(exp) or {}
    pv = {e: pp.get(e) for e in need}
    reasons = []
    if not all(_complete(v) for v in c.values()):
        reasons.append("inputs incomplete")
    if not all(v is not None and v.get("ok") for v in pv.values()):
        reasons.append("mixed or unverified provenance")
    out = dict(primary=True, scope=_clean(need), completeness=c, provenance=pv)
    if frozen_test_dates is not None:
        counts = (dates or {}).get(exp) or {}
        flat = {f"{e}:{k}": n for e, kv in counts.items() for k, n in kv.items()}
        bad = {k: n for k, n in flat.items() if n != int(frozen_test_dates)}
        out["test_dates"] = dict(frozen=int(frozen_test_dates), counts=flat, ok=not bad and bool(flat))
        if bad or not flat:
            reasons.append(f"TEST date count differs from the frozen test_dates = {int(frozen_test_dates)}: "
                           + (", ".join(f"{k} has {n}" for k, n in sorted(bad.items())) or "no dated inputs"))
    out.update(ok=not reasons, reason="; ".join(reasons) if reasons else None)
    return out


ANALYSES: dict[str, Callable] = dict(e0=analyze_e0, e1=analyze_e1, e2=analyze_e2, e3=analyze_e3, e4=analyze_e4,
                                     e5=analyze_e5, e6=analyze_e6, e7=analyze_e7)
_TABLES: dict = {}


def _worker(args):
    exp, cfg = args
    t0 = time.perf_counter()
    try:
        res = ANALYSES[exp](_TABLES, cfg)
    except MissingInput as e:
        res = dict(macros={}, tables={}, hypotheses={}, notes=[f"{exp}: skipped ({e})"])
    res["seconds"] = time.perf_counter() - t0
    return exp, res


def summary_dir(split: str, smoke: bool) -> Path:
    if smoke:
        return SUMMARY / "smoke" / split
    return SUMMARY if split == "test" else SUMMARY / split


def write_json(path: Path, obj, confine: bool = True) -> None:
    """Atomic strict-JSON write (NaN -> null); refuses paths outside the rewrite folder unless confine=False."""
    if confine:
        sio.confine(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=1, allow_nan=False))
    tmp.replace(path)


def run_analysis(split: str, out: Path, *, smoke: bool = False, n_boot: int = N_BOOT,
                 exps: Sequence[str] = EXPERIMENTS, workers: int = 1, force: bool = False, confine: bool = True,
                 src: Mapping[str, Path] | None = None, crossed: bool = True, test_freeze=None,
                 example_to_paper: bool = False, verbose: bool = True, test_dates: int | None = None) -> dict:
    """Analyse ``exps`` and write ``<out>/<exp>.json``, ``holm.json`` and ``example.json``; returns {name: result}.
    TEST gating is the caller's job (:func:`main`)."""
    log = print if verbose else (lambda *a, **k: None)
    src = {k: Path(v) for k, v in (src or sources(split, smoke)).items()}
    cfg = dict(n_boot=n_boot, level=LEVEL, seed=BOOT_SEED, main=MAIN, load_bins=LOAD_BINS, holm_family=HOLM_FAMILY,
               h5b=dict(arm=H5B_PRIMARY_ARM, alpha=H5B_PRIMARY_ALPHA), h2c_arm=H2C_PRIMARY_ARM, split=split,
               smoke=smoke, crossed=crossed)
    if split == "test" and test_dates is not None:       # pre-freeze audit F5: the frozen number of TEST dates
        cfg["frozen_test_dates"] = int(test_dates)
    files = input_files(src)
    digest = input_digest(files, cfg)
    code_sha = sio.file_sha256(__file__)
    t0 = time.perf_counter()
    todo, results = [], {}
    checks: dict = {}

    def fresh(e: str) -> dict | None:
        """The existing summary of ``e`` when it was computed from these inputs by this code (same digest and
        code SHA-256; pre-freeze audit PV-9), else None."""
        p = out / f"{e}.json"
        if not p.exists():
            return None
        try:
            old = json.loads(p.read_text())
        except (ValueError, OSError):
            return None
        same = (old.get("inputs", {}).get("digest") == digest and old.get("code_sha256") == code_sha
                and old.get("split") == split and bool(old.get("smoke")) == bool(smoke))
        return old if same else None
    fam_exps = sorted({HOLM_SOURCE[h][0] for h in HOLM_FAMILY})
    want = list(exps)
    if split == "test":          # PV-9: on TEST a stale family member is re-analysed, never reused
        want += [e for e in fam_exps if e not in want and fresh(e) is None]
    for e in want:
        old = None if force and e in exps else fresh(e)
        if old is not None:
            log(f"{e}: up to date (skipped)")
            results[e] = old
            continue
        todo.append(e)
    if todo:
        notes: list = []
        _TABLES.clear()
        _TABLES.update(load_all(split, smoke, notes, src))
        sizes = {k: int(len(v)) for k, v in _TABLES.items()}
        log(f"inputs: {len(files)} files; rows {sizes}")
        run_cfg = dict(cfg)
        if crossed:
            L = window_loads(_TABLES)
            try:
                run_cfg["memberships"] = memberships(L["unit"].tolist(), split, out / "_memberships.json", workers,
                                                     test_freeze, confine) if len(L) else {}
            except Exception as e:                                          # noqa: BLE001
                notes.append(f"crossed bootstrap: memberships unavailable ({type(e).__name__}: {e})")
                run_cfg["memberships"] = {}
            P5 = _TABLES.get("e5_pairs")
            u5 = sorted(P5[P5["phase"] == "later"]["unit"].unique()) if P5 is not None and len(P5) else []
            try:
                run_cfg["memberships_e5"] = memberships(u5, split, out / "_memberships_e5.json", workers, test_freeze,
                                                        confine, pool_part="B") if u5 else {}
            except Exception as e:                                          # noqa: BLE001
                notes.append(f"crossed bootstrap (E5): memberships unavailable ({type(e).__name__}: {e})")
                run_cfg["memberships_e5"] = {}
        checks = {}
        if split in ("dev", "test") and not smoke:
            try:
                checks = dict(completeness=completeness(_TABLES, split), provenance=provenance(src, _TABLES),
                              primary_dates=primary_dates(_TABLES))
            except Exception as e:                                          # noqa: BLE001
                notes.append(f"input checks unavailable ({type(e).__name__}: {e})")
                checks = dict(error=f"{type(e).__name__}: {e}")
        jobs = [(e, run_cfg) for e in todo]
        if workers > 1 and len(jobs) > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=mp.get_context("fork")) as ex:
                done = list(ex.map(_worker, jobs))
        else:
            done = [_worker(j) for j in jobs]
        for e, res in done:
            res.update(experiment=e, split=split, smoke=smoke, settings=cfg,
                       generated=time.strftime("%Y-%m-%dT%H:%M:%S"), code_sha256=sio.file_sha256(__file__),
                       inputs=dict(digest=digest, sources={k: _rel(v) for k, v in src.items()}, files=len(files),
                                   rows=sizes))
            if "completeness" in checks:
                res["inputs_check"] = inputs_check(e, checks["completeness"], checks["provenance"],
                                                   checks.get("primary_dates"),
                                                   cfg.get("frozen_test_dates") if split == "test" else None)
                if split == "test" and not res["inputs_check"]["ok"]:
                    res.setdefault("notes", []).insert(0, f"{e}: INCOMPLETE or mixed-provenance TEST inputs; the "
                                                          "primary verdicts that read them are not 'supported'")
            res["notes"] = list(notes) + res.get("notes", [])
            write_json(out / f"{e}.json", res, confine)
            results[e] = res
            log(f"{e}: {len(res['macros'])} macros, {len(res['tables'])} tables, {len(res['hypotheses'])} "
                f"hypotheses, {len(res['notes'])} notes ({res['seconds']:.1f} s)")
    # round-3 review: Holm and the final verdicts always use every family member's newest summary in ``out``, not
    # only the experiments of this invocation (a subset run used to enter the others with p = 1)
    fam, stale = {}, []
    for e in fam_exps:
        if e in results:
            fam[e] = results[e]
        elif (out / f"{e}.json").is_file():
            try:
                old = json.loads((out / f"{e}.json").read_text())
            except (ValueError, OSError):
                continue
            if old.get("split") != split or bool(old.get("smoke")) != bool(smoke):
                continue
            if split == "test" and fresh(e) is None:     # PV-9: defensive (stale members were re-analysed above)
                stale.append(e)
                continue
            fam[e] = old
    hs = holm_summary(fam, split=split)
    if stale:
        hs["notes"].append(f"stale summaries (other inputs or code) not used in Holm: {stale}")
    hs.update(experiment="holm", split=split, smoke=smoke, settings=cfg, inputs=dict(digest=digest),
              family_sources={e: ("this run" if e in todo else "existing summary") for e in fam})
    if todo and checks:
        hs["input_checks"] = _clean(checks)
    elif (out / "holm.json").is_file():                     # nothing reanalysed: keep the last input checks
        try:
            prev = json.loads((out / "holm.json").read_text())
            if "input_checks" in prev:
                hs["input_checks"] = prev["input_checks"]
        except (ValueError, OSError):
            pass
    write_json(out / "holm.json", hs, confine)
    ex = example_summary()
    ex.update(experiment="example", split="none", settings=dict(source="theory closed form"))
    write_json(out / "example.json", ex, confine)
    import registry as R
    st = settings_summary(models_root=(R.SMOKE_DIR / "train" / "models") if smoke else None)
    st.update(experiment="settings", split="none")
    write_json(out / "settings.json", st, confine)
    if example_to_paper and out != SUMMARY:
        write_json(SUMMARY / "example.json", ex)          # split-independent theory example for the paper
        write_json(SUMMARY / "settings.json", st)
    log(f"wrote {out} in {time.perf_counter() - t0:.1f} s")
    results["holm"], results["example"], results["settings"] = hs, ex, st
    return results


def refusals_on_test(a: argparse.Namespace) -> list[str]:
    """Reasons to refuse a TEST analysis before the TEST gate (round-3 review; pre-freeze audit PV-8/F10)."""
    out = []
    if a.source:
        out.append("--source redirects inputs and is not allowed on TEST")
    if H5B_SIGN not in (1, -1):
        out.append("H5B_SIGN (the pre-registered sign of H5b) is not set; set it from the DEV pilot before the freeze")
    if a.n_boot is not None and int(a.n_boot) != N_BOOT:
        out.append(f"--n-boot {a.n_boot}: the TEST analysis uses the pre-registered {N_BOOT} replicates")
    if a.no_crossed:
        out.append("--no-crossed: the crossed-bootstrap sensitivity analysis is part of the TEST protocol")
    if a.smoke:
        out.append("--smoke reads driver smoke outputs and is not a TEST analysis")
    if a.out is not None:
        out.append("--out: the TEST summaries go to results/summary/ (read by make_numbers and the figures)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=("train", "dev", "test"))
    ap.add_argument("--smoke", action="store_true", help="read the drivers' smoke outputs, 500 bootstrap replicates")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--exp", default=",".join(EXPERIMENTS), help="comma-separated subset of e0..e7")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="summary directory (default by split)")
    ap.add_argument("--test-freeze", type=Path, default=None, help="frozen PREREG.md (default: the recorded freeze)")
    ap.add_argument("--no-crossed", action="store_true", help="skip the crossed-bootstrap sensitivity analysis")
    ap.add_argument("--source", action="append", default=[], metavar="EXP=DIR",
                    help="override one experiment's input directory (e.g. e2=results/raw/dev/e2_pilot)")
    ap.add_argument("--force", action="store_true", help="recompute even if the input digest is unchanged")
    a = ap.parse_args(argv)
    freeze = frozen_n = None
    if a.split == "test":
        # round-3 review: the TEST analysis reads only the drivers' standard locations, and needs the
        # pre-registered H5b sign (fixed from the DEV pilot, frozen with the code); pre-freeze audit PV-8/F10: no
        # option that changes the protocol (bootstrap replicates, the crossed sensitivity analysis, smoke inputs,
        # another output directory) is accepted on TEST. Every reason is printed before anything is read.
        refusals = refusals_on_test(a)
        if refusals:
            for msg in refusals:
                print(f"refused: {msg}", file=sys.stderr)
            return 3
        from slicemech.data.puffer_fetch import require_test_read_gate
        try:
            rec = require_test_read_gate(a.test_freeze)
        except PermissionError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 3
        freeze = rec.get("path", a.test_freeze)
        frozen_n = rec.get("test_dates")                  # pre-freeze audit F5: checked against each primary's inputs
        if frozen_n is None:
            print("refused: the freeze record has no test_dates (16 or 24)", file=sys.stderr)
            return 3
    exps = [e.strip() for e in a.exp.split(",") if e.strip()]
    bad = [e for e in exps if e not in ANALYSES]
    if bad:
        ap.error(f"unknown experiments {bad}")
    src = sources(a.split, a.smoke)
    for item in a.source:
        exp, _, path = item.partition("=")
        if exp not in src or not path:
            ap.error(f"--source expects EXP=DIR with EXP in {list(src)}, got {item!r}")
        p = Path(path) if Path(path).is_absolute() else ROOT / path
        if a.split != "test" and "test" in p.resolve().parts:
            ap.error("a non-TEST analysis may not read a 'test' directory")
        src[exp] = p
    sio.require_free_gb(3)
    run_analysis(a.split, a.out or summary_dir(a.split, a.smoke), smoke=a.smoke, src=src,
                 n_boot=a.n_boot or (SMOKE_N_BOOT if a.smoke else N_BOOT), exps=exps, workers=a.workers,
                 force=a.force, crossed=not a.no_crossed, test_freeze=freeze, example_to_paper=not a.smoke,
                 test_dates=frozen_n if a.split == "test" else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

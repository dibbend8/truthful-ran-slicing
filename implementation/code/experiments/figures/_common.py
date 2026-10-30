"""Shared style and I/O of the paper figure scripts (``rules/figure_style.md``; dataviz skill).

* Figures read only ``results/summary/*.json`` written by ``experiments/analyze.py`` (TEST: ``results/summary/``;
  ``--split dev``: ``results/summary/dev/``; ``--smoke``: ``results/summary/smoke/<split>/``). No number is typed
  into a plot script; labels, colours and markers come from ``spec/controllers.yaml``.
* Output: vector PDF (``pdf.fonttype = 42``) and a 300-dpi PNG, written for TEST to
  ``paper/expanded/figures/`` and ``results/figures/``, otherwise to ``results/figures/<split>/`` (smoke:
  ``results/figures/smoke/<split>/``) only, so DEV figures never reach the paper.
* IEEE sizes: column 3.5 in, figure* 7.16 in; all text 8 pt at final size; lines 1.2 pt; markers >= 4 pt; 95 % CI
  as thin error bars; light solid grid; one y-axis per panel; units in axis labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / "results" / "summary"
PAPER_FIG = ROOT / "paper" / "expanded" / "figures"
RESULTS_FIG = ROOT / "results" / "figures"
COL_W, FULL_W = 3.5, 7.16
INK, INK2, GRID = "#1f1f1f", "#555555", "#e3e3e3"
CLASS_ORDER = ("video", "urllc", "mmtc")
CLASS_NAME = {"video": "Video", "urllc": "URLLC-like", "mmtc": "mMTC-like"}
#: line styles per controller key (secondary encoding for grayscale print and colour-vision deficiency)
LINESTYLE = {"protected": "-", "fixed_price": "--", "unprotected": "-.", "unprotected_usage": ":",
             "unprotected_envelope": (0, (5, 1, 1, 1)), "meas_only": "--", "static": ":", "equal": (0, (1, 1)),
             "direct": "-.", "oracle": "-", "direct_truerates": (0, (3, 1)), "meas_only_strong": (0, (3, 1, 1, 1)),
             "kappa0": (0, (4, 2)), "retrain_measurement": "-", "retrain_stored": (0, (5, 2)),
             "unprotected_br": (0, (1, 2))}

_CTRL: dict = {}


def controllers() -> dict:
    if not _CTRL:
        _CTRL.update({c["key"]: c for c in yaml.safe_load(open(ROOT / "spec" / "controllers.yaml"))})
    return _CTRL


def style(key: str) -> dict:
    """Colour, marker, label and line style of a controller key (``spec/controllers.yaml``); unknown keys are gray."""
    c = controllers().get(key, {})
    color = c.get("color", "#9a9a9a")
    return dict(color=color, marker=c.get("marker", "o"), label=c.get("label", key), ls=LINESTYLE.get(key, "-"),
                mfc="white" if c.get("open") else color)


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "axes.edgecolor": "#8c8c8c", "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
        "lines.linewidth": 1.2, "lines.markersize": 4.5, "errorbar.capsize": 2.0, "legend.frameon": False,
        "legend.handlelength": 2.2, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "figure.dpi": 150})


def plain_log(ax, axis: str = "y", subs: tuple = (1.0, 2.0, 3.0, 5.0)) -> None:
    """Log axis with plain-number tick labels (no 10^k notation); ``subs`` are the labelled multiples per decade
    (fewer for an axis that spans several decades, so that tick labels do not collide)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=subs))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def wrap_label(label: str) -> str:
    """Break a ``spec/controllers.yaml`` label after its first comma (the label text is unchanged), so that long
    category tick labels do not take half of a column-wide figure."""
    return label.replace(", ", ",\n", 1)


def tiny_scale(top: float) -> tuple[int, float]:
    """Power of ten k and factor 10**-k that bring a positive top value (for example a numerical error bound of
    1e-8) into [1, 10), so that an axis in units of 10^k credits shows the bound instead of a flat zero."""
    import math
    if not top or top <= 0 or not math.isfinite(top):
        return 0, 1.0
    k = math.floor(math.log10(top))
    return k, 10.0 ** (-k)


def zero_line(ax) -> None:
    ax.axhline(0.0, color="#9a9a9a", lw=0.6, zorder=1)


def errorbar(ax, x, y, lo, hi, key: str, *, label: str | None = None, line: bool = True, xerr=None, **kw):
    """Points with 95 % CI bars in the controller's colour, marker and line style."""
    st = style(key)
    yerr = None
    if lo is not None and hi is not None:          # a point without a CI (fewer than MIN_DATES dates) gets no bar
        yerr = [[0.0 if b is None else max(0.0, a - b) for a, b in zip(y, lo)],
                [0.0 if b is None else max(0.0, b - a) for a, b in zip(y, hi)]]
    return ax.errorbar(x, y, yerr=yerr, xerr=xerr, color=st["color"], marker=st["marker"], mfc=st["mfc"],
                       ls=st["ls"] if line else "none", lw=1.2, ms=kw.pop("ms", 4.5), elinewidth=0.8,
                       label=st["label"] if label is None else label, zorder=3, **kw)


def parse_args(argv, description: str):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--split", default="test", choices=("train", "dev", "test"))
    ap.add_argument("--smoke", action="store_true", help="read results/summary/smoke/<split>/")
    ap.add_argument("--summary-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, nargs="*", default=None, help="output directories (default by split)")
    return ap.parse_args(argv)


def summary_dir(args) -> Path:
    if args.summary_dir:
        return Path(args.summary_dir)
    if args.smoke:
        return SUMMARY / "smoke" / args.split
    return SUMMARY if args.split == "test" else SUMMARY / args.split


def out_dirs(args) -> list[Path]:
    if args.out:
        return [Path(p) for p in args.out]
    if args.smoke:
        return [RESULTS_FIG / "smoke" / args.split]
    return [PAPER_FIG, RESULTS_FIG] if args.split == "test" else [RESULTS_FIG / args.split]


def load(sdir: Path, name: str) -> dict | None:
    p = Path(sdir) / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def table(S: dict | None, name: str) -> list:
    return list((S or {}).get("tables", {}).get(name) or [])


def save(fig, name: str, dirs) -> list[Path]:
    """Write ``<name>.pdf`` (vector) and ``<name>.png`` (300 dpi) to every directory."""
    out = []
    for d in dirs:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
            p = d / f"{name}.{ext}"
            fig.savefig(p, **kw)
            out.append(p)
    plt.close(fig)
    return out


def run(make, name: str, needs: tuple, argv=None, description: str = "") -> int:
    """CLI wrapper: load the summaries in ``needs``, call ``make(*summaries)``, save. Exit 2 when inputs are missing."""
    args = parse_args(argv, description or name)
    sdir = summary_dir(args)
    S = [load(sdir, n) for n in needs]
    if any(s is None for s in S):
        print(f"{name}: skipped, missing {[n for n, s in zip(needs, S) if s is None]} in {sdir}", file=sys.stderr)
        return 2
    setup()
    fig = make(*S)
    if fig is None:
        print(f"{name}: skipped, the summaries hold no data for this figure", file=sys.stderr)
        return 2
    for p in save(fig, name, out_dirs(args)):
        print(p)
    return 0

"""Fig. 12 (fig:nr): 5G-LENA replay per slice - delivered traffic within the packet delay budget, 95th-percentile
delay and RLC loss for Protected (truthful), the Unconstrained net (truthful) and the Unconstrained net with the
best-responding tenant (E7, descriptive).

Series styles and labels: ``spec/controllers.yaml`` keys ``protected``, ``unprotected`` (truthful; E7 bills the
fixed-priority price) and ``unprotected_br`` (round-3 review: the truthful case used to wear the usage-tariff style).

Data: ``results/summary/e7.json`` table ``per_slice_mean``: per case and slice class the mean over the replayed
windows (bars: smallest to largest window, not a confidence interval; E7 has no CIs). Delivered traffic in Mbit/s
(Mbit per one-second interval), delay in ms (median over intervals of the per-interval 95th percentile), loss in
percent of the PDCP SDU volume. Drops are always shown next to the delay.

Layout (2026-10-21, plotting only, prereg/DEVIATIONS.md item 1): drawn at the 9/10 text width at which results.tex
includes it (text 8 pt in print), and the log delay axis labels 1 and 3 per decade (with 1, 2, 3 and 5 per decade
the labels 20000/30000 and 200/300 touched); the y-labels and the class names have two lines (the class names
"URLLC-like" and "mMTC-like" touched below the panels).

    python experiments/figures/fig_e7_nr.py [--split test|dev] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_e7_nr"
WIDTH = C.FULL_W * 9 / 10           # results.tex: \resultfig[\dimexpr\textwidth*9/10\relax]{fig_e7_nr}
CASES = ("protected_truthful", "unconstrained_truthful", "unconstrained_br")
PANELS = (("delivered_pdb_mbps", "delivered", "Delivered within the PDB\n(Mbit/s)", False),
          ("delay_p95_ms", "delay", "95th-pct. delay\n(ms)", True),
          ("drop_pct", "drop", "RLC loss\n(%)", False))


def make(S: dict):
    import matplotlib.pyplot as plt
    rows = C.table(S, "per_slice_mean")
    cases = [c for c in CASES if any(r["arm"] == c for r in rows)]
    classes = [c for c in C.CLASS_ORDER if any(r["cls"] == c for r in rows)]
    if not cases or not classes:
        return None
    fig, axes = plt.subplots(1, len(PANELS), figsize=(WIDTH, 2.0), constrained_layout=True)
    off = {c: (j - (len(cases) - 1) / 2) * 0.18 for j, c in enumerate(cases)}
    for ax, (col, pre, ylabel, logy) in zip(axes, PANELS):
        for case in cases:
            rr = {r["cls"]: r for r in rows if r["arm"] == case}
            xs = [i + off[case] for i, c in enumerate(classes) if c in rr and rr[c].get(col) is not None]
            rs = [rr[c] for c in classes if c in rr and rr[c].get(col) is not None]
            if not rs:
                continue
            key, lab = rs[0]["key"], rs[0]["label"]
            st = C.style(key)
            y = [r[col] for r in rs]
            lo = [r.get(f"{pre}_lo", r[col]) if r.get(f"{pre}_lo") is not None else r[col] for r in rs]
            hi = [r.get(f"{pre}_hi", r[col]) if r.get(f"{pre}_hi") is not None else r[col] for r in rs]
            ax.errorbar(xs, y, yerr=[[max(0, a - b) for a, b in zip(y, lo)], [max(0, b - a) for a, b in zip(y, hi)]],
                        color=st["color"], marker=st["marker"], mfc=st["mfc"], ls="none", ms=4.5, elinewidth=0.8,
                        label=lab)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([C.CLASS_NAME[c].replace("-like", "-\nlike") for c in classes])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
            C.plain_log(ax, subs=(1.0, 3.0))
        else:
            ax.set_ylim(bottom=0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncols=len(lab))
    return fig


def main(argv=None) -> int:
    return C.run(make, NAME, ("e7",), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

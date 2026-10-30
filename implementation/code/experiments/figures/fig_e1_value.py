"""Fig. 6 (fig:value): value of reports (E1, H1a/H1b) - realized welfare of Protected (ours) minus that of the
report-free controller ('No reports' in spec/controllers.yaml) versus the measurement delay (and, when the sweeps
exist, versus the offered load and the video lead time).

Data: ``results/summary/e1.json`` tables ``value_by_delay`` (and ``value_by_load``, ``value_by_lead``): mean paired
difference over windows (credits per interval; positive means reports help) with the two-stage cluster-bootstrap
95 % CI across dates. One panel when only the delay sweep exists (column width), otherwise three panels (figure*).

    python experiments/figures/fig_e1_value.py [--split test|dev] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_e1_value"
#: results.tex includes the three-panel figure at 19/20 of the text width; drawn at that width so that the text is
#: 8 pt in print (2026-10-21, plotting only, prereg/DEVIATIONS.md item 1)
WIDE_W = C.FULL_W * 19 / 20
#: pre-freeze audit NUM-09: the comparator's name is its spec/controllers.yaml label (meas_only = 'No reports')
YLABEL = f"Welfare gain over\n{C.style('meas_only')['label']}\n(credits/interval)"
YLABEL_ICC = f"Gain over {C.style('meas_only')['label']}\n(credits/interval)"
PANELS = (("value_by_delay", "d", "Measurement delay (s)", True),
          ("value_by_load", "rho", "Offered load (fraction of capacity)", False),
          ("value_by_lead", "lead_s", "Video lead time (s)", False))


def make(S: dict, icc: bool = False):
    """icc=True: the 6-page variant (column width; delay and load panels only; the lead-time result is stated in the
    text). Same data and styling (2026-10-26, plotting only, prereg/DEVIATIONS.md item 1.14)."""
    import matplotlib.pyplot as plt
    panels = [(t, v, xl, lg) for t, v, xl, lg in PANELS if len(C.table(S, t)) >= (1 if t == "value_by_delay" else 2)]
    if icc:
        panels = [p for p in panels if p[0] != "value_by_lead"]
    if not panels or panels[0][0] != "value_by_delay":
        return None
    wide = len(panels) > 1
    size = (C.COL_W, 1.6) if icc else (WIDE_W if wide else C.COL_W, 2.0)
    fig, axes = plt.subplots(1, len(panels), figsize=size, squeeze=False, constrained_layout=True)
    for j, (ax, (tname, var, xlabel, logx)) in enumerate(zip(axes[0], panels)):
        rows = sorted(C.table(S, tname), key=lambda r: r[var])
        x = [r[var] for r in rows]
        C.zero_line(ax)
        C.errorbar(ax, x, [r["estimate"] for r in rows], [r["lo"] for r in rows], [r["hi"] for r in rows],
                   "protected")
        if logx:
            ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.minorticks_off()
        ax.set_xlabel(xlabel.replace(" (fraction of capacity)", "") if icc
                      else xlabel)
        if j == 0:
            ax.set_ylabel(YLABEL_ICC if icc else YLABEL)
        if wide:
            ax.set_title(f"({'abc'[j]})", loc="left")
    return fig


def main(argv=None) -> int:
    rc = C.run(make, NAME, ("e1",), argv, __doc__.splitlines()[0])
    return rc or C.run(lambda S: make(S, icc=True), NAME + "_icc", ("e1",), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

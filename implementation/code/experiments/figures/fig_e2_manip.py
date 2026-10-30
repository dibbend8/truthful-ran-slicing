"""Fig. 7 (b, c) (fig:manip): manipulation of unprotected designs (E2) - the Unconstrained net's own quota falls as
the own report rises at some grid steps, and honest slices lose on-time traffic when one tenant best-responds.
Panel (a) of fig:manip is ``fig_e2e3_gain.py`` (pre-freeze audit NUM-10: plotting-only script, frozen summaries).

Data: ``results/summary/e2.json`` tables ``nonmonotone_by_class`` ((b): per net and class, the percentage of the
tenant-intervals in which the own quota decreases at some step of the 513-point own-report grid, column
``share_states_nonmonotone``, the per-class version of macro EtwoNonmonoStates that the text quotes; exact counts,
no CI; Protected (ours) is the negative control, 0 by construction and labelled "0" because a zero bar is invisible;
until 2026-10-21 the panel showed ``pct_gridsteps_decrease``, the share of grid steps, prereg/DEVIATIONS.md item 1)
and ``victim_contrasts`` ((c): metric ``d_on_time_victims``, the honest tenants' on-time Mbit per
interval with the deviator best-responding minus all truthful, summed over the honest tenants and averaged over the
deviator roles of a window; negative means the honest slices lose; mean over the E2 replay windows with the
two-stage cluster-bootstrap 95 % CI across dates; the Unconstrained net is the H2c primary cell, the other designs
are secondary).

    python experiments/figures/fig_e2_manip.py [--split test|dev] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_e2_manip"
NETS = ("protected", "unprotected")
DESIGNS = ("fixed_price", "unprotected", "unprotected_usage", "unprotected_envelope")


def make(S: dict):
    import matplotlib.pyplot as plt
    nm = [r for r in C.table(S, "nonmonotone_by_class") if r.get("arm") in NETS]
    vc = [r for r in C.table(S, "victim_contrasts")
          if r.get("metric") == "d_on_time_victims" and r.get("arm") in DESIGNS and r.get("estimate") is not None]
    if not nm and not vc:
        return None
    fig = plt.figure(figsize=(C.COL_W, 3.9), layout="constrained")     # results.tex: 49/100 textwidth = COL_W
    top, bot = fig.subfigures(2, 1, height_ratios=(1.0, 1.1))     # independent layouts: (c) has long tick labels
    a, b = top.subplots(), bot.subplots()
    classes = [c for c in C.CLASS_ORDER if any(r["cls"] == c for r in nm)]
    nets = [n for n in NETS if any(r["arm"] == n for r in nm)]
    width = 0.8 / max(1, len(nets))
    tops = []
    for j, net in enumerate(nets):
        st = C.style(net)
        ys = [100.0 * (next((r.get("share_states_nonmonotone") for r in nm if r["arm"] == net and r["cls"] == c),
                            0.0) or 0.0) for c in classes]
        xs = [i + (j - (len(nets) - 1) / 2) * width for i in range(len(classes))]
        a.bar(xs, ys, width=width * 0.92, color=st["color"], label=st["label"], zorder=3,
              hatch=None if net == "unprotected" else "//", edgecolor="white", linewidth=0.5)
        tops += ys
        for x, y in zip(xs, ys):
            if y == 0.0:                     # a zero bar is invisible: a short baseline mark in the net's colour
                a.hlines(0.0, x - width * 0.42, x + width * 0.42, color=st["color"], lw=2.0, zorder=4)
            # exact counts (no CI): print each share; this also separates the near-zero URLLC-like bar from 0
            a.annotate("0" if y == 0.0 else f"{y:.1f}", (x, y), xytext=(0, 2.0), textcoords="offset points",
                       ha="center", va="bottom", color=C.INK, zorder=4)
    a.set_xticks(range(len(classes)))
    a.set_xticklabels([C.CLASS_NAME[c] for c in classes])
    a.set_ylabel("Tenant-intervals with a\nquota decrease (%)")
    a.set_title("(b)", loc="left")
    if nets:
        a.set_ylim(0.0, 1.45 * max(tops + [1e-9]))          # headroom for the legend inside the panel
        a.legend(loc="upper right", ncols=1)
    rows = [r for d in DESIGNS for r in vc if r["arm"] == d]
    b.axvline(0.0, color="#9a9a9a", lw=0.6, zorder=1)                 # zero line of the horizontal dot plot
    for i, r in enumerate(rows):
        st = C.style(r["arm"])
        lo, hi = r.get("lo"), r.get("hi")
        xerr = None if lo is None or hi is None else [[max(0.0, r["estimate"] - lo)], [max(0.0, hi - r["estimate"])]]
        b.errorbar([r["estimate"]], [i], xerr=xerr, color=st["color"], marker=st["marker"], mfc=st["mfc"],
                   ls="none", ms=4.5, elinewidth=0.8, zorder=3)
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([C.wrap_label(C.style(r["arm"])["label"]) for r in rows])
    b.invert_yaxis()
    b.set_xlabel("Change in honest tenants'\non-time traffic (Mbit/interval)")
    b.set_title("(c)", loc="left")
    return fig


def main(argv=None) -> int:
    return C.run(make, NAME, ("e2",), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

"""E4b frontier (decision D7; no entry in the outline's figure list yet - pre-freeze audit NUM-10; fig:cost is
``fig_e4_cost.py``): realized welfare versus honest payment - scaling the trained report sensitivity kappa
by s trades welfare against the payment honest tenants make; equal, static and direct are references.

Data: ``results/summary/e4.json`` table ``frontier``: per design, mean realized welfare and mean honest payment
(both credits per interval, truthful reports) with two-stage cluster-bootstrap 95 % CIs across dates. The protected
points are joined in the order of s and each is labelled with its s.

Layout (2026-10-21, plotting only, prereg/DEVIATIONS.md item 1): results.tex includes this figure as panel (c) of
fig:cost at 17/50 of the text width next to ``fig_e4_cost`` (31/50); it is drawn at that final width and the height
of ``fig_e4_cost``, so that its text is 8 pt in print (it was drawn at 3.5 in and shrunk to 70 %), it carries the
panel label (c) of the caption, the payment axis is logarithmic (the Direct optimizer's payment is about five times
the others and squeezed every other point into the left sixth of a linear axis), and the legend sits below the plot.

    python experiments/figures/fig_e4_frontier.py [--split test|dev] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_e4_frontier"
REFS = ("equal", "static", "direct")
WIDTH = C.FULL_W * 17 / 50          # results.tex: \resultfig[\dimexpr\textwidth*17/50\relax]{fig_e4_frontier}
HEIGHT = 3.5                        # the height of fig_e4_cost (one row of fig:cost)


def _err(rows, k):
    def d(v):                                       # no bar for a point without a CI (fewer than MIN_DATES dates)
        return 0.0 if v is None else max(0.0, v)
    return [[d(None if r.get(f"{k}_lo") is None else r[k] - r[f"{k}_lo"]) for r in rows],
            [d(None if r.get(f"{k}_hi") is None else r[f"{k}_hi"] - r[k]) for r in rows]]


def make(S: dict):
    import matplotlib.pyplot as plt
    rows = [r for r in C.table(S, "frontier") if r.get("welfare") is not None and r.get("payment") is not None]
    prot = sorted((r for r in rows if r["arm"] == "protected"), key=lambda r: r["kappa_scale"])
    if not prot:
        return None
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), constrained_layout=True)
    st = C.style("protected")
    ax.errorbar([r["payment"] for r in prot], [r["welfare"] for r in prot], xerr=_err(prot, "payment"),
                yerr=_err(prot, "welfare"), color=st["color"], marker=st["marker"], ls="-", lw=1.2, ms=4.5,
                elinewidth=0.8, label=f"{st['label']},\n\u03ba scaled by a factor", zorder=3)
    for arm in REFS:
        rr = [r for r in rows if r["arm"] == arm]
        if rr:
            s2 = C.style(arm)
            ax.errorbar([r["payment"] for r in rr], [r["welfare"] for r in rr], xerr=_err(rr, "payment"),
                        yerr=_err(rr, "welfare"), color=s2["color"], marker=s2["marker"], ls="none", ms=5,
                        elinewidth=0.8, label=s2["label"], zorder=3, mec=s2["color"])
    ax.set_xscale("log")
    C.plain_log(ax, "x", subs=(1.0, 2.0, 5.0))
    pays = [v for r in rows for v in (r["payment"], r.get("payment_lo"), r.get("payment_hi")) if v and v > 0]
    ax.set_xlim(0.6 * min(pays), 1.3 * max(pays))          # room for the s labels left of the points
    # label every s: the right-most protected point on the right, the others on the left (points of nearly the same
    # payment then do not share a side), offset away from the vertical CI bar
    xmax = max(r["payment"] for r in prot)
    for r in prot:
        right = r["payment"] == xmax
        ax.annotate(f"factor {r['kappa_scale']:g}", (r["payment"], r["welfare"]), xytext=(5 if right else -5, 0),
                    textcoords="offset points", ha="left" if right else "right", va="center", color=C.INK,
                    zorder=4)
    ax.set_xlabel("Price paid by truthful tenants\n(credits/interval, log scale)")
    ax.set_ylabel("Realized welfare (credits/interval)")
    ax.set_title("(c)", loc="left")
    fig.legend(loc="outside lower center", ncols=1, handlelength=1.8)
    return fig


def main(argv=None) -> int:
    return C.run(make, NAME, ("e4",), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

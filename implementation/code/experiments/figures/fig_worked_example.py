"""Fig. 1 (fig:example): a misreport moves the priority vector and the quota; the fixed-priority price leaves a gain,
the full-decision price leaves none.

Data: ``results/summary/example.json`` (closed-form example E3 of the theory, written by ``experiments/analyze.py``).
(a) own quota (PRB) versus own report (Mbit/s) with the priority recomputed at every report and fixed at its value for
the report of 20; (b) utility at true demand 8 (credits per interval) under no price, the fixed-priority price and the
full-decision price. In this example the fixed-priority price is 0 at every report, so its utility curve lies on the
no-price curve (drawn dashed over it).

    python experiments/figures/fig_worked_example.py [--split test|dev] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_worked_example"
NO_PRICE = "#9a9a9a"


#: plain-language labels for the 6-page version (2026-10-26, plotting only, prereg/DEVIATIONS.md item 1.15)
PLAIN = dict(a_title="(a) PRBs of tenant 1", a_rec="Priority recomputed", a_fix="Priority held at report {lie:g}",
             a_y="PRBs of tenant 1", x="Report of tenant 1 (Mbit/s)", truth="honest", lie="exaggerated",
             b_title="(b) Value minus price", none="No price", fix="Fixed-priority price",
             full="Full-decision price (ours)", b_y="Tenant 1, true demand {theta:g} Mbit/s\n(credits/interval)",
             gain="gain {g:.2f}", truthful="honest report")


def make(S: dict, plain: bool = False):
    import matplotlib.pyplot as plt
    cur = C.table(S, "curves")
    pts = (C.table(S, "points") or [None])[0]
    if not cur or pts is None:
        return None
    u = [r["report"] for r in cur]
    full, fixed = C.style("protected"), C.style("fixed_price")
    fig, (a, b) = plt.subplots(1, 2, figsize=(C.COL_W, 2.45), constrained_layout=True)
    L = PLAIN if plain else None
    a.plot(u, [r["quota_full"] for r in cur], color=full["color"], ls="-", lw=1.2,
           label=L["a_rec"] if L else "Recomputed")
    a.plot(u, [r["quota_fixed_at_lie"] for r in cur], color=fixed["color"], ls="--", lw=1.2,
           label=L["a_fix"].format(lie=pts["lie"]) if L else f"Fixed at report {pts['lie']:g}")
    for x, y, txt, ha in ((pts["theta"], pts["quota_truth"], L["truth"] if L else "truth", "left"),
                          (pts["lie"], pts["quota_lie"], L["lie"] if L else "lie", "right")):
        a.plot([x], [y], "o", color=full["color"], ms=4.5, mec="white", mew=0.8, zorder=4)
        a.annotate(txt, (x, y), xytext=(0, -11), textcoords="offset points", ha=ha, color=C.INK2)
    a.set_xlabel(L["x"] if L else "Own report (Mbit/s)")
    a.set_ylabel(L["a_y"] if L else "Own quota (PRB)")
    a.set_xlim(0, 1.05 * pts["lie"])
    a.set_ylim(0, None)
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.6,
             title=L["a_title"] if L else "(a) Priority", title_fontsize=8, alignment="left")
    b.plot(u, [r["utility_none"] for r in cur], color=NO_PRICE, ls="-", lw=2.2, label=L["none"] if L else "No price")
    b.plot(u, [r["utility_fixed"] for r in cur], color=fixed["color"], ls="--", lw=1.2,
           label=L["fix"] if L else "Fixed-priority")
    b.plot(u, [r["utility_full"] for r in cur], color=full["color"], ls="-", lw=1.2,
           label=L["full"] if L else "Full-decision")
    x = pts["lie"]
    b.annotate("", xy=(x, pts["utility_fixed_lie"]), xytext=(x, pts["utility_fixed_truth"]),
               arrowprops=dict(arrowstyle="->", color=C.INK, lw=0.8))
    b.axhline(pts["utility_fixed_truth"], color=C.GRID, lw=0.8, ls="-", zorder=0)
    b.annotate(L["gain"].format(g=pts["gain_fixed"]) if L else f"gain {pts['gain_fixed']:.2f}",
               (x, pts["utility_fixed_truth"]), xytext=(-3, -3),
               textcoords="offset points", ha="right", va="top", color=C.INK)
    xt = pts["theta"]
    yt = min(cur, key=lambda r: abs(r["report"] - xt))["utility_full"]
    b.plot([xt], [yt], "o", color=full["color"], ms=4.5, mec="white", mew=0.8, zorder=4)
    b.annotate(L["truthful"] if L else "truthful report", (xt, yt), xytext=(0, -11), textcoords="offset points",
               ha="left", color=C.INK2)
    b.set_xlabel(L["x"] if L else "Own report (Mbit/s)")
    # from the summary (rule 10); unit as in the caption of fig:example (2026-10-21, plotting only, DEVIATIONS item 1)
    b.set_ylabel(L["b_y"].format(theta=pts["theta"]) if L
                 else f"Utility at demand {pts['theta']:g} Mbit/s\n(credits/interval)")
    b.set_xlim(0, 1.05 * pts["lie"])
    b.set_ylim(0, None)
    b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.6,
             title=L["b_title"] if L else "(b) Price", title_fontsize=8, alignment="left")
    return fig


def make_plain(S: dict):
    """6-page variant: same data, plain labels, legends inside the axes (2026-10-26, plotting only, DEVIATIONS 1.15)."""
    import matplotlib.pyplot as plt
    cur = C.table(S, "curves")
    pts = (C.table(S, "points") or [None])[0]
    if not cur or pts is None:
        return None
    u = [r["report"] for r in cur]
    full, fixed = C.style("protected"), C.style("fixed_price")
    fig, (a, b) = plt.subplots(1, 2, figsize=(C.COL_W, 1.95), constrained_layout=True)
    a.plot(u, [r["quota_full"] for r in cur], color=full["color"], ls="-", lw=1.2, label="priority recomputed")
    a.plot(u, [r["quota_fixed_at_lie"] for r in cur], color=fixed["color"], ls="--", lw=1.2,
           label=f"priority held at {pts['lie']:g}")
    a.plot([pts["theta"]], [pts["quota_truth"]], "o", color=full["color"], ms=4.5, mec="white", mew=0.8, zorder=4)
    a.annotate("truthful", (pts["theta"], pts["quota_truth"]), xytext=(2, -11), textcoords="offset points",
               ha="left", color=C.INK2)
    a.plot([pts["lie"]], [pts["quota_lie"]], "o", color=full["color"], ms=4.5, mec="white", mew=0.8, zorder=4)
    a.set_title("(a) PRBs of tenant 1", loc="left")
    a.set_ylabel("PRB")
    a.set_xlabel("Report (Mbit/s)")
    a.set_xlim(0, 1.05 * pts["lie"])
    a.set_ylim(0, None)
    a.legend(loc="lower right", handlelength=1.4, fontsize=6.5, borderaxespad=0.2)
    b.plot(u, [r["utility_none"] for r in cur], color=NO_PRICE, ls="-", lw=2.2, label="no price")
    b.plot(u, [r["utility_fixed"] for r in cur], color=fixed["color"], ls="--", lw=1.2, label="fixed-priority price")
    b.plot(u, [r["utility_full"] for r in cur], color=full["color"], ls="-", lw=1.2, label="full-decision price")
    x = pts["lie"]
    xt = pts["theta"]
    # the gain is measured from the truthful report on the fixed-priority (= no-price) curve: mark that point and draw
    # the reference level from it (r3 layman review: the arrow used to start from an unlabeled gridline)
    b.plot([xt], [pts["utility_fixed_truth"]], "o", color=fixed["color"], ms=4.5, mec="white", mew=0.8, zorder=4)
    b.plot([xt, x], [pts["utility_fixed_truth"]] * 2, color=C.INK2, lw=0.8, ls=":", zorder=2)
    b.annotate("truthful", (xt, pts["utility_fixed_truth"]), xytext=(-2, 5), textcoords="offset points", ha="right",
               color=C.INK2)
    b.annotate("", xy=(x, pts["utility_fixed_lie"]), xytext=(x, pts["utility_fixed_truth"]),
               arrowprops=dict(arrowstyle="->", color=C.INK, lw=0.8))
    # label below the dotted truthful level, clear of the grey curve (r3 layman/fitter note)
    b.annotate(f"gain {pts['gain_fixed']:.2f}", (x, pts["utility_fixed_truth"]), xytext=(-3, -4),
               textcoords="offset points", ha="right", va="top", color=C.INK)
    yt = min(cur, key=lambda r: abs(r["report"] - xt))["utility_full"]
    b.plot([xt], [yt], "o", color=full["color"], ms=4.5, mec="white", mew=0.8, zorder=4)
    b.set_title("(b) Value minus price", loc="left")
    b.set_ylabel("Credits/interval")
    b.set_xlabel("Report (Mbit/s)")
    b.set_xlim(0, 1.05 * pts["lie"])
    b.set_ylim(0, None)
    b.legend(loc="lower right", handlelength=1.4, fontsize=6.5, borderaxespad=0.2)
    return fig


def main(argv=None) -> int:
    rc = C.run(make, NAME, ("example",), argv, __doc__.splitlines()[0])
    return rc or C.run(make_plain, NAME + "_icc", ("example",), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

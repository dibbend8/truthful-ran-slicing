"""Fig. 9 (a, b) (fig:cost): cost of protection (E4) in units an operator understands - what an honest tenant pays
under the full-decision price (fixed-priority part plus premium) and how the realized welfare of every controller
differs from that of Protected (ours). Panel (c) of fig:cost is ``fig_e4_frontier.py``.
(Pre-freeze audit NUM-10: plotting-only script reading the frozen summaries.)

Data: ``results/summary/e4.json`` table ``payments`` ((a): per tenant class, the honest payment of Protected (ours)
in credits per interval, stacked as the fixed-priority part p_fix plus the premium T = p - p_fix; truthful reports)
and ``results/summary/e1.json`` table ``contrasts_main`` ((b): the paired per-window welfare contrast of every
controller with Protected (ours), E1 truthful runs of the main configuration (the runs behind ``e4.json``
``welfare_by_arm``), seeds averaged within a window, with the paired two-stage cluster-bootstrap 95 % CI across
dates; the table holds Protected minus the controller, drawn with the sign flipped, so that a positive value means
that the controller's welfare is higher than that of Protected (ours), the sign of the H4a gap and of fig:scaling (b);
x-axis linear within +-0.1 credits per interval and logarithmic beyond, because the differences range from about
0.03 to about 5 credits per interval). Until 2026-10-21 panel (b) showed the welfare levels of ``welfare_by_arm``
with date-level CIs, which hid the paired comparison (review/open_items.md item 26; prereg/DEVIATIONS.md item 1).

Size: results.tex includes this figure at 31/50 of the text width next to ``fig_e4_frontier`` (17/50); both are
drawn at that final width and the same height, so that the text is 8 pt in print (rules/figure_style.md rule 8);
(a) and (b) are stacked because side by side the class names and the legend of (a) do not fit.

    python experiments/figures/fig_e4_cost.py [--split test|dev] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_e4_cost"
FIXED_PART = "#b5b5b5"
WIDTH = C.FULL_W * 31 / 50          # results.tex: \resultfig[\dimexpr\textwidth*31/50\relax]{fig_e4_cost}
HEIGHT = 3.5                        # the same height as fig_e4_frontier (one row of fig:cost)
LINTHRESH = 0.1                     # credits per interval: linear part of the symmetric-log axis of (b)


def make(S4: dict, S1: dict):
    import matplotlib.pyplot as plt
    pay = [r for r in C.table(S4, "payments") if r.get("cls") in C.CLASS_ORDER and r.get("payment") is not None]
    con = [r for r in C.table(S1, "contrasts_main") if r.get("arm_a") == "protected" and r.get("estimate") is not None]
    if not pay and not con:
        return None
    fig = plt.figure(figsize=(WIDTH, HEIGHT), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=(1.0, 1.55))     # independent layouts: (b) has long tick labels
    a, b = top.subplots(), bot.subplots()
    classes = [c for c in C.CLASS_ORDER if any(r["cls"] == c for r in pay)]
    st = C.style("protected")
    rows = [next(r for r in pay if r["cls"] == c) for c in classes]
    fixed = [r.get("fixed_part") if r.get("fixed_part") is not None else r["payment"] for r in rows]
    prem = [r.get("premium") or 0.0 for r in rows]
    x = range(len(classes))
    a.bar(x, fixed, width=0.6, color=FIXED_PART, label="Fixed-priority price", zorder=3, edgecolor="white", lw=0.5)
    a.bar(x, prem, bottom=fixed, width=0.6, color=st["color"], label="Premium", zorder=3, edgecolor="white", lw=0.5,
          hatch="//")
    a.set_xticks(list(x))
    a.set_xticklabels([C.CLASS_NAME[c] for c in classes])
    a.set_ylabel("Price paid by\ntruthful tenants\n(credits/interval)")
    a.set_title(f"(a) {st['label']}", loc="left")
    if classes:
        a.set_ylim(0.0, 1.45 * max(f + p for f, p in zip(fixed, prem)))     # headroom for the legend
        a.legend(loc="upper center", ncols=2)
        a.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    # (b) paired differences: controller minus Protected (ours) = -(Protected minus controller)
    d = sorted(({"arm": r["arm_b"], "est": -r["estimate"],
                 "lo": None if r.get("hi") is None else -r["hi"],
                 "hi": None if r.get("lo") is None else -r["lo"]} for r in con), key=lambda r: r["est"], reverse=True)
    b.axvline(0.0, color="#9a9a9a", lw=0.6, zorder=1)
    for i, r in enumerate(d):
        s2 = C.style(r["arm"])
        xerr = None if r["lo"] is None or r["hi"] is None else [[max(0.0, r["est"] - r["lo"])],
                                                                [max(0.0, r["hi"] - r["est"])]]
        b.errorbar([r["est"]], [i], xerr=xerr, color=s2["color"], marker=s2["marker"], mfc=s2["mfc"], ls="none",
                   ms=4.5, elinewidth=0.8, zorder=3)
    b.set_xscale("symlog", linthresh=LINTHRESH, linscale=1.0)
    vals = [v for r in d for v in (r["lo"], r["hi"], r["est"]) if v is not None]
    left, right = 2.0 * min(vals + [-LINTHRESH]), 2.0 * max(vals + [LINTHRESH])     # 0.3 decade of margin
    b.set_xlim(left, right)
    ticks = [t for t in (-10.0, -1.0, -LINTHRESH, 0.0, LINTHRESH, 1.0, 10.0) if left <= t <= right]
    b.set_xticks(ticks)
    b.set_xticklabels([f"{t:g}".replace("-", "\u2212") for t in ticks])
    b.xaxis.set_minor_locator(plt.NullLocator())
    b.set_yticks(range(len(d)))
    b.set_yticklabels([C.style(r["arm"])["label"] for r in d])
    b.invert_yaxis()
    b.set_xlabel(f"Welfare minus that of {st['label']}\n(credits/interval; log scale beyond \u00b1{LINTHRESH:g})")
    b.set_title("(b)", loc="left")
    return fig


def main(argv=None) -> int:
    return C.run(make, NAME, ("e4", "e1"), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

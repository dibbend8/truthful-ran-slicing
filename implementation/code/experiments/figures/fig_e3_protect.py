"""Fig. 8 (a, c) (fig:protect): protection by the full-decision price (E3) - no tested report gains more than the
numerical error of the computed price, and the premium that buys this is a share of the payment. Panel (b) of
fig:protect is ``fig_e3_bandits.py`` (pre-freeze audit NUM-10: plotting-only script, frozen summaries).

Data: ``results/summary/e3.json`` tables ``h3a_by_design`` ((a): per monotone design and class, the largest computed
deviation gain over every scored tenant-interval, credits per interval, positive means a misreport pays; the solid
line is the largest guaranteed error bound e of the computed price, the dashed line the certificate 2e; the H3a
allowance is the larger 1e-8 pi_i mu_i Delta, which bounds the gain plus its certificate (orchestrator, 2026-10-20)) and ``premium_by_class`` ((c): the premium T(theta) of an honest tenant as a percentage of its
full-decision payment, per class and over all tenants; H3c defines it relative to the payment, whereas the outline
draft said 'of declared value': reported in the pre-freeze audit, NUM-10).

Layout (2026-10-21, plotting only, prereg/DEVIATIONS.md item 1): the legend of (a) is a figure legend above the
panels (as an axes legend it squeezed both panels into a third of the width, the class names overlapped and the
text shrank below 8 pt in print), and the y-axis of (a) is in units of 10^k credits per interval with limits that
include 2e, so that the error bound e and the certificate 2e (about 1e-8) are visible lines above the zero gains
instead of lying on the zero line.

    python experiments/figures/fig_e3_protect.py [--split test|dev] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_e3_protect"
ORDER = ("protected", "kappa0", "static", "equal", "direct")


def make(S: dict):
    import matplotlib.pyplot as plt
    h = [r for r in C.table(S, "h3a_by_design") if r.get("max_gain") is not None and int(r.get("n", 3)) == 3]
    pr = [r for r in C.table(S, "premium_by_class") if r.get("premium_pct_of_payment") is not None]
    if not h and not pr:
        return None
    fig, (a, b) = plt.subplots(2, 1, figsize=(C.COL_W, 4.2), constrained_layout=True,
                               gridspec_kw=dict(height_ratios=(1.3, 1.0)))
    classes = [c for c in C.CLASS_ORDER if any(r["cls"] == c for r in h)]
    arms = [x for x in ORDER if any(r["arm"] == x for r in h)] + sorted({r["arm"] for r in h} - set(ORDER))
    errs = [r.get("max_err") for r in h if r.get("max_err") is not None]
    e = max(errs) if errs else None
    k, f = C.tiny_scale(max([2 * e if e else 0.0] + [abs(r["max_gain"]) for r in h]))
    width = 0.8 / max(1, len(arms))
    for j, arm in enumerate(arms):
        rr = [next((r for r in h if r["arm"] == arm and r["cls"] == c), None) for c in classes]
        xs = [i + (j - (len(arms) - 1) / 2) * width for i, r in enumerate(rr) if r]
        st = C.style(arm)
        a.plot(xs, [f * r["max_gain"] for r in rr if r], ls="none", marker=st["marker"], color=st["color"],
               mfc=st["mfc"], ms=4.5, label=st["label"], zorder=3)
    if e is not None:
        a.axhline(f * e, color=C.INK2, lw=0.8, ls="-", zorder=2, label="Error bound e")
        a.axhline(f * 2 * e, color=C.INK2, lw=0.8, ls="--", zorder=2, label="Twice the error bound, 2e")
    a.axhline(0.0, color="#9a9a9a", lw=0.6, zorder=1)
    a.set_xticks(range(len(classes)))
    a.set_xticklabels([C.CLASS_NAME[c] for c in classes])
    a.grid(axis="x", visible=False)
    vals = [f * r["max_gain"] for r in h] + ([f * 2 * e] if e else [])
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    pad = 0.15 * ((hi - lo) or 1.0)
    a.set_ylim(lo - pad, hi + pad)
    unit = "credits/interval" if k == 0 else f"10$^{{{k}}}$ credits/interval"
    a.set_ylabel(f"Largest deviation gain\n({unit})")
    a.set_title("(a)", loc="left")
    if h:
        fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncols=2, handlelength=1.6,
                   columnspacing=1.0)
    cls_b = [c for c in C.CLASS_ORDER if any(r["cls"] == c for r in pr)] + (["all"] if any(r["cls"] == "all"
                                                                                           for r in pr) else [])
    st = C.style("protected")
    b.barh(range(len(cls_b)), [next(r["premium_pct_of_payment"] for r in pr if r["cls"] == c) for c in cls_b],
           height=0.6, color=st["color"], zorder=3)
    b.set_yticks(range(len(cls_b)))
    b.set_yticklabels([C.CLASS_NAME.get(c, "All tenants") for c in cls_b])
    b.invert_yaxis()
    b.set_xlabel("Premium (% of price)")
    b.set_title(f"(b) {st['label']}", loc="left")
    return fig


def main(argv=None) -> int:
    return C.run(make, NAME, ("e3",), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

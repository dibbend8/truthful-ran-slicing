"""Fig. 7/8 (fig:manip a, fig:protect): deviation gain versus the report multiple, per price rule - under the
full-decision price no multiple pays, under the fixed-priority price inflation does.

Data: ``results/summary/e3.json`` table ``gain_by_mult`` (E3 driver ``ident``: the protected map on every fourth
scored interval, report min(m theta, C) against truthful others): mean per-interval gain of the deviating tenant over
truth (credits per interval; positive means the misreport pays) with the two-stage cluster-bootstrap 95 % CI across
dates, one panel per tenant class (each panel has its own y-scale).

    python experiments/figures/fig_e2e3_gain.py [--split test|dev] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_e2e3_gain"
ARMS = ("protected", "fixed_price")


def make(S: dict, letters: bool = False):
    import matplotlib.pyplot as plt
    rows = C.table(S, "gain_by_mult")
    classes = [c for c in C.CLASS_ORDER if any(r["cls"] == c for r in rows)]
    if not classes:
        return None
    fig, axes = plt.subplots(1, len(classes), figsize=(C.FULL_W, 2.0), squeeze=False, constrained_layout=True)
    for j, (ax, c) in enumerate(zip(axes[0], classes)):
        C.zero_line(ax)
        for arm in ARMS:
            rr = sorted((r for r in rows if r["cls"] == c and r["arm"] == arm), key=lambda r: r["mult"])
            if rr:
                C.errorbar(ax, [r["mult"] for r in rr], [r["mean_gain"] for r in rr], [r["lo"] for r in rr],
                           [r["hi"] for r in rr], arm,
                           # 6-page variant: both curves use the protected network, so the legend names the price
                           # rule (2026-10-26, plotting only, DEVIATIONS 1.16)
                           label=({"protected": "full-decision price (ours)", "fixed_price": "fixed-priority price"}[arm]
                                  if letters else None))          # spec/controllers.yaml labels (round-3 review)
        # expanded paper: this figure is panel (a) of fig:manip; the 6-page variant letters each class panel
        # (2026-10-21, plotting only, DEVIATIONS item 1.13)
        ax.set_title((f"({'abc'[j]}) " if letters else ("(a) " if j == 0 else "")) + C.CLASS_NAME[c], loc="left")
        ax.set_xlabel("Report / true demand")
        if j == 0:
            ax.set_ylabel("Deviation gain (credits/interval)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper right", ncols=len(lab))
    return fig


def main(argv=None) -> int:
    rc = C.run(make, NAME, ("e3",), argv, __doc__.splitlines()[0])
    # 6-page ICC variant with panels (a)-(c), same data
    return rc or C.run(lambda S: make(S, letters=True), NAME + "_icc", ("e3",), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

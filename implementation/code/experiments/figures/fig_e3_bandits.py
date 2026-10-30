"""Fig. 8b (fig:protect b): report multipliers of learning tenants per episode - under the full-decision price the
tenants converge to truth (multiplier 1), under unprotected designs they do not.

Data: ``results/summary/e3.json`` table ``bandit_trajectories`` (E2 driver bandit runs, EXP3 and epsilon-greedy over
the multipliers 0.5..2): mean played multiplier per episode bin over runs and algorithms, one panel per tenant
class, one line per design (``spec/controllers.yaml`` labels and colours).

    python experiments/figures/fig_e3_bandits.py [--split test|dev] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_e3_bandits"
ORDER = ("protected", "fixed_price", "unprotected", "unprotected_usage")


def make(S: dict):
    import matplotlib.pyplot as plt
    rows = C.table(S, "bandit_trajectories")
    classes = [c for c in C.CLASS_ORDER if any(r["cls"] == c for r in rows)]
    if not classes:
        return None
    arms = [a for a in ORDER if any(r["arm"] == a for r in rows)]
    fig, axes = plt.subplots(1, len(classes), figsize=(C.FULL_W, 2.0), squeeze=False, sharey=True,
                             constrained_layout=True)
    for j, (ax, c) in enumerate(zip(axes[0], classes)):
        ax.axhline(1.0, color="#9a9a9a", lw=0.6, zorder=1)
        for arm in arms:
            rr = sorted((r for r in rows if r["cls"] == c and r["arm"] == arm), key=lambda r: r["episode"])
            if rr:
                st = C.style(arm)
                # markers every few bins (figure_style.md rule 2: every series has a marker and a line style)
                ax.plot([r["episode"] + 0.5 * r.get("bin_width", 1) for r in rr], [r["mean_mult"] for r in rr],
                        color=st["color"], ls=st["ls"], lw=1.2, label=st["label"], marker=st["marker"],
                        mfc=st["mfc"], ms=4.5, markevery=max(1, len(rr) // 6))
        # the caption of fig:protect calls these class panels (c) (2026-10-21, plotting only, DEVIATIONS item 1)
        ax.set_title(("(c) " if j == 0 else "") + C.CLASS_NAME[c], loc="left")
        ax.set_xlabel("Episode")
        if j == 0:
            ax.set_ylabel("Report multiplier\n(1 = truth)")     # one line touched the legend
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper right", ncols=len(lab))
    return fig


def main(argv=None) -> int:
    return C.run(make, NAME, ("e3",), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

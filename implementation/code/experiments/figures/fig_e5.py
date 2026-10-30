"""Fig. 10 (fig:training): later gain of a tenant that distorted its reports during retraining, versus the distortion
factor alpha - zero under measurement-only retraining, not under stored-report retraining.

Data: ``results/summary/e5.json`` table ``later_gain``: per retraining pipeline, target and alpha, the distorting
tenant's mean declared-utility change on the evaluation windows relative to the alpha = 1 arm of the same date and
pipeline (credits per interval; positive means the distortion paid off later), with the two-stage cluster-bootstrap
95 % CI across dates; alpha = 1 is the reference (0 by definition). One panel per target, Protected arm (the H5b
primary arm). Series styles are the ``retrain_measurement`` / ``retrain_stored`` entries of ``spec/controllers.yaml``
(round-3 review: the stored-report pipeline of the Protected arm used to wear the Unconstrained net's orange).
Layout (2026-10-21, plotting only, review/open_items.md item 27, prereg/DEVIATIONS.md item 1): the alpha axis is
logarithmic (a factor and its inverse lie symmetric about alpha = 1, and the tick labels 0.5 and 0.7 no longer
overlap), the y-label has two lines and the legend has one entry per row, so that neither collides with the other and
the figure keeps the column width.

    python experiments/figures/fig_e5.py [--split test|dev] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_e5"
PIPES = (("measurement", "retrain_measurement"), ("stored", "retrain_stored"))
TARGETS = ("video", "urllc")


def make(S: dict):
    import matplotlib.pyplot as plt
    rows = C.table(S, "later_gain")
    arm = "protected" if any(r["arm"] == "protected" for r in rows) else (rows[0]["arm"] if rows else None)
    rows = [r for r in rows if r["arm"] == arm]
    targets = [t for t in TARGETS if any(r["target"] == t for r in rows)]
    if not targets:
        return None
    fig, axes = plt.subplots(1, len(targets), figsize=(C.COL_W, 2.3), squeeze=False, constrained_layout=True)
    for j, (ax, t) in enumerate(zip(axes[0], targets)):
        C.zero_line(ax)
        for pipe, key in PIPES:
            rr = sorted((r for r in rows if r["target"] == t and r["pipeline"] == pipe), key=lambda r: r["alpha"])
            if not rr:
                continue
            ref = dict(alpha=1.0, later_gain=0.0, lo=0.0, hi=0.0)
            rr = sorted(rr + ([ref] if not any(abs(r["alpha"] - 1) < 1e-9 for r in rr) else []),
                        key=lambda r: r["alpha"])
            C.errorbar(ax, [r["alpha"] for r in rr], [r["later_gain"] for r in rr], [r["lo"] for r in rr],
                       [r["hi"] for r in rr], key)
        al = sorted({r["alpha"] for r in rows if r["target"] == t} | {1.0})
        ax.set_xscale("log")
        ax.set_xticks(al)
        ax.set_xticklabels([f"{a:g}" for a in al])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(min(al) / 1.15, max(al) * 1.15)
        ax.set_title(C.CLASS_NAME.get(t, t) + " target", loc="left")
        ax.set_xlabel("Distortion factor \u03b1")
        if j == 0:
            ax.set_ylabel("Later gain\n(credits/interval)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncols=1)
    return fig


def main(argv=None) -> int:
    return C.run(make, NAME, ("e5",), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

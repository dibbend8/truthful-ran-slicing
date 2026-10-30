"""Fig. 5 (fig:worked): one interval on real data - the protected own quota never falls as the own report rises and no
misreport pays under the full-decision price; the unconstrained net's quota path can fall and its fixed-priority price
can leave a gain.

Data: ``results/summary/e0.json`` tables ``worked_meta``, ``worked_staircase`` and ``worked_curves`` (E0 worked
interval, ``experiments/e0_setup.py`` stage 5, carried into the summary by ``experiments/analyze.py``). The interval,
tenant and nets are E0's pre-declared choice (first window of the split, first URLLC-like tenant, seed-0 models).
(a) own quota (PRB) versus own report (Mbit/s), others truthful: the exact own-quota staircases of both maps;
(b) deviation gain (credits per interval; positive means the misreport pays) = declared utility at the true demand
under the design's billed price minus its value at truth: Protected with the full-decision price, the Unconstrained
net with the fixed-priority price; the marker is each design's best report on the grid. The vertical line marks the
true demand. (The outline's "shaded price area" is not drawn: the price is an integral of the declared marginal value,
not of the quota, so no area on the quota axis equals it.)

    python experiments/figures/fig_e0_worked.py [--split test|dev] [--smoke] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_e0_worked"
MAPS = (("protected", "protected"), ("unconstrained", "unprotected"))      # E0 map name -> controllers.yaml key
TIE_CREDITS = 1e-6      # a grid gain this small is a tie with truth (price certificates are ~1e-10 pi C Delta)


def _stairs(rows: list, mp: str, x_end: float):
    pts = sorted((r["start"], r["quota"]) for r in rows if r["map"] == mp)
    if not pts:
        return None, None
    xs = [p[0] for p in pts] + [x_end]
    ys = [p[1] for p in pts] + [pts[-1][1]]
    return xs, ys


def make(S: dict):
    import matplotlib.pyplot as plt
    meta = C.table(S, "worked_meta")
    cur = C.table(S, "worked_curves")
    stairs = C.table(S, "worked_staircase")
    if not meta or not cur:
        return None
    m0 = meta[0]
    theta, cap = float(m0["theta"]), float(m0["C"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(C.COL_W, 2.45), constrained_layout=True)
    handles = []
    for mp, key in MAPS:
        st = C.style(key)
        xs, ys = _stairs(stairs, mp, cap)
        if xs is None:                                  # no staircase table: the 257-point grid curve
            pts = sorted((r["u"], r["quota"]) for r in cur if r["map"] == mp)
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        h, = a.step(xs, ys, where="post", color=st["color"], ls=st["ls"], lw=1.2, label=st["label"])
        handles.append(h)
        g = sorted((r["u"], r["gain_billed"]) for r in cur if r["map"] == mp and r.get("gain_billed") is not None)
        b.plot([p[0] for p in g], [p[1] for p in g], color=st["color"], ls=st["ls"], lw=1.2)
        if g:                                           # the design's best report on the grid (truth on ties)
            ub, gb = max(g, key=lambda p: p[1])
            if gb <= TIE_CREDITS:
                ub, gb = theta, 0.0
            b.plot([ub], [gb], st["marker"], color=st["color"], ms=4.5, mec="white", mew=0.8, zorder=4)
    for ax in (a, b):
        ax.axvline(theta, color=C.INK2, lw=0.7, ls=":", zorder=1)
        ax.set_xlim(0, 1.03 * cap)
        ax.set_xlabel("Own report (Mbit/s)")
    a.annotate("true demand", (theta, 0), xytext=(3, 3), textcoords="offset points", ha="left", va="bottom",
               color=C.INK2)
    a.set_ylim(0, None)
    a.set_ylabel("Own quota (PRB)")
    a.set_title("(a) Quota", loc="left")
    C.zero_line(b)
    b.set_ylabel("Deviation gain (credits per interval)")
    b.set_title("(b) Gain", loc="left")
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               handlelength=2.2)
    return fig


def main(argv=None) -> int:
    return C.run(make, NAME, ("e0",), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

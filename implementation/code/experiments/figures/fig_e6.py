"""Fig. 11 (fig:scaling): pricing time, cost of protection and the largest deviation gain versus the number of
tenants n - pricing stays far inside the one-second interval, the welfare cost of protection stays small and no
report gains more than the numerical error of the computed price as n grows.

Data: ``results/summary/e6.json`` tables ``pricing_time`` (full-decision prices of all tenants in one interval,
Protected arm, one core: median and 95th percentile, ms), ``cost_by_n`` (W(Unconstrained net) - W(Protected),
credits per interval, truthful reports; positive means protection costs welfare; two-stage cluster-bootstrap 95 % CI
across dates) and ``max_gain_by_n`` (pre-freeze audit NUM-10: the gain panel of the outline; largest computed
full-price deviation gain at n = 6 and 9, credits per interval, with the largest guaranteed error bound e of the
computed price as a line). Three panels (figure*) when the gain table exists, else two (column width).

Layout (2026-10-21, plotting only, prereg/DEVIATIONS.md item 1): drawn at the 9/10 text width at which results.tex
includes it (text 8 pt in print); the y-axis of (c) is in units of 10^k credits per interval with limits that include
e, so that the dashed error-bound line is visible above the zero gains (it lay on the zero line), and the legend of (c)
sits above the panel like that of (a) (it covered the data line); the three panels share the n ticks and limits (the
gain is computed at n = 6 and 9 only; n = 3 is fig:protect (a)).

    python experiments/figures/fig_e6.py [--split test|dev] [--out DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig_e6"
WIDTH = C.FULL_W * 9 / 10           # results.tex: \resultfig[\dimexpr\textwidth*9/10\relax]{fig_e6}


def make(S: dict):
    import matplotlib.pyplot as plt
    tm = sorted((r for r in C.table(S, "pricing_time") if r["arm"] == "protected"), key=lambda r: r["n"])
    cost = sorted(C.table(S, "cost_by_n"), key=lambda r: r["n"])
    gain = sorted((r for r in C.table(S, "max_gain_by_n") if r.get("max_gain") is not None), key=lambda r: r["n"])
    if not tm and not cost and not gain:
        return None
    st = C.style("protected")
    if gain:
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDTH, 2.3), constrained_layout=True)
        errs = [r.get("max_err") for r in gain if r.get("max_err") is not None]
        e = max(errs) if errs else None
        k, f = C.tiny_scale(max([e or 0.0] + [abs(r["max_gain"]) for r in gain]))
        c.plot([r["n"] for r in gain], [f * r["max_gain"] for r in gain], color=st["color"], marker=st["marker"],
               ls=st["ls"], label=st["label"], zorder=3)
        if e is not None:
            c.axhline(f * e, color=C.INK2, lw=0.8, ls="--", zorder=2, label="Error bound e")
        c.axhline(0.0, color="#9a9a9a", lw=0.6, zorder=1)
        vals = [f * r["max_gain"] for r in gain] + ([f * e] if e else [])
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        pad = 0.15 * ((hi - lo) or 1.0)
        c.set_ylim(lo - pad, hi + pad)
        unit = "credits/interval" if k == 0 else f"10$^{{{k}}}$ credits/interval"
        c.set_ylabel(f"Largest deviation gain\n({unit})")
        c.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), title="(c)", title_fontsize=8, alignment="left",
                 handlelength=1.8)
    else:
        fig, (a, b) = plt.subplots(1, 2, figsize=(C.COL_W, 2.3), constrained_layout=True)
    if tm:
        n = [r["n"] for r in tm]
        a.plot(n, [r["median_ms"] for r in tm], color=st["color"], marker="o", ls="-", label="Median")
        a.plot(n, [r["p95_ms"] for r in tm], color=st["color"], marker="^", ls="--", label="95th percentile")
        a.set_yscale("log")
        C.plain_log(a)
        a.set_xticks(n)
        a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), title="(a) Protected (ours)", title_fontsize=8,
                 alignment="left", handlelength=1.8)
    else:
        a.set_title("(a)", loc="left")
    a.set_ylabel("Pricing time (ms/interval)")
    if cost:
        C.zero_line(b)
        C.errorbar(b, [r["n"] for r in cost], [r["estimate"] for r in cost], [r["lo"] for r in cost],
                   [r["hi"] for r in cost], "protected")
        b.set_xticks([r["n"] for r in cost])
    ns = sorted({r["n"] for r in tm} | {r["n"] for r in cost} | {r["n"] for r in gain})
    if ns:                                         # one n-axis for all panels (shared x-label)
        for ax in fig.axes:
            ax.set_xticks(ns)
            ax.set_xlim(ns[0] - 0.07 * (ns[-1] - ns[0] or 1), ns[-1] + 0.07 * (ns[-1] - ns[0] or 1))
    fig.supxlabel("Number of tenants n", fontsize=8)
    b.set_ylabel("Cost of protection\n(credits/interval)")
    b.set_title("(b)", loc="left")
    return fig


def main(argv=None) -> int:
    return C.run(make, NAME, ("e6",), argv, __doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())

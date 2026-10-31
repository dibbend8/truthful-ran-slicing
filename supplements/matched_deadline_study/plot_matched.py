#!/usr/bin/env python3
"""Plot verified matched-study summaries; synthetic layouts remain separate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG_LOAD = {"rho0.7": .7, "default": .9, "rho1.1": 1.1}
MULTIPLIERS = (.5, .67, .8, .9, 1.1, 1.25, 1.5, 2.)
TRAFFIC = ("video", "interactive", "background")
COLORS = {"frozen": "#0072B2", "retrained": "#D55E00", "primary": "#0072B2", "fixed_ablation": "#D55E00"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, text):
    if not condition:
        raise ValueError(text)


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2, "legend.fontsize": 7.5, "axes.linewidth": .6,
        "axes.edgecolor": "#8c8c8c", "axes.labelcolor": "#1f1f1f", "text.color": "#1f1f1f",
        "xtick.color": "#555555", "ytick.color": "#555555", "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": None, "figure.dpi": 150})


def load_table(args, config, mode, filename, provenance):
    # A fully validated all-mode analysis is preferred; never use smoke/partial output.
    candidates = [args.analysis_root / config / "both" / m for m in ("all", mode)]
    folder = next((p for p in candidates if (p / "findings.json").is_file() and (p / filename).is_file()), None)
    if folder is None:
        raise FileNotFoundError(f"No completed {mode} analysis for {config}/both")
    findings_path = folder / "findings.json"
    findings = json.loads(findings_path.read_text())
    require(findings.get("scientific_result") is True, "Diagnostic findings cannot produce paper figures")
    require(findings["scope"]["config"] == config and findings["scope"]["profile"] == "both" and
            mode in findings["scope"]["modes"], "Analysis scope mismatch")
    status = [s for s in findings["completion"] if s["mode"] == mode]
    require(len(status) == 1 and status[0]["complete"] is True and status[0]["validated_cases"] == 32 and
            status[0]["expected_cases"] == 32, "Plot requires every planned case in the selected cell")
    b = findings["bootstrap"]
    require(b["replicates"] >= 2000 and b["conditional_on_three_fitted_seeds"] is True,
            "Unexpected confidence-interval procedure")
    require(findings["analysis_sha256"] == sha(ROOT / "analyze_matched.py"), "Analyzer source changed since output")
    require(findings["protocol_sha256"] == sha(args.protocol), "Study protocol differs from analysis")
    table_path = folder / filename
    expected = findings.get("table_files_sha256", {}).get(filename)
    require(expected is not None and expected == sha(table_path), "Analysis table hash missing or mismatched")
    frame = pd.read_csv(table_path)
    require(frame.config.eq(config).all() and frame.profile.eq("both").all(), "CSV contains a different cell")
    require(np.allclose(frame.target_load, CONFIG_LOAD[config]), "CSV load/configuration mismatch")
    provenance.append({"analysis_directory": str(folder.resolve()), "findings_sha256": sha(findings_path),
        "table": filename, "table_sha256": expected, "analysis_sha256": findings["analysis_sha256"],
        "protocol_sha256": findings["protocol_sha256"], "bootstrap": b})
    return frame


def validate_estimates(frame, expected_count):
    require(len(frame) == expected_count, f"Expected {expected_count} plot rows, found {len(frame)}")
    require(np.isfinite(frame[["estimate", "ci_low", "ci_high"]].to_numpy(float)).all(), "Nonfinite plot estimate/CI")
    require((frame.ci_low <= frame.ci_high).all(), "CI endpoints are reversed")
    require(frame.n_dates.eq(16).all() and frame.n_windows.eq(32).all() and frame.n_fitted_seeds.eq(3).all(),
            "Plot rows do not use the complete date/window/seed grid")
    require(frame.finite_bootstrap_replicates.ge(2000).all(), "Insufficient finite bootstrap replicates")


def welfare_rows(args, provenance):
    rows = []
    for config in args.configs:
        table = load_table(args, config, "truth", "truth_comparisons.csv", provenance)
        for comparator in ("direct", "unconstrained"):
            for state in ("frozen", "retrained"):
                right = "reference__direct" if comparator == "direct" else f"{state}__unconstrained"
                selected = table[(table.metric == "relative_welfare_difference") & (table.traffic == "all") &
                    (table.contrast_kind == "protected_vs_baseline") & (table.left == f"{state}__protected") & (table.right == right)]
                validate_estimates(selected, 1)
                row = selected.iloc[0].to_dict()
                rows.append({"config": config, "load": CONFIG_LOAD[config], "comparator": comparator,
                             "state": state, **{k: 100 * float(row[k]) for k in ("estimate", "ci_low", "ci_high")}})
    return pd.DataFrame(rows)


def pricing_rows(args, provenance):
    table = load_table(args, "default", "strategic", "strategic_summary.csv", provenance)
    rows = []
    for tenant, traffic in enumerate(TRAFFIC):
        for bill in ("primary", "fixed_ablation"):
            selected = table[(table.metric == "contract_utility_gain") & (table.state == "retrained") &
                (table.arm == "protected") & (table.traffic == traffic) & (table.deviator == tenant) & (table.bill == bill)].sort_values("multiplier")
            validate_estimates(selected, 8)
            require(np.allclose(selected.multiplier.to_numpy(float), MULTIPLIERS), "Contract plot multiplier grid changed")
            for row in selected.to_dict("records"):
                rows.append({"traffic": traffic, "tenant": tenant, "bill": bill,
                    **{k: float(row[k]) for k in ("multiplier", "estimate", "ci_low", "ci_high")}})
    return pd.DataFrame(rows)


def synthetic_rows():
    """Deliberately artificial layout fixtures, never saved as analysis tables."""
    welfare, pricing = [], []
    for j, (config, load) in enumerate(CONFIG_LOAD.items()):
        for comparator in ("direct", "unconstrained"):
            for state in ("frozen", "retrained"):
                mean = (-4 + .8*j if state == "frozen" else -1 + .4*j) if comparator == "direct" else (-.3+.2*j if state == "frozen" else .4+.15*j)
                radius = .18+.05*j
                welfare.append(dict(config=config, load=load, comparator=comparator, state=state,
                                    estimate=mean, ci_low=mean-radius, ci_high=mean+radius))
    for i, traffic in enumerate(TRAFFIC):
        for bill in ("primary", "fixed_ablation"):
            for multiplier in MULTIPLIERS:
                mean = -(multiplier-1)**2*(.12+i*.08) if bill == "primary" else (multiplier-1)*(.8+i*.7)
                radius = (.012+i*.005)*abs(multiplier-1)
                pricing.append(dict(traffic=traffic, tenant=i, bill=bill, multiplier=multiplier,
                                    estimate=mean, ci_low=mean-radius, ci_high=mean+radius))
    return pd.DataFrame(welfare), pd.DataFrame(pricing)


def series(ax, x, means, low, high, *, color, marker, label, cap_width, linestyle="-", truth_reference=False):
    # Draw CI endpoints directly. A percentile CI need not contain its point
    # estimate; measured intervals are never clipped or changed. Exact truth
    # references, when requested, are drawn separately without bootstrap CIs.
    x, means, low, high = map(np.asarray, (x, means, low, high))
    ax.vlines(x, low, high, color=color, lw=.65, zorder=2)
    ax.hlines(low, x-cap_width, x+cap_width, color=color, lw=.65, zorder=2)
    ax.hlines(high, x-cap_width, x+cap_width, color=color, lw=.65, zorder=2)
    if truth_reference:
        # U(truth) - U(truth) is identically zero. This is a reference, not a
        # ninth measured strategy; the original x/means/CI arrays stay intact.
        line_x, line_means = np.r_[x, 1.], np.r_[means, 0.]
        order = np.argsort(line_x, kind="stable")
        ax.plot(line_x[order], line_means[order], color=color, marker=marker,
                markevery=np.flatnonzero(line_x[order] != 1.).tolist(), markersize=3.5, markeredgewidth=.65,
                markerfacecolor="white" if marker == "s" else color,
                lw=1.05, ls=linestyle, label=label, zorder=3)
        ax.plot([1.], [0.], marker="D", markersize=3.2, markeredgewidth=.65,
                markeredgecolor="#555555", markerfacecolor="white", ls="none", zorder=4)
    else:
        ax.plot(x, means, color=color, marker=marker, markersize=3.5, markeredgewidth=.65,
                markerfacecolor="white" if marker == "s" else color, lw=1.05, ls=linestyle, label=label, zorder=3)


def limits(ax, values, *, symmetric=False):
    values = np.r_[0., np.asarray(values).ravel()]
    low, high = float(np.min(values)), float(np.max(values))
    span = high-low
    pad = .12 * (span if span else max(abs(high), 1e-8))
    ax.set_ylim(low-pad, high+pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.axhline(0, color="#777777", linewidth=.65, zorder=1)
    ax.grid(axis="y", color="#e3e3e3", linewidth=.45)
    ax.tick_params(length=2.5, pad=1.7)


def welfare_figure(rows, diagnostic=False):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.5))
    fig.subplots_adjust(left=.145, right=.985, bottom=.265, top=.71, wspace=.42)
    values = sorted(rows.load.unique())
    for ax, comparator, title in zip(axes, ("direct", "unconstrained"), ("vs Direct", "vs Unconstrained")):
        panel = rows[rows.comparator.eq(comparator)]
        for state, marker, label, shift in (("frozen", "o", "Original training", -.008), ("retrained", "s", "Matched training", .008)):
            data = panel[panel.state.eq(state)].sort_values("load")
            series(ax, data.load.to_numpy()+shift, data.estimate, data.ci_low, data.ci_high,
                   color=COLORS[state], marker=marker, label=label, cap_width=.009,
                   linestyle="-" if state == "frozen" else "--")
        limits(ax, panel[["estimate", "ci_low", "ci_high"]].to_numpy())
        ax.set_xticks(values)
        ax.set_xlim(min(values)-.06, max(values)+.06)
        ax.set_title(title, pad=3)
    axes[0].set_ylabel("Relative welfare gap (%)", labelpad=3)
    fig.text(.56, .035, "Target offered load", ha="center", fontsize=7.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.55, 1.025),
               ncol=2, frameon=False, columnspacing=1.4, handlelength=1.5, handletextpad=.4)
    if diagnostic:
        fig.text(.53, .50, "SYNTHETIC LAYOUT TEST", ha="center", color="#a00000", alpha=.75, fontsize=9, rotation=15)
    return fig


def pricing_figure(rows, diagnostic=False):
    # Manuscript-only compact layout: unchanged data, font sizes, markers and CI styling.
    fig, axes = plt.subplots(1, 3, figsize=(516/72, 118.8/72))
    fig.subplots_adjust(left=.095, right=.99, bottom=.26, top=.75, wspace=.30)
    for ax, traffic, title in zip(axes, TRAFFIC, ("(a) Video", "(b) Interactive", "(c) Background")):
        panel = rows[rows.traffic.eq(traffic)]
        for bill, marker, label in (("primary", "o", "Recomputed-priority pricing"),
                                    ("fixed_ablation", "s", "Frozen-priority pricing")):
            data = panel[panel.bill.eq(bill)].sort_values("multiplier")
            series(ax, data.multiplier, data.estimate, data.ci_low, data.ci_high,
                   color=COLORS[bill], marker=marker, label=label, cap_width=.018,
                   linestyle="-" if bill == "primary" else "--", truth_reference=True)
        limits(ax, panel[["estimate", "ci_low", "ci_high"]].to_numpy())
        ax.axvline(1, color="#aaaaaa", ls=":", lw=.6, zorder=0)
        ax.set_xlim(.44, 2.06)
        ax.set_xticks([.5, 1, 1.5, 2])
        ax.set_title(title, pad=4)
    axes[0].set_ylabel("Contract utility gain\n(credits/interval)", labelpad=3)
    fig.text(.54, .065, "Reporting multiplier", ha="center", fontsize=8)
    fig.text(.985, .065, "Diamond: exact zero gain at truth", ha="right", fontsize=7.2, color="#555555")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.54, 1.015),
               ncol=2, frameon=False, columnspacing=2.0, handlelength=2., handletextpad=.5, fontsize=8)
    if diagnostic:
        fig.text(.52, .48, "SYNTHETIC LAYOUT TEST", ha="center", color="#a00000", alpha=.75, fontsize=14, rotation=10)
    return fig


def save_figure(args, figure, kind, rows, provenance):
    folder = ROOT / "figures" / "diagnostic_layout" if args.diagnostic_layout else ROOT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    stem = ("synthetic_" if args.diagnostic_layout else "matched_") + kind
    pdf, png = folder / f"{stem}.pdf", folder / f"{stem}.png"
    title = "Synthetic layout test; not research data" if args.diagnostic_layout else f"Matched deadline study: {kind.replace('_', ' ')}"
    figure.savefig(pdf, bbox_inches=None, metadata={"Title": title,
        "Subject": "Conditional 95% date/window bootstrap intervals; three fitted seeds; exploratory reused TEST dates" if not args.diagnostic_layout else "Artificial fixture solely for layout validation"})
    figure.savefig(png, dpi=300, bbox_inches=None)
    dimensions = [float(x*72) for x in figure.get_size_inches()]
    plt.close(figure)
    record = {"scientific_result": not args.diagnostic_layout, "kind": kind, "script_sha256": sha(__file__),
        "dimensions_pt": dimensions, "pdf": str(pdf), "pdf_sha256": sha(pdf), "png_sha256": sha(png),
        "source_tables": provenance, "plotted_rows": rows.to_dict("records"),
        "confidence_intervals": "95% percentile, date then window resampling after averaging paired fitted seeds" if not args.diagnostic_layout else "synthetic fixture endpoints; no empirical inference",
        "deterministic_reference_points": [dict(traffic=traffic, bill=bill, multiplier=1., value=0.,
            exact_lower=0., exact_upper=0., origin="U(truth) minus U(truth)",
            measured_trial=False, bootstrap_confidence_interval=False, ci_drawn=False)
            for traffic in TRAFFIC for bill in ("primary", "fixed_ablation")] if kind == "pricing" else [],
        "notes": ["Measured rows are unchanged; pricing curves also include six exact (1,0) truthful references, drawn as diamonds without CIs. These are identities, not new trials." if kind == "pricing" else "No additional reference points.", "CI endpoints plotted directly without clipping to point estimates.",
                  "Welfare gaps are100 times (Protected mean minus baseline mean)/baseline mean.",
                  "Unconstrained comparison uses the same frozen/retrained state as Protected."]}
    (folder / f"{stem}_provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"figure": str(pdf), "scientific_result": not args.diagnostic_layout, "dimensions_pt": dimensions}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-root", type=Path, default=ROOT / "analysis_matched")
    p.add_argument("--protocol", type=Path, default=ROOT / "PROTOCOL.txt")
    p.add_argument("--which", choices=("welfare", "pricing", "all"), default="all")
    p.add_argument("--config", choices=tuple(CONFIG_LOAD), action="append",
                   help="welfare only: explicitly choose completed cells; default requires all three loads")
    p.add_argument("--diagnostic-layout", action="store_true", help="write only watermarked artificial layouts in figures/diagnostic_layout/")
    args = p.parse_args()
    args.analysis_root, args.protocol = args.analysis_root.expanduser().resolve(), args.protocol.expanduser().resolve()
    args.configs = args.config or list(CONFIG_LOAD)
    require(len(set(args.configs)) == len(args.configs), "Duplicate requested load")
    args.configs.sort(key=CONFIG_LOAD.get)
    style()
    fixture = synthetic_rows() if args.diagnostic_layout else None
    # Read and validate all requested inputs before creating a scientific figure.
    rows, sources = {}, {}
    if args.which in ("welfare", "all"):
        sources["welfare"] = []
        rows["welfare"] = fixture[0] if fixture else welfare_rows(args, sources["welfare"])
        rows["welfare"] = rows["welfare"][rows["welfare"].config.isin(args.configs)].copy()
    if args.which in ("pricing", "all"):
        sources["pricing"] = []
        rows["pricing"] = fixture[1] if fixture else pricing_rows(args, sources["pricing"])
    for key, data in rows.items():
        data_before = data.copy(deep=True)
        figure = welfare_figure(data, args.diagnostic_layout) if key == "welfare" else pricing_figure(data, args.diagnostic_layout)
        require(data.equals(data_before), "Plotting changed original measured rows")
        save_figure(args, figure, "welfare_gaps" if key == "welfare" else "pricing", data, sources[key])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"PLOTTING REFUSED: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)

"""Run every paper figure script on the summaries of one split (``results/summary/`` for TEST).

    python experiments/figures/make_all.py [--split test|dev|train] [--smoke] [--out DIR ...]

Exit code 0 when every figure was written; a figure whose summary is missing or empty is reported as skipped.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Script -> outline figure (pre-freeze audit NUM-10): fig:example fig_worked_example; fig:worked fig_e0_worked;
#: fig:value fig_e1_value; fig:manip (a) fig_e2e3_gain, (b, c) fig_e2_manip; fig:protect (a, c) fig_e3_protect,
#: (b) fig_e3_bandits; fig:cost fig_e4_cost; D7 frontier (no outline entry yet) fig_e4_frontier; fig:training
#: fig_e5 (later gain versus alpha; the outline's per-interval panels need E5 episodes); fig:scaling fig_e6;
#: fig:nr fig_e7_nr.
FIGURES = ("fig_worked_example", "fig_e0_worked", "fig_e1_value", "fig_e2e3_gain", "fig_e2_manip", "fig_e3_bandits",
           "fig_e3_protect", "fig_e4_cost", "fig_e4_frontier", "fig_e5", "fig_e6", "fig_e7_nr")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    status = {}
    for name in FIGURES:
        status[name] = importlib.import_module(name).main(argv)
    for name, code in status.items():
        print(f"{name}: {'ok' if code == 0 else 'skipped'}")
    return 0 if all(c == 0 for c in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Pre-registered part-balance check of the E5 pool halves on TRAIN and DEV (round-2 review; not an experiment).

Run from ``code/``:

    PYTHONDONTWRITEBYTECODE=1 /home/darpan/anaconda3/bin/python experiments/diag_part_balance.py

For every candidate block length ``b`` in {1, 2, 5, 10} s it reports, per split and class, the expected per-user
mean rate of the pooled TRACTOR captures and of each interleaved half (``data.tractor.part_balance``), the ratios
to the pooled mean, and the probability that a 200-s user window wraps; and the expected per-PRB rate of each Raca
half (``data.raca.part_balance``). The rule: the block length is the smallest ``b`` whose class means lie within
``1 +- tractor.PART_BALANCE_TOL`` (10 %) of the pooled mean in both halves on TRAIN and on DEV
(``tractor.PART_BLOCK_S`` must equal it). It also composes the first four E5 windows of one TRAIN and one DEV date
per phase and reports their mean measured class loads and wrapped users. TEST is never read.

Output: ``results/raw/diagnostics/part_balance.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from slicemech import io as sio  # noqa: E402
from slicemech.data import compose, raca, tractor  # noqa: E402
from slicemech.data.calibrate import load_calibration  # noqa: E402
from slicemech.tenants import poison  # noqa: E402

OUT = sio.ROOT / "results" / "raw" / "diagnostics" / "part_balance.json"
BLOCKS = (1, 2, 5, 10)
SPLITS = ("train", "dev")
EXAMPLES = (("train", "2024-01-01"), ("dev", "2024-02-28"), ("dev", "2024-03-02"))


def composed_loads(cal, split: str, date: str) -> dict:
    """Mean measured class loads (Mbit/s, scored intervals) and wrapped users of the first four E5 offsets of a
    date, without pool parts and for each E5 phase."""
    offs = [o for o, _ in poison.e5_layout("dev")][:4] if split == "dev" else list(compose.window_offsets(split)[:4])
    out = {}
    for phase in (None, "A", "B"):
        ws = [cal.window(split, date, o, pool_part=phase) for o in offs]
        m = np.mean([w.measured[w.warmup:].mean(0) for w in ws], axis=0)
        wrapped = np.mean([[t.get("wrapped_users", 0) for t in w.meta["tenants"]] for w in ws], axis=0)
        out[str(phase)] = dict(half=ws[0].meta["pool_half"], offsets=offs, mean_measured_mbps=m.tolist(),
                               wrapped_users=wrapped.tolist(), counts=list(cal.counts_for(split, date)),
                               classes=list(cal.classes))
    return out


def main() -> dict:
    sio.require_free_gb(3)
    res = dict(rule="smallest block length in %s s passing the +-%.0f %% class-mean check on TRAIN and DEV"
               % (list(BLOCKS), 100 * tractor.PART_BALANCE_TOL), tractor={}, raca={}, chosen=None,
               PART_BLOCK_S=tractor.PART_BLOCK_S)
    for b in BLOCKS:
        res["tractor"][str(b)] = {s: tractor.part_balance(s, block_s=b) for s in SPLITS}
    res["raca"] = {s: raca.part_balance(s) for s in SPLITS}
    passing = [b for b in BLOCKS if all(res["tractor"][str(b)][s]["balanced"] for s in SPLITS)]
    res["chosen"] = passing[0] if passing else None
    res["raca_balanced"] = bool(all(res["raca"][s]["balanced"] for s in SPLITS))
    cal = load_calibration("n3_default")
    res["composed"] = {f"{s} {d}": composed_loads(cal, s, d) for s, d in EXAMPLES}
    res["flip"] = {d: compose.part_flip(d) for _, d in EXAMPLES}
    sio.write_json(OUT, res)
    for b in BLOCKS:
        row = []
        for s in SPLITS:
            for c, v in res["tractor"][str(b)][s]["classes"].items():
                row.append(f"{s} {c} A/B {v['A_ratio']:.3f}/{v['B_ratio']:.3f}")
        print(f"b = {b:2d} s:", "; ".join(row))
    for s in SPLITS:
        r = res["raca"][s]
        print(f"raca {s}: A/B {r['A_ratio']:.3f}/{r['B_ratio']:.3f}")
    print("chosen block length:", res["chosen"], "| PART_BLOCK_S =", tractor.PART_BLOCK_S)
    print(json.dumps(res["composed"], indent=1)[:3000])
    if res["chosen"] != tractor.PART_BLOCK_S:
        raise SystemExit("tractor.PART_BLOCK_S differs from the block length the rule selects")
    return res


if __name__ == "__main__":
    main()

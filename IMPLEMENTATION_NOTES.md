# Conventions used by the implementation

These notes identify conventions in the preserved source that matter when
interpreting the manuscript or reproducing a calculation.

## Time and delay units

`implementation/spec/params.yaml` sets 106 PRBs, a 1-s controller interval, 1-ms
slots, 15-kHz subcarrier spacing, and 20-MHz bandwidth. A quota is a number of
PRBs available per slot and is held fixed for a controller interval.

The configured deadline budgets are **2002 ms for video**, **150 ms for the
URLLC-like traffic**, and **300 ms for the mMTC-like traffic**. The video budget
is therefore 2.002 s, matching the video chunk period; it is not 2 ms. The
controller's measurement delay is 2 s by default. The report lead is 50 ms and
the default video announcement lead is 4.004 s. These are distinct quantities.

The standalone NR scenario header contains defaults inherited from its earlier
use. The Python NR adapter passes the experiment's deadline budgets explicitly
in slice order. Inspect the driver-generated command when reproducing an NR run.

## Training-channel experiment (E5)

The attack is compared with truthful stored-report retraining (`alpha = 1`) on
the same date and pipeline, rather than simply with an unretrained model. The
primary stored-report cell uses `alpha = 0.5`, fixed in the historical protocol.

`experiments/analyze.py::_e5_cost_net` first computes the signed utility change in
the attack windows and the gain in the later evaluation windows. For each date,
the reported net is:

```text
later gain + attack-phase utility change ×
             (number of attack windows / number of evaluation windows)
```

The resulting unit is **credits per evaluation interval**. A negative attack
change is a cost. With equal attack/evaluation window counts, the net equals
later gain minus attack cost; it is not the mean over both phases combined.
The supplied protocol uses four windows in each phase. The analysis bootstraps
one net value per date and reports the number of evaluation periods needed to
recover the cost when both cost and later gain are positive.

These notes describe the source calculation. They do not substitute for rerunning
the original raw-window analysis.

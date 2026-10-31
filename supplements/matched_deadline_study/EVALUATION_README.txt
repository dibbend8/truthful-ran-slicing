Matched-deadline evaluation
==========================

evaluate_matched.py evaluates the frozen and newly trained policies specified
in PROTOCOL.txt. It imports the previously validated deadline_load_study/
evaluate.py unchanged. It neither trains a model nor selects policies by TEST.
All later TEST results are exploratory: these dates were inspected previously.

Scientific scope per load/deadline setting
Truth: all four learned arms (protected, unconstrained, meas_only, kappa0),
both frozen and retrained, each with seeds0/1/2; plus frozen static, refitted
static, equal weights and Direct once each. This is28 trajectories/window and
896 trajectories over the32 planned windows.

Strategic: frozen/retrained Protected and Unconstrained, seeds0/1/2. Each policy
has its own truthful baseline and each tenant alone uses each multiplier
(.5,.67,.8,.9,1.1,1.25,1.5,2). This is300 trajectories/window and9,600 per cell.
All32 windows are used in both modes: offsets60 and43260 seconds on each of the
same16 original effective TEST dates. There is no configurable TEST subset or
post-outcome strategy refinement. The runner never reuses old frozen replay
results; every necessary trajectory is evaluated with the present exact inputs.

Example commands (use your server paths and compatible Python environment):

  python -B evaluate_matched.py --mode truth --plan
  python -B evaluate_matched.py --mode strategic --plan
  python -B evaluate_matched.py --mode truth --config default --profile both \
    --assets /path/upstream --frozen-study /path/deadline_load_study \
    --output /path/matched_deadline_study --prepare-only
  python -B evaluate_matched.py --mode truth --config default --profile both \
    --assets /path/upstream --frozen-study /path/deadline_load_study \
    --output /path/matched_deadline_study --workers 3
  python -B evaluate_matched.py --mode strategic --config default --profile both \
    --assets /path/upstream --frozen-study /path/deadline_load_study \
    --output /path/matched_deadline_study --workers 3

--config and --profile may each be repeated; their Cartesian product is fixed
before execution. Defaults are default/both. --model-root defaults to
OUTPUT/models; --protocol defaults to OUTPUT/PROTOCOL.txt. --workers defaults
to3 and supports1..16 CPU processes, each with one numerical-library thread.
Multiple workers use portable spawn, not inherited Torch state. --plan performs
no imports, writes, window composition or TEST reads. --prepare-only verifies
model/data/source/protocol files and the original native TEST provenance gate,
then saves case specifications; it performs no simulation or policy evaluation.

Prerequisites and provenance
Every retrained checkpoint must have COMPLETE.json, all eight exact artifact
hashes, a matching scientific job/profile/shape/protocol, the original400/32/M8/
h3 budget and TRAIN/DEV-only validation. Refitted static must have its exact
seven-file completion manifest,232-candidate grid and72-candidate DEV selection.
Changed training/static source is refused. The scientific mode cannot consume
smoke or benchmark models. Frozen model/config and static/candidate files must
match the original recovered GitHub hashes. Prepared Puffer data must match
the separate recovered-archive hashes. Other data and calibrations are verified
against their recovery hashes. The unchanged native TEST read gate verifies
the original frozen assets and logged code/config provenance; this is not a
claim that the new study belonged to the historical registration.

The runner supplies the relocated upstream/prereg/PREREG.md path explicitly
to the gate and TEST composition. Recorded historical absolute paths are not
used to locate the model/data files. No manifest, source, original checkpoint,
preregistration, or frozen-study result is modified. Transfer the complete
recovered upstream tree, the prior study's study_support.py and evaluate.py,
its recovery manifests and shapes/, plus train_matched.py, fit_static_matched.py,
this evaluator and the matched PROTOCOL.txt.

Payments and pairing
Every200-interval window retains its40-interval truthful warm-up. Constant false
reports start at interval40, so truthful and deviating replay share the entire
initial queue history. Reports are clipped through the native evaluator.
Protected uses full envelope prices and also receives the exact frozen-priority
bill on the same quota/queue trajectory. Unconstrained uses frozen-priority
prices. Truthful strategic trajectories price all three tenants; deviations
price the deviator. Missing honest-tenant bills remain NaN/null, never zero.
Truth-only trajectories do not claim net utility when no bill was computed.

All selected prices and certified bounds must be finite. The original evaluator
checks every pricing quota against allocation; the wrapper checks both primary
and fixed bills lie inside their brackets. Every deviation must have identical
warm-up reports, quotas, delivery, expiry and backlog, and identical backlog at
the first scored interval. Contract value always uses true admitted demand,
not the inflated report. Full-price Protected contract gain is checked against
its certified numerical uncertainty; a positive certified lower bound beyond
tolerance fails the case. Any such failure retains its pending outputs for
inspection and is not silently filtered from a completed report.

For each policy/tenant/multiplier, paired_gains records actual-delivery and
contract-utility differences against that policy's own truthful baseline.
For gain=(deviation value - deviation bill) - (truth value - truth bill), the
lower bound subtracts the deviation upper bill and adds the truth lower bill;
the upper bound uses the opposite endpoints. These are numerical payment
bounds, not statistical confidence intervals. Positive observations are kept.
Protected's fixed-price ablation has its own corresponding truthful fixed bill.
Honest harm uses on-time delivery and welfare, never missing utility fields.

Output and resume
One atomic case directory is written per load/profile/date/offset:
  OUTPUT/evaluation/{truth,strategic}/CONFIG/PROFILE/DATE__OFFSET/
It contains result.json, intervals/*.npz and COMPLETE.json. NPZ files retain
every native evaluator array, including full/fixed prices and bounds, quota,
true admitted demand, submitted reports, declared and delivered value, utility,
and queue/delivery/expiry arrays. result.json stores the complete specification,
exact composed-input array hashes, actual scored offered load, per-tenant rows,
trajectory checks/timings, and paired strategy gains. Traffic totals are Mbit
over the scored window; welfare/value/bill/utility summaries are credits per
scored interval. The load denominator is B times the original TRAIN mean
per-PRB rate. Target rho is not realized utilization.

Case files are staged in a unique pending directory and renamed only after
every planned trajectory succeeds. COMPLETE.json hashes result.json and all
interval files. Resume reuses a case only if its entire specification and every
file hash match. Existing incomplete, stale, extra-file or corrupt cases fail;
they are never overwritten. A failed attempt retains FAILED.json and its
available arrays under a pending path. Inspect stale job locks after an unclean
kill before retrying. evaluation_runs/ records the ordered complete case grid,
worker count and a separate completion/error record for each attempt.

TRAIN-only interface validation
  python -B evaluate_matched.py --mode truth --smoke --workers 1
  python -B evaluate_matched.py --mode strategic --smoke --workers 2

Smoke uses only the first TRAIN window, its first48 intervals and the unchanged
40-interval warm-up. It uses existing OUTPUT/smoke/default/both Protected seed0
and static diagnostic completions, plus frozen seed0 models. Truth smoke has9
trajectories; strategic smoke has75 (all eight multipliers for each tenant for
frozen Protected, frozen Unconstrained and diagnostic retrained Protected).
The smaller model grid and shortened scored horizon are explicit in its spec.
Results go under smoke_evaluation/, are marked non-scientific and cannot be
used as evidence for the planned TEST grid. This mode does not invoke a TEST
gate or open TEST Puffer data. It checks interfaces, serialization, pricing,
common warm-up, full-grid completion and atomic resume only.

Aggregation and confidence intervals are deliberately left to a separate
analysis program following PROTOCOL.txt. Do not report these raw paired rows as
independent dates, compare a selected maximum with an aggregate effect, or infer
strategyproofness/3GPP compliance from the finite replay study.

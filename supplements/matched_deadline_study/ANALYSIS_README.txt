Matched-training analysis

analyze_matched.py uses Python 3.11+ with NumPy, pandas and pyarrow. It imports
no training or simulator modules. Run after retrieving complete evaluation
case directories, evaluation_specifications/, evaluation_runs/, the exact
runner sources and protocol, and the corresponding completed fitted models.
The original assets and frozen-study provenance/shape files remain required
for independent hash and value checks. Paths are relocatable.

Examples
  python -B analyze_matched.py --mode truth --cell default/both
  python -B analyze_matched.py --mode all --cell default/both
  python -B analyze_matched.py --mode strategic --cell default/both \
    --cell rho0.7/both --cell rho1.1/both
  python -B analyze_matched.py --mode truth --all
  python -B analyze_matched.py --mode all --cell default/both \
    --root /path/matched_deadline_study --assets /path/upstream \
    --frozen-study /path/deadline_load_study

The default scope is the single default/both cell. --all explicitly requires
all nine cells for the requested mode(s); it is not needed for the revised
three-cell strategic grid. Every selected scientific cell must have all
32 planned windows in every requested mode. All requested cells are validated
before any scientific report is written. Earlier failed attempt records do
not invalidate a subsequently complete, verified case; complete case manifests
and the exact planned grid govern completion. Unfinished cells fail clearly.

--allow-partial validates available cases and prints status only: no research
tables or estimates are saved. --diagnostic-smoke validates only the separate
TRAIN smoke outputs and writes analysis_diagnostics/smoke_validation.json;
it never produces scientific summaries or pretends smoke is TEST data.
Scientific reports default to:
  analysis_matched/CONFIG/PROFILE/{truth,strategic,all}/
Use --output for a separate analysis destination and --model-root when models
were relocated outside ROOT/models (ROOT/smoke for diagnostics).

Independent verification
The analyzer checks the exact model, trajectory, tenant, date/window and
multiplier grids; every case artifact hash; planned evaluation-spec/run
manifests; recovered input origins; runner and fitting source hashes; completed
model and static artifacts; 400-iteration neural budgets and eight DEV checkpoint
choices; the static grid and stable TRAIN-top-72 DEV choice; shape provenance;
and protocol consistency. Scientific mode refuses smoke/benchmark models.
The Puffer fetch manifest permits only its documented top-level freeze.path
relocation: the original bytes are checked against the recovery manifest, all
other content is identical, and the frozen preregistration bytes are verified.

Every NPZ is read without pickle. Contract values are independently reconstructed
from true admitted demand, quotas, stale rates, tariff and TRAIN-fit shape atoms.
Delivery values are reconstructed from on-time Mbit and tariffs. The analyzer
checks report clipping, truthful warm-up, quotas, conservation, summary units,
price missingness, all utility identities, and certified bill brackets. Paired
gains and both bound endpoints are independently reconstructed and compared
with saved paired_gains, including every honest tenant's harm. Truth/strategic
baseline replay and physical/model input hashes must agree when both modes
are requested. Honest-tenant utility is never imputed where bills are missing.

Tables
truth_summary: all frozen/retrained learned arms, both static versions, equal
and Direct, with total/per-traffic welfare, on-time traffic, expiry and backlog.
truth_comparisons: retrained-minus-frozen effects within each architecture and
static; each Protected state versus every other policy group. Contrasts name
both sides explicitly. Different state/architecture comparisons are descriptive.
direct_gap_before_after: Protected minus Direct before and after training, and
the gap change. Since Direct is the same baseline, the absolute gap change is
algebraically Protected's retraining effect, not independent evidence.
strategic_summary: every fixed multiplier separately, with delivery/contract
gain, numerical payment bounds, positive-denominator normalized gain, denominator
values/eligibility counts, positive individual counts and honest-tenant losses.
strategic_training_effects: matched changes in those gains/harms after training,
with certified numerical bounds on absolute gain changes.
strategic_victim_summary: each honest tenant's separate loss.
Raw observations, seed-averaged windows, validated case/trajectory inventories,
and CSV/Parquet versions are retained. Each statistical summary also has a
matching *_per_seed table of individual seed aggregate estimates; seed -1
denotes a deterministic baseline. findings.json records provenance,
completion, actual load mean/range, diagnostic extrema and interpretation limits.
It also records SHA-256 hashes of every generated CSV and Parquet table.

Statistics and units
Each date gets equal weight. Matched seed effects are averaged before resampling
dates and then windows within sampled dates. The default is 2,000 percentile
bootstrap replicates, RNG seed 20261031. Intervals are descriptive and condition
on the three fitted networks; seed ranges are aggregate results by fitted seed.
Seed-free baselines are never counted as three independent replicas.
Traffic summaries supplied by the evaluator are scored-window Mbit; the analyzer
explicitly divides by160 where it reports Mbit/scored interval. Value, welfare,
utility and payment are already credits/scored interval. Repeated total welfare
fields are checked but never summed over tenants.
Expiry is pooled expired/(on-time+expired), not a mean of window percentages.
Relative welfare divides the paired mean difference by the right-hand mean.
Relative utility divides eligible summed gains by eligible summed own-truth
utility, using only positive-truth seed/window pairs. Absolute gains retain all
pairs. Counts and denominators are reported; undefined ratios stay missing.
Positive-instance counts use a 1e-8-credit/interval threshold and separately
require certified lower bounds above that threshold. These counts and maxima
describe correlated seed/window cases; they do not define an adaptive policy.
Numerical payment bounds are distinct from sampling confidence intervals.
No favorable seed, window, profile or multiplier is selected for the reported
fixed-strategy mean. No TEST tuning or confirmatory significance claim is made.

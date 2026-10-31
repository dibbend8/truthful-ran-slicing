Independent analysis of the original-setting five-seed headline replay

This analyzer reads completed output only. It does not train, simulate, download,
modify the historical source/protocol, or publish. The extension uses previously
observed TEST data and is exploratory, outside the original Holm family.

Full analysis (all paths are explicit; output must not exist):
  python -B analyze_original_multiseed.py \
    --campaign /path/to/campaign_v3 --assets /path/to/assets \
    --frozen-support /path/to/frozen_support \
    --runner /path/to/run_original_supplement_v3.py \
    --output /path/to/new_analysis

The named runner must have the exact hash recorded in SPECIFICATION.json. The
local run_original_supplement.py currently has those same bytes. Required
packages are NumPy, pandas and pyarrow; this does not reconstruct the historical
training environment. Campaign specifications retain actual replay versions.

To inspect completion without computing or writing estimates:
  python -B analyze_original_multiseed.py --campaign /path/to/campaign_v3 --status-only

The full path requires the exact original 32 rotated windows on 16 dates, all
five seeds, five trajectories per case and 160 complete case markers. It verifies
the campaign/preflight/specification/protocol records, source and recovered input
hashes, 800 Parquet hashes and all case metadata. It rejects incomplete grids,
wrong identities, extra/missing Parquets, duplicate interval rows, changed
warm-up, seed-dependent exogenous windows and inconsistent byte/quota/utility
identities. It writes no scientific estimates before the complete grid passes.

Every trajectory contains 200 one-second intervals and three tenants. Only
intervals k>=40 are scored. For each same-seed/window/deviating-role triple,
honest means the other two tenants. The two headline estimators are:

  honest on-time loss (%) = 100 * mean(U_truth honest on-time - U_BR honest on-time)
                                  / mean(U_truth honest on-time)
  protection gain (%) = 100 * mean(P_truth honest delivery value - U_BR honest delivery value)
                              / mean(U_BR honest delivery value)

U denotes Unconstrained and P denotes Protected. Delivery value is tariff times
on-time Mbit, with tariffs 1, 3, 0.5 credits/Mbit. A positive loss means harm;
a positive protection gain favors Protected. The numerator and denominator
average deviator roles and seeds within each window before pooling. Percentages
are ratios of pooled means, not averages of individual percentages. Absolute
effects are Mbit or credits per scored interval, summed over honest tenants.

The date/window bootstrap samples 16 dates, then two windows within each drawn
date, with replacement; it uses 10,000 replicates, RNG seed 0 and unadjusted
percentile 95% intervals. Seeds are averaged before resampling. Per-seed point
estimates and intervals, the range and sample standard deviation across five
seed estimates are retained separately. The pooled interval conditions on these
five trained networks; it does not treat 160 seed/windows as independent dates.
The draw algorithm is independently checked against the original stats routine.
Original analysis used RNG seed 20261019; the supplemental protocol fixes seed 0, so
point estimates, rather than exact historical bootstrap samples, are compared.

Outputs include headline_summary, per_seed_summary, case_role_pairs (480 rows),
window_seed_pairs (160), seed_averaged_windows (32) and all 76,800 scored local
best-response decisions. Every table has CSV and Parquet versions. findings.json
records table hashes, case-verification hashes, exact source/input hashes,
individual effect signs, search-gain counts and maximum context. Positive gains
are not discarded or replaced by pooled averages. The original nonmonotone
Unconstrained search is a lower bound, with absent certificates left missing.
No Protected-deviation certification or new packet-simulation claim is made.

The seed 0 point estimates are compared with the original 19.44203832884902% harm
and 16.059932511329347% protection summaries. Their exact v0.1.0 source hashes
and macro names are recorded. A mismatch at the published two-decimal precision
sets ready_for_manuscript=false; it does not hide the newly computed results.
Historical results must not be replaced until a material mismatch is explained.

Diagnostics only, using the existing TRAIN fixture (no TEST output or estimates):
  python -B analyze_original_multiseed.py --self-test \
    --native-stats /path/to/assets/code/slicemech/eval/stats.py \
    --benchmark-fixture /path/to/benchmark_seed1_train

The runner's window_input_sha256 covers selected exported Window fields, not all
raw arrival arrays. This analyzer separately compares exported arrivals and
other exogenous values across all paired trajectories and training seeds; raw
source files and composition code also remain pinned by the specification.

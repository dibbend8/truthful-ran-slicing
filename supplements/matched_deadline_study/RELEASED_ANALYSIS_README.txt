Offline verification of released derived statistics

reanalyze_released.py imports the unchanged, hash-checked analyze_matched.py and
uses its truth_tables, strategic_tables and Bootstrap implementations directly.
It requires Python, NumPy, pandas and a pandas Parquet engine (pyarrow). No
network, traffic traces, trained-model execution or raw interval arrays are used.

Example from the matched_deadline_study directory:

python -B reanalyze_released.py \
  --release-analysis analysis_matched/default/both/all \
  --expected-findings-sha256 HASH_FROM_TRUSTED_RELEASE_INVENTORY \
  --mode all --output /absolute/path/to/a/new/default-both-check

The output directory must not exist and must lie outside the input directory.
The required digest anchors the original findings.json to the independently
obtained release inventory. Computing that digest from an untrusted findings
file alone does not establish authenticity. The script verifies every available
table listed by findings.json. All Parquet and statistical/case CSV tables are
required. The compact release may omit only redundant CSV copies of observation,
window, paired-observation and invariant tables; these omissions are recorded.
--mode truth or --mode strategic checks that mode from a report containing it.
An all-mode report can be used for either subset. --analyzer and --protocol
override the default sibling files for a relocated bundle; their recorded
SHA-256 digests must still match exactly.

Inputs used for recomputation:
  truth_observations.parquet
  strategic_paired_observations.parquet
The latter's JSON-encoded honest-tenant vectors are decoded before aggregation.
Each requested mode's case inventory must have its exact planned 32 windows.
Truth requires all 28 policies/seed instances and three tenant rows per window.
Strategic pairs require both frozen/retrained states, both strategic arms,
three seeds, three deviators, eight multipliers, primary payments for both arms,
and the same-policy fixed-weight payment ablation for Protected: 432 paired
rows per window. Duplicate, missing or extra grid observations cause refusal.

The script checks paired-gain identities and recorded warm-up assertions,
then reproduces all aggregate, paired-window and per-seed tables. Every
regenerated field is compared against Parquet and each included CSV version;
integer counts match exactly, numeric tolerance is atol=rtol=1e-10, and missing
values must agree. It uses the original recorded bootstrap replicate count and
seed. All input hashes and numeric differences appear in the final
DERIVED_ANALYSIS_VERIFICATION.json; the script never overwrites existing work.

Scope limitation: this is derived-data reaggregation, not independent replay.
Without raw traffic and interval arrays it cannot verify the simulator,
controller outputs, payment computation, queue conservation, raw provenance or
original case-file hashes. Case inventories link the released observations to
the anchored historical report, but the omitted files are not rehashed. The
record explicitly states raw_replay_verified=false. It preserves the original
exploratory status and CIs conditional on three fitted seeds; it does not turn
the eight fixed reporting multipliers into an adaptive or exhaustive strategy.

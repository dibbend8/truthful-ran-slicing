Matched-deadline training runner
================================

train_matched.py calls the recovered native PPO implementation without changing
its source. It starts each model from nets.make_net(kind, seed), never from a
fitted checkpoint. It composes only TRAIN and the DEV selection half. No TEST
window is loaded or evaluated by this runner.

Scientific grid
Three load calibrations (rho0.7/default/rho1.1), three changed profiles
(video300/interactive50/both), four learned arms (protected/unconstrained/
meas_only/kappa0), seeds 0/1/2: 108 jobs. Original-deadline trained models remain
the recovered reference. Stage A is default/both (12 jobs); B is the other two
loads/both (24 jobs); C is the isolated profiles at all three loads (72 jobs).
Stages B and C are additional jobs, not cumulative stage commands.

For every scientific job: 400 iterations, 32 sampled windows from the original
180 TRAIN windows, 8 Gaussian samples/state, 4 PPO passes, minibatch 2048,
clip .2, learning rate 3e-4 linearly to 3e-5, initial sigma .6 bounded [.05,1],
entropy coefficient .001, gradient clipping 1, reward horizon explicitly 3,
measurement lag 2, CPU float64 and one numerical-library thread. The existing
per-arm selected hyperparameters are verified, not retuned. DEV selection uses
exactly 66 windows and checkpoints 50,100,...,400, with the earlier checkpoint
winning an exact tie. The remaining 60 DEV windows are excluded from selection.
The same new deadline is applied to queue expiry and contract/context features
in TRAIN and DEV. The operator forecaster and other contract fields are fixed.

Use Python from a compatible environment with numpy, pandas, pyarrow, torch,
numba, scipy and PyYAML. Exact installed versions are included in every job's
specification; a changed environment cannot silently resume an old result.
The validated local interpreter is ../icc2027-check-env/bin/python, relative
to this study's parent work directory. Use the actual server interpreter there.

CLI examples (replace /path values with the relocated bundle paths):

  python -B train_matched.py --plan
  python -B train_matched.py --plan --stage A
  python -B train_matched.py --assets /path/upstream \
    --frozen-study /path/deadline_load_study --output /path/matched_deadline_study \
    --prepare-only --stage A
  python -B train_matched.py --assets /path/upstream \
    --frozen-study /path/deadline_load_study --output /path/matched_deadline_study \
    --job default/both/protected/0

Scientific preparation and training require OUTPUT/PROTOCOL.txt, or an explicit
--protocol PATH. Its exact hash is part of every specification. A draft protocol
is never selected automatically. --plan is read-only and needs no imports/data.
--prepare-only verifies recovered inputs and writes specifications, but does not
compose windows, initialize/train models, or evaluate policies.

Diagnostics, never scientific results:

  python -B train_matched.py --smoke --job default/both/protected/0
  python -B train_matched.py --benchmark --job default/both/protected/0

Smoke uses 2 iterations, 2 TRAIN windows, 2 DEV-selection windows, and evaluation
each iteration. Benchmark uses 3 iterations, the full 180-window sampling pool,
32 windows/iteration, M8, and all66 DEV-selection windows at iteration3. These
short runs compress the learning-rate schedule; their models are diagnostics,
not shortened scientific runs or checkpoints that can resume full training.
Their outputs are under smoke/ and benchmark/, never models/.

Execution and output
One process runs one job at a time. --workers currently accepts only1. A server
orchestrator may launch independent --job subprocesses with separate logs and
a memory-based concurrency limit. --all runs the108 jobs serially; --stage runs
its fixed subset. Within a serial invocation, composed windows are reused for
the same load/profile bucket. Portable spawn/fork orchestration is deliberately
outside this runner, avoiding a change to native sample ordering or PPO math.

Scientific paths: OUTPUT/models/CONFIG/PROFILE/ARM/seedN/
Diagnostic paths: OUTPUT/{smoke,benchmark}/CONFIG/PROFILE/ARM/seedN/
Each result contains selected model.pt, final_model.pt, history.csv, config.json,
initial_rng.json, window_sampling_schedule.npz, window_inputs.json,
prepared_inputs.json and COMPLETE.json. The schedule reproduces the native
seeded window draws; the native PPO call still draws them itself. Input hashes
cover exact arrays, shapes and dtypes; context/allocation-input hashes are also
recorded. The initial RNG record describes the native seeded NumPy and Torch
generators and records the initial network parameter digest. Per-iteration
history preserves state counts, skipped updates and all native diagnostics.

Each job validates the chosen checkpoint by independent native DEV replay,
then safely reloads the serialized checkpoint (torch weights_only=True) and
repeats DEV replay. It also compares the selected model's first TRAIN window
against the previously validated standalone evaluator, without payments.
Each comparison must agree within 1e-9 relative tolerance. No TEST gate is
required or bypassed for this TRAIN/DEV-only work.

Results are assembled in a unique pending directory and renamed to the final
path only when complete. COMPLETE.json hashes all deliverables and the full
specification. Existing complete results are reused only when every hash and
the entire specification match. Incomplete/stale/corrupt results are refused;
nothing is overwritten. A crashed pending directory is retained for inspection.
No optimizer state is saved for in-job continuation: after interruption, use a
new output root and restart from the original seed, recording the restart.
After an unclean kill, inspect and remove only that job's stale .lock before a
new attempt; never remove another active worker's lock.

Recovered assets must remain unmodified. They are verified against the original
GitHub recovery manifest (starkgit91/MTP commit
c5ad2319634209565ae201119ae0c71121eae77f) and the separately recorded user-archive
Parquet hashes. The asset inventory reads only the needed TRAIN/DEV Puffer files,
prepared TRACTOR/Raca caches, load calibrations and historical selection record.
Source, specification, prior TRAIN-only shape and dependency hashes are recorded.
Missing caches fail verification before native readers can build or download
them. The runner never downloads assets or changes the original preregistration.

Static reference
This runner covers the108 learned-model jobs only. Refit static per load/profile
separately using native static.fit_static on the same180 TRAIN windows with
method='grid', steps=20, floor=.05, seed=0. Then use the original
experiments/train.py static_dev_choice rule: top72 distinct TRAIN-ranked
candidates, select by delivery welfare on the same66 DEV-selection windows,
with earlier TRAIN rank winning a tie. Save both TRAIN/DEV candidate scores.
This72-candidate static search is not the same search budget as eight neural
checkpoints. Equal allocation and Direct do not need training; Direct uses the
same new TRAIN-fit shape. Do not substitute frozen static as a refitted result.

Interpretation
This experiment measures local service deadlines in trace replay. It does not
establish 3GPP QoS compliance, an end-to-end packet-delay guarantee, application
playback reliability or practical strategyproofness. The original TEST dates
were already inspected; any later evaluation remains an exploratory extension.

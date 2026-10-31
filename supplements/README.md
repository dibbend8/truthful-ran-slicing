# Matched-deadline supplement — v0.2.0

This directory contains the source and instructions for
[v0.2.0](https://github.com/dibbend8/truthful-ran-slicing/releases/tag/v0.2.0),
dated 2026-10-31. The study completed all 108 neural training runs, nine static
refits, nine truthful-performance settings and three joint-deadline reporting
settings. Together these comprise 384 mode/window cases and 36,864 replayed
trajectories. See the [results](RESULTS.txt) and [validation record](VALIDATION.md).

The study refits four controller architectures and a static
reference under three changed deadline profiles and three target loads. The
frozen protocol specifies the training budget, three seeds, TRAIN/DEV
selection, fixed TEST windows, eight reporting multipliers, and exploratory
interpretation. Each of the twelve mode/settings has 32 windows; reporting
tests cover the three loads with both deadlines tightened. The nine analysis
directories combine truthful and reporting results for those three settings.
See [PROTOCOL.txt](matched_deadline_study/PROTOCOL.txt) and the
[resource addendum](matched_deadline_study/SIX_HOUR_ADDENDUM.txt).
The [execution note](matched_deadline_study/EXECUTION_PARALLELISM_NOTE.txt)
documents the later increase from eight to up to sixteen concurrent training
processes after the strategic evaluations. It changes scheduling only; the
frozen scientific scope, inputs, seeds, budgets and selection remain unchanged.
It also records early execution of the planned `rho1.1/interactive50` truthful
evaluation and analysis when its models became available, independently of TEST
performance.

The retained data schema uses `state=frozen` for original-deadline training and
`state=retrained` for matched-deadline training; figures label these “Original
training” and “Matched training.” Separately, `bill=fixed_ablation` identifies
Protected's frozen-priority payment comparison. These original identifiers stay
unchanged in data and provenance so their recorded hashes remain valid.

## What is in Git

| Directory or file | Purpose |
| --- | --- |
| `matched_deadline_study/` | Exact runner, analysis, plotting, release-validation and packaging sources, with protocols and instructions |
| `deadline_load_study/` | The unchanged prior-study evaluator and deadline-shape/checkpoint support code required by the new runners |
| `SOURCE_MANIFEST.json` | SHA-256 and byte count for each copied source, protocol or instruction file |
| `RESULTS.txt` | Completed study findings and scientific limitations |
| `VALIDATION.md` | Completed software checks and their limits |

The original `implementation/` tree and historical numerical summaries are
unchanged. New sources are copied byte for byte; their old recorded author
paths are provenance, not portable setup instructions. No manuscript, server
connection settings, credentials, private Drive links, audit backups, raw
traffic, models or large result tables are committed in this update.

To inspect the work plan without training or reading traces, run from the
repository root:

```bash
python -B supplements/matched_deadline_study/train_matched.py --plan --stage A
python -B supplements/matched_deadline_study/evaluate_matched.py --mode truth --plan
```

## Use the completed release archive

The [versioned release](https://github.com/dibbend8/truthful-ran-slicing/releases/tag/v0.2.0)
provides `truthful-ran-slicing-v0.2.0-evidence.tar.gz` (104,041,036 bytes), with
complete model artifacts, original baselines, exact specifications,
calibration/shapes, derived observations, statistical tables and figures.
Its SHA-256 is:

```text
6b4b5e649cd17c519cbb3c2d1f8bd1a9d8e3c7ca52fd8f433c3799a9ab97b5cf
```

The archive explicitly identifies all twelve included mode/settings. Its
server validation passed all 384 case records, 117 completed model/refit
artifacts, 356 historical/selected/final checkpoint files and nine analyses.

Extract that archive into a new directory. Its root contains sibling
`matched_deadline_study/` and `deadline_load_study/` directories. This is the
complete artifact layout; this Git source snapshot alone lacks the numerical
prerequisites. From the extracted archive root, use a new output path outside
that directory:

```bash
python -B matched_deadline_study/validate_public_release.py \
  --release-root . --output ../release-validation.json
```

The validator also creates `../release-validation_artifacts/`; both output paths
must be unused. It checks immutable file/model hashes, imports the restored
source, safely loads checkpoints, passes the unchanged historical validation
gate, recomputes derived statistics and regenerates figures without network
access or raw traffic reads.

For targeted statistical reanalysis, use
[RELEASED_ANALYSIS_README.txt](matched_deadline_study/RELEASED_ANALYSIS_README.txt).
The trust anchor is the findings-file hash from the verified release manifest.
The adapter uses the unchanged analyzer's statistical functions and compares
regenerated estimates, confidence intervals and counts with the released tables.
This verifies derived experimental outputs; it does not rerun the simulator or
independently verify omitted interval-level payment and queue computations.

## Full training and replay prerequisites

The public archive omits original raw/prepared traffic and the large raw replay
arrays. It lists their exact filenames, byte counts and SHA-256 hashes in
`EXTERNAL_REQUIREMENTS.json` and `RAW_EVIDENCE_REQUIREMENTS.json`. Obtain those
inputs under the relevant data-access conditions before full replay or raw
verification. Dataset sources are [Puffer](https://puffer.stanford.edu/data-description/),
[TRACTOR](https://github.com/genesys-neu/TRACTOR), and the
[Raca 5G dataset](https://github.com/uccmisl/5Gdataset). Availability of a source
URL is not a claim that exact prepared-cache bytes have been regenerated here.

The original TEST-read gate checks all **309 frozen assets and eight calibration
JSON files**, including historical settings outside the new comparisons. The
release exporter includes this full verified dependency set; it does not remove
or relax the gate. The [gate inventory](matched_deadline_study/FROZEN_GATE_REQUIREMENTS.json)
records the required bytes. Their inclusion does not imply those historical
experiments were rerun.

See the [training instructions](matched_deadline_study/TRAINING_README.txt),
[evaluation instructions](matched_deadline_study/EVALUATION_README.txt), and
[analysis instructions](matched_deadline_study/ANALYSIS_README.txt). Runners take
explicit asset, model and output paths and refuse incompatible hashes or partial
scientific grids. Do not use smoke or benchmark models as research results.

## Environment and citation

A compatible Python 3.11+ environment needs NumPy, pandas, pyarrow, PyTorch,
Matplotlib, SciPy, PyYAML and Numba; these are already listed in the repository's
`requirements.txt`. Exact training and replay versions are recorded in each
release evaluation specification's `dependencies` and each model record's
`spec.dependencies`. They describe those stages, not a reconstructed historical
environment. `requirements-verified.txt` remains the original v0.1.0 macOS
validation environment and is not a lock for the newer server study.

`CITATION.cff` identifies v0.2.0 with its fixed release URL and date. The
[v0.1.0 release](https://github.com/dibbend8/truthful-ran-slicing/releases/tag/v0.1.0)
preserves the original source package and historical summaries separately.
No DOI or new software/data license is assigned. Existing third-party notices
and conditions remain applicable.

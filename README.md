# Truthful Reporting for Learning-Based RAN Slicing

Research implementation accompanying the ICC 2027 manuscript by Prashant Mishra,
Rahul Kushwah, and Dibbendu Roy.

Version **v0.2.0**, dated **2026-10-31**:
[versioned release](https://github.com/dibbend8/truthful-ran-slicing/releases/tag/v0.2.0).
The completed matched-deadline study contains 108 neural training runs, nine
static refits, nine truthful-performance settings and three joint-deadline
reporting settings: 384 mode/window cases and 36,864 replayed trajectories.
See the [results](supplements/RESULTS.txt),
[reproduction instructions](supplements/README.md), and
[validation scope](supplements/VALIDATION.md).

The separately validated release archive is
`truthful-ran-slicing-v0.2.0-evidence.tar.gz` (104,041,036 bytes). Its SHA-256 is:

```text
6b4b5e649cd17c519cbb3c2d1f8bd1a9d8e3c7ca52fd8f433c3799a9ab97b5cf
```

Models and analysis tables belong in that archive, while raw traffic and large
interval-level replay files remain external with exact hash records. Large
artifacts are not committed to this source repository. The original
[v0.1.0 release](https://github.com/dibbend8/truthful-ran-slicing/releases/tag/v0.1.0),
dated 2026-10-30, remains a separate fixed version.

The remaining sections describe the original v0.1.0 package. Its implementation
and recorded numerical summaries are preserved unchanged.

The package contains the allocator and payment mechanisms, controller training
and evaluation code, tests, experiment specifications, and recorded numerical
summaries. The source is drawn from the project associated with the supplied
six-page draft. That original PDF has SHA-256:

```text
eeb1a0fce4573e81954c0e627f346f811b2306e58109c8978c54a5d23688a9e2
```

The PDF match identifies the source project; it does not by itself verify every
experimental result. The full training, trace replay, and ns-3 campaign have
**not been rerun while preparing v0.1.0**. Recorded summaries are historical
research outputs. Regenerating a figure from them reproduces its presentation,
not the underlying experiment.

## Contents

| Path | Purpose |
| --- | --- |
| `implementation/code/slicemech/` | Allocation, scores, prices, queue replay, tenant strategies, controllers, data loading, and NR adapter |
| `implementation/code/experiments/` | Training, E0–E7 evaluation, analysis, and figure scripts |
| `implementation/code/tests/` | Unit, property, and integration tests |
| `implementation/spec/` | Parameters, controller labels, notation, glossary, and claim records |
| `implementation/prereg/` | Historical protocol and post-freeze record |
| `implementation/nr/` | Unbuilt ns-3 scenario, simulator patches, and upstream license information |
| `implementation/results/summary/` | Recorded TEST summaries; DEV and training-smoke summaries in subdirectories |
| `DATA.md` | Required external data, model, and simulator assets |
| `CITATION.cff` | Software citation metadata |
| `PACKAGING_CHANGES.md` | Portable test fixture and patch-header changes |
| `IMPLEMENTATION_NOTES.md` | Units and conventions needed to interpret the results |
| `VALIDATION.md` | Selected test results, figure regeneration, and validation limits |
| `source-provenance.json` | Original mechanism and experiment source hashes |

The mechanism and experiment source files retain their original bytes so that
their provenance can be checked. The test fixture's oracle path was made portable;
the reference oracle is included unchanged. Some original source comments and legacy shell
drivers refer to the research machine or historical documents that are outside
this package. Use the commands below for this layout. Development notes, unrelated
projects, local environments, large raw traces, and trained checkpoints are not
part of this source package.

## Environment

The synthetic checks below passed in a fresh **Python 3.12.14 environment on
macOS arm64**. Other Python versions and operating systems have not been validated
for v0.1.0. The data preparation code uses POSIX file locking. The
requirements list reflects imports in the source and is not a reconstruction of
the historical environment. Use `requirements-verified.txt` for the dependency
versions recorded by the release validation run.

From this repository's root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-verified.txt
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
```

PyTorch is used on the CPU by these experiments; a GPU is not required. Numba compiles the
fluid kernels on first use, so the first test invocation includes compilation.

## Small synthetic checks

These explicit tests use synthetic in-memory inputs. They do not need external
traces, trained checkpoints, or an ns-3 installation, and do not run the full
training campaign. All seven passed during release preparation (10.59 s in the
validation environment). Run from the repository root:

```bash
PYTHONPATH=implementation/code PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -B -m pytest -q -p no:cacheprovider \
  implementation/code/tests/test_fluid.py::test_hand_example_expiry_and_late \
  implementation/code/tests/test_fluid.py::test_zero_delay_budget_serves_only_in_arrival_slot \
  implementation/code/tests/test_fluid.py::test_deadline_spans_interval_boundaries_and_backlog_carries_over \
  implementation/code/tests/test_tenants.py::test_truthful_and_admission_counting \
  implementation/code/tests/test_tenants.py::test_multiplier_series_matches_per_interval \
  implementation/code/tests/test_stats.py::test_holm_known_values \
  implementation/code/tests/test_stats.py::test_interval_metrics_and_welfare
```

The full test directory includes large randomized checks, repeated bootstrap
simulations, integration fixtures, and a training subprocess. The exact rational
oracle used by the mechanism tests is included under
`implementation/code/tests/reference/`; its import path is configured relative to
the fixture. The complete suite should be run only after its additional data and
integration assets are configured.

## Regenerate figures from recorded summaries

This step reads the included summaries and writes figures to a new output folder.
It does not download data, train a controller, or recompute reported estimates.

```bash
python -B implementation/code/experiments/figures/make_all.py \
  --split test --out reproduced-figures
```

The runner reports each figure as `ok` or `skipped`. Check its exit status and the
generated files. To regenerate DEV figures, use `--split dev`; to use the supplied
training-smoke summaries, use `--split train --smoke` and a separate output folder.

## Rerun experiments

See `DATA.md` before running the experiment drivers. A complete run additionally
requires the trace sources, prepared calibrations, trained models and provenance
records, plus the matching patched ns-3/5G-LENA setup for E7. Those assets are not
implied by the presence of the source code or summaries.

After providing the required assets, run drivers from `implementation/code/`.
Start by inspecting the training plan:

```bash
cd implementation/code
python -B experiments/train.py --plan
```

The training driver documents `--plan` as listing jobs and estimated work without
running them. A small TRAIN-only training run, once its data dependencies exist,
is available as:

```bash
python -B experiments/train.py --smoke --split train --workers 1 --mem-gb 4
```

`--smoke` still performs training and reads trace data. Its results are separate
from the paper's recorded TEST results. The full workflow is: prepare/calibrate
data; train and select controllers; run E0 and E1; run E2 followed by E3 and E4;
run E5/E6; run E7 with the configured simulator; then analyze and render figures.
The experiment drivers' `--help` output describes their arguments.

Historical TEST execution is protected by code, configuration, checkpoint, and
pre-registration hashes. This package does not bypass those checks. An incomplete
asset set or a post-freeze change can cause the gate to refuse execution. Consult
`implementation/prereg/README.md` for the distinction between the historical
freeze and this publication package.

## Citation and rights

Use `CITATION.cff` to cite the software version used in a study. A fixed release
URL or a subsequently assigned version DOI should be recorded with experimental
results. No DOI has been assigned in this package.

No standalone software license was present in the selected source subproject.
This release does not assign a new license or change the rights applicable to
third-party data or simulator components. Their source and license notices must
be retained when obtaining or redistributing those components.

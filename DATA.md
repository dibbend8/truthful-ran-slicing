# Data and external assets

This source package supports inspection of the mechanisms, synthetic checks, and
figure regeneration from recorded summaries. A complete rerun requires assets
that are not included here. The table identifies the corresponding code so that
the dependencies can be checked without relying on the original machine layout.

| Asset | Used by | Requirement |
| --- | --- | --- |
| Puffer video-size traces | `slicemech/data/puffer.py`, `puffer_fetch.py` | Original dated video-size objects or verified prepared tables |
| TRACTOR traffic captures | `slicemech/data/tractor.py` | URLLC-like and mMTC-like traffic CSVs with the expected columns |
| Raca 5G production traces | `slicemech/data/raca.py` | Contiguous 5G CQI runs from the referenced production dataset |
| Calibration records | `slicemech/data/calibrate.py` | Training-derived rates, shapes, contracts, and forecasting parameters |
| Controller checkpoints | `experiments/registry.py` | Matching `model.pt` and `config.json`, static fits, and associated provenance |
| Raw evaluation records | `experiments/analyze.py` | Per-window outputs and provenance sidecars for recomputing summaries |
| ns-3/5G-LENA | `slicemech/nr/run.py`, `experiments/e7_nr.py` | Matching scenario source, simulator patches/build, calibration, and run settings |

## Data acquisition and layout

The Puffer fetch module names the public `puffer-data-release` storage bucket and
validates object metadata and downloaded bytes. It performs network access only
when acquisition is explicitly requested. Its TEST download path requires the
historical freeze and associated assets; do not remove these checks to make a
command run.

The preserved Puffer loader looks for historical TRAIN/DEV trace directories in a
sibling project. The TRACTOR and Raca loaders similarly use a sibling `data/raw/`
tree. Prepared caches belong under `implementation/data/`. These legacy input
locations are not a promise that the corresponding files are distributed in this
release. A portable, complete data distribution must provide the exact inputs,
their checksums, and an explicit mapping to the loaders. Honor each dataset's
access conditions and redistribution terms.

Training and evaluation dates are encoded in the source and the archived
pre-registration. They are not the publication date of this package. TRAIN is
used for fitting; DEV is used for selection and development estimates; TEST is
reserved for the frozen evaluation protocol. The recorded TEST rule begins with
16 Sundays/Mondays from September through October 2024, with replacement and
extension rules in the archived record.

## Simulator configuration

The NR adapter reads `NS3_DIR` for the simulator installation. E7 reads `E7_NR_OUT`
for its output base. Set those to appropriate local directories before using E7.
The preserved default values refer to the original research machine and should
not be treated as installation instructions.

The custom scenario source and five NR patches are included under
`implementation/nr/`, with setup notes and license information. They have not
been built or validated in this release environment. A generic ns-3 installation
alone is not sufficient to reproduce the recorded experiment. The frozen
NR calibration records hashes of the simulator binary and library; the TEST
driver checks these. Simulator source and patch provenance must accompany a full
reproduction, and its upstream license notices must be preserved.

## Recorded summaries and limits

The included summaries are copied from the matched research project. TEST
summaries are at `implementation/results/summary/`; DEV and training-smoke
summaries are in their own subdirectories. They contain tables, estimates,
metadata, and identifiers used by the plotting scripts. They do not replace the
raw window records needed to independently recompute statistical estimates.

The optional `exploratory.json` is excluded from this minimal package because it
contains machine-specific provenance paths and is not read by the included
figure scripts. No retained numerical summary was altered during staging.

The complete campaign was not rerun for this release. Passing unit checks or
rendering the supplied summaries should not be described as independently
reproducing all reported experimental results.

# v0.2.0 supplement validation

The matched-deadline study completed all 108 neural training runs, nine static
refits, nine truthful-performance settings and three joint-deadline reporting
settings. This record distinguishes artifact validation from repeating those
scientific runs on another machine.

## Full release archive

The final archive `truthful-ran-slicing-v0.2.0-evidence.tar.gz` has 104,041,036
bytes and SHA-256
`6b4b5e649cd17c519cbb3c2d1f8bd1a9d8e3c7ca52fd8f433c3799a9ab97b5cf`.
Every archive member passed readback verification. The server's final
`RELEASE_VALIDATION.json` reports `complete=true` and has SHA-256
`b6fe647609448030061429c05be500c75da5481ed0a3707815b9b9f8e2bf1521`.
It verified:

- 2,672 included files, twelve mode/settings and 384 evaluation case records;
- 117 completed matched-model/refit artifacts and safe loading of 356 checkpoint
  files: 140 historical, 108 selected and 108 final neural checkpoints;
- the unchanged historical TEST-read gate and exact dependency hashes;
- derived-table reaggregation for all nine analysis directories;
- four regenerated plots and 37,248 external raw-evidence hash references.

The same final archive passed an independent fresh-extraction validation on
macOS after its byte count and SHA-256 were checked. Its local report also
records `complete=true`, with 2,672 files, 384 cases, 117 completed model/refit
artifacts, 356 safe checkpoint loads, nine reanalyses and four regenerated plots.
`LOCAL_RESTORE_VALIDATION.json` has SHA-256
`f0b422304b8f414ff0c42ce64ee5758fd823e4e1ea8584284c506e3493090b13`.
The server used Python 3.11 and PyTorch 2.14; local validation used Python 3.12
and PyTorch 2.15. Neither full validation performed a raw-traffic read,
simulation replay, training or network access. Both record
`raw_replay_verified=false` and preserve
the original scientific file bytes. External evidence hashes are cross-references,
not a claim that omitted raw arrays were read by this offline validator.

## Earlier checkpoint and software checks

An earlier public-archive checkpoint for the completed `default/both` truthful and
strategic cells passed restored-layout validation on the study server with
Python 3.11 and PyTorch 2.14, then from a freshly extracted archive on macOS with
Python 3.12 and PyTorch 2.15. Archive hashes were checked before extraction.
Checks included exact file inventories, safe loading of 164 selected/final or
historical checkpoints, the unchanged native historical TEST-read gate, one
combined analysis and two regenerated figures. No traffic was read and no new
training or simulator replay was performed by this offline validation.

Reaggregation of the actual released default/both observations reproduced all
17 analysis tables, including confidence intervals and seed summaries, with
maximum numeric difference 0.0 in the direct checks. Missing or duplicated grids,
changed findings hashes and existing output destinations were rejected. Compact
archives were checked with redundant CSVs omitted while retaining their Parquet
counterparts. The original analyzer and scientific inputs were unchanged.

The full historical gate dependency set was independently checked: all 309
freeze-listed assets and eight calibration files matched their original hashes.
Every one of the 140 historical `model.pt` files had its matching configuration.
The release validator safely loads models with restricted PyTorch loading;
it does not enable arbitrary pickle globals or disable provenance checks.

The source snapshot in this Git update is verified against the study source
files by `SOURCE_MANIFEST.json`. The public results report differs from its
source only by removal of its final publication/pagination status paragraph;
both hashes and that transformation are recorded separately. Detailed
machine-readable validation records accompany the corresponding release artifact.

Limitations remain explicit: reaggregation checks derived observations, while
raw traffic, interval-level replay and its original computations require the
separately listed external assets. The confidence intervals remain exploratory
and conditional on the fitted seeds. These checks do not establish practical
strategyproofness, packet reliability or end-to-end 3GPP QoS compliance.

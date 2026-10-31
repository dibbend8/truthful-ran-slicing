Public v0.2.0 release preparation

package_public_release.py exports a NEW reviewable archive from payload/ of a
completed bundle_matched.py private snapshot. It never publishes, uploads or
changes an existing repository. Source private payload SHA256SUMS and exact
per-case/model/analysis inventories are checked before copying. Scientific
bytes are preserved; only explicitly labeled release utilities and metadata are
new. Benign historical author filesystem paths remain in original provenance.

Default scope requires9 truthful cells and3 jointly tightened strategic cells,
each with all32 windows. --allow-partial explicitly permits a subset and records
all missing requested cells. Planned run records do not imply completion.

Examples:
  python -B package_public_release.py --private-payload /private/payload --dry-run
  python -B package_public_release.py --private-payload /private/payload \
    --output /new/reviewed-public-v0.2.0

For a private snapshot created before the full historical-gate dependency fix,
also pass --assets /path/to/original/upstream. The exporter verifies and adds all
309 original frozen assets plus8 calibration JSONs without modifying any gate or
scientific source. FROZEN_GATE_REQUIREMENTS.json lists their exact hashes. Later
private bundles already contain these files, so the default upstream suffices.

When available, EXECUTION_PARALLELISM_NOTE.txt is retained unchanged. It records
execution-only scheduling changes: increased training concurrency after the
strategic evaluations and early execution of a planned evaluation once its
models were available. Scientific sources, seeds, budgets and scope remain
unchanged. If the note is present in both the private snapshot and utility source
directory, its hashes must agree. Frozen protocol/addendum bytes are preserved.

Retained: exact source/spec/protocol, original frozen comparator files, matched
selected/final models with their complete training artifacts, calibration/shapes,
original recovery metadata that passed privacy checks, evaluation COMPLETE/spec
records, all analysis Parquet tables and statistical/case CSVs, verified figures.
Excluded: raw/prepared input traffic; per-interval evaluation NPZ and result.json;
redundant observation/window/invariant CSVs; smoke/synthetic artifacts; server,
SSH, credential, Drive-sharing, environment/cache and log files. Omitted raw
files retain exact hashes in RAW_EVIDENCE_REQUIREMENTS.json. Input prerequisites
are listed separately in EXTERNAL_REQUIREMENTS.json. Original findings keep
all their original table hashes, including redundant CSVs deliberately omitted.

The public adapter reanalyze_released.py recomputes statistical tables from
released derived experiment observations. It does not establish a new simulator
run. The original analyze_matched.py remains unchanged and still requires the
original traffic/raw evidence for its complete independent checks. Restoring
those prerequisites preserves its original hash/gate behavior.

Before making the tar.gz, the exporter runs validate_public_release.py against
the new restored layout: exact file/model verification, offline core imports,
safe checkpoint loading, and figure regeneration in a separate temporary output
location. Any failure stops packaging and leaves only the new output directory
for inspection. Existing files are never overwritten. Every final archive member
is streamed back and checked, and archive SHA-256 is recorded. No release URL or
DOI is asserted to exist by this preparation tool. A publication step remains
separate, after review of the concrete resulting archive.

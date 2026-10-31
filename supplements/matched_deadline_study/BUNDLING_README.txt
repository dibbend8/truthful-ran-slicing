Matched-deadline evidence packaging

bundle_matched.py writes a NEW local private evidence snapshot. It never uploads,
publishes, trains, evaluates, downloads data or reads server connection settings.
The default requested scope is strategic/both at default,rho0.7,rho1.1 plus all
nine truthful cells. Only complete AND analyzed cells enter the bundle. Missing
cells are listed in BUNDLE_SCOPE.json; partial scope is explicit. --require-all
refuses any missing requested cell. Existing output directories are refused.

Examples (run from this study directory):
  python -B bundle_matched.py --dry-run
  python -B bundle_matched.py --output /new/path/matched-evidence
  python -B bundle_matched.py --cell truth/default/both --require-all \
    --output /new/path/matched-default-both-truth
  python -B bundle_matched.py --diagnostic-smoke --dry-run

--cell MODE/CONFIG/PROFILE is repeatable and overrides the default requested
scope. --root, --assets, --frozen-study, --analysis-root, --model-root and
--original-release support relocated inputs. Original-release coverage is a
local package inspection, not an assertion about current online attachments.
Default model roots are models/ for scientific data and smoke/ for diagnostics.
The only diagnostic packaging operation is --diagnostic-smoke --dry-run; it
never creates an archive or treats TRAIN smoke as scientific evidence.

Each selected scientific cell must have all32 planned cases, immutable specs
and complete model artifacts. The analyzer independently validates the raw
arrays, paired gains, model/data/source hashes and exact case grid before
packaging. Analysis findings must match the current source/protocol and exact
raw case bytes; every generated CSV/Parquet hash is checked. No scientific
estimate is computed or selected by the packager. Model subset selection follows
included cells, not outcomes. The original v0.1.0 source package lacks the78
frozen checkpoint/static files for the full three-load comparator grid; required
frozen files are therefore included from recovered, hash-verified originals.
The unchanged historical TEST-read gate additionally requires all309 assets in
the original freeze block and all8 calibration JSONs, including checkpoints for
historical arms/settings outside the new comparisons. Their exact verified bytes
are now included and identified in FROZEN_GATE_REQUIREMENTS.json. This addition
does not claim those historical experiments were rerun and does not alter a gate.

The allowlist includes runner code, frozen protocols, instructions, diagnostic
validation JSON, selected completed raw cases, model artifacts, matching specs,
planned run manifests and analysis tables. When present,
EXECUTION_PARALLELISM_NOTE.txt is retained unchanged as an
execution-only scheduling record; it does not replace the frozen protocol or
six-hour addendum. Planned manifests may list additional
cells that are absent; they do not establish completion. The analysis subfolder
can contain an original all-mode report, whose scope remains stated in its
findings. BUNDLE_SCOPE is authoritative about raw cells actually archived.
Scientific figures are included only when their file hashes, plotting source
and every source analysis/table hash match included artifacts. Figures with
unbundled source tables are omitted explicitly; synthetic layouts are excluded.
Original supporting source/spec files, selected shapes/calibrations/prereg and
recovery metadata are included unchanged when they contain no private archive
locator or credential. Historical author filesystem paths in immutable scientific
provenance are retained in this private bundle; generated instructions use
relative paths. This is not a sanitized public-release transformation.

All raw/prepared traffic arrays remain external, including Puffer Parquet,
TRACTOR NPZ and Raca Parquet caches. EXTERNAL_REQUIREMENTS.json gives exact
paths, byte sizes and SHA-256 hashes. Private archive locator metadata, if any,
is also left external with its exact hash. Restore these inputs from the original
frozen study before rerunning. This bundle is self-contained code/evidence and
model provenance, not a standalone data distribution. Existing source/data
notices remain in force; no license is invented.

Excluded: server/, SERVER_STATUS.txt, drive_archive.json, credentials, private
sharing URLs, logs, environment folders, caches, pending/failed jobs, lockfiles,
unrelated projects, optional drafts and synthetic figure outputs. Every selected
text file is scanned for common credential formats and private Drive sharing URLs;
a match fails before archive creation. No matched bytes are printed.

Output has payload/, parts/ and ARCHIVE_MANIFEST.json. Payload SHA256SUMS covers
all payload files except itself. The gzip-compressed tar stream is written
directly into ordered64MiB parts, avoiding a duplicate unsplit archive. The last
part may be smaller. ARCHIVE_MANIFEST supplies part hashes and the concatenated
archive hash. Every archived member is streamed back and checked against its
payload hash. Parts concatenate to one tar.gz; they are not independent archives.
The packager retains payload files for review. A failed write leaves its new
output directory for inspection and never replaces existing evidence.

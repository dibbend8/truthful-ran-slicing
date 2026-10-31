Portable five-seed evidence preparation and offline reanalysis

These tools do not publish. The original analyzer and its complete 160-case
gate remain unchanged. A source-only staging directory contains no results and
cannot pass the scientific release validator.

Prepare source/instructions while the campaign runs:
  python -B prepare_original_package.py --skeleton --output /new/public_package

Only after all 160 cases and the independent analysis pass and the findings are
reviewed, create a NEW final directory (for example a child of public_package):
  python -B prepare_original_package.py \
    --campaign /path/campaign_v3 --analysis /path/complete_analysis \
    --assets /path/assets --frozen-support /path/frozen_support \
    --approved-findings-sha256 EXACT_REVIEWED_FINDINGS_SHA256 \
    --include-replay-tables --output /new/final_package

The optional --include-replay-tables includes all 800 derived episode Parquets.
Without it, the compact package contains paired observations, search decisions,
all analysis tables, original case metadata/completion records and exact
references for the omitted episode tables. It never includes external raw
traffic datasets. Model/code prerequisites for full replay are the immutable
v0.2.0 baseline archive AND its repository source tree, plus the relative
external input/hash list. The archive includes the required checkpoints and
core source; copy the repository's implementation/code/tests tree into the
restored deadline_load_study/upstream/code/tests directory as well. Those 31
test-source files are part of the runner's immutable source inventory, even
though the replay does not execute the tests. The 33 omitted prepared traffic
caches comprise 16 Puffer Parquets, one Raca Parquet and 16 TRACTOR NPZs.
EXTERNAL_INPUT_REQUIREMENTS.json and EXTERNAL_SOURCE_REQUIREMENTS.json retain
all exact relative paths and hashes. The package does not claim that a public
dataset URL guarantees identical prepared-cache bytes.

Full replay additionally requires the simulation dependencies recorded in the
specification and the unchanged native TEST gate. After restoring the v0.2.0
layout and all matching external inputs, create a NEW campaign directory and
copy source/PROTOCOL.txt into that directory. Run:
  python -B source/run_original_supplement.py \
    --assets /restored/deadline_load_study/upstream \
    --frozen-support /restored/deadline_load_study \
    --output /new/campaign --workers 20

The runner records the new environment and locators and preserves the original
source/input hashes. Do not resume original cases with a changed specification.
This full replay has not been rerun as part of package validation.

The packager verifies original source/input/case hashes, preserves the exact
runner/analyzer/protocol and raw scientific case bytes, and creates a separately
named portable specification. Its transformation ledger lists only removed
filesystem/repository locators with hashes of the removed values. The original
specification's file and canonical hashes are retained; the derivative is never
misrepresented as having the original hash. This does not weaken the native
campaign analyzer or require its historical private filesystem paths.

Every included text/Parquet string column is checked for private filesystem
paths, private IPs, credentials and private sharing URLs. Server connection and
sync tools, status logs, environments and unrelated materials are excluded.
Existing public releases remain unchanged.

The packager invokes the portable adapter against the restored payload before
creating an archive, then reads back and hashes every archive member. The
ARCHIVE_MANIFEST.json records archive bytes/SHA-256, the public inventory hash
and the offline verification report hash. The archive contains no private
absolute locators. All output locations must be new; failures are retained for
inspection and are not silently overwritten.

After verifying the downloaded archive SHA-256, extract it into a new folder and
run from its root using the public inventory hash in ARCHIVE_MANIFEST.json:
  python -B source/reanalyze_original_release.py --release-root . \
    --expected-manifest-sha256 EXPECTED_PUBLIC_PACKAGE_MANIFEST_SHA256 \
    --output ../new_reanalysis

The adapter requires exact inventories, all 160 complete case records, the full
480 seed/window/role grid and all 76,800 scored search decisions. It uses the
unchanged analyzer's aggregation functions to regenerate six tables, including
confidence intervals and per-seed results, and compares CSV, Parquet and JSON
representations. Missing/duplicate grids or changed data fail validation.
When episode tables are present it also reconstructs each paired observation,
checks warm-up, physical and utility identities, and compares exogenous arrays
across all five seeds. Otherwise it clearly reports zero episode tables reread.

Neither mode uses raw traffic, loads models, retrains policies or simulates new
traffic. Both are derived-output reanalysis, not a new independent replication
from original external data. Use an interpreter with NumPy, pandas and PyArrow
installed, for example in a dedicated virtual environment:
  python3 -m venv analysis-env
  analysis-env/bin/python -m pip install numpy pandas pyarrow
  analysis-env/bin/python -B source/reanalyze_original_release.py --help

Preparation diagnostics were checked with Python 3.12.14, NumPy 2.5.4,
pandas 3.1.0 and PyArrow 26.0.0. The replay campaign records Python 3.11.16,
NumPy 2.4.6, pandas 3.0.5 and PyArrow 25.0.1, along with its other simulation
dependencies in the portable specification. These are observed environments,
not a reconstructed historical training environment or a dependency lock.

Separate TRAIN-only diagnostics (never accepted as scientific output):
  python -B reanalyze_original_release.py \
    --analyzer analyze_original_multiseed.py \
    --diagnostic-train-fixture /path/benchmark_seed1_train

The paired-observation and confidence-interval methods, exploratory status and
limits are detailed in ANALYSIS_README.txt and PROTOCOL.txt.

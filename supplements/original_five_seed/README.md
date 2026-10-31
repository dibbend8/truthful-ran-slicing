# Five-seed original-setting reporting supplement

This reviewed supplement contains the complete results of an exploratory extension using five original training seeds over 32 prespecified replay windows from 16 recording dates. No network was retrained. All 160 cases and 800 trajectories passed the complete-data and provenance checks.

Honest on-time traffic loss under Unconstrained numerical best response is **19.13%** (conditional 95% CI **16.86–21.12%**; per-seed range **17.91–20.07%**) relative to truthful Unconstrained control. Protected truthful control improves the same honest tenants' weighted delivery welfare by **16.13%** (**13.65–18.48%**; per-seed range **14.40–18.11%**) relative to Unconstrained with a strategically reporting tenant. This second comparison uses separately trained controllers and does not isolate the effect of pricing. The seed-0 rerun reproduces the original 19.44% and 16.06% point estimates to floating-point precision. See [RESULTS.txt](RESULTS.txt), [headline estimates](analysis/headline_summary.csv) and [per-seed estimates](analysis/per_seed_summary.csv) for denominators, absolute units and limitations.

These intervals condition on the five fitted networks and are unadjusted exploratory intervals on reused TEST data. This extension is outside the original Holm family and does not test Protected deviations or rerun packet simulation. The original packet check remains a one-seed result. The v0.2.0 matched-deadline study remains unchanged at 108 neural fits and nine static refits, with three training seeds. Unconstrained search gains are numerical lower bounds, not certified global optima.

The evidence archive is distributed with [release v0.3.0](https://github.com/dibbend8/truthful-ran-slicing/releases/tag/v0.3.0). The immutable v0.2.0 baseline remains at https://github.com/dibbend8/truthful-ran-slicing/releases/tag/v0.2.0.

The full archive `original-five-seed-evidence.tar.gz` contains all 800 derived episode Parquets and all six analysis tables in CSV and Parquet. It is 63,471,274 bytes, SHA-256:

```
628778ba673d85f82d4749d5aa6301d4ff218781ba7f8fefb860a4d72ffbbf55
```

Verify this hash before extracting. From the extracted archive root, run:

```bash
python -B source/reanalyze_original_release.py --release-root . \
  --expected-manifest-sha256 0b841f521770bc904491f5cd9966910bd9e2ee098da09e678c52e2c48bc10175 \
  --output ../new_reanalysis
```

Use Python with NumPy, pandas and PyArrow installed. The full fresh-archive reanalysis passed with Python 3.12.14, NumPy 2.5.4, pandas 3.1.0 and PyArrow 26.0.0. It reread all 800 episode tables, checked the complete pairing grid and regenerated all six tables with maximum numeric difference 0.0. The output directory must be new and outside the extracted archive. The small Git subset alone omits the episode and full observation tables; use the attached archive for this command.

The adapter does not read external raw traffic, load models, train or run a new simulation. Full replay additionally needs the public v0.2.0 archive and repository source tree, plus the exact external prepared traffic caches listed by hash. See `source/PORTABLE_README.txt` and the external-requirements JSON files. The portable specification explicitly records removed locator fields and retains original content hashes; original scientific case bytes in the archive are unchanged.

No credentials, server connection metadata, private sharing URLs, external raw traffic or model blobs are included in this Git subset. `evidence/PUBLIC_PACKAGE_MANIFEST.json` is the inventory of the full evidence archive, not the inventory of this small Git subset. `GIT_SUBSET_MANIFEST.json` inventories this directory.

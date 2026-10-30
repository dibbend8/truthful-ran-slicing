# Release validation — 2026-10-30

Validation used Python 3.12.14 on macOS arm64 in a fresh virtual environment.
The complete installed package versions are in `requirements-verified.txt`.
They describe this validation, not the historical training environment.

## Synthetic checks

The seven queue, tenant, and statistics cases listed in the README passed
in 10.59 seconds. A further 11 mechanism cases passed in 8.44 seconds:

```bash
PYTHONPATH=implementation/code PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -B -m pytest -q -p no:cacheprovider \
  implementation/code/tests/test_allocation.py::test_stage_formulas_by_hand \
  implementation/code/tests/test_allocation.py::test_two_tenants_supported \
  implementation/code/tests/test_allocation.py::test_fill_is_weighted_maxmin_and_scale_invariant \
  implementation/code/tests/test_allocation.py::test_feasibility_bounds_and_zero_reports \
  implementation/code/tests/test_scores.py::test_maps_basic \
  implementation/code/tests/test_scores.py::test_ratio_condition_coefficient_rule \
  implementation/code/tests/test_scores.py::test_necessity_witness_detects_decrease \
  implementation/code/tests/test_pricing.py::test_e3_closed_form_premium \
  implementation/code/tests/test_pricing.py::test_menu_regression_premium_witness \
  implementation/code/tests/test_pricing.py::test_premium_nonnegative_nondecreasing_and_gain_identity
```

The feasibility function is parameterized, hence 11 cases from 10 named
functions. Together these two runs passed **18 cases**. The complete test suite
was not run. Numba and Matplotlib cache directories were redirected outside the
release source during validation; pytest plugins and its cache were disabled.

## Figures from recorded summaries

The README's TEST figure command completed with exit status zero: all 12 scripts
succeeded, producing 15 PDF/PNG pairs, including their manuscript variants.
PNG decoding and PDF headers were checked. This does not constitute a visual
equivalence check or a recomputation of the numerical estimates.

No training, trace-replay experiment, statistical analysis, or NR simulation
was rerun. The NR scenario and patches were not built or applied in this
validation environment. See `DATA.md` for assets required for a complete rerun.

## Provenance

All 150 source, specification, and summary files covered by the before/after
validation comparison were unchanged. Independently, all 74 packaged mechanism
and experiment Python/shell source files match the supplied project byte for
byte; their hashes are in `source-provenance.json`. Test-fixture and NR patch
header adaptations are documented in `packaging-transformations.json`.

`SHA256SUMS` records the committed package files other than itself. Generated
figures and temporary validation caches are excluded from the release.

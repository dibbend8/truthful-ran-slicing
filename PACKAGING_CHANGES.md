# Changes made for the publication package

The allocator, pricing, learning, simulation, and experiment-driver source under
`implementation/code/slicemech/` and `implementation/code/experiments/` is copied
unchanged from the matched research project. The original `PREREG.md` and
`DEVIATIONS.md` are also unchanged. New documentation and dependency files describe
this release rather than the original development environment.

## Portable test oracle

`implementation/code/tests/reference/rational_general.py` is copied unchanged
from the earlier exact-reference module used by the supplied tests. SHA-256:

```text
91c32b63a5b86cb29e17a1a8e04d9e9e88605bdec242af2add0b08230bfbcdef
```

The only change to `_core_fixtures.py` is the `_RG_DIR` assignment. The original
machine-specific absolute directory is replaced by:

```python
_RG_DIR = Path(__file__).resolve().parent / "reference"
```

The fixture's original SHA-256 is
`6abe9560f0d23a707199f222f88c8907b432d6572bf47df0eae547c156ccd1b1`;
the packaged SHA-256 is
`36617fd4243041b134fe4608ad9345122187c103d122c73d739a14d7de4dc985`.
The oracle and this fixture are outside the two directories used by the
historical analysis-code tree hash.

## NR source and patches

The scenario source is copied unchanged. Each supplied simulator patch has only
its first two filename headers normalized to `a/contrib/nr/model/...` and
`b/contrib/nr/model/...`. This removes machine-specific absolute paths and permits
standard patch handling from a simulator checkout. Every patch hunk, including
copyright and SPDX text, is preserved. This is a packaging change, not a claim
that the patches were successfully applied or that the simulator was rebuilt.

`packaging-transformations.json` records the source identifier, original hash,
packaged hash, and transformation for each affected or additionally included file.
The source issue note about symbol counting was not copied: it describes an early
pre-fix state and contains local trace paths. The NR README summarizes the issue
and distinguishes it from the later patch record.

## Included results and exclusions

Retained JSON summaries are copied without numerical or metadata edits. The
optional `exploratory.json` is omitted; no supplied figure script reads it.
Internal development notes, review correspondence, large raw data, model weights,
local environments, and unrelated projects are excluded. No historical freeze
check is disabled or rewritten, and no new study execution is implied by packaging.

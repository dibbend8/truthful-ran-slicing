# NR scenario and simulator patches

This directory contains the unbuilt scenario `vslice-nr-5g.cc` and five patches
from the supplied research project. It does not contain an ns-3 checkout,
compiled binaries, the original build environment, or simulator run outputs.
The sources have not been compiled or the patches applied during this release
validation.

The scenario identifies **ns-3.46 with 5G-LENA nr-4.1.1** as its base. Two patches
backport behavior from NR v5.0; the RLC patch records upstream commit
`6b9da7a06e43a875a449e12bba189160217839ae`. This describes the supplied source
provenance, not a claim that an arbitrary current NR checkout is compatible.

| Patch | Recorded purpose |
| --- | --- |
| `nr-gnb-phy-symbol-union.patch` | Count the union of occupied symbols for allocation statistics when DCIs overlap |
| `nr-rlc-um-v5.0-backport.patch` | Backport RLC-UM reassembly behavior |
| `nr-amc-v5.0-tbsize-backport.patch` | Backport transport-block size calculation |
| `nr-rlc-um-resync.patch` | Handle unexpected RLC-UM framing by resynchronizing and dropping a stranded partial SDU |
| `nr-ofdma-ul-notch.patch` | Remove an uplink check that used the downlink notch mask |

The early issue note reported a symbol-count assertion before the patches were
available. Later historical records describe additional RLC and uplink fixes.
The symbol-count patch affects statistics; the RLC patches can affect packet
delivery and must not be treated as cosmetic changes.

## Setup outline

Use a separate checkout of the matching upstream simulator and NR module. Keep
its upstream license and provenance records. From that checkout, inspect and
apply the supplied patches using their normalized relative paths. Apply the
RLC v5.0 backport **before** the RLC resynchronization patch. Check each patch
against the intended source version before applying it; an unexpected mismatch
requires investigation rather than silently omitting or forcing a change.

Copy `vslice-nr-5g.cc` into the checkout's `scratch/` directory. The scenario's
recorded build command, run from that checkout, is:

```bash
./ns3 build vslice-nr-5g
```

The Python adapter expects the scenario executable at
`build/scratch/ns3.46-vslice-nr-5g-default` beneath `NS3_DIR`. Set `NS3_DIR` to
your simulator checkout and `E7_NR_OUT` to a writable output base before running
the E7 driver. The adapter uses offline replay (`--port=0`) with budget,
per-millisecond schedule, and KPI files. Other modes in the C++ source include a
TCP client; they are not needed for the recorded Python offline replay.

The E7 protocol also requires training-based NR calibration, matching models,
and raw traces. Its TEST checks compare the executable and NR library hashes
with the frozen calibration record. A fresh build may have different hashes;
it does not reproduce the historical build identity merely because compilation
succeeds. The publication package leaves those checks intact.

## Provenance and licenses

The scenario was copied unchanged from the supplied `ICC/code/ns3` source. Its
header records derivation from an earlier local scenario; no standalone license
header was present in the supplied file, and none has been invented here.

The AMC and RLC backport patches preserve upstream CTTC copyright notices and
`SPDX-License-Identifier: GPL-2.0-only`. `COPYING.GPL-2.0` contains the GNU GPL v2
text obtained from the [GNU license archive](https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).
Retain the applicable ns-3 and 5G-LENA notices when building or redistributing
their code. Including this upstream license does not assign that license to the
Python research implementation or resolve the scenario's missing standalone
license declaration.

Only patch filename headers were normalized for portability. Original and
packaged SHA-256 values are recorded in the repository's
`packaging-transformations.json`; all patch hunks and notices are unchanged.

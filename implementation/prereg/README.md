# Historical protocol records

`PREREG.md` and `DEVIATIONS.md` are byte-for-byte copies of the records supplied
with the matched research project. Their dates describe that project's history,
not new experiments performed when this release was prepared.

The record states a freeze on 2026-10-20 and a TEST pass spanning 2026-10-20 to
2026-10-21. The deviation record also describes later plotting changes and
descriptive analyses. These are historical statements from the supplied material;
the full campaign was not independently rerun during publication packaging.

The exact `PREREG.md` contents matter because the original TEST acquisition
manifest records its checksum. Its fenced `freeze` block also lists code,
configuration, and checkpoint hashes. Keeping the record intact does not provide
the missing checkpoints, calibrations, raw traces, or simulator build.

Post-freeze figure edits changed the analysis-tree hash. The supplied deviation
record itself notes that a later exploratory operation was refused by the TEST
read gate when a complete updated hash was not recorded. This release preserves
the gate and does not claim that an unchanged historical freeze authorizes every
later source state.

References to internal development records in these files are historical context;
those records are not required for the synthetic checks or summary-based figure
regeneration documented in the main README.

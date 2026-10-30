# Pre-registration — strategy-proof learning-based RAN slicing (protocol text final 2026-10-20; the freeze block with the
# code, configuration and model hashes is appended below at the freeze, before any TEST data is downloaded)

Study: strategy-proof learning-based RAN slicing (ICC rewrite). Design source: `design/experiment_design.md`.
This document is written **before any test data exists on disk**. At the freeze we record: the SHA-256 of this file, of
`code/slicemech/**`, of `code/experiments/**` (dot-directories such as lint caches, `__pycache__` and `*.pyc` excluded), of every
trained checkpoint and of the frozen config; then, and only then,
`data/puffer_fetch.py` downloads the test dates. Every rerun after the freeze is logged in `prereg/DEVIATIONS.md` with its reason.

## 0. Decisions taken on TRAIN/DEV before the freeze (orchestrator, 2026-10-19)

- **D1 GBR fractions.** TRAIN GBR outage was 0.077 (> 0.05). Following the pre-declared order (GBR fractions first), every
  class's GBR fraction is scaled by 0.8: g/mu = 0.40 (video), 0.64 (URLLC-like), 0.16 (mMTC-like). On TRAIN this gives
  outage 0.029 and all five regime targets pass (code/STATUS.md §5.2).
- **D2 Offered-load normalization.** For every date of every split, the integer number of viewers/users/devices of each
  class is set by a deterministic rule so that the date's mean offered traffic of class c equals
  share_c x rho x B x rbar_TRAIN (shares 0.6/0.3/0.1; rbar_TRAIN = TRAIN mean per-PRB rate). The rule reads only offered
  traffic, never a controller outcome, is identical for every compared controller, and is applied to TEST dates only after
  the freeze. Realized per-window loads are reported. Reason: DEV was much lighter than TRAIN under TRAIN-fitted counts
  (0.68 vs 0.91), so load would otherwise be an uncontrolled variable.
- **D3 Gate G4 dropped.** The lookahead oracle is not an upper bound (it fell below a fixed 66-point weight grid on 19 % of
  DEV windows), so the gate is ill-defined. H1c and H1d are secondary comparisons reported with CIs and with the largest
  advantage the data rule out; windows are also analysed by offered-load subgroup (exogenous).
- **D4 Harm to honest tenants.** H2c and H4b are measured on the non-deviating tenants' own outcomes (on-time Mbit, deadline
  misses, their realized value). Total tariff-weighted welfare is descriptive, because a high-tariff deviating tenant can
  raise total welfare while honest tenants lose service.
- **D5 E5 protocol.** Retraining windows 1-4 and evaluation windows 5-8 of each date use interleaved 2-s halves of every
  URLLC/mMTC capture and CQI run, with roles swapped on alternate dates (balance within 7 % on TRAIN/DEV); the alpha = 1
  stored-report arm is the H5b reference; DEV also has 8 windows per date so the protocol can be piloted.
- **D6 Learner objective.** Unserved traffic carries over between intervals, so the one-interval reward has the wrong sign on
  about half of light windows. The learner maximises the realized on-time value of the current interval and the next three
  (lookahead return, h = 3) under its own policy; the paper calls it short-horizon reinforcement learning.
- **D7 E4b frontier (descriptive).** The trained report sensitivity kappa is scaled by s in {0, 0.5, 1, 2} at deployment; each
  scaled map stays monotone and is priced by the full-decision price; welfare and honest payments trace the trade-off,
  with equal, static and direct for reference.
- **D8 H5b cell and sign (2026-10-20, after the E5 DEV pilot).** On DEV, inflating reports during the retraining windows
  LOWERED the distorting tenant's later quota and utility (protected arm, stored reports, alpha = 2: later gain -3.44
  credits per interval on the estimation half, 0/10 dates positive; alpha = 1.5: -2.15), while under-reporting RAISED them
  (alpha = 0.5: +0.93, 10/10 dates; alpha = 0.7: +0.77, 10/10; the selection half agrees: +1.12 and +0.92). The retrained
  network learns to favour a tenant whose stored reports understated its realized traffic. H5b is a hypothesis about a
  later GAIN, so its primary cell moves from the draft's alpha = 2 to alpha = 0.5, with pre-registered sign +1
  (code: analyze.H5B_PRIMARY_ALPHA = 0.5, analyze.H5B_SIGN = +1). The alpha = 2 cell and the other alphas are reported as
  secondary. Measurement-only retraining was bit-identical across all profiles on every DEV date (H5a).
- **D9 H1a effect rule kept (2026-10-20, after the E1 DEV run).** On DEV the value of reports at d = 2 s is +0.13 credits per
  interval, 0.215 % of the report-free welfare on the estimation half (CI [0.175, 0.258] %; 123 of 126 DEV windows positive).
  The pre-declared rule of H1a (CI lower bound > 0 AND effect >= 0.5 %) is kept unchanged, so H1a is EXPECTED TO BE NOT
  SUPPORTED on TEST through its effect-size part; the paper reports the measured size, its dependence on delay and load
  (H1b, subgroups) and per-class deadline misses. The 1 % minimum effect of interest is descriptive (the size we would call
  practically important); it enters no decision.
- **D10 Pre-freeze audit (2026-10-20).** A read-only audit (review/prefreeze_audit.md) found one blocker (concurrent
  preparation of TEST dates) and 14 major issues in the TEST tooling, the analysis and the macros. They are fixed in the code
  before the freeze (code/CHANGELOG.md); none changes a hypothesis, a decision rule or a DEV estimate used above. The
  protocol statements they required are added here: the H2c primary cell (§9.10), the power method (§9.4), the date rule
  (§2), the seeds of E5 and E7 (§9.6), the source of H4a (§9.11), the E5 pool-half rule (§9.12) and the descriptive
  PRB-efficiency quantities (H4c).
- **Disclosure.** Before a review fix, a data report computed TEST-split TRACTOR and Raca statistics (per-user means, per-PRB
  rate) once; the numbers were withheld afterwards and no decision used them. The Puffer TEST dates are fresh.

## 1. Questions and estimands (kept separate on purpose)

| Estimand | Definition (per scored interval, paired on identical windows) | Good direction |
|---|---|---|
| Value of reports | realized welfare(protected) − realized welfare(measurement-only) | > 0 |
| Value of learning | realized welfare(protected) − realized welfare(static) | > 0 (two-sided test) |
| Deviation gain | U_i(best-response report; θ) − U_i(θ; θ) for one deviating tenant, others truthful | ≤ 0 under the full-decision price |
| Victim harm | change in honest tenants' on-time Mbit when one tenant best-responds | 0 under protection |
| Cost of protection | realized welfare(unconstrained, truthful) − realized welfare(protected, truthful); honest payment / declared value | small |
| Training channel | later-window gain of a tenant that distorted reports during retraining windows | 0 under measurement-only retraining |

Realized welfare = Σ_i π_i × (on-time Mbit of tenant i in the interval) from the ms-level fluid replay with true arrivals and
true per-PRB rates. Declared value V_i uses the public burst shape and is used only for prices and the incentive theorems.

## 2. Data and splits

- Puffer (video tenants): TRAIN 2024-01-01..02-25 (15 dates), DEV 2024-02-28..05-06 (21 dates), TEST = Sundays and Mondays
  2024-09-01..2024-10-21 (16 dates), downloaded after the freeze. A missing or corrupt date is replaced by the next calendar
  day that is neither a TEST date (including the 24-date extension) nor an earlier replacement; after 7 failed candidates the
  fetch stops and the failure is logged as a deviation. "Corrupt" means: the object fails the pinned size/MD5 check twice,
  cannot be decompressed or parsed, fails the validation of the chunk-size parser, or has none of the seven channels with all
  four formats. A missing channel's viewers are reassigned uniformly to the channels present. If the power analysis on DEV
  (§9.4) asks for it, the TEST range extends by the Sundays and Mondays 2024-10-27..2024-11-18 (8 more dates, 24 in total) —
  decided before the freeze and recorded in the freeze block (field test_dates).
- TRACTOR: TRAIN {urll, urll_2, urllc_3, urll_11_18, mmtc, mmtc_2, mmtc_3, mmtc_11_18}; DEV {urllc_05_18, urllc_06_12, mmtc_05_18};
  TEST {urllc_03_03, urllc_04_10, urllc_04_16, mmtc_03_03, mmtc_04_10}. Raca CQI: TRAIN 2019-11-20..12-06, DEV 2019-12-14..2020-01-06,
  TEST 2020-01-16..02-27. Only the Puffer test dates are fresh; TRACTOR and Raca test parts were inspected in earlier work.
- Windows: 8 per test date (every 3 h), 200 s each, 40 s warm-up and 160 scored intervals. Composition draws are nested so
  load sweeps share randomness.

## 3. Fixed configuration (from `spec/params.yaml`; values set on DEV are marked as such)

B = 106 PRB (20 MHz, 15 kHz SCS); Δ = 1 s; γ = 0.2; η = 2; integer quotas. Classes (×μ): video g 0.40, M 3, C 5, π 1.0, δ 2002 ms;
URLLC-like g 0.64, M 3, C 5, π 3.0, δ 150 ms; mMTC-like g 0.16, M 4, C 8, π 0.5, δ 300 ms (D1). Load ρ = 0.9 with shares 0.6/0.3/0.1
(final: GBR fractions as above after D1; ρ = 0.9 unchanged; all five regime targets hold on TRAIN and on DEV — DEV pooled:
ΣH > B in 98.1 %, all caps fit 0 %, weights change quotas 97.7 %, weighted share 0.452, GBR outage 0.6 %). Measurement delay d = 2 s default; sweep {1, 2, 5, 15} s. Video lead time
4.004 s default; sweep {0, 1.05, 4.004} s. Regime targets (checked on TRAIN, then DEV): share of intervals with ΣH > B ≥ 0.6;
all caps fitting ≤ 0.05; weights change integer quotas ≥ 0.5; mean share of PRBs placed by weighted filling ≥ 0.25; GBR outage ≤ 0.05.

## 4. Compared controllers (names fixed in `spec/controllers.yaml`)

Protected (ours: own-report score network, κ ≥ 0, full-decision price, measurement-only retraining); protected net with the
fixed-priority price; unconstrained network (MLP of context and reports) with the fixed-priority price; unconstrained with the usage
tariff; unconstrained with the envelope formula; measurement-only (no reports; usage tariff); κ = 0 ablation; static weights;
equal weights; direct optimizer (declared value, envelope price); direct with true current rates (diagnostic); hindsight oracle.
All learned controllers: identical DEV tuning grid lr ∈ {1e-4, 3e-4}, σ0 ∈ {0.3, 0.6}, M ∈ {4, 8}; 5 training seeds at the default
configuration, 3 seeds in sweeps and at n = 6, 9; checkpoint selected on DEV realized welfare. Selected cell (results/models/selection.json, DEV selection half, mean over
5 tuning seeds): lr = 3e-4, σ0 = 0.6, M = 8 for every learned arm (protected, unconstrained, measurement-only, κ = 0);
the same cell is reused for every sweep configuration. Every arm chose the largest value of each grid dimension; this is
reported as a limitation (a wider grid could raise every learned arm).

## 5. Tenant behaviours

Truthful; per-interval best response of one deviating tenant (role rotated over classes) against truthful others (exact cell
enumeration for monotone maps; grid + refinement lower bound otherwise); learning tenants (EXP3 and ε-greedy over multipliers
{0.5, 0.67, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2}, 400 episodes × 4 runs per (design, algorithm), i.e. 8 runs per design; the H3b modal
multiplier pools both algorithms); training-channel distortion α ∈ {0.5, 0.7, 1.5, 2}
during the retraining windows 1–4 of a date, truthful in windows 5–8.

## 6. Hypotheses and decision rules

Primary (Holm-corrected family, α = 0.05): **H1a, H2c, H4a, H5b** (H1c moved to secondary by D3; H2c, the honest-tenant harm, joins the primary family). All others descriptive or deterministic.

| ID | Hypothesis | Decision rule / failure criterion |
|---|---|---|
| H1a | Protected − measurement-only realized welfare > 0 at d = 2 s | supported if the 95% CI lower bound > 0 and the effect ≥ 0.5% of measurement-only welfare (D9: kept; expected to fail its effect part); the 1 % minimum effect of interest is descriptive |
| H1b | The H1a difference is nondecreasing in d and in the video lead time | slope CI over the sweep cells; reported descriptively |
| H1c | Protected − static realized welfare ≠ 0 | two-sided, secondary (D3); reported with CI, the largest advantage ruled out, and by load subgroup |
| H1d | Protected vs direct optimizer | two-sided; the direct-with-true-rates diagnostic is reported next to it to explain the sign |
| H2a | Under the fixed-priority price or the usage tariff, the best-response deviation gain is positive for at least one class | mean gain CI lower bound > 0 for that class |
| H2b | The trained unconstrained network violates the ratio condition (non-monotone own-quota path) in a nonzero share of states | rate with exact counts; descriptive |
| H2c | Under the Unconstrained net with the fixed-priority price (primary cell, §9.10), honest (non-deviating) tenants lose on-time Mbit when one tenant best-responds (D4); the other unprotected designs and the deadline-miss change are secondary | CI upper bound of the honest on-time change < 0 |
| H3a | Under the full-decision price every monotone design (including the protected net with κ scaled by 0, 0.5 and 2) has certified maximal deviation gain (the computed gain plus its numerical certificate) ≤ 1e-8 · π_i μ_i Δ in every scored tenant-interval | deterministic certificate; any violation is reported and investigated |
| H3b | Learning tenants under the full-decision price end with multiplier 1 (no gain over truth) | final-episode modal multiplier and gain |
| H3c | The priority premium T(θ) is reported in credits and as a share of payment; the gain identity holds numerically | every residual of the identity computed against an independent second price computation lies within that computation's guaranteed error bound (D10); the same-engine residual (< 1e-9) is reported as a check |
| H3d | Under realized-service utility, the maximal gain over constant multipliers ≤ 1% of truthful realized utility | otherwise reported as a limitation of the declared-value model |
| H4a | Welfare cost of protection is within 1% (non-inferiority: one-sided 95% upper bound of W(unconstrained) − W(protected) < 1% of W(protected)) | primary |
| H4b | Honest tenants' realized value under the protected design (truthful) ≥ under unprotected designs with a best-responding or learning deviator (D4) | CI lower bound > 0 |
| H4c | Honest payments as % of declared value, tenant surplus, pricing runtime; PRB efficiency (G6) from E1 at the default configuration under truthful reports: share of reserved capacity used for on-time traffic, share of cell PRBs busy, on-time Mbit/s per reserved PRB, for every controller | descriptive |
| H5a | Measurement-only retraining leaves parameters bit-identical across distortion levels | deterministic |
| H5b | Stored-report retraining produces a later-window gain for a tenant that under-reports (alpha = 0.5) during the retraining windows, relative to the alpha = 1 stored-report arm (D5, D8) | pre-registered sign +1 (D8); two-sided test in the Holm family; supported only if the CI lies above 0; an effect of the other sign is reported as contrary; a null is reported as a null |
| H6a | At n = 6 and 9: zero certified gain; p95 pricing time ≤ 100 ms per interval at n = 9 on one core | deterministic / measured |
| H6b | Cost of protection at n = 9 ≤ 2% | descriptive |
| H7a–c | 5G-LENA replay: victim effect has the same sign in ≥ 3/4 windows and is within 2× of the fluid prediction; victims receive more under protection; per-interval delivered correlation ≥ 0.8 for calibrated fluid | descriptive; no CIs |

## 7. Statistical analysis

Unit = window, paired across designs on identical compositions and random numbers; cluster = Puffer test date. Two-stage
bootstrap (dates, then windows within dates), 10,000 replicates, percentile 95% CIs; per-date sign counts reported. Training
seeds are averaged within a window; seed spread is reported separately. Physical units first (Mbit, credits per interval, PRB,
ms); every normalised number names its denominator. A sensitivity analysis clusters by TRACTOR trace pool.

## 8. Reporting rules

Every hypothesis above is reported, including adverse and null results, in the order of this document. No test-time tuning,
no re-selection of windows, controllers or hypotheses. If a pipeline bug is found after the freeze, the fix and the rerun are
logged; both the original and corrected numbers are kept in the evidence (the drivers archive overwritten outputs). A figure
script that only plots frozen summaries may be added or changed after the freeze; such changes are listed in DEVIATIONS.md but
need no rerun.

## 9. Further fixed choices (added 2026-10-19 from code/EXPERIMENTS.md §2 D9)

1. **DEV halves.** Every DEV choice (checkpoint selection every 50 iterations, the hyper-parameter cell, the static
   comparator) uses only the DEV *selection half* (the 11 DEV dates at even positions, 66 windows). The *estimation half*
   (10 dates, 60 windows) is used only for the DEV effect estimates of the power analysis and for the H5b sign (D8, whose
   choice of the alpha = 0.5 cell was also informed by the selection half, where every alpha showed the same direction). The static comparator is the
   selection-half best of the 72 best TRAIN candidates, so it has the same DEV access as a learned arm.
2. **Selected hyper-parameters** per learned arm: lr = 3e-4 (decayed linearly to 3e-5), σ0 = 0.6, M = 8 for protected,
   unconstrained, measurement-only and κ = 0 (results/models/selection.json), reused for every configuration.
3. **H5b primary cell.** Protected arm, stored-report retraining, alpha = 0.5 (D8; the draft had alpha = 2), video and URLLC-like targets pooled (mean per
   evaluation window), against the alpha = 1 stored-report arm of the same date; two-sided p-value in the Holm family;
   supported only when the CI excludes 0 on the side of the pre-registered sign (+1, from the estimation half: +0.93 credits per interval); an
   effect of the other sign is reported as contrary.
4. **Number of TEST dates: 16** (freeze field test_dates = 16). Decided by the DEV estimation-half power analysis
   (results of 2026-10-20: simulated power at G = 16 with the effect at the CI bound nearest the null = 1.00 for H1a's CI and
   Holm part, 0.9975 for H2c, 1.00 for H4a and 1.00 for H5b; H1a's effect part is unattainable, D9). Method: for each primary,
   the estimation-half dates (centred date residuals plus the DEV effect) are resampled to G = 16 and G = 24 dates and the
   registered procedure (two-stage bootstrap, the primary's rule, Holm level alpha/4; alpha/8 per side for H5b) is applied;
   the rejection share estimates the power, with an analytic cross-check that inflates the date-level standard error by the
   bootstrap design factor. 24 dates are used if any of H2c, H4a, H5b, or the CI-and-Holm part of H1a, has power below 0.8 at
   G = 16 (the simulated rejection share at the CI bound nearest the null decides; the analytic number is a cross-check). H1a's
   effect part is unattainable on DEV (D9) and does not drive the choice. DEV has 6 windows per date, TEST 8.
5. **E7 window.** The window at offset 43,260 s (23:01 UTC, the 5th of the 8 windows of a TEST date) of each of the first
   four TEST dates, chosen on TRAIN because the
   first window of the day (60 s) bound quotas in only 35-40 % of intervals; code constant e7_nr.CASE_OFFSET_S.
   Simulator: ns-3.46 with nr-4.1.1 in a private copy patched with a statistics-only fix of nr-gnb-phy.cc and the
   RLC-UM reassembly and AMC transport-block-size code of nr v5.0 (the unpatched module aborted in calibration and in
   the pilot); the DEV rehearsal of the 12 cases (2026-10-20) found two further aborts on one window (an RLC-UM
   reassembly assertion and an uplink scheduling check that used the downlink notch mask), fixed by resynchronising the RLC
   receiver (the stranded partial SDU is dropped) and by removing the misplaced check; the patches and the shared-library
   SHA-256 are recorded (nr/patches/, E7 fingerprint), and the calibration was repeated with the final library. Calibration
   (TRAIN, saturating runs, RngRuns 1-4): radius grid {30, 60, 120, 200, 250, 300, 400} m; chosen --discRadius = 250 m
   (NR rate 0.422 Mbit/s per used PRB = 0.964 x r_bar_TRAIN). Pilot (TRAIN, offset 43,260 s, 200 epochs): quotas bind in
   99 % and 100 % of the scored intervals (gate >= 50 %).
6. **Seeds.** E2, E3 and E4 use checkpoint seed 0 only (cost); H2c, H4b and H3a therefore do not average seeds, and the paper
   says so. E1 averages all five seeds within a window. E5 uses checkpoint seeds 0, 1, 2 of the default configuration (the
   seeds of the DEV pilot), averaged within an evaluation window before the date bootstrap; H5a is checked per (date, seed,
   arm). E7 replays checkpoint seed 0.
7. **Load subgroups (D3).** Subgroup variable = the window's offered load (summed measured demand over B x rbar_TRAIN,
   averaged over all 200 intervals); bins < 0.8, [0.8, 1.0), >= 1.0; paired contrast and two-stage date bootstrap within
   each bin, plus the OLS slope of the per-window paired difference on offered load with a date-cluster bootstrap CI;
   bins or slopes with fewer than 5 dates are reported without a CI.
8. **Inference rules.** No CI, p-value or decision from fewer than 5 dates. CI rules use the two-sided 95 % percentile
   interval (H1a lower bound, H2c upper bound); H4a uses the one-sided 95 % upper bound; Holm p-values are one-sided in the
   pre-registered direction for H1a, H2c, H4a and two-sided for H5b; a primary is SUPPORTED only if its pre-registered
   rule holds AND Holm rejects at FWER 0.05 AND (on TEST) its inputs are complete and of a single provenance.
9. **E7 hypotheses.** H7a needs the same nonzero sign and the within-2x criterion in >= 3/4 windows; H7c is tested on the
   fluid model with the calibration-stage NR rate (out of sample); the fluid model with NR-measured rates is a
   decomposition only.
10. **H2c primary cell (D10).** Arm: Unconstrained net with the fixed-priority price (E2 arm unc_fixed, code constant
   analyze.H2C_PRIMARY_ARM). Sample: the E2 replay windows, 2 per date (windows t mod 8 and (t+4) mod 8 of the t-th date;
   32 TEST windows, 48 with 24 dates). Per window: the honest on-time change (deviation minus truthful, summed over the
   non-deviating tenants, Mbit per interval, mean over the 160 scored intervals; the deviator is truthful in the warm-up),
   averaged over the 3 deviator roles; checkpoint seed 0. Inference: two-stage date bootstrap; rule = CI upper bound < 0;
   one-sided ('less') p in Holm. Secondary with unadjusted CIs: the deadline-miss change, the honest realized value, total
   welfare, and the other three unprotected arms (protected net with fixed price, usage tariff, envelope formula).
11. **Source of H4a.** H4a uses E1 at the default configuration (5 seeds averaged per window, truthful reports): W(Unconstrained
   net) minus W(Protected), one-sided 95 % upper bound of the bootstrapped difference as a percentage of W(Protected). The E4
   seed-0 number is descriptive.
12. **E5 pool halves.** The interleaved 2-s halves swap roles by the ordinal parity of the date within its split (D5); on DEV all
   10 estimation-half dates use the swapped assignment, on TEST the assignment is balanced.
```freeze
analysis_code_sha256: b08f92b026476efcebda246ddd1c284c79d9b6da4a5d82913bf1273ef6a06406
checkpoints:
  results/models/d1/kappa0/seed0/config.json: ae006ec11f0b249fe73ab6ddf41f13f0a478c699e4a13011f863e3e57c59330c
  results/models/d1/kappa0/seed0/model.pt: 841b3ae78df49067ef3e5f974486f23bd9b42dfbfb7c3bc6001e9113a04fdcbb
  results/models/d1/kappa0/seed1/config.json: eab34ddbb3c1b34a16cee88740d3108b0fb0201b194eceb53abc68b95c8708d5
  results/models/d1/kappa0/seed1/model.pt: f6c8b95f72eefe6fd2adb25abce9f066c13503488e0bba85694747b78d6d07c4
  results/models/d1/kappa0/seed2/config.json: c4663146a3268eb85359d0c4c5f3824115961ca7563bd7e19b374d0c810a97c5
  results/models/d1/kappa0/seed2/model.pt: 11645e14ad9cd22636967befa9f264b34a5d600858545bed95d5d74ab9d6d290
  results/models/d1/meas_only/seed0/config.json: a2b67b1d9d0fd94b62debc0c63ac941970f7f8c498289c2f5446b4e87ecdef5b
  results/models/d1/meas_only/seed0/model.pt: 6d0e0f071997e2a47f7f117fd343e782be984b3d14e533e7aaf0eb26450a0c85
  results/models/d1/meas_only/seed1/config.json: 2e95510c0014a8a97790023905d68d56ff4c2544f127512d8811147c9bd29651
  results/models/d1/meas_only/seed1/model.pt: 8a0b1b6c06090623438079a863c64aabb27654b8e613eb8fe26563034f68743d
  results/models/d1/meas_only/seed2/config.json: 8df7e9536dbf7294bbe134442725cf6fef706f913c4fbeac96d472d8ee5061ce
  results/models/d1/meas_only/seed2/model.pt: abda1b663c3df2199ebe6e36fd2aa8b751ed2f4089eab0af9609552466847ea1
  results/models/d1/protected/seed0/config.json: 91906a0c3dbec7cb9810463c3ffaa7be0568fe6acaa04b3fbf9346066d2c75ed
  results/models/d1/protected/seed0/model.pt: 0b04f6aad871d0c7b8623e97b5f22e125f58c193087e1138714388da72a117a4
  results/models/d1/protected/seed1/config.json: 589c36c9cfba01737e1e58cf22abb1400ed2a5d87c53f3ea8f28a96b630c08a7
  results/models/d1/protected/seed1/model.pt: 6ca7ef2b191d588e5beee467ac588a58749708d16a51e008a6902b3df0f50758
  results/models/d1/protected/seed2/config.json: 20e8a01cd0282ee42342994f868d2ce777f6952f8ebf043986b0ef7793c233d8
  results/models/d1/protected/seed2/model.pt: fcf01806583a3bc7ba4a6cf38377dc866eab4fc3c262331db27e200febc27693
  results/models/d1/unconstrained/seed0/config.json: fbc5098319e6df4e78392a8c33720ff2b68236437e37eb56a04207cbaf6cec3b
  results/models/d1/unconstrained/seed0/model.pt: 6bca7181fde16c7dece62650021d55484a92b530e12a241640f3d87c54b585c3
  results/models/d1/unconstrained/seed1/config.json: 2bf5f7eff0d50102c6770e42712a62762919a4dca8a37294cd0f64c84fce74be
  results/models/d1/unconstrained/seed1/model.pt: 02f828fabf35205928dd9c064cd949a46cb79c1b516d6257edc98312de1d1ba5
  results/models/d1/unconstrained/seed2/config.json: dd6b43948bd8b36a3a79f08309643db2a37c1b61fb345a174c18ba8dfa7bb92b
  results/models/d1/unconstrained/seed2/model.pt: 52b4d71b264aa5390a8e3090f861382339dfe41a2aab08e47251461e113c58f2
  results/models/d15/kappa0/seed0/config.json: 37936a70ad603df4bbc43750ea3d6cbe59532e793e81cfd7cf7e2ee8f392c88f
  results/models/d15/kappa0/seed0/model.pt: cf5588e001b82f1e4fdb93bfec55a6ae05e76cf04d028a8e364821aac47912c6
  results/models/d15/kappa0/seed1/config.json: 205e0cba4b3a9c2364ff432af40203ccfcbebb7a1ef9ae950af06137c1e3a421
  results/models/d15/kappa0/seed1/model.pt: 6b0fd15fc09bd1c97628c1e26857fe30a55f60e0b888bf68d95e6b73484ccbff
  results/models/d15/kappa0/seed2/config.json: 0baef571c11d8167ea3fee6580d6b836ba35aed0d9615e11553a80474865fea1
  results/models/d15/kappa0/seed2/model.pt: 5662bd4214478b564ba01c45eba2f224939a3f0824cf2e2c13c551703872093b
  results/models/d15/meas_only/seed0/config.json: 8bd0393dbc024b51a97bb402da58e76a67701001127c986445c9e4922892aa80
  results/models/d15/meas_only/seed0/model.pt: 8b8d5c6343f4187e1bf5d2147842980330897cd9d52d0332cbd0b47e58b634f5
  results/models/d15/meas_only/seed1/config.json: 0398905433c6f15b839919977cff55500dd14f94e9ebb12ba4c42de9e84ba568
  results/models/d15/meas_only/seed1/model.pt: 7b153c2c4d20979879f5392c60e3aa8116c2d9910da46a67d4e07cbcbc1d52cc
  results/models/d15/meas_only/seed2/config.json: ef9c7ead7b6ee98650040d73324400416b8fbc062f40f09133af8895f9b994ae
  results/models/d15/meas_only/seed2/model.pt: f1464aca15ada282ac3c134ace70590c0b8058412f5da90651a6d09c484f7db7
  results/models/d15/protected/seed0/config.json: 2fecae71c5373aa1554bd1a35ca8c8a944fcc1ae3fa9f861529ee098be152734
  results/models/d15/protected/seed0/model.pt: 02258a244ae67872e6d2fb90e234f859e5e7de2cf63505081e99486eb92e27ee
  results/models/d15/protected/seed1/config.json: 9b523e4e64d23780821a57ace466b30b1abd326fc4edfed51fa4a1eee4b0d966
  results/models/d15/protected/seed1/model.pt: e636a81662053b9f7a7a7aee57887d159bcab9529d174d276e27a1c9d410940f
  results/models/d15/protected/seed2/config.json: 1e6c90b3871ea789fdedbf851bc24f84c88113c91d5ac059d3d037b9b3d6438e
  results/models/d15/protected/seed2/model.pt: 4241d4f5e37985373e87e5869f0ab84b09617cdacde5a4eab76f52bd14c63d2f
  results/models/d15/unconstrained/seed0/config.json: c78e4beae2c140f0cfe87c5584873e3049812226ff0698c53363f4131f09c00d
  results/models/d15/unconstrained/seed0/model.pt: 3aa860ed3b235586b30bf8d443d856899ff8ce283f486de9f6947127bae55ee8
  results/models/d15/unconstrained/seed1/config.json: ed9fcebb0e29276722d15e890cc52b27bd4704919afd56b2ef705e089504a973
  results/models/d15/unconstrained/seed1/model.pt: f9d047a3f03a872d99615b1351cd40f0c094cd8072bbeacf94ccc1685b9a7c5b
  results/models/d15/unconstrained/seed2/config.json: 2b2e3fd24959f8680be0026b836ef43af73285f5d882e13f794cdd98aaa29b13
  results/models/d15/unconstrained/seed2/model.pt: 000b137987db515d4dd70c9a0a324be164db79d05872912230c5de76f9d11010
  results/models/d5/kappa0/seed0/config.json: 8026ebd929b46d048e702b350259737f6675a6c32acf8f07f6427346c35bbf45
  results/models/d5/kappa0/seed0/model.pt: d8055d2564cb66a094e34a7ed39a32f5013f0bed0eb5c08d5e90a3617f7325d8
  results/models/d5/kappa0/seed1/config.json: 2447a00e29077fc4f6279c904b42c3f253d8df7604cad72dc40474150faa5bae
  results/models/d5/kappa0/seed1/model.pt: 06d98f2dcabcb18967fe7253b23b5230caf36abd7a5bf6bafdd46a9b8eaf4775
  results/models/d5/kappa0/seed2/config.json: 9309869de61cd5e77b363b236d58950ce6f0f9972a3969c3bedee2d172cdd1f3
  results/models/d5/kappa0/seed2/model.pt: 208b764d183e53193740e4ce322db5e53a53c96f5da9eb9320b16468a4d4087f
  results/models/d5/meas_only/seed0/config.json: 479dacb5dfcb6c78998d6cb15b35e915c00feee81e4a28977814441b19c21b75
  results/models/d5/meas_only/seed0/model.pt: 96e82b3a0f49e2f63d8980fe69ca46a7086ddee3c23cf22871111587a957fa5c
  results/models/d5/meas_only/seed1/config.json: 9bab89c70e4994fc57e50cf47c07de4b04e4e7c96c0664eba240694b6f0cbda7
  results/models/d5/meas_only/seed1/model.pt: 00c06f702df330d690a9d171fc85ed0eb579adbc7ddb845c59456ec1c5738b1d
  results/models/d5/meas_only/seed2/config.json: 0f7c172fccb7b5d915c32f1b3e66bbed4dfc944f332c6a9cf064401d5e055e5a
  results/models/d5/meas_only/seed2/model.pt: 0845071a97ae47346d9c1fecdb0a8094669435d123766a769bf03a69ab6b4744
  results/models/d5/protected/seed0/config.json: 24ebf6994ff844c816423c973e9eed06c6590b359289c6c85156819b839cea93
  results/models/d5/protected/seed0/model.pt: a6d65d5d3e3af58a941e2a83c69b6b80a8455d0cd52440166470eef3e7c4d2a1
  results/models/d5/protected/seed1/config.json: 74f8daca1ca5def6597fd36180054e16708f07792658dd05c01e33b93195fe19
  results/models/d5/protected/seed1/model.pt: 7389b7fd7fdf94be0801436260266531947465b2caf6c0df1b4250c42e3c24a9
  results/models/d5/protected/seed2/config.json: 6afe2875d2203f11e874dd50c2d4b662e24df2e8b15e253dbbd56a223e7b508a
  results/models/d5/protected/seed2/model.pt: 70ff1705e937b0781a933ecc30e461e90ee07fa7be7abc749f10b905ebdaca4a
  results/models/d5/unconstrained/seed0/config.json: d08a475cfbd168f3249b81cfa31c6080ffb77407cbb4ec07ac54876fe3034c98
  results/models/d5/unconstrained/seed0/model.pt: e71f8a58dcf49f6c5048b00fb5131261f1f3da1b713dddcfc23aca1f4f0f92ef
  results/models/d5/unconstrained/seed1/config.json: 00a873100ee22fa914974b4f9cd4d4b8d7c1059b8d6057a6a907e0e33e4d8f0f
  results/models/d5/unconstrained/seed1/model.pt: fcb5dad4006d277aa246cc01f798a2ef55c90c1e9151fb8d1d9f116151769493
  results/models/d5/unconstrained/seed2/config.json: a35dee07ee71d9ceab1eef1b19da670fc57485cb8b185cd599b87b04e873f9c7
  results/models/d5/unconstrained/seed2/model.pt: 57ae63c14cd1ec02359c3b02717d1fa4c9dacfdfd5fd2cbf564d943bc1130802
  results/models/default/kappa0/seed0/config.json: 420639606b322d818292b3f3ad30145cc5967c0f96cec4936d51eab595cc09c2
  results/models/default/kappa0/seed0/model.pt: 2c9092086328bf3241857c8363fb190e8e9dc0864f4f683b4600ecc62ca9338f
  results/models/default/kappa0/seed1/config.json: 129e39e871928fb161c869d05068ad4925c2f661f5e0488bc3f62fc4fa758ad6
  results/models/default/kappa0/seed1/model.pt: 916213f861d80865918862ff00d527f84ed4ed608511ef27cb0d46382dc6d255
  results/models/default/kappa0/seed2/config.json: e44ba3e372549910c52bd71aa46b25bfa0a6ef4084fe7fd090cc7ce2f25f455c
  results/models/default/kappa0/seed2/model.pt: 622967f41f3c43f2a79db7983abb78d75f808dd2c858ff9c19dce0c51e7112c8
  results/models/default/kappa0/seed3/config.json: dc64f7bad34e3061e7e97efe000151be5f3a01ecde1c5c9bc09cc92274203c21
  results/models/default/kappa0/seed3/model.pt: 31948c7117f7c5f2c8c094d1c19b282b77e89a0b7deae6a4d8b8f994191fa3ad
  results/models/default/kappa0/seed4/config.json: ef166cd89d7c30010ec88d3b0e75a566d8a3f058306b433158721a6e35349883
  results/models/default/kappa0/seed4/model.pt: a44b0a6d2ed2ec9aaa7227344f4fa4b9aa3f605b0a54f831318f71eb38435898
  results/models/default/meas_only/seed0/config.json: 5baba0dafebd096fd19700a2a1bf188d2379aca1be46c17f5f5b7b27913ddef2
  results/models/default/meas_only/seed0/model.pt: a779b6dcff51648e082ccd8a9bab59056ceadb17bc61126026017ccb522f08ef
  results/models/default/meas_only/seed1/config.json: 9cebae56ee7a670289e9577780a241b34926dafc9366d2f30d1d26651098d3bc
  results/models/default/meas_only/seed1/model.pt: 8fdb4dfbf6052de636083841debfe10fa9eda38a1b13da0b67ae9ef7d21b9260
  results/models/default/meas_only/seed2/config.json: c23c10e06127c94565392fb353b434bfed20f6d2736fbdb4ca4e7ccb16089b16
  results/models/default/meas_only/seed2/model.pt: e26546f916d8f62d60dc1712fcbfd09527a4b491088be030c702afdd41641d81
  results/models/default/meas_only/seed3/config.json: f393fad3d511249ffcb5f3afaa88711d842ce0dada6e34297f8ff11bcc10de75
  results/models/default/meas_only/seed3/model.pt: 982231c2193aea6189af6c613546f59d46d1dcdc6a2f23df49e7b12682494ded
  results/models/default/meas_only/seed4/config.json: 1d805fa0b18cc30a04517c7023cc3e48d0591aafde8f574da0cbc85d4232c7ca
  results/models/default/meas_only/seed4/model.pt: 89f3c65326ff00fd864929c886f4fec4de3af2700687fde17f75677fefe4bd29
  results/models/default/protected/seed0/config.json: 8ef3620badd3532aaf3370c838089edc7f7d31f2a4c6eca58a8a78f30a1af27d
  results/models/default/protected/seed0/model.pt: 05c93925109456739a61f04d3faddf0f1f5947118a303df9e3bbdb7229a6938e
  results/models/default/protected/seed1/config.json: e317fbe4994a7bd02f969a7ee12e7d2d9ee87cb1efe0c69130cc2fad9f2eb5cf
  results/models/default/protected/seed1/model.pt: f7a8f64247fd1b7bebbc290c2336e16cb5fd8e40a10ef1a567e3c0076097c98d
  results/models/default/protected/seed2/config.json: 1508c31c4db91f5a6960a626c2a0280835ff2ba414925800ec49f7355e942e5e
  results/models/default/protected/seed2/model.pt: 350ff40b9d3a55f75da7a8dcd505e2057d08ce34ea3a3d4e74455a2b42d8a535
  results/models/default/protected/seed3/config.json: 0daf7375b927a3f389660d6be85b3f685c4c95bc044f6d143b74fe01dca9b935
  results/models/default/protected/seed3/model.pt: d091eaf62da7101f5f3210cbda98ae0132c1eba07a5b8a5b2b03ac1c6cce2d85
  results/models/default/protected/seed4/config.json: 7f1f4463cdc6b092159457dab8787d3b33393e47278f205c9dfdd07aceaa0855
  results/models/default/protected/seed4/model.pt: b337b89deaac2c3a67bd42c679ed8b94f6bd951654ab7bd6b8c0d32377a7c07d
  results/models/default/static/candidates.npz: dc1a344983c2c248d3bd1065060b21db93eb03c5fcc153a3254506b7f5c6603d
  results/models/default/static/static.json: f154806ed7bc869306b808e16d5c7836c782b9475946321b4cf596b1875f5a42
  results/models/default/unconstrained/seed0/config.json: de70141214f3f30e266e8a30280d82bb67aac19ec746a7c91caadce94bbc946c
  results/models/default/unconstrained/seed0/model.pt: 244246f9ea8dbe0b27c98ce785f64b51535b69c37d5e7e3693d687b63b13b41d
  results/models/default/unconstrained/seed1/config.json: 412fa229331b00f1dd9c6a6a94d2d73786ba54251f4ee2d7de551f9ec82f3598
  results/models/default/unconstrained/seed1/model.pt: 513c9474065c52d1e45227eac592d50d206132f3ff467da6f7f415578abb1007
  results/models/default/unconstrained/seed2/config.json: f3046c817650e80e5a8877c7e6bef7009068e8aaf5eff0fa46c59493175e3253
  results/models/default/unconstrained/seed2/model.pt: 0e405d5307522239e344892e70f87717a06e3d84c69a3b3b87f2bd92d1ce83f9
  results/models/default/unconstrained/seed3/config.json: b1d0dddc6fd7ef1a8a092a39c871ccf4783ea3af2e7404dd0b9f5535585bb19e
  results/models/default/unconstrained/seed3/model.pt: adb23c629861e4487c794300d65e3a5982be533408e797b5d74e3951a7f18d3c
  results/models/default/unconstrained/seed4/config.json: 32a73e941962fa5fadc9a17cf507e59580ba9a5921759f4e0c96260b55555e99
  results/models/default/unconstrained/seed4/model.pt: 467fa10e09cb99186200230ccaf52379e4d2d87d2338b8cf147521101eb6ba73
  results/models/lead0/kappa0/seed0/config.json: ea1d17f6c8f2ac15b13184b36d91c5e6e9782b82824824ac2eb18d102d9b26ab
  results/models/lead0/kappa0/seed0/model.pt: 0de64d64f7968d4bbc2b47dfe2d77fba683a5f7d92ac84e6d098cb8d55f1c3ee
  results/models/lead0/kappa0/seed1/config.json: c71b75ae14c50cc2889c9ab1b238c3252892d09e10a2fb90342485664557bd50
  results/models/lead0/kappa0/seed1/model.pt: 95c7b68b9d95f74908981d2461b54eeb0fc27a0e52ca49a75091a29c4945631b
  results/models/lead0/kappa0/seed2/config.json: 9de59cc223fd9ab80fd2a74af66e8f12bc458dfffc77139db304261d8b06294b
  results/models/lead0/kappa0/seed2/model.pt: de096484ecc35e263976316307bb7b3d1378032e0bd03b56f0273d0cfc0275a7
  results/models/lead0/meas_only/seed0/config.json: 743a5e356424de37f0de5d22a79f9b24122e8014cdd3c4b715571de93463ff64
  results/models/lead0/meas_only/seed0/model.pt: 3aac9fbf3c1060bfffa3883755eb15c31f74ee939a7a49d86d602096600331ae
  results/models/lead0/meas_only/seed1/config.json: a9bf4a86aab9c5930a2b231e000996adf64c34989973f15764764f1f24abda6c
  results/models/lead0/meas_only/seed1/model.pt: cf75303c2e0117d6da3c337941693c398b499d795f3674495c4cede90c41d277
  results/models/lead0/meas_only/seed2/config.json: 7fdee55b7c9dae834efaa695e48e091e177f46d25915e3c5b2f58c9cdf59774a
  results/models/lead0/meas_only/seed2/model.pt: fef609f701609b9cfc447593c7e92ab5ccbe34701ba3b4676d8d2b2ddb348110
  results/models/lead0/protected/seed0/config.json: 049b1a9eed0756c68253c12413de3e594d52b40fc60fb76c632bbc882a162037
  results/models/lead0/protected/seed0/model.pt: c7ceba3a0bb68d0a62bd03b3fe88aa19fe160288ffdcf0626f86a255049e0397
  results/models/lead0/protected/seed1/config.json: d624099e150c5a7ce67f2ea246686aecfb8fd4538acff29822df8fcab601a179
  results/models/lead0/protected/seed1/model.pt: f4a8c2fba8244058a5fbf51d7ef6f8ad14a05400a13c69fd5ff754b503cb3d78
  results/models/lead0/protected/seed2/config.json: 244d554215c98f20b6d4685a63f2385c15f37ad8061b5dcdce60eadc699a9fcf
  results/models/lead0/protected/seed2/model.pt: 8bbca65601fd4050b9af12854f4a3fe7cb0607c433a99d6d72316ae47f26affb
  results/models/lead0/static/candidates.npz: 95bff5c2879e1c4af7cd574ac329e63a2f3e7d58a6f345d01f3718c61dfd45ab
  results/models/lead0/static/static.json: b828f23dce7a89bc502537cb8ad74375bcb5f46f3737c67bc8a1dc61e26ae905
  results/models/lead0/unconstrained/seed0/config.json: 1f048bc31541a02b6c7a6226ead2d03d56f1dd9e040e464e80ecdb590b90ab50
  results/models/lead0/unconstrained/seed0/model.pt: 1012ac8d37ea6912a29b837fe24f82cf776521a16b0daccf42bbea35241923f7
  results/models/lead0/unconstrained/seed1/config.json: 6eb43ec66d7edbdcefd0752bd9ccf1074db45455a59c869ce91f90cfc52e2702
  results/models/lead0/unconstrained/seed1/model.pt: 89e2b600e7defcea438d756d967ead3ee6a55c68d55de3dda339663d8f3a80c7
  results/models/lead0/unconstrained/seed2/config.json: 96ca9089a951a5a955d5c4d94989428e768b11ad7b0910612f16b6f555cb0664
  results/models/lead0/unconstrained/seed2/model.pt: b7b6f436950d75c0f26f72e49fec79abb6ec4185c0cb93cdd91d253ebcd92966
  results/models/lead1.05/kappa0/seed0/config.json: b29e34d83e1fd9b57d4983c4d04b7a5cd554a6435afb98f725c48085a2cd00b9
  results/models/lead1.05/kappa0/seed0/model.pt: 91dbb8704e0bd20db405e5e12b1a0795332bf2dfb2c626cc810600d611e41c26
  results/models/lead1.05/kappa0/seed1/config.json: 535a93b89b12455007ea51334c9fd4ed9ead07c451d79aeb2fa4a79735ed7664
  results/models/lead1.05/kappa0/seed1/model.pt: 4e51061d52f46019c4c7c97a7b4b21a63605fc9ec7d6aa06b63dda869e631c43
  results/models/lead1.05/kappa0/seed2/config.json: 5f8023237f66fafce24bdc18d5eab78764190e3a2b47cf6f9b296e574d06ef1c
  results/models/lead1.05/kappa0/seed2/model.pt: e2952a055682cec787a58c96c8a3bf0cceac9fed9020636491340591833d6e54
  results/models/lead1.05/meas_only/seed0/config.json: 93130ccc8de52b48970b28f391fd120de53485dfea3aec4a7e1c77e6b0691208
  results/models/lead1.05/meas_only/seed0/model.pt: 579b354e5f11d49a40b75473512718f1bcc413b87fff971401db75c879920894
  results/models/lead1.05/meas_only/seed1/config.json: 0aaa1a805e3d1a043e0bf01c82d9f3b6e58e9bc441374c129463eafe0dbb2415
  results/models/lead1.05/meas_only/seed1/model.pt: 5b221dbf5891651b02b695fa19d3bf56a18f20e2d9bd90f3a3d8e9b9daf30df7
  results/models/lead1.05/meas_only/seed2/config.json: 8a52fd908aafb4a9afb2b6d0eb69d66d86fefa7e1bcb4ccf2d1456df8b38f1b7
  results/models/lead1.05/meas_only/seed2/model.pt: cf7b3d9080baca5e9ea730c588ffe731b217e9556d60864f0824a7ef4b0c1fcf
  results/models/lead1.05/protected/seed0/config.json: e28d56ca5cc165e237e1a45c618f95b8f4518a130d62a0d363d04e589e7de0d9
  results/models/lead1.05/protected/seed0/model.pt: 5fb97105f0e8a3a1297e0d07b0ea5c3d91e1cf9226c481a8f659a7ce5986e3fb
  results/models/lead1.05/protected/seed1/config.json: 4eec1c9ff3a7fc90eebb4f1e2fd029478f6b22e77e7833b8a45cdbadcf1ce0e6
  results/models/lead1.05/protected/seed1/model.pt: 76234f2b1a13dfcaa0ab1b3b0fe1ed3660029c7d4e3b78ed959c82800973ef04
  results/models/lead1.05/protected/seed2/config.json: 2c48c3daa005ab1c79da69ee03865c5161d915f515f2c5e411de6c4c6452c7e0
  results/models/lead1.05/protected/seed2/model.pt: 65b00e8e35cf33d702441b1ea258b01db742ad441a4213449ce8345f81b37ae0
  results/models/lead1.05/static/candidates.npz: 37cf4e620590bab0d47b6d3cfc87126e479de42827aec23db08dde288d288011
  results/models/lead1.05/static/static.json: a1ae911e10b129d55e1865d93778133747d88ce42a022169e2e938837b93f5a2
  results/models/lead1.05/unconstrained/seed0/config.json: 979ccb4dc0a1043f6f0b5551382dd4d78fefd7b543fac9769fcca93fafdabdb3
  results/models/lead1.05/unconstrained/seed0/model.pt: d81104160397a025b0238b7b8d0594968fa8db7a83425430806be0c380e58d3d
  results/models/lead1.05/unconstrained/seed1/config.json: c3cd2427cf9299ac8cceb298cd93e425c3f3739f0b61f04ba839c1bdd0519541
  results/models/lead1.05/unconstrained/seed1/model.pt: d65b19c62f1789b55bc100bab47461f1e656aea17d24ec70045451c00d4edb1f
  results/models/lead1.05/unconstrained/seed2/config.json: 0daa71058924e60628a8aedf4751a1bc72a5d6f1d3764998079ca41950f67a0a
  results/models/lead1.05/unconstrained/seed2/model.pt: 9f981cccba01fff28e5cf076196d1804eed4b42bc24f6322a8512bce1ad3ede9
  results/models/n6/kappa0/seed0/config.json: 5d7b3cb7300dc96f3b4c943d89ec74ae6fca98b044d6e6fd5762e3986e3e9e1a
  results/models/n6/kappa0/seed0/model.pt: 517fc71aea4764cc3af89321d15a73635de1576b3d70b662b5ec3bdfdf2f1c67
  results/models/n6/kappa0/seed1/config.json: cff1724b097196c3456cb99868bd7d016bd7cc13b6a1d7cdcc6ba44951302077
  results/models/n6/kappa0/seed1/model.pt: c2b2430749205a69234bcbcdb09c92b701a4d08e2f9ca5f37f884032d8d10da4
  results/models/n6/kappa0/seed2/config.json: 4de739ab62d8ea055cadd2a89df42c56e5ad6fb0311f5623a05099eebbccf5f6
  results/models/n6/kappa0/seed2/model.pt: 6fbfe62a4e7136b664cf72ee0580bfe189a634dd08b8665be82af34406e16636
  results/models/n6/meas_only/seed0/config.json: 3f1cef5305953b7c52dbeaf97144642a8932f054719cda8ea8805c7accba63de
  results/models/n6/meas_only/seed0/model.pt: 121a7c6cfe45089ff736fcffccec24c30f4f5972a5db3781a08b12fefbac6601
  results/models/n6/meas_only/seed1/config.json: 5b095875f39928d963050f7608f1ed4814d169863e33c6f3d120dd8b12598b01
  results/models/n6/meas_only/seed1/model.pt: e5fb2c48b1ebd5f5ff447dec13d96e727ec82ba0de0eece26b013a7f562e136d
  results/models/n6/meas_only/seed2/config.json: 7390bcc7852f7a9804ee00332e6e8a9be2b988f5da2657ebf0c918e815bd0edb
  results/models/n6/meas_only/seed2/model.pt: be757e0dd6a991b549a71e1fa18bdc76c9cfa9363c95f37c131fdb403ed7ac94
  results/models/n6/protected/seed0/config.json: 0f2e0002609d6c08fba114bb66929975bf5ab21cac7f27cd21e1dd38369af7a1
  results/models/n6/protected/seed0/model.pt: e6040ef49c1173ff9243721d2ac262a4dbefb13af36760063c075c3b549a9a66
  results/models/n6/protected/seed1/config.json: c0d2600e4795661ad032800b168ceb5bf6029dfcff5f8432ede839c1485f0510
  results/models/n6/protected/seed1/model.pt: ef811cf45b62620653c44d81e87d530c83b85df8756e269a7a8b4fd9354891d3
  results/models/n6/protected/seed2/config.json: 96198f0003cffb9f6f3a9f795df84ff6901b2418f097f0a3bb657a462a19b08a
  results/models/n6/protected/seed2/model.pt: ee962af66becb6be9ecc8c1034b9d6ec97f08e72fb9d39bf28a5c83696573d04
  results/models/n6/static/candidates.npz: 6e4f04ccd52b2c13d212985015e4991bdfbc87a520011d369e016fc7fc3f3d50
  results/models/n6/static/static.json: 5e9d5a9ff9f85f69809015cbb7fd80aec768661e0f7e1bc2a4f80127f735ff8c
  results/models/n6/unconstrained/seed0/config.json: d33afabfa86a067b687614801e8015751c21aee2814a0aaa2948317beffa49d4
  results/models/n6/unconstrained/seed0/model.pt: 003891222e7fa76bc0da5654dcecf9911457c6d3dda49c2352da82916a063cd3
  results/models/n6/unconstrained/seed1/config.json: 70ccccc231a6988023a7e82f371ff7bd24f6c0caf4d546e5cb50e54eb9115df5
  results/models/n6/unconstrained/seed1/model.pt: 695547427312912b592f8988cb8616cacdb2484206aad86c470830caea855682
  results/models/n6/unconstrained/seed2/config.json: 5486794b74e9e6438dfd6955dff8b07a50efcf97947c38309e35fb9be8012f2a
  results/models/n6/unconstrained/seed2/model.pt: 5dc8799179541e4120db08af7a1e30be243d31c2bad8db98bb7e6e25f6666595
  results/models/n9/kappa0/seed0/config.json: 0373de87c3591eab3d6770f4bf61e7fe415a2ef9ea5fa78570cb28144bdea159
  results/models/n9/kappa0/seed0/model.pt: 8668b7db172c71e4b8775fa5655e44192b4cda5e706ba6edd36d08e1e17b3426
  results/models/n9/kappa0/seed1/config.json: 006d0cf0b9a86af29f83e7e08b4d3511fcf752c034e11df30ac5f8d13a690469
  results/models/n9/kappa0/seed1/model.pt: e437323fa662a4fd112f8d3b0553532b4bc8be3cd6d53acfab2c20f02972e6a0
  results/models/n9/kappa0/seed2/config.json: 678420a1a13584563b731052613fdc96607781d5a36326ed517fdc20a0258bf3
  results/models/n9/kappa0/seed2/model.pt: 93f10d9dee04e3b2f56073e15c09b9af124b0dfee04fe9114fbc69bda3ea8401
  results/models/n9/meas_only/seed0/config.json: b57d7945ac4f95f8bcd5f124ab34a347b52bf81ae8ae27c85d9055d7e5db24a3
  results/models/n9/meas_only/seed0/model.pt: 2c9c3b541117c87f04b2fce63c5d2f0f59c2f1c95a0997bab8ddf8b306c4b49e
  results/models/n9/meas_only/seed1/config.json: 6249136fd460ae71f3418691bf6b8c0333147694286a3b6f7baa424eb0de0aab
  results/models/n9/meas_only/seed1/model.pt: 0dd9d0e23c6fccecf5e108f62122543d4b563f331a966be2af39ac7c448cd7ad
  results/models/n9/meas_only/seed2/config.json: 44559eccfd0f33eeebef20b6ba77d3201021a6cb1446866c0062f9dc39e4cc19
  results/models/n9/meas_only/seed2/model.pt: fa67d26d579a9b333e5afb9c6399e40339c3b22876db9d7f60c5018e259c8f9e
  results/models/n9/protected/seed0/config.json: 3ba106494516412999aa123b7ac705ce02d01c7f35b10aa9299db71ef3521f40
  results/models/n9/protected/seed0/model.pt: 4b3e681cd773002619cfa7c9bdc566d16ff294deca4c0a72dab3de983794909f
  results/models/n9/protected/seed1/config.json: d1c5da863ae7de209878281de90c5a67068af806b55f7743b11fd40f0f946bd8
  results/models/n9/protected/seed1/model.pt: cde7cdbac8d257fff6e992442a0824c938567861b4820d04ff90b7d607175d20
  results/models/n9/protected/seed2/config.json: 0f29b7a724f6194bab69f0e1695451240edca1ebc3532852b74b66b050135de9
  results/models/n9/protected/seed2/model.pt: a21d92a9cdba87bdc4d64ce95318074263bae0d184fce16ebae92604dd3951f2
  results/models/n9/static/candidates.npz: b416a7787bb71ee6322511c17d54d9b6a501d69468d1dde416721d7cea8d1f62
  results/models/n9/static/static.json: be7733a734723c6cb3f942f91f35fd0acbc8cf173af501b0f7b00aa870a27e52
  results/models/n9/unconstrained/seed0/config.json: 01b56b2f937cd3d02f58d79ad35f9710978499ea6ee4d3556d5e463e14d8ce5a
  results/models/n9/unconstrained/seed0/model.pt: d3979d1360a6c6e29b46a65d31937f820cb9d5b3a6031cdb718e2ff084fa7fcc
  results/models/n9/unconstrained/seed1/config.json: d6b83a39771adedba1937daac23c77afd8da4d8c7d62538d10c50f22930bf24a
  results/models/n9/unconstrained/seed1/model.pt: efdafbbd1164f94f22406d7b303f0bf6df5f551d7d664c87327a024e234fb89b
  results/models/n9/unconstrained/seed2/config.json: d463007e41d411a4b416a2507730838e8fb3546c2be4c02ccde315bcaa3285f3
  results/models/n9/unconstrained/seed2/model.pt: 34f052cb10c8d47ef71e6609e0a73a043fa0fc83fd0f5f5c8a13da327b008a18
  results/models/rho0.5/kappa0/seed0/config.json: 9e82b47faa8f12058ea4d98c0c875679514042a4e9f070d95d385b846693dd15
  results/models/rho0.5/kappa0/seed0/model.pt: 219e4fa48c0ad88eceddc83d19f45eca1bafe6f191b07e064ee28fbc07938d8d
  results/models/rho0.5/kappa0/seed1/config.json: 7e1e822189d5991f6d37c6c93ea0023ab49b6670c194e07c7282b48ec64558b3
  results/models/rho0.5/kappa0/seed1/model.pt: 3ebc009ab49feade15c26d2ec93b31a5670df8c2f210f474fd57fe24c06e9ba8
  results/models/rho0.5/kappa0/seed2/config.json: 0b7b970a835eeab2f71bc52d981bb926b46b46ecfb45a4c4337cdd29ce9c7707
  results/models/rho0.5/kappa0/seed2/model.pt: cadd64cf9d949dd706da896b953c74d28b2aab428fee8f1c6d217bcb755be10c
  results/models/rho0.5/meas_only/seed0/config.json: 7fcd7f4b29e53793b4da8dd2d27989ca5dbe908a60afe6a9f2c3b8234ee09bd0
  results/models/rho0.5/meas_only/seed0/model.pt: d2d917abd178393fa69d4482fda0097c89bd441e58574bc08553a75f88335eb1
  results/models/rho0.5/meas_only/seed1/config.json: 7b0d16e19ee588328e8f32f394bae5b51541e4e52b5b4f0cf9a96c59bda12976
  results/models/rho0.5/meas_only/seed1/model.pt: 5a07b290d959b8e9caccd856109f05f758d5836bb0b35f705ec687cbda558e36
  results/models/rho0.5/meas_only/seed2/config.json: 51ffff31b0ef48af5d1d8a8c24939695b0d9dc14b822890eb404ffd4cf522e2a
  results/models/rho0.5/meas_only/seed2/model.pt: 82d376eb479360c22566cb9a0794936ca57b3672a8cad66e9699b5d81d50d45e
  results/models/rho0.5/protected/seed0/config.json: 9469726b05fdc27ee78736bff1501c4865fe010512b429dd5368e445986c6710
  results/models/rho0.5/protected/seed0/model.pt: 39bd77997fadc93ac9c867832eaa16aa4f656e24daf48ebfb69b5367d32ae31a
  results/models/rho0.5/protected/seed1/config.json: 36441a565272c7c8525cbdc16f69e542c2444a685696b546292cfa21b512b98f
  results/models/rho0.5/protected/seed1/model.pt: 9d683423bbd84922107ea8bf8ce75260fe813bc301fb5ae490b01c08dd25e8a1
  results/models/rho0.5/protected/seed2/config.json: 9b6834ec1d026e67d9f9a9c74f828a49843e1a7376ba429464ee05864d524072
  results/models/rho0.5/protected/seed2/model.pt: e1715dbba14a62c15d75d4ad53b2e7b8226aae0b9a1d6182794dc2f2309b5ade
  results/models/rho0.5/static/candidates.npz: 875c8893b0e3ec3a770aea7f116059031eda58b1ed803134e3fe830237bd720d
  results/models/rho0.5/static/static.json: 8e142437d63822cdb86913b5d8b592c5c0fe1c5934d04e469039f3fc91b6106f
  results/models/rho0.5/unconstrained/seed0/config.json: 096d6c4b9976236a083f1f6ecc4aa5c50470288bda76400455a68bd1f059a3bd
  results/models/rho0.5/unconstrained/seed0/model.pt: 1ad8ade08d20d165c2f54487a6f9c0933639571772f5dc16f40d268a8bccb4a1
  results/models/rho0.5/unconstrained/seed1/config.json: 3969dd1ef78a779272f5a7f98cc20af0ba899cb68f098037a776760a65925f0b
  results/models/rho0.5/unconstrained/seed1/model.pt: 04b8f53023b140ff8c011c571916039a08f7269c890ebeb55cbb738cab02eb0b
  results/models/rho0.5/unconstrained/seed2/config.json: 6dae40dab7e44e15b4666a7e4c93b86116086ca93729b9e97baef2a5b85ec9d2
  results/models/rho0.5/unconstrained/seed2/model.pt: c768a95e8ca1df20681ba446a15916c57d2926ce2d465c7f222c98884e3d4865
  results/models/rho0.7/kappa0/seed0/config.json: b2e854e3a6f0feaf794f2786907d25ff28ac9b513e33890a860427475bb9ea70
  results/models/rho0.7/kappa0/seed0/model.pt: b44191097a4827f180b85c8c95ff8efbb6c88f69fcf8462585775ee0e48ede31
  results/models/rho0.7/kappa0/seed1/config.json: 5fceb743712e0a52aaf8016e0ccf742c2cdf034ed343326743bf412bb3c0ac4c
  results/models/rho0.7/kappa0/seed1/model.pt: 50ce8ff3bfff3b240af6e51489f23129b72daea9d101bb5cf2a9519168328641
  results/models/rho0.7/kappa0/seed2/config.json: b7bec57aaa986b8883140bbb9132f4475ed839389a1f1b9b65ed0d1449025f8f
  results/models/rho0.7/kappa0/seed2/model.pt: 1331460c60dc607824dcd744eaee70251eb1f2e6d0f406ee0e77719d450a9f93
  results/models/rho0.7/meas_only/seed0/config.json: 2c87c58dea6d820545cbdaf07c7de8b60491408ee1fcf4f8622a3877eb8e2eac
  results/models/rho0.7/meas_only/seed0/model.pt: 03c79bdafc151f7e9a72bc323a8093dc50d9dc0a609dae86bde4f9429dea79a1
  results/models/rho0.7/meas_only/seed1/config.json: 5188178e665fa7c88ad04b9381ccc2c78ce9928ecebadba3dfb30d333c29d993
  results/models/rho0.7/meas_only/seed1/model.pt: c68a1302516fc4585630c6a4706a2ca3ecd76113eaacdad5ef651b92ddbe8363
  results/models/rho0.7/meas_only/seed2/config.json: 293db1dc764a1a06782f72e763ce0725616eb86b8349be9499dd70172918f454
  results/models/rho0.7/meas_only/seed2/model.pt: e9cd7e95945c9731db837a83d1ac861e16dfc2162165bbb89d261abb46b64de3
  results/models/rho0.7/protected/seed0/config.json: 7786abaccfd583b80b146387a6e683625875aa4f67beb9375ab13e7798fe7539
  results/models/rho0.7/protected/seed0/model.pt: b9fdf3d2aecf5e682a7da28840cb62182a22bfee14a252132dce24c3c4896d0c
  results/models/rho0.7/protected/seed1/config.json: b29b56bbedd6672cb761593485dc6bae0569327cc421b1d37c3453ec43a1aff4
  results/models/rho0.7/protected/seed1/model.pt: 9973d5e5ef064cd2e386c36fd9192654804768eb2ad6a4248e9d7c1291e60baa
  results/models/rho0.7/protected/seed2/config.json: bd2d13f0fb64307190f0698073c5af6f5334669e56b938f4719d42629ee5f628
  results/models/rho0.7/protected/seed2/model.pt: 93116ae221d6a0a3534ecb9cdf2c491f8a26bdce05f75b2549d10d81e74b3449
  results/models/rho0.7/static/candidates.npz: 8ffee1e189d0109242d52d62a7d7ba264f4118b55920195ab3016f0525a17a31
  results/models/rho0.7/static/static.json: c3b3e532b90554937d482ce33527c30362cce569fdb5abbfa74f5d76cc76550c
  results/models/rho0.7/unconstrained/seed0/config.json: 4c041a7e9e7a6acc853fff9cb4615f057f7893fd223b3afaca92ff540ad99e3c
  results/models/rho0.7/unconstrained/seed0/model.pt: 7521494adb6836476d52f24ad47f33c59f84eb5d0b2ea479a504c7164f73905a
  results/models/rho0.7/unconstrained/seed1/config.json: c74cc1227294d403936afbb64ed3d43a49562dd2d76d009ace594148c10045ec
  results/models/rho0.7/unconstrained/seed1/model.pt: 76c67d4c903400e7a27a53e78285556f0803435adf130cc1f308e4bf786c07ff
  results/models/rho0.7/unconstrained/seed2/config.json: a0e51a910530779355dd3e571e558ce7563486585efc1f2cab944dd898d4a6e8
  results/models/rho0.7/unconstrained/seed2/model.pt: 2fdb1f9f78ed044d7acd1997e8a5815132f49c534b8c017b6d99b3e60103bde1
  results/models/rho1.1/kappa0/seed0/config.json: adfe39d937c135ff20fd0335f6943c11e42b96e771b329d00f2db896071bda4e
  results/models/rho1.1/kappa0/seed0/model.pt: 2fd8c3c949b953fa29f2f18fa5f0452725e47516960837b4abd6cf842d614285
  results/models/rho1.1/kappa0/seed1/config.json: 76a4b0cfaabb8e8b60788284e585c5e9266b5c61830c232fcdf86e7e85888343
  results/models/rho1.1/kappa0/seed1/model.pt: 83e5b4a232a77350dcc163e34a6571c0c061e54b69e6b661732ee1ead3a4704e
  results/models/rho1.1/kappa0/seed2/config.json: a9e1f36e8ce51a3ebfc63f1f8af937c9e065467832670af78693eeec6ad4c656
  results/models/rho1.1/kappa0/seed2/model.pt: c04b699935462102167db904225a89bfd54724e1602fd8cdd319e2ee1b181092
  results/models/rho1.1/meas_only/seed0/config.json: 0555f9e31c0cb761a7baa49f9785fc7bbdd161804561cf448bcc558e7a8cd910
  results/models/rho1.1/meas_only/seed0/model.pt: 12c7ade512a0f27704c190f270ad71284325a2d7fcd76a34ff70743440c856db
  results/models/rho1.1/meas_only/seed1/config.json: fd998317c990a9fade1c469299ecd0b73a48c18f3937156b19d9ca8917d383f9
  results/models/rho1.1/meas_only/seed1/model.pt: 893a134189123d26ae30e13805a048b7d66c2291e483ba5026954a23b8f7b845
  results/models/rho1.1/meas_only/seed2/config.json: 0be9767f2c425647a143f370d2ca32a621f8c5cac06afcff9a42aaef4fa2889f
  results/models/rho1.1/meas_only/seed2/model.pt: 2792cbb385532ac9f1d419230458dd682bd9d6682797ab98d14eaae586071e9d
  results/models/rho1.1/protected/seed0/config.json: 144b968d98b50126e7db13234c97555cfef18cb4e2e7095283e91a6cd39dac97
  results/models/rho1.1/protected/seed0/model.pt: 06c82d06ca1a3f96fee5c3451b80eaf4a300596751271e6c3634b197a074648e
  results/models/rho1.1/protected/seed1/config.json: 50adac6dc93075c9cdc098266d00432ad378c5b7ee4a9af64fa8055b507ea6e0
  results/models/rho1.1/protected/seed1/model.pt: 1104a879dfcb7d414a5e252f7bf862068e571c413cf32e463bde82f09715e63b
  results/models/rho1.1/protected/seed2/config.json: 850100f907b7ef83ce6aca999285d9ad08aae4b086d85572605f181a4a42063a
  results/models/rho1.1/protected/seed2/model.pt: e178de6812fdc4927629ce59d4e506c777d2d816592a32f5b386c6b3286a93af
  results/models/rho1.1/static/candidates.npz: 15fbdb4451fe50e920bef4edc6d0ee583f65cda0dd248a07634f76caf01d0abc
  results/models/rho1.1/static/static.json: 5eeb1798e925e3c0659b1d31a60245f7f76d11c5c9a86cff4074d8b76f6e09db
  results/models/rho1.1/unconstrained/seed0/config.json: 6a920117c055ee12471a5316d9cf874127e9f3ce596b2243f63684c59751d752
  results/models/rho1.1/unconstrained/seed0/model.pt: d8d73dae0af236449a06e46498457b8747ef4d32193e9618b645affa5184663b
  results/models/rho1.1/unconstrained/seed1/config.json: 1b05dd957404ad0a978de6f22d503d74bc9aab7db74db7cf8ab3935af5a0d635
  results/models/rho1.1/unconstrained/seed1/model.pt: 5031c459e456f8a89e037d05d421d6b2ca44a50bccaac880b002a36b52fd3f4c
  results/models/rho1.1/unconstrained/seed2/config.json: dbdddc7be1bd78d9feeb9b270ffa39e91a637fb40e2faea34569957f2d4f64a6
  results/models/rho1.1/unconstrained/seed2/model.pt: 715e0d3f7dcdad804916d9fbbd0a05d7a4cdee44f539e539203ff93e60a754fe
  results/models/selection.json: 1433881942f4d5fbef59ce6567f3f26e80306bdadd20d0294936280a59e1a906
  results/raw/e7/nr_calibration.json: 583ed29613d7d5afef40eb2bacacdf2c1be337786a55433bd3fbcb2400184734
  results/raw/train/e1/d1/usage_nu.json: 1d3fd545c0c29574ed0a1ed7d346808a53bab7dbb1c0f1bc822a85487b4aedca
  results/raw/train/e1/d15/usage_nu.json: b19afaaee4e6a12b0b4ed72358cc200ba55f0721ba24e3feb043ee2215587479
  results/raw/train/e1/d5/usage_nu.json: 2644b698547dd51694898a5b81151a55ff2078d7e9fb43e632868b780cddc7f0
  results/raw/train/e1/lead0/usage_nu.json: d123936b044897173379716067aa8056c1677f38833b0a3c1cbb6ca65b87d8b8
  results/raw/train/e1/lead1.05/usage_nu.json: ebd08debe6b9fd60735c13034749cedecc02a622f4b9e4584dd704e191fc45a3
  results/raw/train/e1/n6/usage_nu.json: 7d270f911ca5b912a3bcf8e3c72315ddf69fd01fa0a5ed75f0fa99522a613b43
  results/raw/train/e1/n9/usage_nu.json: 1ecf2240093fbd4b8b66697a8e78f56ad42668c657d87aae04bef0f1578c9306
  results/raw/train/e1/rho0.5/usage_nu.json: 919203980778d1c312a924c665f1c9c9e998456603619452c6e312536ecaf283
  results/raw/train/e1/rho0.7/usage_nu.json: 872b0a02f940c75d9fa45e0c46beafb8b2fbcf8920f039a23857a9f115f13260
  results/raw/train/e1/rho1.1/usage_nu.json: f0c8e1af166acb351d8807999f09ebeeffbb0d764bd87dc79a901e5e8686f98b
  results/raw/train/e2/usage_tariff/nu_protected_seed0_05c939251094.json: 35ea3f5304a2870ce40626efb0d9d7b8763f01ac4d6fd0e4f0365141c891f110
config_sha256: b426ab365e8e6a7cd789b04f2d9279a49381d67e6b8d02cdb940f987e45e1a8c
date_rule: Sundays and Mondays 2024-09-01..2024-10-21 (16 dates); a missing or corrupt date is replaced by the next calendar day that is neither a TEST date (including the 24-date extension) nor an earlier replacement; after 7 failed candidates the fetch stops and the failure is logged as a deviation; a date is missing when its object metadata returns HTTP 404, and corrupt when its object fails the pinned size/MD5 check on two attempts, cannot be decompressed or parsed, fails a validation of puffer.parse_video_size, or has none of the seven channels with all four formats
test_dates: 16
```

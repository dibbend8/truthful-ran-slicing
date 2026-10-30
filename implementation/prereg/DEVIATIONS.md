# Deviations after the freeze (prereg/PREREG.md, freeze 2026-10-20 22:48, PREREG sha256 ca0e4c246b75e213…)

The single TEST pass ran 2026-10-20 23:10 to 2026-10-21 08:37 (tools/run_test_pass.sh); every step exited 0, all 16 TEST dates
were downloaded without replacement, and holm.json reports complete, single-provenance inputs for the four primaries.

1. 2026-10-21 — plotting-only changes to figure scripts (PREREG §8). The figure scripts under code/experiments/figures/ are
   changed after the TEST pass only in what they draw from the frozen summaries (no driver, analysis or summary changes; no
   rerun of any experiment). Each change and the reason are listed here when made; the analysis-code hash of the freeze
   therefore no longer matches the tree for these files only.

   1.1 `fig_e2_manip.py` (fig:manip b, c; review/open_items.md items 24, 25). (b) now plots column `share_states_nonmonotone`
       of e2.json `nonmonotone_by_class`: the percentage of tenant-intervals in which the own quota falls at some step of
       the own-report grid (the per-class version of macro EtwoNonmonoStates, 21.0 % pooled, which the text quotes) instead
       of `pct_gridsteps_decrease`, the percentage of grid steps (at most 0.17 %). y-label "Tenant-intervals with a quota
       decrease (%)". The zero bars of Protected (ours) get a short baseline mark in its colour, and every bar prints its
       exact value (0, 45.9, 0.7, 16.4; the shares are exact counts without a CI). (c) x-label wrapped onto two lines
       ("Change in honest tenants' / on-time traffic (Mbit/interval)"; it was clipped, and "slices'" now reads "tenants'"
       as in the caption); design labels break after their comma (label text unchanged). Why: the panel did not show the
       quantity the text cites, the control's zero bars were invisible, and the label was clipped.
   1.2 `fig_e4_cost.py` (fig:cost a, b; item 26). (b) now plots the paired per-window welfare difference of every
       controller minus Protected (ours) with its paired two-stage cluster-bootstrap 95 % CI across dates, from e1.json
       `contrasts_main`, which holds Protected minus controller for the eight other controllers of the same E1 truthful
       main-configuration runs that e4.json `welfare_by_arm` summarises. The sign is flipped, so a positive value means the
       controller has higher welfare, as for the H4a gap and fig:scaling (b). The x-axis is linear within ±0.1 and
       logarithmic beyond, because the differences run from 0.03 to 4.9 credits per interval. The script now also reads
       e1.json. The paired table exists, so the levels fallback was not needed. Layout: the figure is drawn at its
       included width, 31/50 of the text width (4.44 in), and 3.5 in tall, so it is printed at scale 1. Before, it was
       3.5 × 4.0 in, enlarged 1.27× in print and 5.1 in tall. (a) and (b) stay stacked. The (a) legend entry
       "Fixed-priority part" now reads "Fixed-priority price", the term of the caption. Why: the level panel's date-level
       CIs all overlapped and hid the paired comparison that H4a makes.
   1.3 `fig_e4_frontier.py` (fig:cost c). The figure is drawn at its included width, 17/50 of the text width (2.43 in),
       and at the height of fig_e4_cost. It was drawn 3.5 in wide and shrunk to 70 % in print, which gave 5.6 pt text and
       a panel a third as tall as its neighbour. Other changes: panel label "(c)", which the caption uses; a logarithmic
       payment axis (the Direct optimizer's payment of 9.4 squeezed every other point into the left sixth); every
       protected point labelled with its s (the path zigzags; before, only s = 0 and s = 2 were labelled, and "s = 0"
       overlapped a y tick label); and the legend moved below the plot.
   1.4 `fig_e5.py` (fig:training; item 27). The α axis is logarithmic with ticks at the tested factors, so 0.5 and 0.7 no
       longer overlap and a factor and its inverse sit symmetric about 1. The y-label has two lines. The legend has one
       entry per row above the panels; it was wider than the column and collided with the y-label. Height 1.9 → 2.3 in.
   1.5 `fig_e3_protect.py` (fig:protect a, b). The legend of (a) is now a figure legend above the panels. As an axes
       legend it squeezed both panels into a third of the width, the class names overlapped, and the figure was 4.3 in
       wide and printed at 81 %. The y-axis of (a) is in units of 10^-8 credits per interval, with limits that include
       2e. The caption's lines for e and 2e (about 1.7e-8 and 3.3e-8) lay on the zero line of a ±0.05 axis and now sit
       visibly above the zero gains. Also: legend entry "Twice the error bound, 2e", x-grid off, height 4.0 → 4.2 in.
   1.6 `fig_e6.py` (fig:scaling). The y-axis of (c) is in units of 10^-9 credits per interval, with limits that include
       e, so the caption's dashed error-bound line is visible; before, it lay on the zero line. The legend of (c) moved
       above the panel, like that of (a); it covered the data line. All three panels share the n ticks 3, 6, 9 and the
       same limits. (c) has no point at n = 3, because E6 computes the gain at n = 6 and 9; n = 3 is fig:protect (a).
       The figure is drawn at its included width, 9/10 of the text width; its text was 7.2 pt in print.
   1.7 `fig_e7_nr.py` (fig:nr). The figure is drawn at 9/10 of the text width (text was 7.2 pt in print). The delay axis
       labels 1 and 3 per decade; with 1, 2, 3 and 5 per decade, 20000 touched 30000 and 200 touched 300. The y-labels
       and the class names have two lines; the class names touched at the narrower width.
   1.8 `fig_e1_value.py` (fig:value). The figure is drawn at its included width, 19/20 of the text width; its text was
       7.6 pt in print. The content is unchanged.
   1.9 `fig_e2e3_gain.py`, `fig_e3_bandits.py` (fig:manip a, fig:protect c). The first class panel is titled "(a) Video"
       or "(c) Video", because the captions refer to these panels by those letters and no panel carried them. The
       y-label of fig_e3_bandits has two lines; it touched the legend.
   1.10 `fig_worked_example.py` (fig:example). The panel (b) y-label unit changes from "(credits)" to
       "(credits/interval)", on two lines, as in the caption.
   1.11 `_common.py`. `plain_log` gains a `subs` argument; its default is unchanged. Two helpers are new: `wrap_label`
       inserts a line break after the first comma of a spec/controllers.yaml label without changing its text, and
       `tiny_scale` picks the power of ten for an axis of values around 1e-8. Input/output, colours, markers, summary
       paths and `make_all.py` are unchanged.
   Regenerated 2026-10-21 08:55 with `make_all.py --split test`; all 12 figures were written to paper/expanded/figures/
   and results/figures/ from the frozen results/summary/*.json. sha256 prefixes, before → after: _common e76cb36b026d7fe1 →
   e2815d669351a590, fig_e1_value 047f25ab93d0bb6d → a1db155dbf5508b3, fig_e2e3_gain b7bed47e6db763bc → 1178a35fe899d119,
   fig_e2_manip ba942db667a853c4 → 73976de0e38af636, fig_e3_bandits 6f79788f266c9d6f → 4bc0affb78585ceb, fig_e3_protect
   2cf58907b5196ae3 → 956a53e853ccd64c, fig_e4_cost 48c92557b3b34ad6 → dd33488b8f548bcc, fig_e4_frontier
   c44b28b837897842 → f121112208cd669b, fig_e5 54d9ace6d7a57057 → d5a605f8b4625007, fig_e6 099b11752c35ffbe →
   9cd1661f496dae2a, fig_e7_nr 569ddc2b89633016 → 95a67fc62964fcc0, fig_worked_example bf2e9b41e64177c0 → 222242c3bbc6e56d;
   fig_e0_worked and make_all are unchanged. The captions in paper/expanded/sections/results.tex were not edited.
   Fig. manip (b), fig. cost (b) and (c), fig. protect (a) and fig. scaling (c) now draw something the caption text
   does not yet describe (listed in the figure hand-off).
   1.12 2026-10-21, final round (review/final_r1/adjudication.md, R-F1 to R-F4; labels and one marker only, no data change):
        1.12a `fig_worked_example.py`: the truthful report (8 Mbit/s) is marked on the full-decision curve of panel (b); the
        y-label reads "Utility at demand 8 Mbit/s (credits/interval)".
        1.12b `fig_e3_protect.py`: panel (b) x-label "Premium (% of price)" (was "payment").
        1.12c `fig_e4_cost.py` and `fig_e4_frontier.py`: axis label "Price paid by truthful tenants" (was "Honest payment");
        the kappa-scale point labels read "factor ..." (was "s = ..."), and the legend "kappa scaled by a factor".
        1.12d `fig_e7_nr.py`: y-label "Delivered within the PDB" (was "Delivered in budget").
   1.13 2026-10-21, 6-page ICC version: `fig_e2e3_gain.py` also writes `fig_e2e3_gain_icc` (same data and styling; the three
        class panels are lettered (a)-(c) because the 6-page version shows this figure alone). The expanded-paper file is
        unchanged (its text layer is identical).
   1.14 2026-10-26, 6-page ICC version: `fig_e1_value.py` also writes `fig_e1_value_icc` (column width; the delay and load
        panels only, same data and styling; the lead-time result is stated in the text).
   1.15 2026-10-26, 6-page ICC version: `fig_worked_example.py` also writes `fig_worked_example_icc` (same curves; plain
        labels, legends inside the axes, after the professor could not read Fig. 1). Expanded-paper files unchanged.
   1.16 2026-10-26, 6-page ICC version: in `fig_e2e3_gain_icc` the legend names the price rule ("full-decision price
        (ours)", "fixed-priority price"), since both curves use the protected network; data unchanged.

2. 2026-10-21 — two EXPLORATORY analyses, made after the TEST pass. They are not pre-registered, test no hypothesis
   and decide nothing: no hypothesis verdict, threshold, selection or reported confirmatory number depends on them.
   No file under code/slicemech, code/experiments or results/raw and no existing results/summary/*.json was changed,
   and no experiment driver was rerun. The scripts import the frozen package read-only and lie outside the analysis
   tree, in the session scratchpad; their sha256 values are recorded in the summary under `scripts`. So they do not
   change the analysis-code hash. The one new project file is results/summary/exploratory.json (experiment
   "exploratory", split test, macros `Exp...`). tools/make_numbers.py reads it like any other summary; `--final`
   passes.

   2.1 Why do the video announcements add nothing measurable (e1.json: EoneGainVideo 0.01 against EoneGainUrllc
       0.10 credits per interval; the lead-time slope is zero)? Per class, the operator's own forecast is compared
       with the truthful report of the same interval. The forecast is the TRAIN-fitted AR(6) on measurements
       released d = 2 s late, built as sim/episode.py builds the context; it is what the No-reports controller
       allocates on. Both are also compared with the realized demand. Each error is the mean absolute error as a
       percent of the class's mean demand.
       TEST windows could not be composed through the frozen code. Its TEST read gate
       (data.puffer_fetch.require_test_read_gate, which analyze.py also calls) now refuses. The reason is that the
       analysis-code tree hash changed with the figure edits of item 1 (now 85fb05df5c4fad73…), and that full hash
       is not logged here. The gate was not bypassed.
       The forecast-against-report comparison therefore uses the 126 DEV windows (21 dates), composed through the
       frozen pipeline. TEST quantities that need no composition come from frozen TEST outputs:
       * report against realized demand: the E0 TEST window rows;
       * forecast against realized demand: the demand reconstructed from the E1 TEST interval rows (on_time +
         dropped + change in backlog; exact to about 1e-6 when checked on DEV), intervals 48 to 199;
       * the share of intervals in which a tenant's quota left no idle capacity: the same E1 rows.
       Result (descriptive): the forecast misses the video report by 14.9 % of mean video demand. That is about
       9 PRB per interval, and 18.0 % on TEST, where the video report equals the realized demand. It misses the
       URLLC-like report by 3.5 % (1.1 PRB) and the mMTC-like report by 4.9 % (0.5 PRB). So the video report
       carries the most new information, not the least, and the forecast does not already know it. Its small value
       fits a different pattern instead: the video quota leaves no idle capacity in 77 % of TEST intervals under
       both Protected (ours) and No reports (URLLC-like 32 %, mMTC-like 79 %). This explanation was not tested
       causally.
   2.2 Why does the 5G-LENA replay deliver so little on time (e7.json per_slice_mean)? Inputs, read only: the 12
       E7 TEST case outputs and the E7 NR calibration (250 m, the same RngRuns). Per slice, the analysis reports
       offered traffic, NR-delivered traffic, quota, MCS, and new-TB and delivered Mbit per used PRB. It also
       reports offered traffic as a percent of the quota's NR capacity, where capacity = quota × the per-PRB rate
       NR achieved in that case. For comparison it uses the fluid replays of the same quotas that e7_nr stored.
       Result (descriptive; 4 windows × 3 arms; no CIs):
       * Offered traffic exceeds the quotas' NR capacity: video 114 %, URLLC-like 131 %, mMTC-like 221 % (the
         mMTC-like slice in all 12 cases).
       * The NR per-PRB rates are 18–29 % below the calibration's. Single badly placed UE drops dominate: URLLC at
         MCS 0 in RngRun 2, mMTC at MCS 5.7 against 13.7 in the calibration in RngRun 4.
       * The mMTC-like slice's lower rate is not mainly per-packet overhead. Its packets average 648 B (the
         calibration's average 1125 B), and 5.4 % of its new-TB bytes are not delivered, against 1.9 % in the
         calibration.
       * RLC UM never discards late packets, so the excess fills the per-UE RLC buffers: drops are 23 / 32 / 58 %.
         Delivered packets wait about one buffer-drain time, so p95 delays are 3.0 / 7.8 / 26.7 s.
       * The fluid model replaying the same quotas without deadline expiry also delivers 0 % of the mMTC-like
         traffic on time. With expiry it delivers 56 %.

3. 2026-10-21 — derived reporting macros (results/summary/final_r1_derived.json). 18 numbers that the frozen TEST summaries already
   contain in their tables (price by class, welfare when the report sensitivity is zeroed at deployment, the Direct optimizer's price,
   the gain at d = 5 and 15 s as a percentage, the Oracle contrast, the worked interval's best report and quota) are copied, without
   recomputation, into macros so that the final text can cite them; each macro names its source file, table and column
   (review/final_r1/scripts/make_derived.py). No analysis or experiment is rerun.

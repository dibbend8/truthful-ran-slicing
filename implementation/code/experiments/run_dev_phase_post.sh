#!/bin/bash
# DEV pass with the pre-freeze code (orchestrator, 2026-10-20): after the audit fixes (review/prefreeze_audit.md) the code
# fingerprints changed, so E1 (new columns cap_Mbit/unused_Mbit/prb_busy/r_true, code_fp) is recomputed at every
# configuration; E5 is re-run on two DEV dates as a regression check of the new job bookkeeping (the numbers must equal the
# archived ones); E3/E4/E6 rehearse on six DEV dates; then the DEV analysis (power analysis for the 16/24 decision) and figures.
# E2 is kept from the earlier code (DEV provenance is informational). Worker counts are limited by RAM (31 GB; systemd-oomd).
set -u
ROOT=/home/darpan/Desktop/MTP/MTP-Droy/ICC/rewrite_2026-10-18
cd "$ROOT/code"
export PYTHONDONTWRITEBYTECODE=1
PY=/home/darpan/anaconda3/bin/python
LOG=$ROOT/results/logs
STATUS=$LOG/dev_phase_status.txt
REHEARSAL_DATES="2024-02-28 2024-03-02 2024-03-04 2024-03-06 2024-03-09 2024-03-10"
echo "[$(date '+%F %T')] POST-AUDIT DEV PASS (run_dev_phase_post.sh, code swapped in)" >> "$STATUS"
step() {  # step <name> <command...>
  local name=$1; shift
  local t0=$(date +%s)
  echo "[$(date '+%F %T')] START $name" >> "$STATUS"
  "$@" >> "$LOG/dev_${name}.out" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END   $name rc=$rc ($(( $(date +%s) - t0 )) s)" >> "$STATUS"
  df -h / | tail -1 >> "$STATUS"
  return 0
}
step e1_default_v2 $PY experiments/e1_value.py --split dev --config default --workers 16 --fit-nu
step e5_regress    $PY experiments/e5_training_channel.py --split dev --workers 12 --dates 2024-02-28,2024-03-02
for c in d1 d5 d15 rho0.5 rho0.7 rho1.1 lead0 lead1.05 n6 n9; do
  step "e1_$c"     $PY experiments/e1_value.py --split dev --config "$c" --workers 16 --fit-nu
done
step e3_rehearsal  $PY experiments/e3_protection.py --split dev --workers 14 --dates $REHEARSAL_DATES
step e4_rehearsal  $PY experiments/e4_cost.py --split dev --workers 14 --dates $REHEARSAL_DATES
step e6_rehearsal  $PY experiments/e6_scaling.py --split dev --workers 14 --dates "${REHEARSAL_DATES// /,}"
step analyze       $PY experiments/analyze.py --split dev --workers 8
step figures       $PY experiments/figures/make_all.py --split dev
echo "[$(date '+%F %T')] DEV PHASE DONE" >> "$STATUS"

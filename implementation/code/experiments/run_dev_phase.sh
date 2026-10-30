#!/bin/bash
# DEV-phase evaluation sequence (orchestrator, 2026-10-20). Decision-critical steps first (E0, E1 default, E5, E2),
# then the E1 sweeps, then reduced rehearsals of E3, E4, E6 on six DEV dates (they decide nothing on DEV but must run
# cleanly before the single TEST pass), then the DEV analysis and figures. Each step logs to results/logs/dev_<step>.out
# and the script continues after a failed step (the failure is recorded in results/logs/dev_phase_status.txt).
set -u
ROOT=/home/darpan/Desktop/MTP/MTP-Droy/ICC/rewrite_2026-10-18
cd "$ROOT/code"
export PYTHONDONTWRITEBYTECODE=1
PY=/home/darpan/anaconda3/bin/python
LOG=$ROOT/results/logs
STATUS=$LOG/dev_phase_status.txt
REHEARSAL_DATES="2024-02-28 2024-03-02 2024-03-04 2024-03-06 2024-03-09 2024-03-10"
: > "$STATUS"

step() {  # step <name> <command...>
  local name=$1; shift
  local t0=$(date +%s)
  echo "[$(date '+%F %T')] START $name" >> "$STATUS"
  "$@" > "$LOG/dev_${name}.out" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END   $name rc=$rc ($(( $(date +%s) - t0 )) s)" >> "$STATUS"
  df -h / | tail -1 >> "$STATUS"
  return 0
}

step e0            $PY experiments/e0_setup.py --split dev --config default --workers 16
step e1_default    $PY experiments/e1_value.py --split dev --config default --workers 20 --fit-nu
step e5            $PY experiments/e5_training_channel.py --split dev --workers 16
step e2            $PY experiments/e2_manipulation.py --split dev --workers 20
for c in d1 d5 d15 rho0.5 rho0.7 rho1.1 lead0 lead1.05 n6 n9; do
  step "e1_$c"     $PY experiments/e1_value.py --split dev --config "$c" --workers 20 --fit-nu
done
step e3_rehearsal  $PY experiments/e3_protection.py --split dev --workers 20 --dates $REHEARSAL_DATES
step e4_rehearsal  $PY experiments/e4_cost.py --split dev --workers 20 --dates $REHEARSAL_DATES
step e6_rehearsal  $PY experiments/e6_scaling.py --split dev --workers 20 --dates $REHEARSAL_DATES
step analyze       $PY experiments/analyze.py --split dev --workers 8
step figures       $PY experiments/figures/make_all.py --split dev
echo "[$(date '+%F %T')] DEV PHASE DONE" >> "$STATUS"

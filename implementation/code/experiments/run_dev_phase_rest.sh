#!/bin/bash
# Resume of run_dev_phase.sh after the orchestrating session ended at 09:31 on 2026-10-20 (E0, E1 default and E5 had
# finished; E2 had started and was killed). Same steps and options as run_dev_phase.sh from E2 on; the drivers are
# resumable (finished jobs are skipped by their provenance fingerprint). Appends to results/logs/dev_phase_status.txt.
set -u
ROOT=/home/darpan/Desktop/MTP/MTP-Droy/ICC/rewrite_2026-10-18
cd "$ROOT/code"
export PYTHONDONTWRITEBYTECODE=1
PY=/home/darpan/anaconda3/bin/python
LOG=$ROOT/results/logs
STATUS=$LOG/dev_phase_status.txt
REHEARSAL_DATES="2024-02-28 2024-03-02 2024-03-04 2024-03-06 2024-03-09 2024-03-10"
echo "[$(date '+%F %T')] RESUME (run_dev_phase_rest.sh) after the session ended during e2" >> "$STATUS"

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

step e2            $PY experiments/e2_manipulation.py --split dev --workers 14
for c in d1 d5 d15 rho0.5 rho0.7 rho1.1 lead0 lead1.05 n6 n9; do
  step "e1_$c"     $PY experiments/e1_value.py --split dev --config "$c" --workers 16 --fit-nu
done
step e3_rehearsal  $PY experiments/e3_protection.py --split dev --workers 14 --dates $REHEARSAL_DATES
step e4_rehearsal  $PY experiments/e4_cost.py --split dev --workers 14 --dates $REHEARSAL_DATES
step e6_rehearsal  $PY experiments/e6_scaling.py --split dev --workers 14 --dates $REHEARSAL_DATES
step analyze       $PY experiments/analyze.py --split dev --workers 8
step figures       $PY experiments/figures/make_all.py --split dev
echo "[$(date '+%F %T')] DEV PHASE DONE" >> "$STATUS"

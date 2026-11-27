#!/bin/bash
# E1 debug (REVIEW_PLAN E1, reading "low | low": our pipeline does not
# reproduce published GABAR - stop and debug). Trains and tests GABAR with the
# code as it was BEFORE any GADAR change (536c352, 2025-08-26) and the README's
# flags, on one domain, on the same test instances (PDDLEnv<dom>Test-v0).
#   campaign/jobs/gabar_prejuly.sh logistics_ipcc 96
# Reading: near the published coverage -> a regression introduced since July
# (bisect it); the same ~0% as today -> not our code but the setup (test
# instances, training-set size, hyperparameters, plan-length bound).
# Runs in its own worktree with its OWN cache/ and models/ (the old cache
# format must never touch the real ones). Only the built planner is shared.
set -u
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD
DOM=${1:?usage: gabar_prejuly.sh <domain> <n_test>}; NTEST=${2:?number of test problems}
BASE=${BASE:-536c352}
WT="$REPO/campaign/worktrees/prejuly_$DOM"
mkdir -p campaign/worktrees campaign/reports logs
[ -e "$WT/.git" ] || git worktree add --detach "$WT" "$BASE" || exit 1
# Fast Downward is built inside the checkout (ploi/planning/FD); in a fresh
# worktree that is an empty submodule directory, and the old code would try to
# clone and build it. Point it at the built one.
if [ ! -e "$WT/ploi/planning/FD/fast-downward.py" ]; then
    rm -rf "$WT/ploi/planning/FD"
    ln -s "$REPO/ploi/planning/FD" "$WT/ploi/planning/FD"
fi
[ -e "$WT/ploi/planning/FD/fast-downward.py" ] || { echo "no built planner at $REPO/ploi/planning/FD"; exit 1; }
mkdir -p "$WT/cache/results" "$WT/models"
LOG="$REPO/logs/prejuly_${DOM}.log"
echo "GABAR at $BASE on $DOM, $NTEST test problems; log: $LOG"
(cd "$WT" && python main.py --method ltp --domain "$DOM" --all-problems \
    --lr 0.0005 --n-heads 1 --attention-dropout 0 --dropout 0 --weight-decay 0.00 \
    --wandb False --num-train-problems 200 --num-test-problems "$NTEST" --epochs 500 \
    --gnn-rounds 9 --batch-size 16 --mode train_test --max-plan-length 500 \
    --problems-per-division 10 --run-learned-model True --run-non-optimal False \
    --use-global-node True --device cuda:0) > "$LOG" 2>&1
rc=$?
OUT=campaign/reports/e1_gabar_${DOM}_prejuly.txt
{ echo "== GABAR at $BASE (pre-GADAR code), $DOM, $NTEST test problems, rc=$rc"
  echo "== full log: $LOG"
  echo "== planner failures while collecting training plans: $(grep -ac 'Planning failed' "$LOG")"
  grep -aiE "testing model from epoch|success|failed:|coverage|plan quality" "$LOG" | tail -40
  echo "== last lines"; tail -15 "$LOG"
} > "$OUT"
echo "wrote $OUT"
exit $rc

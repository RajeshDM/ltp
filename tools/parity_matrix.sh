#!/usr/bin/env bash
# Batch vs sequential parity across device x determinism.
#
# Eight runs: {cpu,gpu} x {det,nodet} x {sequential,batch}. Within each
# (device, determinism) cell the two harnesses are compared two ways:
#
#   trajectory  - compare_score_traces.py --action-only: did the two paths
#                 SELECT the same action at every step? Score values may
#                 differ (float accumulation order); only "forked" matters.
#   outcome     - the coverage and plan-quality lines from each run.
#
# Trajectory parity is the strict test; outcome parity is what actually
# affects reported numbers. A cell can fork and still land on the same
# coverage, and a cell that never forks cannot differ in outcome.
#
# Usage:
#   bash tools/parity_matrix.sh [CONFIG] [TEST_DOMAINS] [OUTDIR]
#
# Defaults are the visitall timing fixture at 5 problems, which is enough
# to expose a fork and finishes in minutes. Raise the problem count only
# after a cell shows a fork and you want to know how often it happens.

set -u

CONFIG="${1:-configs/ab_visitall.yaml}"
TEST_DOMAINS="${2:-visitall_ipcc:5}"
OUT="${3:-parity_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT/traces" "$OUT/logs"
echo "config=$CONFIG  test-domains=$TEST_DOMAINS  out=$OUT"
echo

COMMON="--config $CONFIG --mode test --test-model-metrics combined \
--num-models-to-test 1 --test-domains $TEST_DOMAINS"

run_one() {           # device determinism harness
  local dev="$1" det="$2" harness="$3"
  local tag="${dev}_${det}_${harness}"
  local log="$OUT/logs/$tag.log"
  local trace="$OUT/traces/$tag.jsonl"

  local -a env_vars=(GABAR_TRACE_SCORES="$trace")
  [ "$det" = "det" ] && env_vars+=(GABAR_DETERMINISTIC=1 PYTHONHASHSEED=42)
  [ "$harness" = "batch" ] && env_vars+=(GABAR_BATCH_EVAL=1)

  local devflag="--device cpu"
  [ "$dev" = "gpu" ] && devflag="--device cuda:0"

  printf "  %-22s " "$tag"
  local t0=$SECONDS
  if env "${env_vars[@]}" python main.py $COMMON $devflag \
        --expid "parity_$tag" > "$log" 2>&1; then
    printf "ok   %4ds\n" "$((SECONDS - t0))"
  else
    printf "FAIL %4ds  (see %s)\n" "$((SECONDS - t0))" "$log"
  fi
}

# The first "Succ:" line is the learned model; the second is the reference
# planner. Plan Quality is printed once, after both.
outcome() {
  local log="$1"
  local succ pq
  succ=$(grep -aoE "Succ: [0-9.]+%/[0-9.]+%" "$log" | head -1 | sed 's/Succ: //')
  pq=$(grep -a "Plan Quality" "$log" | head -1 | awk '{print $NF}')
  echo "${succ:-none} pq=${pq:-none}"
}

echo "=== running 8 configurations ==="
for dev in cpu gpu; do
  for det in det nodet; do
    for harness in seq batch; do
      run_one "$dev" "$det" "$harness"
    done
  done
done

echo
echo "=== parity within each cell (sequential vs batch) ==="
printf "%-12s %-34s %-34s %s\n" "cell" "sequential" "batch" "trajectory"
for dev in cpu gpu; do
  for det in det nodet; do
    s="$OUT/logs/${dev}_${det}_seq.log"
    b="$OUT/logs/${dev}_${det}_batch.log"
    ts="$OUT/traces/${dev}_${det}_seq.jsonl"
    tb="$OUT/traces/${dev}_${det}_batch.jsonl"

    verdict="no traces"
    if [ -s "$ts" ] && [ -s "$tb" ]; then
      cmp_out=$(python compare_score_traces.py "$ts" "$tb" --action-only 2>&1)
      echo "$cmp_out" > "$OUT/logs/compare_${dev}_${det}.txt"
      # "N forked" is the only number that means different actions were taken.
      forked=$(echo "$cmp_out" | grep -aoE "[0-9]+ forked" | head -1 | awk '{print $1}')
      if [ "${forked:-x}" = "0" ]; then
        verdict="SAME actions"
      elif [ -n "${forked:-}" ]; then
        verdict="$forked FORKED"
      else
        verdict="unparsed"
      fi
    fi
    printf "%-12s %-34s %-34s %s\n" "$dev/$det" "$(outcome "$s")" "$(outcome "$b")" "$verdict"
  done
done

echo
echo "detail: $OUT/logs/compare_<cell>.txt"
echo
echo "Reading this table:"
echo "  outcomes equal + SAME actions -> parity holds, batching is safe here"
echo "  outcomes equal + N FORKED     -> trajectories differ, coverage happened"
echo "                                   to agree; not safe to rely on"
echo "  outcomes differ               -> batching changes reported numbers"

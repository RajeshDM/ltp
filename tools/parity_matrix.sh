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
# POOL_WORKERS=N enables the `pool` harness (batched + N rollout workers).
POOL_WORKERS="${POOL_WORKERS:-0}"
# Which harnesses and devices to run. The FIRST harness is the reference;
# every other is compared against it.
#   full check (default):  seq batch [pool]      x {cpu,gpu} = 8-12 runs
#   quick check:           HARNESSES="batch pool" DEVICES=gpu  = 4 runs
# Keep `seq` first whenever the featurizer or executor changed: it is the
# reference implementation, and batch-vs-pool cannot detect a bug both
# inherit from it.
DEFAULT_HARNESSES="seq batch"
[ "$POOL_WORKERS" -gt 0 ] && DEFAULT_HARNESSES="$DEFAULT_HARNESSES pool"
HARNESSES="${HARNESSES:-$DEFAULT_HARNESSES}"
DEVICES="${DEVICES:-cpu gpu}"
REFERENCE="${HARNESSES%% *}"

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
  if [ "$harness" = "pool" ]; then
    # Tracing is refused with workers (the harness says so and falls back),
    # so the pool cell is compared on OUTCOMES only.
    env_vars=(GABAR_BATCH_EVAL=1 GABAR_FEATURIZE_WORKERS="$POOL_WORKERS")
    [ "$det" = "det" ] && env_vars+=(GABAR_DETERMINISTIC=1 PYTHONHASHSEED=42)
  fi

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

N_RUNS=0
for dev in $DEVICES; do for det in det nodet; do for h in $HARNESSES; do
  N_RUNS=$((N_RUNS + 1)); done; done; done
echo "=== running $N_RUNS configurations (reference: $REFERENCE) ==="
for dev in $DEVICES; do
  for det in det nodet; do
    for harness in $HARNESSES; do
      run_one "$dev" "$det" "$harness"
    done
  done
done

echo
echo "=== parity within each cell (vs $REFERENCE) ==="
for dev in $DEVICES; do
  for det in det nodet; do
    ref_log="$OUT/logs/${dev}_${det}_${REFERENCE}.log"
    ref_trace="$OUT/traces/${dev}_${det}_${REFERENCE}.jsonl"
    printf "%-10s %-8s %-34s %s\n" "$dev/$det" "$REFERENCE" \
           "$(outcome "$ref_log")" "(reference)"

    for harness in $HARNESSES; do
      [ "$harness" = "$REFERENCE" ] && continue
      log="$OUT/logs/${dev}_${det}_${harness}.log"
      trace="$OUT/traces/${dev}_${det}_${harness}.jsonl"

      if [ "$(outcome "$log")" = "$(outcome "$ref_log")" ]; then
        verdict="outcome matches"
      else
        verdict="OUTCOME DIFFERS"
      fi

      # Trajectory comparison only when both sides traced (the pool does not).
      if [ -s "$ref_trace" ] && [ -s "$trace" ]; then
        cmp_out=$(python compare_score_traces.py "$ref_trace" "$trace" --action-only 2>&1)
        echo "$cmp_out" > "$OUT/logs/compare_${dev}_${det}_${harness}.txt"
        forked=$(echo "$cmp_out" | grep -aoE "[0-9]+ forked" | head -1 | awk '{print $1}')
        if [ "${forked:-x}" = "0" ]; then
          verdict="$verdict, SAME actions"
        elif [ -n "${forked:-}" ]; then
          verdict="$verdict, $forked FORKED"
        fi
      fi
      printf "%-10s %-8s %-34s %s\n" "" "$harness" "$(outcome "$log")" "$verdict"
    done
  done
done

echo
echo "detail: $OUT/logs/compare_<cell>.txt"
echo
echo "Reading this table:"
echo "  outcome matches + SAME actions -> parity holds, that harness is safe"
echo "  outcome matches + N FORKED     -> trajectories differ, coverage happened"
echo "                                    to agree; not safe to rely on"
echo "  OUTCOME DIFFERS                -> that harness changes reported numbers"
echo "  (no trajectory verdict shown when a harness does not emit traces,"
echo "   e.g. pool: workers are refused while GABAR_TRACE_SCORES is set)"

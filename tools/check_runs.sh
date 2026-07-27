#!/bin/bash
# Is the 5-split joint_chain launch actually doing what we asked?
#
# Run from the repo root on EACH machine; the union is the real picture.
#
# IMPORTANT: pgrep only sees processes on THIS host. On a login/submit node it
# returns nothing even when every job is healthy, so process state is reported
# as "not visible from this node" -- never as a failure. The authoritative
# signals come from the LOG:
#
#   periodic  - the word 'periodic' in a "Saved new best model for metrics:"
#               line proves --checkpoint-every took effect. Its absence proves
#               nothing until epoch N has been reached.
#   age       - seconds since the log was last written. A training run touches
#               it every ~10 epochs; a featurizing run every few seconds. Many
#               minutes of silence on an unfinished run means it died.
#   _ty2      - graphs rebuilt with type compilation, not reused from a
#               pre-compilation cache.
#   types     - compiled-type counts. 0 for grid/logistics is correct; 0 for
#               gripper/miconic/spanner/rovers is a bug.
#
#     bash tools/check_runs.sh

RUNS="loo8_joint_chain_no_manyblocks loo8_joint_chain_no_gripper
      loo8_joint_chain_no_miconic loo8_joint_chain_no_grid
      loo8_joint_chain_no_logistics"

NOW=$(date +%s)

for n in $RUNS; do
    L="logs/$n.log"
    echo "=================================================================="
    if [ ! -f "$L" ]; then
        echo "$n"
        echo "  not on this machine (no logs/$n.log)"
        continue
    fi

    AGE=$(( NOW - $(stat -c %Y "$L" 2>/dev/null || echo "$NOW") ))
    LAST=$(grep '^Epoch' "$L" | tail -1)
    # Match BOTH the logger line 'lifted layer (joint_chain):' and the
    # print 'lifted layer [joint_chain]:'. print() is block-buffered under
    # nohup, so early in a run only the logger line exists and counting just
    # the print made a featurizing run look like it had not started.
    NTYPE=$(grep -cE 'lifted layer [\[(]' "$L")

    if [ -n "$LAST" ]; then
        case "$LAST" in
            *"$(grep -o 'Epoch [0-9]*/[0-9]*' "$L" | tail -1 | cut -d/ -f2)/"*) : ;;
        esac
        STATE="TRAINING"
    elif [ "$NTYPE" -gt 0 ]; then
        STATE="FEATURIZING ($NTYPE/7 domains)"
    else
        STATE="starting / loading raw cache"
    fi
    grep -q 'Testing\|=== zero-shot\|=== in-domain' "$L" && STATE="TESTING"

    echo "$n    [$STATE]"
    printf '  log written:      %dm %ds ago\n' $((AGE/60)) $((AGE%60))
    [ "$AGE" -gt 1800 ] && [ "$STATE" != "TESTING" ] && \
        echo "  *** 30+ min of silence -- likely dead, or this log is from an OLD run"

    if grep -q 'metrics:.*periodic' "$L"; then
        echo "  checkpoint-every: CONFIRMED (periodic checkpoints being saved)"
    elif [ -n "$LAST" ]; then
        echo "  checkpoint-every: ** no periodic save yet -- expected by epoch 75 **"
    else
        echo "  checkpoint-every: unverifiable until the first epoch"
    fi

    CMD=$(pgrep -af "configs/$n.yaml" 2>/dev/null | head -1)
    if [ -z "$CMD" ]; then
        echo "  process:          not visible from this node (says nothing)"
    else
        echo "  process:          alive, $(pgrep -fc "configs/$n.yaml") pids"
        case "$CMD" in
            *"--test-domains"*) echo "  test scope:       scoped" ;;
            *) echo "  test scope:       ** full 9-domain sweep (~12h) **" ;;
        esac
    fi

    echo "  sidecars _ty2:    $(grep -c '_ty2' "$L") lines"
    echo "  type warnings:    $(grep -c 'WARNING: declared type' "$L")"
    grep -oE 'lifted layer [\[(].*' "$L" | sed 's/.*(\([0-9]*\) type-compiled.*/  \1 type(s)/;s/.*: \([0-9]*\) declared.*/  \1 type(s)/' \
        | sort | uniq -c | sed 's/^/    /'
    echo "  last epoch:       ${LAST:-none yet}"
    FAIL=$(grep -ciE 'traceback|CUDA error|out of memory|size mismatch' "$L")
    [ "$FAIL" != "0" ] && echo "  *** $FAIL error lines -- grep -iE 'traceback|cuda error' $L"
done

echo
echo "Healthy = log written recently + state advancing. A run still"
echo "FEATURIZING has no epochs and no periodic saves yet; that is normal."
echo "Compiled-type counts: gripper 4, miconic 2, visitall 1, spanner 4,"
echo "rovers 7, grid 0, logistics 0, manyblocks 1 -- each appears only in"
echo "runs where it is a TRAINING domain (6 others + itself absent = 7)."

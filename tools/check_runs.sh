#!/bin/bash
# Is the 5-split joint_chain launch actually doing what we asked?
#
# Run from the repo root on EACH machine; the union of the two outputs is the
# real picture. Checks the things that fail SILENTLY and are only discovered
# after ten hours of training:
#
#   flags     - --checkpoint-every missing => the 'periodic' bucket stays
#               empty => the test phase finds no models and the run is lost
#   sidecar   - _ty2 tag => graphs were rebuilt with type compilation, not
#               reused from a pre-compilation cache
#   types     - per-domain compiled-type counts; 0 for grid/logistics is
#               correct, 0 for gripper/miconic/spanner/rovers is a bug
#   warnings  - a declared type no object reports => type hierarchy, that
#               slot's has_type path is dead
#
#     bash tools/check_runs.sh

RUNS="loo8_joint_chain_no_manyblocks loo8_joint_chain_no_gripper
      loo8_joint_chain_no_miconic loo8_joint_chain_no_grid
      loo8_joint_chain_no_logistics"

for n in $RUNS; do
    L="logs/$n.log"
    CMD=$(pgrep -af "configs/$n.yaml" 2>/dev/null | head -1)
    P=$(pgrep -fc "configs/$n.yaml" 2>/dev/null || true); P=${P:-0}

    echo "=================================================================="
    echo "$n    processes=$P"
    if [ ! -f "$L" ]; then
        echo "  not on this machine (no logs/$n.log)"
        continue
    fi

    # The flag that loses a run if it is missing.
    case "$CMD" in
        *"--checkpoint-every"*) CE="yes" ;;
        *)                      CE="** MISSING -- periodic bucket will be empty **" ;;
    esac
    case "$CMD" in
        *"--test-domains"*) TD="scoped" ;;
        *)                  TD="** full 9-domain sweep (~12h) **" ;;
    esac
    echo "  checkpoint-every: $CE"
    echo "  test scope:       $TD"
    [ -n "$CMD" ] && echo "  cmd: ...${CMD#*main.py}"

    echo "  sidecars _ty2:    $(grep -c '_ty2' "$L") lines"
    echo "  type warnings:    $(grep -c 'WARNING: declared type' "$L")"
    grep 'lifted layer' "$L" | sed 's/^ *//' | sort | uniq -c \
        | sed 's/^/    /'
    echo "  last epoch:       $(grep '^Epoch' "$L" | tail -1)"
    echo "  saved metrics:    $(grep -o 'metrics: .*' "$L" | tail -1)"
    FAIL=$(grep -ciE 'traceback|CUDA error|out of memory|size mismatch' "$L")
    [ "$FAIL" != "0" ] && echo "  *** $FAIL error lines -- grep -iE 'traceback|cuda error' $L"
done

echo
echo "Expected: 5 runs across the two machines, each with checkpoint-every=yes,"
echo "test scope=scoped, 7 'lifted layer' lines, 0 type warnings."
echo "Compiled-type counts: gripper 4, miconic 2, visitall 1, spanner 4,"
echo "rovers 7, grid 0, logistics 0, manyblocks 1 (each appears only in the"
echo "runs where it is a TRAINING domain)."

#!/bin/bash
# Collect one experiment's results into campaign/reports/<name>.txt (small,
# committed; the raw dumps stay in cache/results/).   report.sh e0|e1|e2
set -u
cd "$(dirname "$0")/../.." || exit 1
mkdir -p campaign/reports
OUT=campaign/reports/${1:?usage: report.sh e0|e1|e2}.txt
case "$1" in
  e0)
    { echo "== rule baselines (cache/results/e0/*.txt) and the random floor"
      cat cache/results/e0/*.txt cache/results/random_floor_summary.txt 2>/dev/null
    } > "$OUT" ;;
  e1)
    { echo "== GABAR through our pipeline, and optimal-plan runs (combined-loss checkpoint)"
      python tools/analyze_results.py 2>/dev/null \
          | grep -E "^(experiment|baseline_|sd_joint_chain_.*_opt)"
      echo; echo "== single-domain GADAR and the seven-domain models, for comparison"
      python tools/indomain_table.py --since 20261101
    } > "$OUT" ;;
  e2)
    RUNGS=joint_chain,jc_no_bind,jc_no_chain,jc_no_eff,jc_no_goal,joint,joint_lite
    FOLDS=manyblocks,visitall,grid
    { echo "== in-domain, same folds in every column ($FOLDS held out)"
      python tools/indomain_table.py --rungs "$RUNGS" --folds "$FOLDS" --since 20261101
      echo; echo "== zero-shot, test split: per-fold e*, MEAN over all snapshots"
      python tools/crossfold_select.py --since 20261101
      echo; echo "== zero-shot, training-scale split"
      python tools/crossfold_select.py --split train --since 20261101
    } > "$OUT" ;;
  *) echo "unknown report $1"; exit 1 ;;
esac
echo "wrote $OUT"

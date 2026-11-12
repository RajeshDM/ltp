#!/bin/bash
# Gate: the featurizer at HEAD must emit byte-identical graphs to the commit
# before the C2 ablation modes (CLAUDE.md 1.1, tools/graph_fingerprint.py) for
# the modes that already existed. Needs the featurized sidecars in cache/results.
#   BASE=82b901f TAGS="joint_chain_kp3_ka6_ty2 joint_lite_kp3_ka6_ty2" \
#   DOMAINS=visitall_ipcc,grid_ipcc,gripper_ipcc,logistics_ipcc campaign/jobs/gate_fingerprint.sh
# Writes campaign/reports/gate_fingerprint.txt only if every comparison passes.
set -eu
cd "$(dirname "$0")/../.."
BASE=${BASE:-82b901f}
TAGS=${TAGS:-"joint_chain_kp3_ka6_ty2 joint_lite_kp3_ka6_ty2"}
DOMAINS=${DOMAINS:-visitall_ipcc,grid_ipcc,gripper_ipcc,logistics_ipcc}
TMP=$(mktemp -d); WT="$TMP/base"; REPO=$PWD
cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; rm -rf "$TMP"; }
trap cleanup EXIT
git worktree add --detach "$WT" "$BASE" >/dev/null
ln -s "$REPO/cache" "$WT/cache"
mkdir -p campaign/reports; OUT=campaign/reports/gate_fingerprint.txt; : > "$OUT.tmp"
for T in $TAGS; do
    # the tool against ITSELF first: a drifting instrument reports avalanche
    # mismatches that look like a broken refactor (CLAUDE.md 1.1)
    for run in a b; do
        (cd "$WT" && PYTHONHASHSEED=42 python tools/graph_fingerprint.py \
            --domains "$DOMAINS" --tag "$T" --out "$TMP/base_${T}_$run.json") >/dev/null
    done
    python tools/graph_fingerprint.py --compare "$TMP/base_${T}_a.json" "$TMP/base_${T}_b.json" >/dev/null \
        || { echo "instrument not self-consistent for $T"; exit 1; }
    PYTHONHASHSEED=42 python tools/graph_fingerprint.py --domains "$DOMAINS" --tag "$T" \
        --out "$TMP/head_$T.json" >/dev/null
    { echo "== $T: $BASE vs HEAD"
      python tools/graph_fingerprint.py --compare "$TMP/base_${T}_a.json" "$TMP/head_$T.json"
    } | tee -a "$OUT.tmp"
done
mv "$OUT.tmp" "$OUT"

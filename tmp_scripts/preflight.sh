#!/bin/bash
# preflight.sh — prove this node can run the campaign unattended.
#
#   ./tmp_scripts/preflight.sh          # ~10 minutes, run on BOTH nodes
#
# Run this before walking away. It is the difference between "the imports
# work" and "a real run trained, evaluated, and its numbers are visible on
# wandb from my phone". Every check is the real thing, not a proxy:
#
#   1  environment           imports, cores, CUDA, free disk
#   2  wandb round trip      init -> log -> finish, then read it BACK from the
#                            server. A valid API key with no network path
#                            still passes a credentials check and still loses
#                            two days of results.
#   3  end-to-end run        4 epochs of the real pipeline through
#                            tmp_scripts/preflight.yaml: multi-domain harness,
#                            joint_chain featurization, training, checkpoint,
#                            batched+pooled evaluation
#   4  artifacts             checkpoint on disk, results JSON on disk
#   5  results online        the run's test metrics read back from the wandb
#                            API - the thing actually being relied on
#   6  aggregation           tools/analyze_results.py parses what was written
#
# Exit code is 0 only if every check passed.
#
# THROWAWAY - see tmp_scripts/README.md.
set -u

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42

PROJECT="ltp_gnn_gru_pyg"
LOG="logs/preflight.log"
: > "$LOG"
PASS=0; FAIL=0
ok()   { printf "  \033[32mok\033[0m    %s\n" "$*"; PASS=$((PASS+1)); }
bad()  { printf "  \033[31mFAIL\033[0m  %s\n" "$*"; FAIL=$((FAIL+1)); }
info() { printf "  info  %s\n" "$*"; }

echo "=== preflight on $(hostname), $(date '+%F %T') ==="
echo "repo: $PWD"
echo "full output: $LOG"
echo

# ---------------------------------------------------------------- 1. env ---
echo "[1/6] environment"
if python -c "import torch, pddlgym, wandb, torch_geometric, torch_scatter" \
        >>"$LOG" 2>&1; then
    ok "imports (torch, pddlgym, wandb, torch_geometric, torch_scatter)"
else
    bad "imports failed - wrong conda env? (conda activate <di_ltp_1|ltp_3>)"
    tail -5 "$LOG" | sed 's/^/        /'
fi

CORES=$(python -c "import os; print(len(os.sched_getaffinity(0)))" 2>/dev/null || echo 0)
if [ "$CORES" -ge 8 ]; then
    ok "$CORES cores visible"
    # Print the DERIVED budget, not just the count. The campaign was planned
    # for 32 and this node has been seen with 2, 16 and 32; a silently
    # oversubscribed node just makes every run slower with nobody watching.
    . tmp_scripts/budget.sh
    compute_budget "$CORES"
    info "budget: $SLOTS trainings x (1 + $DL_WORKERS dataloader) = \
$((SLOTS * (1 + DL_WORKERS))) cores, eval lane $EVAL_BUSY workers \
($EVAL_IDLE once training is done)"
    [ "$CORES" -lt 28 ] && info "fewer than 28 cores: dataloaders halved to \
keep 4 training slots (see tmp_scripts/budget.sh)"
else
    bad "$CORES cores visible, too few to run the campaign"
fi

if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    ok "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
else
    bad "no CUDA device visible"
fi

# The campaign writes checkpoints, caches and sidecars for two days. Running
# out mid-way is the classic unattended failure, and on these allocations a
# spent quota shows as Avail 0 with low Used.
AVAIL_G=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$AVAIL_G" ] && [ "$AVAIL_G" -ge 50 ]; then ok "${AVAIL_G}G free disk"
elif [ -n "$AVAIL_G" ]; then bad "${AVAIL_G}G free disk - the campaign needs tens of GB"
else info "could not read free disk"; fi

# -------------------------------------------------------------- 2. wandb ---
echo
echo "[2/6] wandb round trip (this is the one you asked about)"
WB_OUT=$(python - "$PROJECT" <<'PY' 2>>"$LOG"
import sys, wandb
project = sys.argv[1]
try:
    run = wandb.init(project=project, name="preflight", job_type="preflight",
                     tags=("preflight",), reinit=True, settings=wandb.Settings(silent=True))
except Exception as exc:
    print("INIT_FAILED", type(exc).__name__, exc); sys.exit(1)
path = f"{run.entity}/{run.project}/{run.id}"
url = run.url
wandb.log({"preflight/probe": 1.0})
wandb.summary["preflight/probe"] = 1.0
wandb.finish()

# Read it BACK. A key that authenticates but cannot reach the server, or a
# project the account cannot write to, both pass every local check and lose
# the whole campaign.
try:
    api = wandb.Api()
    fetched = api.run(path)
    got = fetched.summary.get("preflight/probe")
except Exception as exc:
    print("READBACK_FAILED", type(exc).__name__, exc); sys.exit(1)
if got != 1.0:
    print("READBACK_MISMATCH", got); sys.exit(1)
print("OK", run.entity, url)
PY
)
WB_RC=$?
if [ "$WB_RC" -eq 0 ] && [ "${WB_OUT%% *}" = "OK" ]; then
    WB_ENTITY=$(echo "$WB_OUT" | awk '{print $2}')
    ok "wrote to wandb and read it back (entity: $WB_ENTITY)"
    info "project: https://wandb.ai/${WB_ENTITY}/${PROJECT}"
else
    bad "wandb round trip failed: $WB_OUT"
    info "fix with: wandb login    (or set WANDB_API_KEY)"
    WB_ENTITY=""
fi

# ------------------------------------------------------- 3. end-to-end -----
echo
echo "[3/6] end-to-end run (train + evaluate; a few minutes)"
# Marker file for step 4. Comparing two files on the SAME filesystem is
# immune to clock skew between this host and the NFS server, which a
# "newer than N seconds ago" test is not - a freshly written checkpoint can
# carry a server mtime that looks older than the client's own clock.
rm -f .preflight_marker && touch .preflight_marker
T0=$SECONDS
GABAR_WANDB_GROUP=preflight GABAR_BATCH_EVAL=1 GABAR_FEATURIZE_WORKERS=4 \
  python main.py --config tmp_scripts/preflight.yaml --device cuda:0 \
      --wandb True >>"$LOG" 2>&1
RUN_RC=$?
ELAPSED=$((SECONDS - T0))
if [ "$RUN_RC" -eq 0 ]; then
    ok "train_test completed in ${ELAPSED}s"
else
    bad "train_test exited $RUN_RC after ${ELAPSED}s"
    echo "        last lines:"; tail -15 "$LOG" | sed 's/^/        /'
fi
grep -aq "No models found" "$LOG" && bad "evaluation found no checkpoint to test"

# --------------------------------------------------------- 4. artifacts ----
echo
echo "[4/6] artifacts on disk"
# Ask the process, not the filesystem: traineval.py prints this line as it
# saves. Filesystem-independent, so no clock-skew or mtime-granularity
# failure mode, and it names the exact path.
CKPT=$(grep -a "Saved model checkpoint" "$LOG" | tail -1 | sed 's/.*checkpoint \([^,]*\),.*/\1/')
if [ -n "$CKPT" ]; then
    ok "checkpoint written: $CKPT"
else
    # Fall back to the filesystem before calling it a failure - the log line
    # could change, and the artifact is what actually matters.
    CKPT=$(find models -name '*.pt' -newer .preflight_marker 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        ok "checkpoint written: $CKPT (found on disk; no save line in the log)"
    else
        bad "no checkpoint written under models/ during the run"
        info "checkpoints save every 10 epochs (save_iter in traineval.py);"
        info "a config with fewer epochs than that writes none"
    fi
fi
rm -f .preflight_marker

RJSON=$(ls -t cache/results/preflight/results_*.json 2>/dev/null | head -1)
if [ -n "$RJSON" ]; then ok "results JSON written: $RJSON"
else bad "no results JSON under cache/results/preflight/"; fi

# ----------------------------------------------------- 5. results online ---
echo
echo "[5/6] those results visible on wandb"
if [ -n "$WB_ENTITY" ]; then
    RB=$(python - "$WB_ENTITY" "$PROJECT" <<'PY' 2>>"$LOG"
import sys, wandb
entity, project = sys.argv[1], sys.argv[2]
api = wandb.Api()
runs = [r for r in api.runs(f"{entity}/{project}", order="-created_at")
        if "preflight" in (r.group or "")][:1]
if not runs:
    print("NO_RUN"); sys.exit(1)
r = runs[0]
# The keys log_model_metrics writes into summary: "<domain>_<metric>/<planner>/..."
keys = [k for k in r.summary.keys() if "success_rate_monitor" in k]
if not keys:
    print("NO_METRICS", r.name, r.url); sys.exit(1)
print("OK", r.url, len(keys), keys[0], r.summary[keys[0]])
PY
)
    if [ "${RB%% *}" = "OK" ]; then
        ok "coverage metrics are on wandb ($(echo "$RB" | awk '{print $3}') keys)"
        info "run: $(echo "$RB" | awk '{print $2}')"
        info "e.g. $(echo "$RB" | awk '{print $4" = "$5}')"
    elif [ "${RB%% *}" = "NO_METRICS" ]; then
        bad "the run reached wandb but carries no coverage metrics"
        info "training loss will still be visible; test numbers will not"
    else
        bad "could not read the run back from wandb: $RB"
    fi
else
    bad "skipped (step 2 failed)"
fi

# ------------------------------------------------------- 6. aggregation ----
echo
echo "[6/6] aggregation"
if python tools/analyze_results.py >>"$LOG" 2>&1; then
    ok "tools/analyze_results.py parses the results tree"
else
    bad "tools/analyze_results.py failed"
fi

# ------------------------------------------------------------ verdict ------
echo
echo "=== $PASS passed, $FAIL failed ==="
if [ "$FAIL" -eq 0 ]; then
    # Record WHICH interpreter passed, not just that something did. The
    # campaign compares against this and refuses a launch from a shell whose
    # python is different - the failure mode that cost a day was preflight
    # passing in a conda shell and the campaign being launched later from a
    # plain one, where every run died on `import torch`.
    {
        echo "python=$(command -v python)"
        echo "date=$(date '+%F %T')"
        echo "host=$(hostname)"
        echo "cores=$CORES"
    } > .preflight_ok
    echo "wrote .preflight_ok ($(command -v python))"
    echo
    echo "This node is ready. Launch and walk away, FROM THIS SHELL:"
    echo "    ./tmp_scripts/run_campaign.sh <m|d>"
    exit 0
fi
echo "Do NOT launch the campaign until these are fixed. Detail: $LOG"
exit 1

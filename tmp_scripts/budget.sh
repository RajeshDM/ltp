# budget.sh — core arithmetic, shared by preflight.sh and run_campaign.sh.
# Sourced, not executed. THROWAWAY, see tmp_scripts/README.md.
#
# The campaign was planned for 32 cores; dgxh-2 has been seen with 2, 16 and
# 32 depending on the allocation (PERFORMANCE.md says so), so the concurrency
# has to be derived rather than assumed. Getting this wrong is silent: too
# many trainings and every one of them slows down, with nobody watching.
#
# Cost model, measured: one training run needs 1 core for the main process
# plus --num-workers dataloader cores; one evaluation scales to ~16 workers
# and plateaus there.

compute_budget() {          # cores -> SLOTS DL_WORKERS EVAL_BUSY EVAL_IDLE
    local cores="$1"
    if [ "$cores" -ge 28 ]; then
        # Roomy: 4 trainings x 5 cores = 20, leaving 12 for an eval lane.
        SLOTS=4; DL_WORKERS=4
    elif [ "$cores" -ge 14 ]; then
        # Tight: keep 4 trainings (4 is the GPU power ceiling either way) but
        # halve the dataloaders, 4 x 3 = 12, leaving 4 for the eval lane.
        # Dataloader workers idle during the forward/backward pass, so this
        # costs far less than dropping a whole training slot would.
        SLOTS=4; DL_WORKERS=2
    elif [ "$cores" -ge 8 ]; then
        SLOTS=3; DL_WORKERS=1
    else
        # Below 8 the campaign should not be running at all (preflight fails);
        # this branch only keeps the arithmetic from going negative.
        SLOTS=1; DL_WORKERS=0
    fi
    CORES_PER_TRAINING=$((1 + DL_WORKERS))
    EVAL_BUSY=$(( cores - SLOTS * CORES_PER_TRAINING ))
    [ "$EVAL_BUSY" -lt 1 ] && EVAL_BUSY=1
    # Once training is done the eval lane gets the machine, capped at the
    # measured plateau.
    EVAL_IDLE=$(( cores < 16 ? cores : 16 ))
}

# EVAL_BUSY above assumes every training slot is occupied. A phase that runs
# FEWER trainings than SLOTS should hand the difference to the eval lane -
# node D runs 2 seed replicates on a 16-core box, so the lane is 10 workers,
# not the 4 that SLOTS=4 implies. Evaluation is the CPU-bound half of the
# work; leaving cores idle there is the expensive mistake.
eval_workers_for() {        # concurrent-trainings -> echoes worker count
    local running="$1" free
    free=$(( CORES - running * CORES_PER_TRAINING ))
    [ "$free" -lt 1 ] && free=1
    [ "$free" -gt "$EVAL_IDLE" ] && free="$EVAL_IDLE"
    echo "$free"
}

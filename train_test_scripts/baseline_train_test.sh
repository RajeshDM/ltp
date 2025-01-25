#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <domain> <mode>"
    echo "Available domains: blocks, gripper, miconic, visitall, grid, logistics, spanner"
    echo "Modes: train, test"
    exit 1
fi

SIMPLE_DOMAIN=$1
MODE=$2

if [[ "$MODE" != "train" && "$MODE" != "test" ]]; then
    echo "Error: Mode must be either 'train' or 'test'"
    exit 1
fi

# Domain configurations
declare -A domain_configs=(
    ["blocks"]="name=manyblocks_ipcc_big;train=200;test=200"
    ["gripper"]="name=gripper_ipcc;train=147;test=173"
    ["miconic"]="name=miconic_ipcc;train=228;test=119"
    ["visitall"]="name=visitall_ipcc;train=125;test=50"
    ["grid"]="name=grid_ipcc;train=192;test=48"
    ["logistics"]="name=logistics_ipcc;train=192;test=96"
    ["spanner"]="name=spanner_ipcc;train=234;test=96"
)

CONFIG=${domain_configs[$SIMPLE_DOMAIN]}
if [[ -z "$CONFIG" ]]; then
    echo "Error: Invalid domain '$SIMPLE_DOMAIN'"
    echo "Available domains: blocks, gripper, miconic, visitall, grid, logistics, spanner"
    exit 1
fi

# Parse domain configuration into separate variables
FULL_DOMAIN_NAME=$(echo "$CONFIG" | grep -o 'name=[^;]*' | cut -d'=' -f2)
NUM_TRAIN=$(echo "$CONFIG" | grep -o 'train=[^;]*' | cut -d'=' -f2)
NUM_TEST=$(echo "$CONFIG" | grep -o 'test=[^;]*' | cut -d'=' -f2)

workspace="$(pwd)/../../pddlgym/pddlgym/pddl/${FULL_DOMAIN_NAME}"


loss_functions=("selfsupervised_optimal" "selfsupervised_suboptimal")
aggregations=("max" "add")
#The actual values of the patience don't matter too much - it's just to ensure we run 2 models of each config (and maybe in some cases the actual value matters) 
patience_values=(100)
max_epochs=500

run_training() {
    local domain=$1
    local workspace=$2
    local loss=$3
    local agg=$4
    local patience=$5

    echo "Running training: Domain=${domain}, Loss=${loss}, Aggregation=${agg}, Patience=${patience}"
    python -m ploi.baselines.exp_3.train \
        --domain "${domain}" \
        --train "${workspace}_small" \
        --validation "${workspace}_val" \
        --loss "${loss}" \
        --aggregation "${agg}" \
        --patience "${patience}" \
	--max_epochs "${max_epochs}"
}

run_testing() {
    local domain=$1
    local num_train=$2
    local num_test=$3

    echo "Running testing: Domain=${domain}" 
    python main.py \
        --domain "${domain}" \
        --all-problems \
        --lr 0.0005 \
        --n-heads 4 \
        --attention-dropout 0.1 \
        --dropout 0 \
        --weight-decay 0.00 \
        --wandb False \
        --mode test \
        --starting-test-number 0 \
        --num-test-problems ${num_test} \
        --num-train-problems ${num_train} \
        --epochs 700 \
        --gnn-rounds 9 \
        --batch-size 16 \
        --max-plan-length 200 \
        --problems-per-division 10 \
        --run-learned-model False \
        --run-non-optimal True \
        --use-global-node True \
        --exp-baseline-3 True
}

# Main execution loop
for loss in "${loss_functions[@]}"; do
    for agg in "${aggregations[@]}"; do
        for patience in "${patience_values[@]}"; do
            if [ "$MODE" = "train" ]; then
                run_training "$FULL_DOMAIN_NAME" "$workspace" "$loss" "$agg" "$patience"
            fi
        done
    done
done
run_testing "$FULL_DOMAIN_NAME" "$NUM_TRAIN" "$NUM_TEST" 

#!/bin/bash

decay=0.000
lr=0.0005
other_drop=0
#expid="blocks"
domain="gripper_ipcc"
gnn_round=9
epochs=600
num_test_problems=173
num_train_problems=147
mode="test"
wandb="False"
timestamp=$(date +"%Y_%m_%d_%H_%M")
max_plan_length=200

run_model() {
    local epoch_num=$1  # First argument to function will be epoch number

    echo "python ../main.py \
        --domain ${domain} \
        --all-problems \
        --lr ${lr} \
        --n-heads ${n_head} \
        --attention-dropout ${attn_drop} \
        --dropout ${other_drop} \
        --weight-decay ${decay} \
        --wandb ${wandb} \
        --mode ${mode} \
        --gnn-rounds ${gnn_round} \
        --epochs ${epochs} \
        --num-test-problems ${num_test_problems} \
        --num-train-problems ${num_train_problems} \
        --epoch-number ${epoch_num}
        --max_plan_length=${max_plan_length}"

    python main.py \
        --domain ${domain} \
        --all-problems \
        --lr ${lr} \
        --n-heads ${n_head} \
        --attention-dropout ${attn_drop} \
        --dropout ${other_drop} \
        --weight-decay ${decay} \
        --wandb ${wandb} \
        --mode ${mode} \
        --gnn-rounds ${gnn_round} \
        --epochs ${epochs} \
        --num-test-problems ${num_test_problems} \
        --num-train-problems ${num_train_problems} \
        --epoch-number ${epoch_num}
        --max-plan-length=${max_plan_length}
}

run_experiments() {
    local attn_drop=$2
    local epochs=($3)

    echo "Running: n_head=${n_head}, attention_dropout=${attn_drop}, epochs=${epochs[@]}"
    attention_dropput=$attn_drop
    for epoch in ${epochs[@]}; do
        run_model $epoch
    done
    echo
}

# n_head=1 experiments
n_head=1
#run_experiments 1 0.1 "310 540 580"
#run_experiments 1 0.4 "180 540"

# n_head=2 experiments
n_head=2
#run_experiments 2 0.2 "270"
#run_experiments 2 0.4 "530 570 590"

# n_head=4 experiments
n_head=4
run_experiments 4 0.1 "400"
run_experiments 4 0.2 "260 520"
run_experiments 4 0.4 "520 550"

# n_head=8 experiments
n_head=8
run_experiments 8 0.1 "580"
run_experiments 8 0.2 "160 180 430 490 550"
run_experiments 8 0.3 "220"
run_experiments 8 0.4 "250"

#!/bin/bash

decays=(0.000)
lrs=(0.0005)
other_drops=(0)
#expid="blocks"
domain="gripper_ipcc"
gnn_rounds=(9)
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
        --n-heads ${head} \
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

    python ../main.py \
        --domain ${domain} \
        --all-problems \
        --lr ${lr} \
        --n-heads ${head} \
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
        --max_plan_length=${max_plan_length}
}


n_head=1

attention_dropput=0.1
good_epochs=(310 540 580)

for i in ${good_epochs[@]};
do
	run_model $i
done

attention_dropput=0.4
good_epochs=(180 540)

for i in ${good_epochs[@]};
do
	run_model $i
done


n_head=2

attention_dropput=0.2
good_epochs=(270)

for i in ${good_epochs[@]};
do
	run_model $i
done

attention_dropput=0.4
good_epochs=(530 570 590)

for i in ${good_epochs[@]};
do
	run_model $i
done

n_head=4
attention_dropput=0.1
good_epochs=(400)

for i in ${good_epochs[@]};
do
	run_model $i
done


attention_dropput=0.2
good_epochs=(260 520 )

for i in ${good_epochs[@]};
do
	run_model $i
done


attention_dropput=0.4
good_epochs=(520 550)

for i in ${good_epochs[@]};
do
	run_model $i
done

n_heads=8
attention_dropput=0.1
good_epochs=(580)

for i in ${good_epochs[@]};
do
	run_model $i
done

attention_dropput=0.2
good_epochs=(160 180 430 490 550)

for i in ${good_epochs[@]};
do
	run_model $i
done


attention_dropput=0.3
good_epochs=(220)

for i in ${good_epochs[@]};
do
	run_model $i
done

attention_dropput=0.4
good_epochs=(250)

for i in ${good_epochs[@]};
do
	run_model $i
done

#!/bin/bash

heads=(4 8)
lrs=(0.0005)
decays=(0.000)
attn_drops=(0.1 0.2 0.3 0.4)
other_drops=(0)
#expid="blocks"
domain="gripper_ipcc"
gnn_rounds=(9)
epochs=600
num_test_problems=128
num_train_problems=147
mode="train_test"
#mode="test"
wandb="True"
timestamp=$(date +"%Y_%m_%d_%H_%M")

for head in ${heads[@]};
do
   for decay in ${decays[@]}; 
   do
      for attn_drop in ${attn_drops[@]};
      do
         for other_drop in ${other_drops[@]}; 
	 do
             for lr in ${lrs[@]}; 
	     do
		 for gnn_round in ${gnn_rounds[@]};
		 do
			 #echo "python main.py --all-problems --lr $lr --n-heads $head --attention-dropout $attn_drop --dropout $other_drop --weight_decay $decay" |& tee "cache/results/${expid}/lr_${lr}_n_heads_${head}_attn_drop_${attn_drop}_drop_${other_drop}_decay_${decay}.txt" 
			 echo "python main.py --domain ${domain} --all-problems --lr $lr --n-heads $head --attention-dropout $attn_drop --dropout $other_drop --weight-decay $decay --wandb ${wandb} --mode ${mode} --gnn-rounds $gnn_round --epochs $epochs --num-test-problems $num_test_problems --num-train-problems ${num_train_problems}" 
			 python main.py --domain ${domain} --all-problems --lr $lr --n-heads $head --attention-dropout $attn_drop --dropout $other_drop --weight-decay $decay --wandb ${wandb} --mode ${mode} --gnn-rounds $gnn_round --epochs $epochs --num-test-problems $num_test_problems --num-train-problems ${num_train_problems} |& tee -a "cache/results/${expid}/lr_${lr}_n_heads_${head}_attn_drop_${attn_drop}_drop_${other_drop}_decay_${decay}.txt" 
		done
             done
	 done
      done
   done
done

mode="test"
wandb="False"
for head in ${heads[@]};
do
   for decay in ${decays[@]}; 
   do
      for attn_drop in ${attn_drops[@]};
      do
         for other_drop in ${other_drops[@]}; 
	 do
             for lr in ${lrs[@]}; 
	     do
		 for gnn_round in ${gnn_rounds[@]};
		 do
			 echo "python main.py --domain ${domain} --all-problems --lr $lr --n-heads $head --attention-dropout $attn_drop --dropout $other_drop --weight-decay $decay --wandb ${wandb} --mode ${mode} --gnn-rounds $gnn_round --epochs $epochs --num-test-problems $num_test_problems --num-train-problems ${num_train_problems}" 
			 python main.py --domain ${domain} --all-problems --lr $lr --n-heads $head --attention-dropout $attn_drop --dropout $other_drop --weight-decay $decay --wandb ${wandb} --mode ${mode} --gnn-rounds $gnn_round --epochs $epochs --num-test-problems $num_test_problems --num-train-problems ${num_train_problems} 
		done
             done
	 done
      done
   done
done

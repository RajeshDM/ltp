#!/bin/bash

heads=(4 8)
lrs=(0.0003 0.0005)
decays=(0.00015)
attn_drops=(0.1 0.4)
other_drops=(0)
expid="blocks"
gnn_rounds=(7)
epochs=650
num_test_problems=100
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
			 echo "python main.py --all-problems --lr $lr --n-heads $head --attention-dropout $attn_drop --dropout $other_drop --weight_decay $decay" |& tee "cache/results/${expid}/lr_${lr}_n_heads_${head}_attn_drop_${attn_drop}_drop_${other_drop}_decay_${decay}.txt" 
			 python main.py --all-problems --lr $lr --n-heads $head --attention-dropout $attn_drop --dropout $other_drop --weight-decay $decay --wandb "True" --gnn-rounds $gnn_round --epochs $epochs --num-test-problems $num_test_problems |& tee -a "cache/results/${expid}/lr_${lr}_n_heads_${head}_attn_drop_${attn_drop}_drop_${other_drop}_decay_${decay}.txt" 
		done
             done
	 done
      done
   done
done

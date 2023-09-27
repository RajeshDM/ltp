import ploi.server_information as server_information
#File with all the constants that will be varied during experimentation

'''
orig_v1_r7 - original version with 7 rounds of GNN (with 98 acc on 500 dataset)
orig_v2 - added one feature for object (orig_v2_r7 - same with 7 ronuds of GNN)
pred_v1 - added unary predicates as nodes
pred_v2 - added binary predicates as nodes
pred_v3 - added all predicates as nodes
pred_v4 - add all predicate instances as nodes
prev_v4_1 - without the explicit goal predicate in nodes
pred_v5 - moved all of the action edge computation into a function, removed
         predicate from edge information, added 0 sized predicate nodes, updated action
         object edge removal process, udpating beam search score computation
pred_v5_1 - removed extra features from action object scores
pred_v6 - new object score computation using action edge vector instead of object score vector
pred_v6_1 - updated aggregation to symmetric normalization - (original object score cmoputation)
            updated object node aggretation from mean to divide by sqrt for overall
            symmetric normalization
pred_v6_1_1 - updated symmetric + removing goal info from predicates
pred_v6_1_2 - updated symmetric + adding goal as a single value
pred_v6_1_3 - updated symmetric + added in predicate information for 0 arity predicates
pred_v6_1_4 - no symmetric + added in predicate (v5_1 + zero predicate)
pred_v6_1_5 - no symmetric + added in predicate + removing goal predicates but keeping one signal for goal
# pred_v6_1_6 - no symmetric + added in predicate + removing goal predicates (all traces of goal) (useless)
pred_v6_1_7 - no symmetric + added in predicate + same pred for goal and normal ones (all traces of goal here too - even same predicate instance)
pred_v6_1_8 - symmetric + added in predicate + same pred for goal and normal ones (all traces of goal here too - even same predicate instance)
pred_v6_1_9 - symmetric + no added predicate info (removed from list of nodes as well)
pred_v6_2 - updated object score computation and edge,node update function (with shuffle of)
pred_v6_3 - v6_2 with shuffle on for dataloaders
pred_v6_4 - purely for testing
pred_v6_5 - for running with pyg
pred_v6_6 - final version of this for testing
pred_v7_1 - with Attention on node computation(weights on edge aggregation for computing node values)
pred_v7_2 - Attention on global computation only
pred_v7_3 - Attention on nodes and global computation
pred_v7_4 - AdamW with weight decay of 0.01 (only node attention)
pred_v7_5 - AdamW with weight decay of 0.01 (only global attention)
pred_v8_1 - No attention - but with AdamW
pred_v8_2 - Node attention (with updated edge representation) (With adam)
pred_v8_3 - Node attention (with updated node and edge representation) (With adam)
pred_v8_4 - v6_7 without node and edge decoder params
pred_v8_5 - 8_4 with separate encoders and separate update functions 
pred_v8_6 - original code with separate encoders and separate update functions 
pred_v8_7 - original code + attention
pred_v8_8 - original code with separate encoders and separate update functions + attention
pred_v8_9 - original code with lonely nodes instantiated correctly
pred_v8_10 - original code with lonely nodes instantiated correctly + attention
'''
major_concepts = ['orig', 'pred']
versions = ['v1', 'v2', 'v3', 'v4', 'v4_1',
            'v5', 'v5_1', 'v5_2', 'v6','v6_1',
            'v6_1_1','v6_1_2','v6_1_3','v6_1_4','v6_1_5',
            'v6_1_6','v6_1_7','v6_1_8','v6_1_9','v6_2',
            'v6_3','v6_4','v6_5','v6_6','v6_7',
            'v7_1','v7_2','v7_3','v7_4','v7_5',
            'v8_1','v8_2','v8_3','v8_4','v8_5',
            'v8_6','v8_7','v8_8','v8_9','v8_10']
rounds = ['r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13']

#constants for guidance
num_epochs = 750
concept_loc = 2
version_loc = 38
#epoch_number = 740

use_gpu = server_information.use_gpu
server = server_information.server
continue_training = False

#constants for main

starting_epoch = 0
problem_number = 0
number_problems_each_division = 10
max_plan_length_permitted = 90
opt_planner = False
#non_opt_planner = False
non_opt_planner = True
external_monitor_bool = False
heuristic_planner = False
plot_aggregates = False
max_file_open = 50
debug_level = -1
seed = 10
batch_size = 16
pyg = True
learning_rate = 5* 1e-4
#cheating_input = True
cheating_input = False
representation_size = 64
gnn_rounds = 7
gru_layers = 3
n_heads = 1

'''
Debug level information:

Level 3 : will only have model name and it's results
Level 2 : will have general hyperparameter details and results (current output)
Level 1 : will have full plan outputs
Level 0 : Full everything output - action param list etc.
'''
max_debug_level = 3

# model_outfile = self._save_model_prefix+"__dagger_{}Test_2.pt".format(train_env_name)
# model_outfile = "/tmp/model80.pt"
# model_outfile = "/tmp/ManyBlocks_ipcc_model60.pt"
# model_outfile = "/tmp/ManyBlocks_ipcc_big_model130.pt"
# model_outfile = "/tmp/ManyBlocks_ipcc_big_seed0_model10.pt"
# model_outfile = "/tmp/ManyBlocks_ipcc_big_seed0_model80_95_2_200_data.pt"
# model_outfile = "/tmp/Discrete_tamp_seed0_model70.pt"
# model_outfile = "/tmp/Discrete_tamp_seed0_model50_near_0_loss.pt"
# model_outfile = "/tmp/Discrete_tamp_seed0_model70_fully_working.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model150_rounds5_pred_v1_r5.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model160_rounds7_pred_v1_r7.pt"
# model_outfile = "/tmp/Manygripper_seed0_model80_rounds5_orig_v1_r5.pt"
# model_outfile = "/tmp/Manygripper_seed0_model110_rounds7_pred_v1_r7_very_high_acc.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model110_rounds7_pred_v1_r7_very_high_acc"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model160_rounds7_pred_v1_r7_97_5_200.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model150_rounds7_pred_v1_r7.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model210_rounds9_pred_v4_r9.pt" # currently the main one is the beset
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model260_rounds10_pred_v4_r10.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model270_rounds11_pred_v4_r11.pt" #270 is best of 11 currently
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model340_rounds9_pred_v5_r9.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model300_rounds13_pred_v5_r13.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model320_rounds9_pred_v5_r9.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model290_rounds11_pred_v5_r11.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model290_rounds13_pred_v5_r13.pt"
# model_outfile = "/tmp/N_puzzle_ipcc_seed0_model260_rounds9_pred_v5_r9.pt"
# model_outfile = "/tmp/N_puzzle_ipcc_seed0_model160_rounds5_pred_v5_r5.pt"
# model_outfile = "/tmp/Ferry_ipcc_seed0_model260_rounds9_pred_v5_r9.pt"
# model_outfile = "/tmp/Ferry_ipcc_seed0_model160_rounds5_pred_v5_r5.pt"
# model_outfile = "/tmp/Manygripper_seed0_model210_rounds9_pred_v5_r9.pt"
# model_outfile = "/tmp/Manygripper_seed0_model290_rounds5_pred_v5_r5.pt"
# model_outfile = "/tmp/Manygripper_seed0_model190_rounds7_pred_v5_r7.pt"
# model_outfile = "/tmp/Gripper_ipcc_seed0_model300_rounds5_pred_v5_r5.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model320_rounds5_pred_v5_r5.pt"
# model_outfile = "/tmp/Sokoban_ipcc_seed0_model10_rounds5_pred_v5_r5.pt"
# model_outfile = "/tmp/Manygripper_seed0_model60_rounds5_pred_v5_2_r5.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model270_rounds9_pred_v5_3_r9.pt"
# model_outfile = self._save_model_prefix+"__dagger_{}Test_2.pt".format(train_env_name)
# model_outfile = "/tmp/model80.pt"
# model_outfile = "/tmp/ManyBlocks_ipcc_model60.pt"
# model_outfile = "/tmp/ManyBlocks_ipcc_big_model130.pt"
# model_outfile = "/tmp/ManyBlocks_ipcc_big_seed0_model10.pt"
# model_outfile = "/tmp/ManyBlocks_ipcc_big_seed0_model80_95_2_200_data.pt"
# model_outfile = "/tmp/Discrete_tamp_seed0_model70.pt"
# model_outfile = "/tmp/Discrete_tamp_seed0_model50_near_0_loss.pt"
# model_outfile = "/tmp/Discrete_tamp_seed0_model70_fully_working.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model150_rounds5_pred_v1_r5.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model160_rounds7_pred_v1_r7.pt"
# model_outfile = "/tmp/Manygripper_seed0_model80_rounds5_orig_v1_r5.pt"
# model_outfile = "/tmp/Manygripper_seed0_model110_rounds7_pred_v1_r7_very_high_acc.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model110_rounds7_pred_v1_r7_very_high_acc"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model160_rounds7_pred_v1_r7_97_5_200.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model150_rounds7_pred_v1_r7.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model210_rounds9_pred_v4_r9.pt" # currently the main one is the beset
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model260_rounds10_pred_v4_r10.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model270_rounds11_pred_v4_r11.pt" #270 is best of 11 currently
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model300_rounds9_pred_v5_r9.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model320_rounds11_pred_v5_r11.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model260_rounds11_pred_v5_1_r11.pt"
# model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model260_rounds7_pred_v5_1_r7.pt"
#model_outfile = "/tmp/Manyblocks_ipcc_big_seed0_model290_rounds9_pred_v5_1_r9.pt"
'''
old_pred_v5 - removed double predicate informaion from nodes and edges - now half in nodes and half in edges
#old_pred_v5_1 - Added back predicate info in edges between nodes and predicate
old_pred_v5_2 - Removed one bug from action edge computation
old_pred_v5_3 - Removed extra edges from predicate to nodes -
may not be required as edges are added
old_pred_v6 - Removed extra elements in object action scores
old_pred_v6_1 - added softmax for scores being used in loss
old_pred_v6_2 - Removed softmax - did not work. Updated score calculation in
beam search - need to update it further for larger number of action params
old_pred_v6_3  - Without float for both action scores as well as action_object scores but
with requires grad for both
'''

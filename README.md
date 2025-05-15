# Modern scene-graph task planners

```sh
pip install numpy==1.26.0
pip install protobuf==3.20.0
#pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
#pip install torch-geometric==2.3.1
#pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

#pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
#pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cu118.html

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install torch-geometric==2.6.1
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install tensorboard
pip install torchviz==0.0.2
pip install pytorch-lightning==2.0.1
pip install wandb
pip install icecream
pip install pymimir==0.9.71
pip install pyperplan==2.1
pip install pandas
#for running baselines
pip install termcolor
pip install tarski
pip install clingo
```

### Additional Requirements

For use with pddlgym, we require our fork of [pddlgym](https://anonymous.4open.science/r/pddlgym-0F03/), which houses our custom domains and problems.

Download and build the plan validation tool available at https://github.com/KCL-Planning/VAL, then make a symlink called validate on your path that points to the build/Validate binary, e.g. `ln -s <path to VAL>/build/Validate /usr/local/bin/validate`. If done successfully, running validate on your command line should give an output that starts with the line: `VAL: The PDDL+ plan validation tool`.

## Running the Code

To run the planner, use `main.py` with the appropriate command-line arguments. The framework supports various learning-based planners and domains.

### Command-line Arguments

```
usage: main.py [-h] [--seed SEED] [--method {ltp,ltp_no_cd,ltp_no_ag,ltp_no_ag_no_cd,ltp_val,scenegraph,hierarchical,ploi}]
               [--mode {train,test,train_test,visualize}] [--domain DOMAIN]
               [--all-problems] [--lr LR] [--n-heads N_HEADS]
               [--attention-dropout ATTENTION_DROPOUT] [--dropout DROPOUT]
               [--weight-decay WEIGHT_DECAY] [--wandb {True,False}]
               [--starting-test-number STARTING_TEST_NUMBER]
               [--num-train-problems NUM_TRAIN_PROBLEMS]
               [--num-test-problems NUM_TEST_PROBLEMS]
               [--epochs EPOCHS] [--gnn-rounds GNN_ROUNDS]
               [--batch-size BATCH_SIZE] [--max-plan-length MAX_PLAN_LENGTH]
               [--problems-per-division PROBLEMS_PER_DIVISION]
               [--run-learned-model {True,False}]
               [--run-non-optimal {True,False}] [--run-optimal {True,False}]
               [--monitor {True,False}] [--use-global-node {True,False}]
               [--ablation {no_ag_no_cd,main_val,main}]
               [--search-strat {dfs,bfs}] [--test-with-seed {true,false}]
               [--expid EXPID] [--logdir LOGDIR] [--device {cpu,cuda:0}]
               [--criterion {bce}] [--pos-weight POS_WEIGHT]
               [--print-every PRINT_EVERY] [--gamma GAMMA]
               [--force-collect-data]
```

### Key Arguments Explained

- `--method`: The planning method to use (options: ltp, ltp_no_cd, ltp_no_ag, ltp_no_ag_no_cd, ltp_val, scenegraph, hierarchical, ploi)
- `--mode`: Execution mode (options: train, test, train_test, visualize)
- `--domain`: Name of the pddlgym domain to use
- `--all-problems`: Use all available problems in the domain
- `--lr`: Learning rate for training
- `--n-heads`: Number of attention heads in the model
- `--dropout` and `--attention-dropout`: Dropout rates for regularization
- `--weight-decay`: Weight decay parameter for optimizer
- `--wandb`: Whether to use Weights & Biases for logging (True/False)
- `--num-train-problems`: Number of training problems to use
- `--num-test-problems`: Number of test problems to use
- `--epochs`: Number of training epochs
- `--gnn-rounds`: Number of message-passing rounds in the GNN
- `--batch-size`: Batch size for training
- `--max-plan-length`: Maximum allowed plan length
- `--run-learned-model`: Whether to run the learned model at test time (True/False)
- `--run-non-optimal`: Whether to run the non-optimal planner (True/False)
- `--use-global-node`: Whether to use a global node in the graph (True/False)
- `--ablation`: Ablation study type (options: no_ag_no_cd, main_val, main)
- `--search-strat`: Search strategy to use (options: dfs, bfs)

### Example Commands

#### Training and Testing on Blocks Domain

```sh
python main.py --method ltp --domain manyblocks_ipcc_big --all-problems --lr 0.0005 --n-heads 1 --attention-dropout 0 --dropout 0 --weight-decay 0.00 --wandb False --num-train-problems 50 --num-test-problems 5 --epochs 300 --gnn-rounds 9 --batch-size 16 --mode train_test --max-plan-length 500 --problems-per-division 10 --run-learned-model True --run-non-optimal True --use-global-node True
```

This command:
- Uses the LTP (Learning to Plan) method
- Trains on the "manyblocks_ipcc_big" domain
- Uses all available problems with 50 for training and 5 for testing
- Sets a learning rate of 0.0005
- Uses a model with 1 attention head and no dropout
- Disables Weights & Biases logging
- Trains for 300 epochs with batch size 16
- Performs 9 message-passing rounds in the GNN
- Runs in train_test mode (trains then immediately tests)
- Sets maximum plan length to 500
- Uses 10 problems per division
- Runs both the learned model and non-optimal planner
- Uses a global node in the graph

#### Testing a Pre-trained Model

```sh
python main.py --method ltp --domain manyblocks_ipcc_big --all-problems --num-test-problems 10 --mode test --max-plan-length 500 --run-learned-model True --use-global-node True
```

#### Running with Ablations

```sh
python main.py --method ltp_no_ag --domain manyblocks_ipcc_big --all-problems --lr 0.0005 --epochs 300 --mode train --max-plan-length 500 --ablation no_ag_no_cd
```

This runs the planner without action grounding (no_ag) ablation.
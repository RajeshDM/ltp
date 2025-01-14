# Modern scene-graph task planners

```sh
pip install numpy==1.26.0
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install torch-geometric==2.3.1
pip install torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1 torchviz==0.0.2
pip install wandb
pip install icecream
pip install pymimir
pip install pyperplan==2.1
```

For use with pddlgym, we require our fork of [pddlgym](https://github.com/taskography/pddlgym), which houses our custom domains and problems.

Another essential requirement is `torch_geometric`, which is best installed by following [these instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

Download and build the plan validation tool available at https://github.com/KCL-Planning/VAL, then make a symlink called validate on your path that points to the build/Validate binary, e.g. `ln -s <path to VAL>/build/Validate /usr/local/bin/validate`. If done successfully, running validate on your command line should give an output that starts with the line: `VAL: The PDDL+ plan validation tool`.

## Running PLOI/SCRUB/SEEK on a registered pddlgym environment

To train a planner on an environment already registered with `pddlgym`, simply run `main.py` passing the appropriate commandline arguments.

For example, PLOI may be run on a domain `Taskographyv2tiny10` by executing the following command. It trains on 40 problem instances for 401 epochs, and tests on all validation problem instances.
```
python main.py --domain taskographyv2tiny10 --method ploi --num-train-problems 40 --epochs 401 --mode train  --timeout 30 --expid taskographyv2tiny10_ploi --logdir cache/results --all-problems
```

To run evaluation using a pretrained model a PLOI baseline on the domain, set the `--mode` argument to `test` instead. The code will then pick up the best model from the directory pointed to by `--expid`.

Here's the list of supported commandline arguments across all planners.
```sh
usage: main.py [-h] [--seed SEED] [--method {scenegraph,hierarchical,ploi}]
               [--mode {train,test,visualize}] [--domain DOMAIN]
               [--train-planner-name {fd-lama-first,fd-opt-lmcut}]
               [--eval-planner-name {fd-lama-first,fd-opt-lmcut}]
               [--num-train-problems NUM_TRAIN_PROBLEMS]
               [--num-test-problems NUM_TEST_PROBLEMS]
               [--do-incremental-planning] [--timeout TIMEOUT] [--expid EXPID]
               [--logdir LOGDIR] [--device {cpu,cuda:0}] [--criterion {bce}]
               [--pos-weight POS_WEIGHT] [--epochs EPOCHS] [--lr LR]
               [--load-model] [--print-every PRINT_EVERY] [--gamma GAMMA]
               [--force-collect-data]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Random seed
  --method {scenegraph,hierarchical,ploi}
  --mode {train,test,visualize}
                        Mode to run the script in
  --domain DOMAIN       Name of the pddlgym domain to use.
  --train-planner-name {fd-lama-first,fd-opt-lmcut}
                        Train planner to use
  --eval-planner-name {fd-lama-first,fd-opt-lmcut}
                        Eval planner to use
  --num-train-problems NUM_TRAIN_PROBLEMS
                        Number of train problems
  --num-test-problems NUM_TEST_PROBLEMS
                        Number of test problems
  --do-incremental-planning
                        Whether or not to do incremental planning
  --timeout TIMEOUT     Timeout for test-time planner
  --expid EXPID         Unique exp id to log data to
  --logdir LOGDIR       Directory to store all expt logs in
  --device {cpu,cuda:0}
                        torch.device argument
  --criterion {bce}     Loss function to use
  --pos-weight POS_WEIGHT
                        Weight for the positive class in binary cross-entropy
                        computation
  --epochs EPOCHS       Number of epochs to run training for
  --lr LR               Learning rate
  --load-model          Path to load model from
  --print-every PRINT_EVERY
                        Number of iterations after which to print training
                        progress.
  --gamma GAMMA         Value of importance threshold (gamma) for PLOI.
  --force-collect-data  Force data collection (ignore pre-cached datasets).
```
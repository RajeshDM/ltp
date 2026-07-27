import argparse
import sys
import ploi.constants as constants


def apply_config_defaults(parser):
    """Apply a YAML config file (if --config was passed) as parser defaults.

    Precedence: constants.py < YAML config < explicit CLI flags.  Call after
    ALL add_argument calls and before parse_args.  YAML keys use the argparse
    dest names (underscores, e.g. num_train_problems).  Unknown keys are a
    hard error so a typo cannot silently fall back to a default.
    """
    config_path = None
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
        elif tok.startswith("--config="):
            config_path = tok.split("=", 1)[1]
    if config_path is None:
        return

    import os
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config {config_path} must be a mapping, got {type(config)}")

    # 'base' layers a shared-defaults file under this config (one level only):
    # precedence constants < base < experiment file < CLI flags.  Path is
    # relative to the experiment file's directory.
    base_name = config.pop('base', None)
    if base_name is not None:
        base_path = os.path.join(os.path.dirname(config_path), base_name)
        with open(base_path) as f:
            base = yaml.safe_load(f) or {}
        if not isinstance(base, dict):
            raise ValueError(f"Base {base_path} must be a mapping, got {type(base)}")
        if 'base' in base:
            raise ValueError(f"Base {base_path} may not itself have a 'base' key")
        base.pop('description', None)
        config = {**base, **config}

    # 'description' is documentation-only inside experiment configs
    config.pop('description', None)

    known_dests = {action.dest for action in parser._actions}
    unknown = set(config) - known_dests
    if unknown:
        raise ValueError(
            f"Unknown keys in {config_path}: {sorted(unknown)}. "
            f"Keys must match argparse dest names (underscores).")

    parser.set_defaults(**config)
    print(f"Loaded config: {config_path} ({len(config)} settings)")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_ploi_argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                        type=int,
                        default=constants.SEED,
                        help="Random seed")
    parser.add_argument(
        "--method",
        type=str,
        choices=["scenegraph", "hierarchical", "ploi", "ltp", 
                 "ltp_no_cd", "ltp_no_ag", "ltp_val", "ltp_no_ag_no_cd"],
        default="ltp",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["no_cd", "no_ag", "no_ag_no_cd", "val","main", "main_val"] ,
        default="main",
    )
    parser.add_argument(
        "--test-with-seed",
        type=str2bool,
        choices=[True, False],
        default="false",
    )
    parser.add_argument(
        "--search-strat",
        type=str,
        choices=['greedy', 'dfs', 'mcts'],
        default="greedy",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default="1",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "visualize","debug", "train_test"],
        default=constants.MODE,
        help="Mode to run the script in",
    )

    parser.add_argument(
        "--domain",
        type=str,
        default=constants.DOMAIN,
        help="Name of the pddlgym domain to use.",
    )
    parser.add_argument(
        "--train-planner-name",
        type=str,
        choices=["fd-lama-first", "fd-opt-lmcut"],
        default="fd-lama-first",
        #default="fd-opt-lmcut",
        help="Train planner to use",
    )
    parser.add_argument(
        "--eval-planner-name",
        type=str,
        choices=["fd-lama-first", "fd-opt-lmcut"],
        default="fd-lama-first",
        help="Eval planner to use",
    )
    parser.add_argument(
        "--num-train-problems", 
        type=int, 
        default=constants.NUM_TRAIN_PROBLEMS, 
        help="Number of train problems"
    )
    parser.add_argument(
        "--problems-per-division", 
        type=int, 
        default=constants.NUMBER_PROBLEMS_EACH_DIVISION, 
        help="Problems per division"
    )
    parser.add_argument(
        "--num-test-problems", type=int, default=constants.NUM_TEST_PROBLEMS, help="Number of test problems"
    )
    parser.add_argument(
        "--do-incremental-planning",
        action="store_true",
        help="Whether or not to do incremental planning",
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Timeout for test-time planner"
    )

    parser.add_argument(
        "--expid", type=str, 
        default=constants.EXPID, 
        help="Unique exp id to log data to"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="cache/results/",
        help="Directory to store all expt logs in",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda:0"],
        default="cpu",
        help="torch.device argument",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        choices=["bce"],
        default="bce",
        help="Loss function to use",
    )

    parser.add_argument(
        "--pos-weight",
        type=float,
        default=10.0,
        help="Weight for the positive class in binary cross-entropy computation",
    )
    parser.add_argument(
        "--epochs", type=int, 
        default=constants.NUM_EPOCHS, 
        help="Number of epochs to run training for"
    )

    parser.add_argument(
        "--epoch-number", type=int, 
        default=constants.EPOCH_NUMBER, 
        help="Model epoch number to run"
    )

    parser.add_argument(
        "--load-model", action="store_true", help="Path to load model from"
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Number of iterations after which to print training progress.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Value of importance threshold (gamma) for PLOI.",
    )

    parser.add_argument(
        "--force-collect-data",
        action="store_true",
        help="Force data collection (ignore pre-cached datasets).",
    )

    parser.add_argument("--model_version", 
                        type=int,
                        default=0
    )

    parser.add_argument(
        "--gnn-rounds",
        type=int,
        default=constants.GNN_ROUNDS,
        help="Number of rounds of GNN",
    )
    parser.add_argument(
        "--gru-layers",
        type=int,
        default=constants.GRU_LAYERS,
        help="Number of layers in GRU",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=constants.BATCH_SIZE,
        help="Batch size for training",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=constants.NUM_WORKERS,
        help="num workers for data loaders",
    )

    parser.add_argument(
        "--representation-size",
        type=int,
        default=constants.REPRESENTATION_SIZE,
        help="Representation size for all embeddings",
    )

    parser.add_argument(
        "--n-heads",
        type=int,
        default=constants.N_HEADS,
        help="Number of heads to use for node update attention layer",
    )
    parser.add_argument(
        "--starting-epoch",
        type=int,
        default=constants.STARTING_EPOCH,
        help="Starting epoch for training",
    )

    parser.add_argument(
        "--starting-test-number",
        type=int,
        default=constants.starting_test_number,
        help="Starting problem number for testing", 
    )

    parser.add_argument(
        "--data-augmentation",
        type=str2bool,
        default=constants.DATA_AUGMENTATION,
        help="Whether to use augmented data or not",
    )

    parser.add_argument(
        "--continue-training",
        type=str2bool,
        default=constants.CONTINUE_TRAINING,
        help="Training from a checkpoint or not",
    )

    parser.add_argument(
        "--pyg",
        type=str2bool,
        default=constants.PYG,
        help="Using pytorch geometric or not",
    )

    parser.add_argument(
        "--monitor",
        type=str2bool,
        default=constants.EXTERNAL_MONITOR_BOOL,
        help="Using pytorch geometric or not",
    )

    parser.add_argument(
        "--cheating-input",
        type=str2bool,
        default=constants.CHEATING_INPUT,
        help="Using cheating input or not",
    )
    parser.add_argument(
        "--server",
        type=str2bool,
        default=constants.SERVER,
        help="Using server or not",
    )
    parser.add_argument(
        "--use-gpu",
        type=str2bool,
        default=constants.USE_GPU,
        help="Using GPU or not",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=constants.DROPOUT,
        help="Dropout in GNN",
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=constants.ATTENTION_DROPOUT,
        help="Dropout in attention layer of GNN",
    )

    parser.add_argument(
        "--lr", 
        type=float,
        default=constants.LEARNING_RATE, 
        help="Learning rate")

    parser.add_argument(
        "--weight-decay", 
        type=float,
        default=constants.WEIGHT_DECAY, 
        help="L2 regularization weight decay")


    parser.add_argument(
        "--debug-level", 
        type=int,
        default=constants.DEBUG_LEVEL,
        help="Debug level")

    parser.add_argument(
        "--max-file-open", 
        type=int,
        default=constants.MAX_FILE_OPEN,
        help="Maximum number of files to open at once")

    parser.add_argument(
        "--max-plan-length", 
        type=int,
        default=constants.MAX_PLAN_LENGTH_PERMITTED,
        help="Maximum plan length to stop at")

    parser.add_argument(
        "--concept-loc", 
        type=int,
        default=constants.CONCEPT_LOC,
        help="Experiment concept number (for tracking with model filename)")

    parser.add_argument(
        "--wandb", 
        type=str2bool,
        default=False,
        help="Whether to log on Wandb or not")

    parser.add_argument(
        "--run-learned-model", 
        type=str2bool,
        default=True,
        help="Whether to run the learned model or not")

    parser.add_argument(
        "--run-non-optimal", 
        type=str2bool,
        default=True,
        help="Whether to run the non-optimal planner or not")

    parser.add_argument(
        "--run-optimal", 
        type=str2bool,
        default=False,
        help="Whether to run the optimal planner or not")

    parser.add_argument(
        "--use-global-node", 
        type=str2bool,
        default=True,
        help="Whether to use the global node for the learning method")

    parser.add_argument(
        "--exp-baseline", 
        type=str2bool,
        default=False,
        help="Whether to run the exp baseline model")

    parser.add_argument(
        "--exp-train-model", 
        type=str,
        default=None,
        help="Location of model to continue training from")

    parser.add_argument(
        "--exp-baseline-2", 
        type=str2bool,
        default=False,
        help="Whether to run the exp 2 baseline model")

    parser.add_argument(
        "--exp-baseline-3", 
        type=str2bool,
        default=False,
        help="Whether to run the exp 2 baseline model")

    parser.add_argument(
        "--exp-2-train-model", 
        type=str,
        default=None,
        help="Location of model to continue training from")

    parser.add_argument(
        "--object-options",
        type=int,
        default=constants.OBJECT_OPTIONS,
        help="Number of object options to consider during testing at each step")

    parser.add_argument(
        "--action-options",
        type=int,
        default=constants.ACTION_OPTIONS,
        help="Number of actions to consider")

    parser.add_argument(
        "--num-mlp-layers-gnn",
        type=int,
        default=constants.NUM_MLP_LAYERS_GNN,
        help="Number of layers in MLP of the GNN")

    # Multi-domain harness (CLAUDE.md Phase 0/1, claim C1). When --domains is
    # given it supersedes --domain; --domain alone keeps the published
    # single-domain GABAR behavior (parity gate).
    parser.add_argument(
        "--domains",
        type=str,
        default="",
        help="Comma-separated training domains, each optionally with a "
             "problem count, e.g. 'blocks:100,gripper'. Empty = single-domain "
             "mode via --domain.")

    parser.add_argument(
        "--heldout-domains",
        type=str,
        default="",
        help="Comma-separated domains excluded from training and used only "
             "for zero-shot evaluation (C1).")

    parser.add_argument(
        "--test-domains",
        type=str,
        default="",
        help="Comma-separated domains to test on, each optionally with a "
             "problem count, e.g. 'blocks:200,gripper:173,spanner:96'. "
             "Overrides the default (test on training domains). Domains not "
             "in the training set are treated as zero-shot. If empty, tests "
             "on training domains + held-out domains as before.")

    parser.add_argument(
        "--featurization",
        type=str,
        choices=["per_domain", "union", "structural", "joint_lite", "joint",
                 "joint_chain"],
        default="per_domain",
        help="Feature dictionary mode: 'per_domain' = published GABAR "
             "(Phase 0 parity), 'union' = shared union vocabulary across "
             "training domains (Baseline 0, C1 control), 'structural' = "
             "symbol-free structural classes (Method 0, zero-shot capable), "
             "'joint_lite' = structural + lifted domain layer in every state "
             "graph (GADAR-BIND ablation), 'joint' = joint_lite + occurrence "
             "nodes + binding layer (full GADAR, Method B), 'joint_chain' = "
             "full GADAR + schema chaining + goal-relevance features + "
             "grounded effect edges.")

    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["default", "toy", "sweep", "spot"],
        default="default",
        help="Execution mode: 'toy' = 2 epochs on CPU with tiny data subset, "
             "'sweep' = short runs with AMP and 10%% data for HP tuning, "
             "'spot' = full training with AMP, per-epoch checkpointing, and "
             "auto-resume for interruptible instances, 'default' = unchanged.")

    parser.add_argument(
        "--auto-shutdown",
        action="store_true",
        help="Shut down the instance on exit (success or crash). SPOT mode only.")

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop training after this many val-loss checks with no improvement. "
             "0 = disabled. Set automatically by --run-mode if not specified.")

    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=2,
        help="Checkpoints retained per selection metric (ModelManager). The "
             "default 2 can lose the EARLY epochs that transfer best - "
             "zero-shot peaks well before training loss bottoms out - "
             "especially when an older run's checkpoints squat the list. "
             "Use 6+ for zero-shot runs.")

    parser.add_argument(
        "--max-pred-arity",
        type=int,
        default=0,
        help="Lower bound on the canonical max predicate arity used for "
             "structural/joint featurization (0 = derive from the training "
             "set). Raise it when a held-out test domain has higher-arity "
             "predicates than any training domain.")

    parser.add_argument(
        "--max-action-arity",
        type=int,
        default=0,
        help="Lower bound on the canonical max schema arity for "
             "structural/joint featurization (0 = derive from the training "
             "set). Raise it when a held-out test domain has higher-arity "
             "schemas (e.g. Rovers under a no-rovers training set).")

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="YAML experiment config. Values become argparse defaults, so "
             "explicit CLI flags still override. Keys use argparse dest "
             "names (e.g. num_train_problems, test_domains).")

    parser.add_argument(
        "--num-models-to-test",
        type=int,
        default=1,
        help="How many of the best checkpoints to test per metric "
             "(ModelManager keeps 2; 1 = best only).")

    parser.add_argument(
        "--test-model-metrics",
        type=str,
        default="validation,training,combined",
        help="Comma-separated checkpoint-selection metrics to test "
             "(subset of validation,training,combined,periodic). 'periodic' "
             "needs --checkpoint-every and is ordered EARLIEST epoch first.")

    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="Also snapshot every N epochs regardless of loss, into the "
             "'periodic' metric. Loss-ranked slots always end up holding "
             "late epochs; zero-shot transfer peaks early. Use 50 for "
             "zero-shot runs, then test --test-model-metrics periodic.")

    return parser
import argparse
import ploi.constants as constants

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
        choices=["scenegraph", "hierarchical", "ploi", "ltp"],
        default="ltp",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default="1",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "visualize","debug"],
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
        "--num-test-problems", type=int, default=5, help="Number of test problems"
    )
    parser.add_argument(
        "--do-incremental-planning",
        action="store_true",
        help="Whether or not to do incremental planning",
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout for test-time planner"
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
        "--concept-loc", 
        type=int,
        default=constants.CONCEPT_LOC,
        help="Experiment concept number (for tracking with model filename)")

    parser.add_argument(
        "--wandb", 
        type=str2bool,
        default=False,
        help="Whether to log on Wandb or not")


    return parser

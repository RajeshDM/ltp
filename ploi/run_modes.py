import os
import sys
import atexit
import random
from datetime import datetime


class TeeStream:
    """Duplicates writes to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def fileno(self):
        return self.original.fileno()

    def isatty(self):
        return self.original.isatty()


def build_log_filename(domain, args):
    """Build log filename: domain, timestamp, then hyperparameters.

    Regex patterns in ModelLogParser.extract_hyperparameters() are
    position-independent, so prepending domain + timestamp is safe.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"{domain}_{timestamp}"
            f"_lr_{args.lr}"
            f"_n_heads_{args.n_heads}"
            f"_attn_drop_{args.attention_dropout}"
            f"_drop_{args.dropout}"
            f"_decay_{args.weight_decay}"
            f"_g_node_{args.use_global_node}"
            f"_ablation_{args.ablation}"
            f"_search_strat_{args.search_strat}"
            f"_{args.run_mode}")
    return name + ".txt"


def setup_logging(domain, args):
    """Redirect stdout and stderr to both console and a log file.

    Activates for spot and sweep modes only. Log files go in
    cache/results/{domain}/ to stay compatible with analyse_domain_results.py.
    Returns the log file path, or None if logging is not active.
    """
    if args.run_mode not in ('spot', 'sweep'):
        return None

    log_dir = os.path.join("cache", "results", domain)
    os.makedirs(log_dir, exist_ok=True)

    log_filename = build_log_filename(domain, args)
    log_path = os.path.join(log_dir, log_filename)

    log_file = open(log_path, 'w')

    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    def cleanup():
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.flush()
        log_file.close()

    atexit.register(cleanup)

    print(f"Logging to: {log_path}")
    return log_path


def apply_run_mode(args):
    """Adjust args in-place based on --run-mode. Call after arg parsing."""
    mode = args.run_mode

    if mode == 'toy':
        args.epochs = min(2, args.epochs)
        args.num_workers = 0
        args.batch_size = min(4, args.batch_size)
        args.num_train_problems = min(5, args.num_train_problems)
        args.device = 'cpu'
        args.use_gpu = False
        args.use_amp = False
        args.spot_resume = False
        args.data_fraction = None
        args.toy_max_graphs = 50
        if not args.early_stopping_patience:
            args.early_stopping_patience = 0

    elif mode == 'sweep':
        args.epochs = min(5, args.epochs)
        args.use_amp = True
        args.spot_resume = False
        args.toy_max_graphs = None
        args.data_fraction = 0.1
        if not args.early_stopping_patience:
            args.early_stopping_patience = 2

    elif mode == 'spot':
        args.use_amp = True
        args.spot_resume = True
        args.toy_max_graphs = None
        args.data_fraction = None
        if not args.early_stopping_patience:
            args.early_stopping_patience = 20

    else:
        args.use_amp = False
        args.spot_resume = False
        args.toy_max_graphs = None
        args.data_fraction = None
        if not args.early_stopping_patience:
            args.early_stopping_patience = 0

    return args


def subset_graphs(graphs, fraction=None, max_count=None, seed=42):
    """Return a deterministic subset of graphs for TOY/SWEEP modes."""
    if fraction is not None:
        count = max(1, int(len(graphs) * fraction))
    elif max_count is not None:
        count = min(max_count, len(graphs))
    else:
        return graphs

    if count >= len(graphs):
        return graphs

    rng = random.Random(seed)
    indices = list(range(len(graphs)))
    rng.shuffle(indices)
    selected = sorted(indices[:count])
    return [graphs[i] for i in selected]


def get_spot_checkpoint_path(domain):
    """Return the path for the spot-mode latest checkpoint."""
    return os.path.join("cache", "results", domain, "latest_checkpoint.pt")


def save_spot_checkpoint(path, model, optimizer, epoch):
    """Save full training state for spot-instance resume."""
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, path)


def load_spot_checkpoint(path, model, optimizer, device='cpu'):
    """Load spot checkpoint. Returns the epoch to resume from, or 0."""
    import torch
    if not os.path.exists(path):
        return 0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    resume_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from spot checkpoint at epoch {checkpoint['epoch']}, "
          f"starting at {resume_epoch}")
    return resume_epoch


def trigger_auto_shutdown():
    """Shut down the instance. Registered via atexit when --auto-shutdown."""
    print("Auto-shutdown triggered.", flush=True)
    os.system('sudo shutdown -h now')


def notebook_setup(domain, num_train_problems=50, device='cpu', **override_args):
    """Load data and metadata for interactive Jupyter use.

    Returns (all_graphs, graph_metadata, action_space, args).

    Example::

        from ploi.run_modes import notebook_setup
        graphs, metadata, action_space, args = notebook_setup("manyblocks_ipcc_big")
        single_sample = graphs[0]  # one unbatched PyG HeteroData graph
    """
    from ploi.argparsers import get_ploi_argument_parser
    from ploi.datautils_ltp import (
        _create_graph_dataset_ltp, process_pddl_to_graphs,
    )
    from ploi.run_planner_with_ltp_v1 import _create_planner

    parser = get_ploi_argument_parser()
    args = parser.parse_args([])
    args.domain = domain.capitalize()
    args.device = device
    args.use_gpu = device.startswith('cuda')
    args.num_train_problems = num_train_problems
    for k, v in override_args.items():
        setattr(args, k.replace('-', '_'), v)

    train_planner = _create_planner(args.train_planner_name)
    all_graphs, graph_metadata, action_space = process_pddl_to_graphs(
        args.domain, train_planner, args.num_train_problems, args,
        _create_graph_dataset_ltp)

    return all_graphs, graph_metadata, action_space, args

from sys import argv, stdout
from pathlib import Path
from termcolor import colored
from timeit import default_timer as timer
import argparse, logging
import torch

from generators import compute_traces_with_augmented_states, load_pddl_problem_with_augmented_states
from architecture import g_model_classes

def _get_logger(name : str, logfile : Path, level = logging.INFO):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    # add stdout handler
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s')
    console = logging.StreamHandler(stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # add file handler
    if logfile != '':
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s')
        file_handler = logging.FileHandler(str(logfile), 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def _parse_arguments(exec_path : Path):
    default_aggregation = 'max'
    default_debug_level = 0
    default_cycles = 'avoid'
    default_logfile = 'log_plan.txt'
    default_max_length = 500
    default_registry_filename = '../DerivedPredicates/registry_rules.json'

    # required arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, type=Path, help='domain file')
    parser.add_argument('--model', required=True, type=Path, help='model file')
    parser.add_argument('--problem', required=True, type=Path, help='problem file')

    # optional arguments
    parser.add_argument('--aggregation', default=default_aggregation, nargs='?', choices=['add', 'max', 'addmax', 'attention'], help=f'aggregation function for readout (default={default_aggregation})')
    parser.add_argument('--augment', action='store_true', help='augment states with derived predicates')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    parser.add_argument('--cycles', type=str, default=default_cycles, choices=['avoid', 'detect'], help=f'how planner handles cycles (default={default_cycles})')
    parser.add_argument('--debug_level', dest='debug_level', type=int, default=default_debug_level, help=f'set debug level (default={default_debug_level})')
    parser.add_argument('--ignore_unsolvable', action='store_true', help='ignore unsolvable states in policy controller')
    parser.add_argument('--logfile', type=Path, default=default_logfile, help=f'log file (default={default_logfile})')
    parser.add_argument('--max_length', type=int, default=default_max_length, help=f'max trace length (default={default_max_length})')
    parser.add_argument('--print_trace', action='store_true', help='print trace')
    parser.add_argument('--readout', action='store_true', help='use global readout')
    parser.add_argument('--registry_filename', type=Path, default=default_registry_filename, help=f'registry filename (default={default_registry_filename})')
    parser.add_argument('--registry_key', type=str, default=None, help=f'key into registry (if missing, calculated from domain path)')
    parser.add_argument('--spanner', action='store_true', help='special handling for Spanner problems')
    args = parser.parse_args()
    return args

def _load_model(args):
    try:
        Model = g_model_classes[(args.aggregation, args.readout, 'base')]
    except KeyError:
        raise NotImplementedError(f"No model found for {(args.aggregation, args.readout, 'base')} combination")
    return Model

def _main(args):
    global logger
    start_time = timer()

    # load model
    use_cpu = args.cpu #hasattr(args, 'cpu') and args.cpu
    use_gpu = not use_cpu and torch.cuda.is_available()
    device = torch.cuda.current_device() if use_gpu else None
    Model = _load_model(args)
    model = Model.load_from_checkpoint(checkpoint_path=str(args.model), strict=False).to(device)
    elapsed_time = timer() - start_time
    logger.info(f"Model '{args.model}' loaded in {elapsed_time:.3f} second(s)")

    logger.info(f"Loading PDDL files: domain='{args.domain}', problem='{args.problem}'")
    registry_filename = args.registry_filename if args.augment else None
    pddl_problem = load_pddl_problem_with_augmented_states(args.domain, args.problem, registry_filename, args.registry_key, logger)
    del pddl_problem['predicates'] #  Why?

    logger.info(f'Executing policy (max_length={args.max_length})')
    start_time = timer()
    is_spanner = args.spanner and 'spanner' in str(args.domain)
    unsolvable_weight = 0.0 if args.ignore_unsolvable else 100000.0
    action_trace, state_trace, value_trace, is_solution, num_evaluations = compute_traces_with_augmented_states(model=model, cycles=args.cycles, max_trace_length=args.max_length, unsolvable_weight=unsolvable_weight, logger=logger, is_spanner=is_spanner, **pddl_problem)
    elapsed_time = timer() - start_time
    logger.info(f'{len(action_trace)} executed action(s) and {num_evaluations} state evaluations(s) in {elapsed_time:.3f} second(s)')

    if is_solution:
        logger.info(colored(f'Found valid plan with {len(action_trace)} action(s) for {args.problem}', 'green', attrs=[ 'bold' ]))
    else:
        logger.info(colored(f'Failed to find a plan for {args.problem}', 'red', attrs=[ 'bold' ]))

    if args.print_trace:
        for index, action in enumerate(action_trace):
            value_from = value_trace[index]
            value_to = value_trace[index + 1]
            logger.info('{}: {} (value change: {:.2f} -> {:.2f} {})'.format(index + 1, action.name, float(value_from), float(value_to), 'D' if float(value_from) > float(value_to) else 'I'))


if __name__ == "__main__":
    # setup timer and exec name
    entry_time = timer()
    exec_path = Path(argv[0]).parent
    exec_name = Path(argv[0]).stem

    # parse arguments
    args = _parse_arguments(exec_path)

    # setup logger
    log_path = exec_path
    logfile = log_path / args.logfile
    log_level = logging.INFO if args.debug_level == 0 else logging.DEBUG
    logger = _get_logger(exec_name, logfile, log_level)
    logger.info(f'Call: {" ".join(argv)}')

    # do jobs
    _main(args)

    # final stats
    elapsed_time = timer() - entry_time
    logger.info(f'All tasks completed in {elapsed_time:.3f} second(s)')

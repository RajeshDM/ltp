
import argparse
import json
import os
import pickle
import warnings

# Deterministic execution: eliminate FP non-determinism from scatter/cuBLAS ops.
# Off by default (faster). Enable with GABAR_DETERMINISTIC=1 for reproducible results.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
if os.environ.get('GABAR_DETERMINISTIC'):
    torch.use_deterministic_algorithms(True)

import wandb
import time
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pyg_dataloader
from ploi.model_checkpointing import ModelManager 
from pathlib import Path
from typing import Dict, List, Optional, Any,Set
from torch_geometric.data import Data
from ploi.argparsers import get_ploi_argument_parser
from ploi.datautils import (
    collect_training_data,
    create_graph_dataset,
    create_graph_dataset_hierarchical,
    GraphDictDataset,
)
from torch_geometric.loader import DataLoader as PyGDataLoader
import gc
from ploi.baselines.exp_1.train import exp_baseline_train
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ploi.datautils_ltp import (
    _collect_training_data_ltp,
    _collect_training_data,
    _create_graph_dataset_ltp,
    _state_to_graph_ltp,
    get_filenames,
    TorchGraphDictDataset,
    graph_dataset_to_pyg_dataset,
    process_pddl_to_graphs,
    remove_actions_all_graphs,
)
from ploi.run_planner_with_ltp_v1 import (
    run_planner_with_gnn_ltp,
    _create_planner,
)
#from ploi.run_planner_with_ltp_2 import PlannerTester, PlannerConfig, PlannerType
from ploi.test_utils import (
    PlannerConfig, PlannerType, PlanningResult, PlannerMetrics,
    LearnedSearchStrat,
    compute_metrics,
    compute_combined_metrics,
    learned_planner_types,
    baselines
)
from ploi.run_planner_with_ltp_v2 import PlannerTester, PlannerConfig, PlannerType
from ploi.test_utils import format_metrics, log_model_metrics ,format_metrics_non_opt
from ploi.guiders import HierarchicalGuidance, PLOIGuidance, SceneGraphGuidance
from ploi.modelutils import (
    GraphNetwork,
)
from ploi.modelutils_ltp import (
    GNN_GRU,
)
from ploi.ablations import (
    GNN_non_AG_CD,
    GNN_non_CD,
    GNN_Val,
)
from ploi.planning import IncrementalPlanner
from ploi.planning.incremental_hierarchical_planner import (
    IncrementalHierarchicalPlanner,
)
from ploi.planning.scenegraph_planner import SceneGraphPlanner
from ploi.traineval import (
    test_planner,
    train_model_graphnetwork,
    train_model_graphnetwork_ltp_batch,
    train_model_graphnetwork_ltp_batch_val,
    train_model_graphnetwork_ltp_batch_allows_both,
    train_model_graphnetwork_ltp_batch_profiling,
    train_model_graphnetwork_ltp_batch_val_profiling,
    train_model_hierarchical,
)
from ploi.baselines.exp_1.utils import load_checkpoint 
from ploi.baselines.exp_2.train import load_model
from ploi.baselines.exp_2.architecture.supervised.optimal import MaxModel, AddModel
from ploi.baselines.exp_3.architecture import g_model_classes

#import ploi.constants as constants
from icecream import ic
import subprocess
import numpy as np
import pddlgym

'''
baselines = [PlannerType.EXP_BASELINE, 
             PlannerType.EXP_BASELINE_2, 
             PlannerType.EXP_BASELINE_3] 

learned_planner_types = [
    PlannerType.LEARNED_MODEL,
    PlannerType.LEARNED_MODEL_VAL
]
'''
def get_free_gpu():
    # Get GPU memory usage using nvidia-smi
    command = "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader"
    memory_use = subprocess.check_output(command.split()).decode("utf-8").strip().split("\n")
    memory_use = [int(x) for x in memory_use]
    
    # Find the GPU with the least memory usage
    free_gpu = np.argmin(memory_use)
    return free_gpu

def set_seed(args):
    seed = args.seed
    #torch.manual_seed(seed)
    if args.test_with_seed is True:
    #if True :
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        #np.random.seed(seed)
        #random.seed(seed)
        torch.cuda.manual_seed_all(seed)
    else :
        pass
        #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def initialize_model(model_class, args, action_space):
    if args.use_gpu:
        '''
        try :
            free_gpu_idx = get_free_gpu()
            device = f"cuda:{free_gpu_idx}"
            #print (f"Found free GPU at {device}")
        except Exception as e : 
            #print (f"ISsue with finding the right GPU {e}")
            #print ("Using default GPU at CUDA 0")
        '''
        device = "cuda:0"
    else:
        device = "cpu"

    return model_class(
        n_features=args.num_node_features,
        n_edge_features=args.num_edge_features,
        n_global_features = args.num_global_features,
        n_hidden=args.representation_size,
        gnn_rounds= args.gnn_rounds,
        num_decoder_layers = args.gru_layers ,
        dropout = args.dropout,
        attn_dropout = args.attention_dropout,
        action_space= action_space,
        batch_size=args.batch_size,
        n_heads = args.n_heads,
        g_node = args.use_global_node,
        num_mlp_layers_gnn=args.num_mlp_layers_gnn,
        device=device,
        action_options=args.action_options,
        object_options=args.object_options,
        ablation=args.ablation
    )


def _apply_arity_override(model, args):
    """Lift the decoder's parameter capacity to --max-action-arity.

    The capacity is derived from the training action space at init; a wider
    override (used to featurize at canonical arities above the training
    set's, so a high-arity held-out domain decodes at full width) must
    extend the beam loop bound too. No weights are sized by it - the
    parameter loop is length-only (CLAUDE.md 5.7).
    """
    if getattr(args, 'max_action_arity', 0) > 0:
        model.max_number_action_parameters = max(
            model.max_number_action_parameters, args.max_action_arity)
    return model


def run_tests(
            curr_manager,
            model_class,
             train_env_name: str,
             seed: int,
             hyperparameters: Dict,
             test_function,
             metric: str = 'validation',
             device: str = "cuda:0",
             args = None,
             action_space = None,
             tested_epoch_numbers: Set[int] = None,
             num_models_to_test: int = 2,
             starting_model_num: int = 0,
             planner_types = [PlannerType.LEARNED_MODEL],
             baseline_models = None,
             ignore_defaults : Dict[str, Any] = None,
             decode_action_space = None) -> List[Dict]:
    """
    Run tests on best models for a specific configuration

    decode_action_space: the TEST domain's own action space. Beam decoding
    terminates each beam when its depth reaches the arity of the selected
    schema, looked up by graph-local schema index. The model's dict is built
    from the (possibly merged multi-domain) training action space, whose
    ordering does not match graphs of any domain but the first, so the dict
    must be rebuilt from the domain being tested. Everything else on the
    model stays as trained.
    """
    
    # Get best models for this configuration
    best_models = curr_manager.load_best_models(
        train_env_name=train_env_name,
        seed=seed,
        hyperparameters=hyperparameters,
        metric=metric,
        ignore_defaults=ignore_defaults,
        #device=device
    )
    
    if not best_models:
        logger.warning("No models found to test")
        planner_types_copy = planner_types[:]
        if PlannerType.LEARNED_MODEL in planner_types_copy:
            planner_types_copy.remove(PlannerType.LEARNED_MODEL)

        if PlannerType.LEARNED_MODEL_VAL in planner_types_copy:
            planner_types_copy.remove(PlannerType.LEARNED_MODEL_VAL)

        if len(planner_types_copy) == 0:  
            return []
    
    results = []
    curr_model = None
    #if PlannerType.LEARNED_MODEL in planner_types or PlannerType.LEARNED_MODEL_VAL in planner_types: 
    for planner_type in planner_types :
        if planner_type not in learned_planner_types :
            continue 
        for model_info in best_models[::-1][starting_model_num:starting_model_num+num_models_to_test]:
            # Create fresh model instance
            #model = model_class()
            curr_models = {}
            curr_model = _apply_arity_override(
                initialize_model(model_class, args, action_space), args)

            # Load model state. Checkpoints from configs that share the
            # training-domain set + seed + hyperparameters land in the SAME
            # model dir and tracking list, because `featurization` is not part
            # of the ModelManager key - so a union checkpoint (wide symbol
            # vocabulary) can be offered to a structural model (narrow fixed
            # width). Widths differ per featurization, so skip mismatches
            # loudly and keep looking instead of crashing the whole test run.
            try:
                curr_model.load_state_dict(model_info['state_dict'])
            except RuntimeError as _e:
                if 'size mismatch' not in str(_e):
                    raise
                print(f"SKIP checkpoint epoch {model_info['epoch']}: built for "
                      f"a different featurization (width mismatch). Looking "
                      f"for one matching --featurization {args.featurization}.")
                continue
            curr_model.to(device)
            curr_model.eval()

            if decode_action_space is not None:
                # Clamp to the model's trained parameter capacity: a test
                # domain with higher-arity schemas (e.g. Rovers under a
                # ka=4-trained model) would otherwise never satisfy the beam
                # finish condition and index past the padded parameter dim
                # (CUDA device-side assert). Clamped schemas decode truncated
                # groundings -> invalid actions -> counted as failures, not
                # crashes.
                _cap = curr_model.max_number_action_parameters
                _clamped = [str(schema) for schema, op in
                            decode_action_space.items()
                            if len(op.params) > _cap]
                if _clamped:
                    print(f"WARNING: schemas exceed the model's parameter "
                          f"capacity ({_cap}) and cannot be fully decoded "
                          f"zero-shot: {_clamped}. Train with "
                          f"--max-action-arity >= their arity to lift this.")
                curr_model.action_parameter_number_dict = {
                    i: min(len(op.params), _cap)
                    for i, op in enumerate(decode_action_space.values())}

            if model_info['epoch'] in tested_epoch_numbers:
                print ("Already tested model from epoch ",model_info['epoch'])
                continue 
            else :
                tested_epoch_numbers.add(model_info['epoch'])

            if args.epoch_number != -1 : 
                if model_info['epoch'] != args.epoch_number :
                    continue

            #curr_models[PlannerType.LEARNED_MODEL] = (curr_model,model_info['epoch'])
            curr_models[planner_type] = (curr_model,model_info['epoch'])

            # Run tests
            print ("Testing model from epoch ",model_info['epoch'])
            test_results, run_metrics = test_function(curr_models)
            results.append({
                'epoch': model_info['epoch'],
                'validation_loss': model_info['validation_loss'],
                'training_loss': model_info['training_loss'],
                'combined_loss': model_info['combined_loss'],
                'test_results': run_metrics, 
                'all_plan_results': test_results
            })
            #metrics = results[-1]['test_results'][PlannerType.LEARNED_MODEL]
            metrics = results[-1]['test_results'][planner_type]
            _fails = metrics.failures
            _per_div = {d: len(v) for d, v in _fails.items() if v}
            print(f"failed: {sum(_per_div.values())} problems | "
                  f"per division: {_per_div}")
            if args.debug_level > 0:
                print("failed problem ids: ", _fails)
            #_ = format_metrics(results[-1]['test_results'][PlannerType.LEARNED_MODEL], model_info['epoch'])
            _ = format_metrics(results[-1]['test_results'][planner_type], model_info['epoch'])
            #print (test_results[PlannerType.LEARNED_MODEL][-1].plan)

            if PlannerType.NON_OPTIMAL in planner_types :
                 _ = format_metrics(results[-1]['test_results'][PlannerType.NON_OPTIMAL], model_info['epoch'])
            # Plan quality is defined relative to the non-optimal planner's
            # plans; skip when that planner didn't run (--run-non-optimal False)
            if PlannerType.NON_OPTIMAL in results[-1]['all_plan_results']:
                combnined_metrics = compute_combined_metrics(results[-1]['all_plan_results'], planner_type)
                print ("Plan Quality : ", combnined_metrics.plan_quality)
            else:
                print ("Plan Quality :  n/a (non-optimal planner disabled)")

    for baseline in baselines:
        if baseline in planner_types:
            #curr_models[baseline],_ = (load_checkpoint(baseline_models[baseline], device),-1)
            #curr_models[baseline],_ = load_model(baseline_models[baseline], device, MaxModel)#(load_checkpoint(baseline_models[baseline], device),-1)
            print ("Baseline : ",baseline)
            for model_path in baseline_models[baseline]:
                curr_models = {}
                model_filename = str(model_path).split("/")[-1]
                aggregation = model_filename.split("_")[1]
                loss_fn = model_filename.split("_")[2] + "_" + model_filename.split("_")[3]
                baseline_model_class = g_model_classes[(aggregation, False, loss_fn)]
                loaded_model, _  = load_model(model_path, device, baseline_model_class)
                curr_models[baseline] = (loaded_model,-1, {'aggregation': aggregation,'loss_fn' : loss_fn})
                test_results, run_metrics = test_function(curr_models)
                combnined_metrics = compute_combined_metrics(test_results, baseline)
                #print (f"Combined Metrics for {model_type} : ", combnined_metrics)
                print ("Model : ",model_filename)
                _ = format_metrics(run_metrics[baseline], None )
                print ("Plan Quality : ", combnined_metrics.plan_quality)

    return results

if __name__ == "__main__":

    parser = get_ploi_argument_parser()

    parser.add_argument(
        "--all-problems",
        action="store_true",
        help="Run testing on all problems in domain",
    )
    from ploi.argparsers import apply_config_defaults
    apply_config_defaults(parser)
    args = parser.parse_args()

    if args.wandb:  
        run = wandb.init(
            # Set the project where this run will be logged
            project="ltp_gnn_gru_pyg",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "representation_size": args.representation_size,
                "gnn_rounds": args.gnn_rounds,
                "seed": args.seed,
                "domain": args.domain,
                "model_version": args.model_version,
                "mode" : args.mode,
                "server": args.server,
                "continue_training": args.continue_training,
                "starting_epoch": args.starting_epoch,
                "train_planner_name": args.train_planner_name,
                "num_train_problems": args.num_train_problems,
                "num_test_problems": args.num_test_problems,
                "expid": args.expid,
                "gru_layers" : args.gru_layers,
                "n_heads" : args.n_heads,
                "attention_dropout" : args.attention_dropout,
                "dropout" : args.dropout,
                "augmentation" : args.data_augmentation,
                "weight_decay" : args.weight_decay,
                "monitor" : args.monitor,
                "ablation" : args.ablation ,
            })

    mode = args.mode
    if args.data_augmentation is True :
        args.batch_size = 64
    if mode == "debug" :
        args.representation_size = 4
        args.batch_size = 1

    # Apply run-mode overrides (toy/sweep/spot) before anything else
    from ploi.run_modes import apply_run_mode
    apply_run_mode(args)

    # Seed RNG
    set_seed(args)
    #torch.manual_seed(args.seed,args)

    # Create dir to log files to
    args.expdir = os.path.join(args.logdir, args.expid)
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir, exist_ok=True)

    # Capitalize the first letter of the domain name
    args.domain = args.domain.capitalize()

    # Multi-domain: parsed once, used for both data and test loops
    train_domain_names = []
    if args.domains:
        from ploi.multidomain import parse_domain_arg
        _parsed = parse_domain_arg(args.domains, args.heldout_domains,
                                    args.num_train_problems)
        train_domain_names = [n for n, _, held_out in _parsed if not held_out]
        if not train_domain_names:
            raise ValueError("--domains produced no training domains")
        args.domain = "MULTI-" + "-".join(train_domain_names)

    # This datafile is the same for ploi and hierarchical variants
    args.datafile = os.path.join(args.logdir, f"ploi_{args.domain}.pkl")
    if args.domain.endswith("scrub"):
        args.datafile = os.path.join(args.logdir, f"ploi_{args.domain[:-5]}.pkl")

    # Set up logging (tee to file for spot/sweep) and auto-shutdown
    from ploi.run_modes import setup_logging
    if args.auto_shutdown:
        import atexit
        from ploi.run_modes import trigger_auto_shutdown
        atexit.register(trigger_auto_shutdown)
    setup_logging(args.domain, args)

    print(f"Domain: {args.domain}")
    print(f"Run mode: {args.run_mode}")
    print(f"Train planner: {args.train_planner_name}")
    print(f"Test planner: {args.eval_planner_name}")

    eval_planner = _create_planner(args.eval_planner_name)
    is_strips_domain = True

    train_planner = _create_planner(args.train_planner_name)
    model_dir = os.path.join(os.path.dirname(__file__), "cache")
    model_dir = os.path.join(model_dir, "results")
    _dataset_file_prefix=os.path.join(model_dir, "training_data")

    training_data = None
    print("Collecting training data")
    graphs_inp, graphs_tgt, graph_metadata = None, None, None
    if not os.path.exists(args.datafile) or args.force_collect_data:
        if 'ltp' in args.method :
            args.datafile = _dataset_file_prefix + "_{}.pkl".format(args.domain)

            if args.domains:
                # Multi-domain path (CLAUDE.md Phase 0/1, C1)
                from ploi.multidomain import (merge_action_spaces,
                    merge_feature_metadata, parse_domain_arg)
                from ploi.datautils_ltp import load_domain_metadata
                import random as _random

                domains = parse_domain_arg(args.domains, args.heldout_domains,
                                           args.num_train_problems)
                train_domains = [(n, c) for n, c, held_out in domains if not held_out]

                # Domain-set tag: sorted names prevent stale cache when the
                # domain combination changes (e.g. {A,B} vs {A,B,C}).
                _domain_set_tag = "_" + "_".join(
                    sorted(n.lower() for n, _ in train_domains))

                # Pass 1: per-domain metadata + action_space (fast, no graphs)
                per_domain_meta = {}
                for name, num_problems in train_domains:
                    md, aspace = load_domain_metadata(
                        name, train_planner, num_problems, args,
                        _create_graph_dataset_ltp)
                    per_domain_meta[name] = (md, aspace)

                if args.featurization == 'union':
                    # Pass 2: re-featurize with shared union vocab (Baseline 0).
                    # Held-out domains contribute nothing to the union.
                    union_md = merge_feature_metadata(
                        [md for (md, _) in per_domain_meta.values()])
                    tag = "_union" + _domain_set_tag
                    per_domain_graphs = {}
                    for name, num_problems in train_domains:
                        graphs, _, _ = process_pddl_to_graphs(
                            name, train_planner, num_problems, args,
                            _create_graph_dataset_ltp,
                            metadata_override=union_md, cache_tag=tag)
                        per_domain_graphs[name] = graphs
                    graph_metadata = union_md
                    action_space = merge_action_spaces(
                        [a for (_, a) in per_domain_meta.values()])
                elif args.featurization in ('structural', 'joint', 'joint_lite',
                                            'joint_chain'):
                    from ploi.structural import build_structural_metadata
                    from ploi.lifted_layer import build_lifted_metadata
                    kp = max(p.arity for (md, _) in per_domain_meta.values()
                             for p in md['all_predicates'])
                    ka = max(len(op.params) for (_, a) in per_domain_meta.values()
                             for op in a.values())
                    # Optional overrides: train at wider canonical arities
                    # than the training set needs, so a held-out domain with
                    # higher-arity symbols (e.g. Rovers under a no-rovers
                    # set) featurizes and decodes at full width. Sidecars are
                    # shared with any set computing the same (kp, ka).
                    if args.max_pred_arity > 0:
                        kp = max(kp, args.max_pred_arity)
                    if args.max_action_arity > 0:
                        ka = max(ka, args.max_action_arity)
                    # Structural/lifted metadata for a domain depends on the
                    # TRAINING SET only through (kp, ka) - each domain is
                    # featurized from its own action space at those canonical
                    # arities. So tag sidecars by (feat, kp, ka), NOT the
                    # domain set: every config whose set shares the same
                    # arity maxima (in practice, every set containing rovers)
                    # reuses the same featurized graphs. Union keeps
                    # domain-set tags (its merged vocab genuinely differs).
                    tag = f"_{args.featurization}_kp{kp}_ka{ka}"
                    if args.cheating_input:
                        tag += "_cheat"  # widths differ: never share sidecars
                    per_domain_graphs = {}
                    for name, num_problems in train_domains:
                        if args.featurization == 'structural':
                            struct_md = build_structural_metadata(
                                per_domain_meta[name][1], kp, ka,
                                cheating=args.cheating_input)
                        else:
                            struct_md = build_lifted_metadata(
                                per_domain_meta[name][1], kp, ka,
                                args.featurization,
                                cheating=args.cheating_input)
                        graphs, _, _ = process_pddl_to_graphs(
                            name, train_planner, num_problems, args,
                            _create_graph_dataset_ltp,
                            metadata_override=struct_md, cache_tag=tag)
                        per_domain_graphs[name] = graphs
                    graph_metadata = struct_md
                    action_space = merge_action_spaces(
                        [a for (_, a) in per_domain_meta.values()])
                else:
                    if len(train_domains) > 1:
                        raise ValueError("per_domain featurization cannot mix domains; use --featurization union, structural, joint_lite, joint, or joint_chain")
                    name = train_domains[0][0]
                    graphs, graph_metadata, action_space = process_pddl_to_graphs(
                        name, train_planner, train_domains[0][1], args,
                        _create_graph_dataset_ltp)
                    per_domain_graphs = {name: graphs}

                # Update num_global_features from actual graph data: union/
                # structural metadata predates graph creation, so its value
                # may not match the wider globals produced by the merged vocab.
                for _dg in per_domain_graphs.values():
                    if _dg:
                        try:
                            graph_metadata['num_global_features'] = _dg[0]['globals'].x.shape[-1]
                        except (KeyError, AttributeError, IndexError):
                            pass
                        break

                # Tag each graph with its domain index for per-domain loss tracking.
                # Stratified train/val split: 10% from EACH domain goes to
                # validation, so every domain is proportionally represented.
                _domain_names_ordered = list(per_domain_graphs.keys())
                args._domain_names_ordered = _domain_names_ordered
                for dom_idx, (name, dom_graphs) in enumerate(per_domain_graphs.items()):
                    for g in dom_graphs:
                        g.domain_id = dom_idx

                rng = _random.Random(args.seed)
                input_hetero_graphs = []
                val_hetero_graphs = []
                for name, dom_graphs in per_domain_graphs.items():
                    _random.Random(args.seed).shuffle(dom_graphs)
                    n_val = max(1, int(len(dom_graphs) * 0.1))
                    val_hetero_graphs.extend(dom_graphs[:n_val])
                    input_hetero_graphs.extend(dom_graphs[n_val:])
                    print(f"  {name}: {len(dom_graphs)} graphs "
                          f"(train={len(dom_graphs) - n_val}, val={n_val})")
                rng.shuffle(input_hetero_graphs)
                rng.shuffle(val_hetero_graphs)

            else:
                all_input_graphs , graph_metadata,action_space =  process_pddl_to_graphs(
                    args.domain,
                    train_planner,
                    args.num_train_problems,
                    args,
                    _create_graph_dataset_ltp,
                )
                # Subset data for toy/sweep modes
                if getattr(args, 'toy_max_graphs', None):
                    from ploi.run_modes import subset_graphs
                    print(f"TOY mode: subsetting {len(all_input_graphs)} graphs to {args.toy_max_graphs}")
                    all_input_graphs = subset_graphs(all_input_graphs, max_count=args.toy_max_graphs, seed=args.seed)
                elif getattr(args, 'data_fraction', None):
                    from ploi.run_modes import subset_graphs
                    target = max(1, int(len(all_input_graphs) * args.data_fraction))
                    print(f"SWEEP mode: subsetting {len(all_input_graphs)} graphs to {target} ({args.data_fraction:.0%})")
                    all_input_graphs = subset_graphs(all_input_graphs, fraction=args.data_fraction, seed=args.seed)

                num_validation = max(1, int(len(all_input_graphs) * 0.1))
                input_hetero_graphs = all_input_graphs[num_validation:]
                val_hetero_graphs = all_input_graphs[:num_validation]

            # Pad action_object_scores to uniform width across all graphs so
            # PyG Batch.from_data_list can collate them. Different domains (or
            # problems with different object counts) need uniform tensor widths.
            from ploi.datautils_ltp import pad_pyg_action_scores
            pad_pyg_action_scores(input_hetero_graphs + val_hetero_graphs)
        else :
            training_data = collect_training_data(
                args.domain, train_planner, num_train_problems=args.num_train_problems
            )
            with open(args.datafile, "wb") as f:
                pickle.dump(training_data, f)
            with open(args.datafile, "rb") as f:
                print("Loading training data from file")
                training_data = pickle.load(f)
    else:
        print("Training data already found on disk")

        with open(args.datafile, "rb") as f:
            print("Loading training data from file")
            training_data = pickle.load(f)

    if args.method in ["hierarchical"]:
        graphs_inp, graphs_tgt, graph_metadata = create_graph_dataset_hierarchical(
            training_data
        )
    elif 'ltp' in args.method:
        if 'no_ag' in args.ablation :
            input_hetero_graphs = remove_actions_all_graphs(input_hetero_graphs)
            val_hetero_graphs = remove_actions_all_graphs(val_hetero_graphs)

    else:
        graphs_inp, graphs_tgt, graph_metadata = create_graph_dataset(training_data)

    # Use 10% for validation
    '''
    num_validation = max(1, int(len(graphs_inp) * 0.1))
    train_graphs_input = graphs_inp[num_validation:]
    train_graphs_target = graphs_tgt[num_validation:]
    valid_graphs_input = graphs_inp[:num_validation]
    valid_graphs_target = graphs_tgt[:num_validation]
    '''
    pyg = args.pyg
    batch_size = args.batch_size

    #args.num_node_features_object = train_graphs_input[0]['nodes'][0].shape[-1]
    #args.num_edge_features_object = train_graphs_input[0]['edges'][0].shape[-1]

    #if 'globals' in train_graphs_input[0]:
    #    args.num_global_features = train_graphs_input[0]['globals'][0].shape[-1]
    args.num_node_features_object = graph_metadata['num_node_features']
    args.num_edge_features_object = graph_metadata['num_edge_features']
    args.num_node_features = args.num_node_features_object
    args.num_edge_features = args.num_edge_features_object
    if 'num_global_features' in graph_metadata :
        args.num_global_features = graph_metadata['num_global_features']

    if pyg == False:
        # Set up dataloaders
        graph_dataset = GraphDictDataset(train_graphs_input, train_graphs_target)
        graph_dataset_val = GraphDictDataset(valid_graphs_input, valid_graphs_target)

    else :
        print ("Size of dataset : ",len(input_hetero_graphs) + len(val_hetero_graphs))
        #train_graphs_pyg = graph_dataset_to_pyg_dataset(train_graphs_input)
        #train_graphs_target_pyg = graph_dataset_to_pyg_dataset(train_graphs_target)

        #val_graphs_pyg = graph_dataset_to_pyg_dataset(valid_graphs_input)
        #val_graphs_target_pyg = graph_dataset_to_pyg_dataset(valid_graphs_target)

        #graph_dataset = pyg_dataloader(train_graphs_pyg, batch_size=batch_size,shuffle=True)
        #graph_dataset_val = pyg_dataloader(val_graphs_pyg,batch_size=batch_size,shuffle=True)
        num_workers = args.num_workers
        shuffle = True
        graph_dataset = PyGDataLoader(
            input_hetero_graphs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
        graph_dataset_val =  PyGDataLoader(
            val_hetero_graphs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )

        '''
        graph_dataset = graph_dataset_to_pyg_dataset(
            train_graphs_input, 
            batch_wise=True, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers
        )
        gc.collect()
        graph_dataset_val = graph_dataset_to_pyg_dataset(
            valid_graphs_input, 
            batch_wise=True, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
        '''

    datasets = {"train": graph_dataset, "val": graph_dataset_val}
    #dataloaders = {"train": dataloader, "val": dataloader_val}

    object_level_model = GraphNetwork(
        n_features=args.num_node_features_object,
        n_edge_features=args.num_edge_features_object,
        n_hidden=16,
    )

    if args.method == "scenegraph":

        if args.mode == "train":
            import sys

            warnings.warn("No training mode for scenegraph planner.")
            sys.exit(0)

        scenegraph_guidance = SceneGraphGuidance(graph_metadata)
        planner_to_eval = SceneGraphPlanner(
            is_strips_domain=is_strips_domain,
            base_planner=eval_planner,
            guidance=scenegraph_guidance,
        )
        test_stats, global_stats = test_planner(
            planner_to_eval,
            args.domain,
            args.num_test_problems,
            args.timeout,
            all_problems=args.all_problems,
        )

        statsfile = os.path.join(args.expdir, "scenegraph_test_stats.py")
        json_string = json.dumps(test_stats, indent=4)
        json_string = "STATS = " + json_string
        with open(statsfile, "w") as f:
            f.write(json_string)

        globalstatsfile = os.path.join(
            args.expdir, f"{args.domain.lower()}_{args.method}_test.json"
        )
        with open(globalstatsfile, "w") as fp:
            json.dump(global_stats, fp, indent=4, sort_keys=True)

    elif args.method == "hierarchical":

        args.num_node_features_room = datasets["train"][0]["graph_input"]["room_graph"][
            "nodes"
        ].shape[-1]
        args.num_edge_features_room = datasets["train"][0]["graph_input"]["room_graph"][
            "edges"
        ].shape[-1]

        room_level_model = GraphNetwork(
            n_features=args.num_node_features_room,
            n_edge_features=args.num_edge_features_room,
            n_hidden=32,
            # dropout=0.2,
        )

        if args.mode == "train":

            optimizer_room = torch.optim.Adam(room_level_model.parameters(), lr=1e-4)
            optimizer_object = torch.optim.Adam(
                object_level_model.parameters(), lr=1e-3
            )
            pos_weight = args.pos_weight * torch.ones([1])
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            room_level_model_dict = train_model_hierarchical(
                room_level_model,
                datasets,
                criterion=torch.nn.BCEWithLogitsLoss(pos_weight=2 * torch.ones([1])),
                optimizer=optimizer_room,
                use_gpu=False,
                epochs=args.epochs,
                save_folder=args.expdir,
                model_type="room",
                eval_every=10,
            )
            object_level_model_dict = train_model_hierarchical(
                object_level_model,
                datasets,
                criterion=criterion,
                optimizer=optimizer_object,
                use_gpu=False,
                epochs=args.epochs,
                save_folder=args.expdir,
                model_type="object",
            )
            room_level_model.load_state_dict(room_level_model_dict)
            object_level_model.load_state_dict(object_level_model_dict)

        elif args.mode == "test":

            with torch.no_grad():

                room_model_outfile = os.path.join(args.expdir, "room_best.pt")
                object_model_outfile = os.path.join(args.expdir, "object_best.pt")
                room_level_model.load_state_dict(torch.load(room_model_outfile))
                object_level_model.load_state_dict(torch.load(object_model_outfile))
                print(
                    f"Loaded saved models from {room_model_outfile}, {object_model_outfile}"
                )

                hierarchical_guider = HierarchicalGuidance(
                    room_level_model, object_level_model, graph_metadata
                )
                planner_to_eval = IncrementalHierarchicalPlanner(
                    is_strips_domain=is_strips_domain,
                    base_planner=eval_planner,
                    #base_planner=train_planner,
                    search_guider=hierarchical_guider,
                    seed=args.seed,
                    gamma=args.gamma,
                    threshold_mode="geometric",
                    # force_include_goal_objects=False,
                )

                test_stats, global_stats = test_planner(
                    planner_to_eval,
                    args.domain,
                    args.num_test_problems,
                    args.timeout,
                    all_problems=args.all_problems,
                )

                statsfile = os.path.join(args.expdir, "hierarchical_test_stats.py")
                json_string = json.dumps(test_stats, indent=4)
                json_string = "STATS = " + json_string
                with open(statsfile, "w") as f:
                    f.write(json_string)
                    # json.dump(test_stats, f, indent=4)

                globalstatsfile = os.path.join(
                    args.expdir, f"{args.domain.lower()}_{args.method}_test.json"
                )
                with open(globalstatsfile, "w") as fp:
                    json.dump(global_stats, fp, indent=4, sort_keys=True)


    elif args.method == "ploi":
        # PLOI training / testing

        args.num_node_features = datasets["train"][0]["graph_input"]["nodes"].shape[-1]
        args.num_edge_features = datasets["train"][0]["graph_input"]["edges"].shape[-1]

        model = GraphNetwork(
            n_features=args.num_node_features,
            n_edge_features=args.num_edge_features,
            n_hidden=16,
        )

        print("====================================")
        print(f"==== Expid: {args.expid} ==========")
        print("====================================")

        if args.mode == "train":
            """
            Train PLOI on pre-cached dataset of states and targets
            """
            if not args.load_model:
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                pos_weight = args.pos_weight * torch.ones([1])
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                model_dict = train_model_graphnetwork(
                    model,
                    datasets,
                    criterion=criterion,
                    optimizer=optimizer,
                    use_gpu=False,
                    epochs=args.epochs,
                    save_folder=args.expdir,
                )
                model.load_state_dict(model_dict)

        if args.mode == "test":
            """
            Test phase
            """

            model_outfile = os.path.join(args.expdir, "object_best.pt")
            try:
                object_level_model.load_state_dict(torch.load(model_outfile))
                print(f"Loaded saved model from {model_outfile}")
            except Exception as e1:
                try:
                    object_level_model.load_state_dict(
                        torch.load(os.path.join(args.expdir, "best.pt"))
                    )
                except Exception as e2:
                    raise IOError(f"No model file {model_outfile} or best.pt")

            ploiguider = PLOIGuidance(object_level_model, graph_metadata)
            planner_to_eval = IncrementalPlanner(
                is_strips_domain=is_strips_domain,
                base_planner=eval_planner,
                search_guider=ploiguider,
                seed=args.seed,
                gamma=args.gamma,
                # force_include_goal_objects=False,
            )

            ic (planner_to_eval)
            test_stats, global_stats = test_planner(
                planner_to_eval,
                args.domain,
                args.num_test_problems,
                args.timeout,
                all_problems=args.all_problems,
            )
            statsfile = os.path.join(args.expdir, "ploi_test_stats.py")
            json_string = json.dumps(test_stats, indent=4)
            json_string = "STATS = " + json_string
            with open(statsfile, "w") as f:
                f.write(json_string)

            globalstatsfile = os.path.join(
                args.expdir, f"{args.domain.lower()}_{args.method}_test.json"
            )
            with open(globalstatsfile, "w") as fp:
                json.dump(global_stats, fp, indent=4, sort_keys=True)

    elif args.method == "exp_baseline":
        exp_baseline_train(args)

    elif 'ltp' in args.method:
        ic ("LTP start")
        representation_size = args.representation_size
        gnn_rounds = args.gnn_rounds
        n_heads = args.n_heads
        #action_space = training_data[3]

        if args.ablation == 'main' or args.ablation == 'main_val':
            model_class = GNN_GRU
        elif 'no_cd' in  args.ablation:
            model_class = GNN_non_CD
        elif args.ablation == 'no_ag' :
            model_class = GNN_non_AG_CD
        elif args.ablation == 'val' :
            model_class = GNN_Val

        # NOTE: no _apply_arity_override here. Training data has exactly
        # as many parameter slots as the TRAINING action space's max arity;
        # the decode loop indexes action_object_scores by slot, so raising
        # capacity above the data's width indexes out of bounds (CUDA
        # device-side assert). The override is test-time only, where graphs
        # come from the test domain's own action space.
        _model = initialize_model(model_class, args, action_space)

        training_hyperparameters = {
            'lr': args.lr,
            'gnn_rounds': args.gnn_rounds,
            'd' : args.num_train_problems,
            'ad' : args.attention_dropout,
            'wd' : args.dropout,
            'heads' : args.n_heads,
            'g_node' : args.use_global_node,
            #'model_class' : model_class.__name__,
            'abl_' : args.ablation ,
            'mlp_layers' : args.num_mlp_layers_gnn,
        }

        testing_hyperparameters = {
            'domain_name' : args.domain,
            'search' : args.search_strat,
        }

        ignore_defaults = {
            #'g_node' : True ,
            #'model_class' : GNN_GRU.__name__
            #'abl_' : 'main'
            'mlp_layers' : 2,
        }

        continue_training = args.continue_training
        train_env_name = args.domain
        save_model_prefix=os.path.join(
            model_dir, "bce10_model_seed{}".format(args.seed)),
        dataset_size = len(input_hetero_graphs) + len(val_hetero_graphs)#len(training_data[0])
        save_folder = os.path.join(Path.cwd(),"models")
        manager = ModelManager(save_folder, hyperparameters=training_hyperparameters,
                               max_checkpoints_per_metric=args.keep_checkpoints,
                               train_env_name=train_env_name,seed=args.seed, ignore_defaults=ignore_defaults)

        model_outfile, message_string,save_folder = get_filenames(dataset_size,train_env_name,
                                                        args.epochs,args.model_version,
                                                        representation_size,
                                                        save_model_prefix,args.seed,
                                                        args, model_class)
        
        #if args.mode == 'train' and (not os.path.exists(model_outfile) or continue_training == True):
        if args.mode == 'train'  or args.mode == 'train_test' :
            optimizer = torch.optim.Adam(_model.parameters(),lr=args.lr,weight_decay=args.weight_decay) 
            enable_profiling = True
            enable_profiling = False

            if args.ablation != 'val' :# not in args.ablation  :
                pos_weight = args.pos_weight * torch.ones([1])
                criterion = torch.nn.CrossEntropyLoss()
                if enable_profiling :
                    train_func = train_model_graphnetwork_ltp_batch_profiling
                else :
                    #train_func = train_model_graphnetwork_ltp_batch
                    train_func = train_model_graphnetwork_ltp_batch_allows_both

            else :
                pos_weight = None 
                criterion = torch.nn.MSELoss() 
                if enable_profiling :
                    train_func = train_model_graphnetwork_ltp_batch_val_profiling
                else :
                    train_func = train_model_graphnetwork_ltp_batch_val

            if continue_training:
                _resumed = False
                # Try ModelManager checkpoints first (current save format)
                _best = manager.load_best_models(
                    train_env_name=train_env_name, seed=args.seed,
                    hyperparameters=training_hyperparameters,
                    metric='validation', ignore_defaults=ignore_defaults)
                if _best:
                    _ckpt = _best[-1]  # best val model (sorted ascending)
                    _model.load_state_dict(_ckpt['state_dict'])
                    _ckpt_full = torch.load(_ckpt['save_path'], map_location='cpu')
                    if 'optimizer' in _ckpt_full:
                        optimizer.load_state_dict(_ckpt_full['optimizer'])
                    args.starting_epoch = _ckpt['epoch'] + 1
                    print(f"Resuming from ModelManager checkpoint epoch {_ckpt['epoch']}")
                    _resumed = True
                # Fall back to legacy model_outfile
                if not _resumed and os.path.exists(model_outfile):
                    _model_state = torch.load(model_outfile)
                    _model.load_state_dict(_model_state['state_dict'])
                    if 'optimizer' in _model_state:
                        optimizer.load_state_dict(_model_state['optimizer'])
                    args.starting_epoch = _model_state.get('epoch', 0) + 1
                    print(f"Resuming from legacy checkpoint epoch {args.starting_epoch - 1}")
                    _resumed = True
                if not _resumed:
                    print("No checkpoint found, starting fresh")

            # Spot-mode resume (overrides --continue-training if checkpoint exists)
            _spot_path = None
            if getattr(args, 'spot_resume', False):
                from ploi.run_modes import get_spot_checkpoint_path, load_spot_checkpoint
                _spot_path = get_spot_checkpoint_path(train_env_name)
                _device = "cuda:0" if args.use_gpu else "cpu"
                _resume_epoch = load_spot_checkpoint(_spot_path, _model, optimizer, _device)
                if _resume_epoch > 0:
                    args.starting_epoch = _resume_epoch

            #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # Train model
            #model_dict = train_model_graphnetwork_ltp_batch(_model,
            _train_kwargs = dict(
                criterion=criterion, optimizer=optimizer,
                use_gpu=args.use_gpu,
                starting_epoch=args.starting_epoch,
                save_folder=save_folder,
                final_epoch=args.epochs,
                train_env_name=train_env_name, seed=args.seed,
                message_string=message_string,
                log_wandb=args.wandb,
                ablation=args.ablation,
                chpkt_manager=manager,
                enable_profiling=enable_profiling,
                use_amp=getattr(args, 'use_amp', False),
                spot_checkpoint_path=_spot_path,
                patience=getattr(args, 'early_stopping_patience', 0),
            )
            if hasattr(args, '_domain_names_ordered'):
                _train_kwargs['domain_names'] = args._domain_names_ordered
            train_func(_model, datasets, **_train_kwargs)
            ic (args.attention_dropout)
            ic (args.dropout)
            ic (args.weight_decay)
            ic (args.n_heads)
            ic (args.gnn_rounds)
            ic (args.lr)

        if args.mode != 'test' and args.mode != 'train_test' :
           exit() 

        planner_types = [] 
        baseline_models = {}

        if args.run_learned_model is True :
            if args.ablation == 'val' :
                planner_types.append(PlannerType.LEARNED_MODEL_VAL)
            else :
                planner_types.append(PlannerType.LEARNED_MODEL)

        if args.run_non_optimal is True :
            planner_types.append(PlannerType.NON_OPTIMAL)
            
        if args.run_optimal is True :
            planner_types.append(PlannerType.OPTIMAL)

        if args.exp_baseline is True:
            planner_types.append(PlannerType.EXP_BASELINE)
            folder = os.path.join(Path.cwd(),"models")
            folder = os.path.join(folder,args.domain+"_exp_baseline")
            file = os.path.join(folder,"best.pth")
            baseline_models[PlannerType.EXP_BASELINE] = [file]

        if args.exp_baseline_2 is True:
            planner_types.append(PlannerType.EXP_BASELINE_2)
            folder = os.path.join(Path.cwd(),"models")
            folder = os.path.join(folder,args.domain+"_exp_2")
            file = os.path.join(folder,"model_best.pth")
            baseline_models[PlannerType.EXP_BASELINE_2] = [file]

        if args.exp_baseline_3 is True:
            planner_types.append(PlannerType.EXP_BASELINE_3)
            folder = os.path.join(Path.cwd(),"models")
            folder = os.path.join(folder,args.domain+"_exp_3")
            files = list(Path(folder).glob("*.ckpt")) #os.path.join(folder,"model_best.pth")
            baseline_models[PlannerType.EXP_BASELINE_3] = files[:]

        #ONLY DONE FOR FINALY DAY TESTING - REMOVE LATER
        #planner_types = [PlannerType.LEARNED_MODEL,PlannerType.NON_OPTIMAL]
        learned_search_strat = []

        if args.search_strat == 'greedy' : 
            learned_search_strat.append(LearnedSearchStrat.GREEDY)
        elif args.search_strat == 'dfs' : 
            learned_search_strat.append(LearnedSearchStrat.DFS)
        elif args.search_strat == 'bfs' : 
            learned_search_strat.append(LearnedSearchStrat.BFS)
        elif args.search_strat == 'mcts' : 
            learned_search_strat.append(LearnedSearchStrat.MCTS)

        # Build the list of (domain_name, test_count, is_zero_shot) to eval on.
        # --test-domains overrides everything; otherwise default to training
        # domains + held-out domains.
        from ploi.multidomain import parse_domain_arg
        _train_set = set(train_domain_names) if train_domain_names else {args.domain}

        # A '@train' suffix on a test domain evaluates on that domain's TRAIN
        # env (PDDLEnv<name>-v0: the easy problems) instead of the Test env
        # (hard, size-generalization problems). For held-out domains those
        # easy problems are unseen, so this is the zero-shot easy split.
        eval_plan = []  # [(name, count, is_zero_shot, split), ...]
        if args.test_domains:
            for name, count, _ in parse_domain_arg(args.test_domains, "",
                                                    args.num_test_problems):
                split = 'test'
                if name.endswith('@train'):
                    name = name[:-len('@train')]
                    split = 'train'
                is_zs = name not in _train_set
                eval_plan.append((name, count, is_zs, split))
        else:
            _default_domains = train_domain_names if train_domain_names else [args.domain]
            for name in _default_domains:
                eval_plan.append((name, args.num_test_problems, False, 'test'))
            if args.heldout_domains:
                for name, count, _ in parse_domain_arg("", args.heldout_domains,
                                                        args.num_test_problems):
                    eval_plan.append((name, count, True, 'test'))

        _valid_metrics = {'validation', 'training', 'combined'}
        all_model_types = [m.strip() for m in args.test_model_metrics.split(',')
                           if m.strip()]
        _bad = set(all_model_types) - _valid_metrics
        if _bad:
            raise ValueError(f"Invalid --test-model-metrics entries: {sorted(_bad)} "
                             f"(valid: {sorted(_valid_metrics)})")
        num_models_to_test = args.num_models_to_test
        starting_model_num = 0
        all_results = {}

        for test_domain, requested_count, is_zero_shot, test_split in eval_plan:
            tag = "zero-shot" if is_zero_shot else "in-domain"
            display_name = (test_domain if test_split == 'test'
                            else f"{test_domain}@train")
            print(f"\n=== {tag} evaluation: {display_name} ===")

            # Cap against the problem count of the env that will actually be
            # evaluated ('test' -> PDDLEnv<name>Test-v0; 'train' -> the base
            # env with the easy problems).
            _env_suffix = "Test" if test_split == 'test' else ""
            try:
                t_env = pddlgym.make(f"PDDLEnv{test_domain}{_env_suffix}-v0")
            except Exception:
                t_env = pddlgym.make(f"PDDLEnv{test_domain}-v0")
            domain_test_count = min(requested_count, len(t_env.problems))
            if domain_test_count < requested_count:
                print(f"  {display_name}: capping test at {domain_test_count} "
                      f"(requested {requested_count})")

            # The domain's own action space. Used for (a) structural test
            # metadata and (b) overriding the decoder's arity lookup: beam
            # termination indexes action_parameter_number_dict by GRAPH-LOCAL
            # schema position, but the dict is built from the action space the
            # model was initialized with (merged, in multi-domain runs), so
            # any domain after the first maps schemas to the wrong arities and
            # decodes malformed actions. Only this dict is swapped; the
            # schema-embedding offset is graph-local (per-graph n_action) so
            # it needs no per-domain handling. Checkpoints trained before the
            # graph-local offset fix need GABAR_LEGACY_ACTION_OFFSET=1.
            test_action_space = t_env.action_space._action_predicate_to_operators

            # Build per-domain graph metadata for testing. Structural/lifted
            # modes rebuild from the test domain's own action space at the
            # canonical training arities, so any domain (zero-shot included)
            # featurizes at the trained widths.
            _feat = graph_metadata.get('featurization')
            _cheat_cols = ('node_feature_to_index' in graph_metadata and
                           'is_correct_action' in graph_metadata['node_feature_to_index'])
            if _feat == 'structural':
                from ploi.structural import build_structural_metadata
                test_md = build_structural_metadata(
                    test_action_space,
                    graph_metadata['max_pred_arity'],
                    graph_metadata['max_action_arity'],
                    cheating=_cheat_cols)
            elif _feat in ('joint', 'joint_lite', 'joint_chain'):
                from ploi.lifted_layer import build_lifted_metadata
                test_md = build_lifted_metadata(
                    test_action_space,
                    graph_metadata['max_pred_arity'],
                    graph_metadata['max_action_arity'],
                    _feat,
                    cheating=_cheat_cols)
            elif is_zero_shot:
                test_md = dict(graph_metadata)
                test_md['allow_unknown_symbols'] = True
            else:
                test_md = graph_metadata

            test_hypers = {**testing_hyperparameters, 'domain_name': display_name}
            config = PlannerConfig(
                planner_types=planner_types,
                domain_name=test_domain,
                num_problems=domain_test_count,
                timeout=30.0,
                enable_state_monitor=args.monitor,
                max_plan_length=args.max_plan_length,
                problems_per_division=args.problems_per_division,
                eval_planner_name=args.eval_planner_name,
                train_planner_name=args.train_planner_name,
                model_hyperparameters=training_hyperparameters,
                ignore_defaults=ignore_defaults,
                testing_hyperparameters=test_hypers,
                learned_search_strat=learned_search_strat,
                test_split=test_split,
            )

            tester = PlannerTester(config)
            problems_to_solve = list(range(args.starting_test_number,
                                           args.starting_test_number + domain_test_count))

            def test_function_v2(curr_models, _tester=tester,
                                 _problems=problems_to_solve, _md=test_md):
                return _tester.test_planners(problems_to_solve=_problems,
                                             models=curr_models,
                                             graph_metadata=_md)

            curr_test_function = test_function_v2
            tested_epoch_numbers = set()

            # Zero-shot domains test one metric only (validation if selected)
            if is_zero_shot:
                _model_types = (['validation'] if 'validation' in all_model_types
                                else all_model_types[:1])
            else:
                _model_types = all_model_types

            for model_type in _model_types:
                results = run_tests(
                    curr_manager=manager,
                    model_class=model_class,
                    train_env_name=train_env_name,
                    seed=42,
                    hyperparameters=training_hyperparameters,
                    test_function=curr_test_function,
                    metric=model_type,
                    args=args,
                    action_space=action_space,
                    decode_action_space=test_action_space,
                    tested_epoch_numbers=tested_epoch_numbers,
                    num_models_to_test=num_models_to_test,
                    starting_model_num=starting_model_num,
                    planner_types=planner_types,
                    baseline_models=baseline_models if not is_zero_shot else {},
                    ignore_defaults=ignore_defaults,
                )
                prefix = "zeroshot_" if is_zero_shot else ""
                suffix = f"_{model_type}" if len(eval_plan) > 1 else model_type
                all_results[f"{prefix}{display_name}{suffix}"] = results

        # Structured results dump for cross-run aggregation
        # (tools/analyze_results.py). One JSON per invocation, keyed by
        # <domain>_<metric>; wandb/stdout logging is unchanged.
        import dataclasses
        _summary = {}
        for _key, _results in all_results.items():
            _summary[_key] = []
            for _r in _results:
                _entry = {
                    'epoch': _r['epoch'],
                    'validation_loss': _r.get('validation_loss'),
                    'training_loss': _r.get('training_loss'),
                    'metrics': {}
                }
                for _ptype, _m in _r['test_results'].items():
                    _entry['metrics'][str(_ptype)] = dataclasses.asdict(_m)
                _summary[_key].append(_entry)
        _dump = {
            'experiment': args.expid,
            'train_domain': args.domain,
            'featurization': args.featurization,
            'test_model_metrics': all_model_types,
            'num_models_to_test': num_models_to_test,
            'eval_plan': [
                {'domain': (d if s == 'test' else f"{d}@train"),
                 'count': c, 'zero_shot': z, 'split': s}
                for d, c, z, s in eval_plan],
            'results': _summary,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        }
        _dump_path = os.path.join(
            args.expdir, f"results_{args.domain}_{_dump['timestamp']}.json")
        with open(_dump_path, 'w') as _f:
            json.dump(_dump, _f, indent=2, default=str)
        print(f"\nResults written to {_dump_path}")

        _ = log_model_metrics(all_results,args)
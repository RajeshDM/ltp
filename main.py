
import argparse
import json
import os
import pickle
import warnings
import torch
import wandb
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pyg_dataloader
from ploi.model_checkpointing import ModelManager 
from pathlib import Path
from typing import Dict, List, Optional, Set
from torch_geometric.data import Data
from ploi.argparsers import get_ploi_argument_parser
from ploi.datautils import (
    collect_training_data,
    create_graph_dataset,
    create_graph_dataset_hierarchical,
    GraphDictDataset,
)
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ploi.datautils_ltp import (
    _collect_training_data_ltp,
    _create_graph_dataset_ltp,
    _state_to_graph_ltp,
    get_filenames,
    TorchGraphDictDataset,
    graph_dataset_to_pyg_dataset,
)
from ploi.run_planner_with_ltp_v1 import (
    run_planner_with_gnn_ltp,
    _create_planner,
)
#from ploi.run_planner_with_ltp_2 import PlannerTester, PlannerConfig, PlannerType
from ploi.test_utils import (
    PlannerConfig, PlannerType, PlanningResult, PlannerMetrics,
    compute_metrics,
    compute_combined_metrics,
)
from ploi.run_planner_with_ltp_v2 import PlannerTester, PlannerConfig, PlannerType
from ploi.test_utils import format_metrics, log_model_metrics 
from ploi.guiders import HierarchicalGuidance, PLOIGuidance, SceneGraphGuidance
from ploi.modelutils import (
    GraphNetwork,
)
from ploi.modelutils_ltp import (
    GraphNetworkLtp,
    GNN_GRU,
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
    train_model_hierarchical,
)
from ploi.baselines.utils import load_checkpoint 

#import ploi.constants as constants
from icecream import ic

baselines = [PlannerType.EXP_BASELINE] 


def set_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    if args.server == True:
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        #np.random.seed(seed)
        #random.seed(seed)
        torch.cuda.manual_seed_all(seed)

def initialize_model(model_class, args, action_space):
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
    )


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
             baseline_models = None
             ) -> List[Dict]:
    """
    Run tests on best models for a specific configuration
    """
    
    # Get best models for this configuration
    best_models = curr_manager.load_best_models(
        train_env_name=train_env_name,
        seed=seed,
        hyperparameters=hyperparameters,
        metric=metric,
        #device=device
    )
    
    if not best_models:
        logger.warning("No models found to test")
        return []
    
    results = []
    curr_model = None
    if PlannerType.LEARNED_MODEL in planner_types : 
        for model_info in best_models[::-1][starting_model_num:starting_model_num+num_models_to_test]:
            # Create fresh model instance
            #model = model_class()
            curr_models = {}
            curr_model = initialize_model(model_class, args, action_space)
            
            # Load model state
            curr_model.load_state_dict(model_info['state_dict'])
            curr_model.to(device)
            curr_model.eval()

            if model_info['epoch'] in tested_epoch_numbers:
                print ("Already tested model from epoch ",model_info['epoch'])
                continue 
            else :
                tested_epoch_numbers.add(model_info['epoch'])

            if args.epoch_number != -1 : 
                if model_info['epoch'] != args.epoch_number :
                    continue

            curr_models[PlannerType.LEARNED_MODEL] = (curr_model,model_info['epoch'])

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
            metrics = results[-1]['test_results'][PlannerType.LEARNED_MODEL]
            print ("failed : ",metrics.failures)
            _ = format_metrics(results[-1]['test_results'][PlannerType.LEARNED_MODEL], model_info['epoch'])
            #print (test_results[PlannerType.LEARNED_MODEL][-1].plan)

            #if PlannerType.NON_OPTIMAL in planner_types : 
            combnined_metrics = compute_combined_metrics(results[-1]['all_plan_results'], PlannerType.LEARNED_MODEL)
            #print (f"Combined Metrics for {model_type} : ", combnined_metrics)
            print ("Plan Quality : ", combnined_metrics.plan_quality)

    for baseline in baselines:
        curr_models = {}
        if baseline in planner_types:
            curr_models[baseline],_ = (load_checkpoint(baseline_models[baseline], device),-1)
            test_results, run_metrics = test_function(curr_models)
            combnined_metrics = compute_combined_metrics(test_results, baseline)
            #print (f"Combined Metrics for {model_type} : ", combnined_metrics)
            print ("Baseline : ",baseline)
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
            })

    mode = args.mode
    if args.data_augmentation is True :
        args.batch_size = 64
    if mode == "debug" :
        args.representation_size = 4
        args.batch_size = 1

    # Seed RNG
    set_seed(args)
    #torch.manual_seed(args.seed,args)

    # Create dir to log files to
    args.expdir = os.path.join(args.logdir, args.expid)
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir, exist_ok=True)

    # Capitalize the first letter of the domain name
    args.domain = args.domain.capitalize()

    # This datafile is the same for ploi and hierarchical variants
    args.datafile = os.path.join(args.logdir, f"ploi_{args.domain}.pkl")
    if args.domain.endswith("scrub"):
        args.datafile = os.path.join(args.logdir, f"ploi_{args.domain[:-5]}.pkl")

    print(f"Domain: {args.domain}")
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
    if not os.path.exists(args.datafile) or args.force_collect_data:
        if args.method == 'ltp' :
            args.datafile = _dataset_file_prefix + "_{}.pkl".format(args.domain)
            training_data,_,_ = _collect_training_data_ltp(
                args.domain, train_planner, _num_train_problems=args.num_train_problems,
                outfile=args.datafile,args=args)
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

    graphs_inp, graphs_tgt, graph_metadata = None, None, None
    if args.method in ["hierarchical"]:
        graphs_inp, graphs_tgt, graph_metadata = create_graph_dataset_hierarchical(
            training_data
        )
    elif args.method in ["ltp"]:
        graphs_inp, graphs_tgt, graph_metadata = _create_graph_dataset_ltp(training_data,args=args)
    else:
        graphs_inp, graphs_tgt, graph_metadata = create_graph_dataset(training_data)

    # Use 10% for validation
    num_validation = max(1, int(len(graphs_inp) * 0.1))
    train_graphs_input = graphs_inp[num_validation:]
    train_graphs_target = graphs_tgt[num_validation:]
    valid_graphs_input = graphs_inp[:num_validation]
    valid_graphs_target = graphs_tgt[:num_validation]

    pyg = args.pyg
    batch_size = args.batch_size

    args.num_node_features_object = train_graphs_input[0]['nodes'][0].shape[-1]
    args.num_edge_features_object = train_graphs_input[0]['edges'][0].shape[-1]

    args.num_node_features = args.num_node_features_object
    args.num_edge_features = args.num_edge_features_object
    if 'globals' in train_graphs_input[0]:
        args.num_global_features = train_graphs_input[0]['globals'][0].shape[-1]

    if pyg == False:
        # Set up dataloaders
        graph_dataset = GraphDictDataset(train_graphs_input, train_graphs_target)
        graph_dataset_val = GraphDictDataset(valid_graphs_input, valid_graphs_target)

    else :
        print ("Size of dataset : ",len(graphs_inp))
        train_graphs_pyg = graph_dataset_to_pyg_dataset(train_graphs_input)
        #train_graphs_target_pyg = graph_dataset_to_pyg_dataset(train_graphs_target)

        val_graphs_pyg = graph_dataset_to_pyg_dataset(valid_graphs_input)
        #val_graphs_target_pyg = graph_dataset_to_pyg_dataset(valid_graphs_target)

        graph_dataset = pyg_dataloader(train_graphs_pyg, batch_size=batch_size,shuffle=True)
        graph_dataset_val = pyg_dataloader(val_graphs_pyg,batch_size=batch_size,shuffle=True)

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

    elif args.method == 'ltp' :
        ic ("LTP start")
        representation_size = args.representation_size
        gnn_rounds = args.gnn_rounds
        n_heads = args.n_heads
        action_space = training_data[3]

        _model = initialize_model(GNN_GRU, args, action_space)

        training_hyperparameters = {
            'lr': args.lr,
            'gnn_rounds': args.gnn_rounds,
            'd' : args.num_train_problems,
            'ad' : args.attention_dropout,
            'wd' : args.dropout,
            'heads' : args.n_heads,
            'g_node' : args.use_global_node,
        }

        continue_training = args.continue_training
        train_env_name = args.domain
        save_model_prefix=os.path.join(
            model_dir, "bce10_model_seed{}".format(args.seed)),
        dataset_size = len(training_data[0])
        save_folder = os.path.join(Path.cwd(),"models")
        manager = ModelManager(save_folder, hyperparameters=training_hyperparameters,
                               train_env_name=train_env_name,seed=args.seed)

        model_outfile, message_string,save_folder = get_filenames(dataset_size,train_env_name,
                                                        args.epochs,args.model_version,
                                                        representation_size,
                                                        save_model_prefix,args.seed,
                                                        args)
        
        #if args.mode == 'train' and (not os.path.exists(model_outfile) or continue_training == True):
        if args.mode == 'train'  or args.mode == 'train_test' :
            optimizer = torch.optim.Adam(_model.parameters(),lr=args.lr,weight_decay=args.weight_decay) 
            #optimizer = torch.optim.AdamW(self._model.parameters(), lr=5 * 1e-4,weight_decay=0.01)
            if continue_training == True and os.path.exists(model_outfile) :
                _model_state = torch.load(model_outfile)
                _model.load_state_dict(_model_state['state_dict'])
                optimizer.load_state_dict(_model_state['optimizer'])
                _starting_epoch = _model_state['epochs'] + 1

            pos_weight = args.pos_weight * torch.ones([1])
            #pos_weight = self._bce_pos_weight*torch.ones([1])
            criterion = torch.nn.CrossEntropyLoss()
            #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # Train model
            #model_dict = train_model_graphnetwork_ltp_batch(_model, 
            train_model_graphnetwork_ltp_batch(_model, 
                                    datasets,
                                    #dataloaders,
                                    criterion=criterion, optimizer=optimizer,
                                    use_gpu=args.use_gpu,
                                    starting_epoch = args.starting_epoch,
                                    save_folder = save_folder,
                                    final_epoch= args.epochs,
                                    train_env_name=train_env_name,seed=args.seed,
                                    message_string=message_string,
                                    log_wandb=args.wandb,
                                    chpkt_manager=manager,)
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
            baseline_models[PlannerType.EXP_BASELINE] = file

        #ONLY DONE FOR FINALY DAY TESTING - REMOVE LATER
        #planner_types = [PlannerType.LEARNED_MODEL,PlannerType.NON_OPTIMAL]

        config = PlannerConfig(
            #planner_types=[PlannerType.NON_OPTIMAL],
            #planner_types=[PlannerType.LEARNED_MODEL],
            #planner_types=[PlannerType.LEARNED_MODEL, PlannerType.NON_OPTIMAL],
            planner_types=planner_types,
            domain_name=args.domain , 
            num_problems=args.num_test_problems,
            timeout=30.0,
            enable_state_monitor=args.monitor,  # Enable monitoring
            max_plan_length=args.max_plan_length,
            problems_per_division=args.problems_per_division,
            eval_planner_name = args.eval_planner_name,
            train_planner_name = args.train_planner_name,
            model_hyperparameters = training_hyperparameters,
        )

        tester = PlannerTester(config)
        problems_to_solve = list(range(args.starting_test_number, args.starting_test_number + args.num_test_problems))

        def test_function_v2(curr_models):
            return tester.test_planners(problems_to_solve=problems_to_solve,
                                        models=curr_models, 
                                        graph_metadata=graph_metadata)

        #all_model_types = ['validation','training','combined']
        #all_model_types = ['validation','training']
        #all_model_types = ['validation']#,'training']
        all_model_types = ['training' ]

        #curr_test_function = test_function
        curr_test_function = test_function_v2
        num_models_to_test = 2
        starting_model_num = 0

        def run_tests_model_type(model_type, tested_epoch_numbers):
            return run_tests(
                curr_manager=manager,
                model_class=GNN_GRU,
                train_env_name=train_env_name,
                seed=42,
                hyperparameters=training_hyperparameters,
                test_function=curr_test_function,
                metric=model_type,  # or 'training' or 'combined',
                args=args,
                action_space=action_space,
                tested_epoch_numbers=tested_epoch_numbers,
                num_models_to_test=num_models_to_test,
                starting_model_num=starting_model_num,
                planner_types=planner_types,
                baseline_models=baseline_models,
            )

        tested_epoch_numbers = set()

        all_results = {}
        for model_type in all_model_types:
            results = run_tests_model_type(model_type, tested_epoch_numbers)
            all_results[model_type] = results

        # Log all results and get best model info
        best_model_type, best_epoch, best_success_rate = log_model_metrics(all_results, args)

        '''
        curr_model = None
        if PlannerType.NON_OPTIMAL in planner_types : 
            test_results, run_metrics = curr_test_function(curr_model)
            _ = format_metrics(run_metrics[PlannerType.NON_OPTIMAL],"NON OPTIMAL") 
        if PlannerType.OPTIMAL in planner_types : 
            test_results, run_metrics = curr_test_function(curr_model)
            _ = format_metrics(run_metrics[PlannerType.OPTIMAL],"NON OPTIMAL") 
        '''

        # Print best model info
        if best_model_type is not None:
            print(f"\nBest Model (With Monitor):")
            print(f"Type: {best_model_type}")
            print(f"Epoch: {best_epoch}")
            print(f"Success Rate: {best_success_rate:.2%}")
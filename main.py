
import argparse
import json
import os
import pickle
import warnings
import torch
import wandb
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pyg_dataloader
from torch_geometric.data import Data
from ploi.argparsers import get_ploi_argument_parser
from ploi.datautils import (
    collect_training_data,
    create_graph_dataset,
    create_graph_dataset_hierarchical,
    GraphDictDataset,
)
from ploi.datautils_ltp import (
    _collect_training_data_ltp,
    _create_graph_dataset_ltp,
    _state_to_graph_ltp,
    get_filenames,
    TorchGraphDictDataset,
    graph_dataset_to_pyg_dataset,
)
from ploi.run_planner_with_ltp import (
    run_planner_with_gnn
)
from ploi.guiders import HierarchicalGuidance, PLOIGuidance, SceneGraphGuidance
from ploi.modelutils import (
    GraphNetwork,
)
from ploi.modelutils_ltp import (
    GraphNetworkLtp,
    GNN_GRU,
)
from ploi.planning import FD, IncrementalPlanner
from ploi.planning.incremental_hierarchical_planner import (
    IncrementalHierarchicalPlanner,
)
from ploi.planning.scenegraph_planner import SceneGraphPlanner
from ploi.traineval import (
    test_planner,
    #test_planner_ltp,
    train_model_graphnetwork,
    train_model_graphnetwork_ltp,
    train_model_graphnetwork_ltp_batch,
    train_model_hierarchical,
)

import ploi.constants as constants
from icecream import ic

def _create_planner(planner_name):
    if planner_name == "fd-lama-first":
        return FD(alias_flag="--alias lama-first")
    if planner_name == "fd-opt-lmcut":
        return FD(alias_flag="--alias seq-opt-lmcut")
    raise ValueError(f"Uncrecognized planner name {planner_name}")

if __name__ == "__main__":

    parser = get_ploi_argument_parser()

    parser.add_argument(
        "--all-problems",
        action="store_true",
        help="Run testing on all problems in domain",
    )
    args = parser.parse_args()

    run = wandb.init(
        # Set the project where this run will be logged
        project="ltp_gnn_gru_pyg",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": constants.learning_rate,
            "epochs": args.epochs,
            "batch_size": constants.batch_size,
            "representation_size": constants.representation_size,
            "gnn_rounds": constants.gnn_rounds,
            "decoder_layers": constants.gru_layers,
            "seed": constants.seed,
            "domain": args.domain,
            "model_version": args.model_version,
            "server": constants.server,
            "continue_training": constants.continue_training,
            "starting_epoch": constants.starting_epoch,
            "train_planner_name": args.train_planner_name,
            "num_train_problems": args.num_train_problems,
            "num_test_problems": args.num_test_problems,
            "expid": args.expid,
            "gru_layers" : constants.gru_layers,
        })


    # Seed RNG
    torch.manual_seed(args.seed)

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
                outfile=args.datafile)
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
        graphs_inp, graphs_tgt, graph_metadata = _create_graph_dataset_ltp(training_data)
    else:
        graphs_inp, graphs_tgt, graph_metadata = create_graph_dataset(training_data)

    # Use 10% for validation
    num_validation = max(1, int(len(graphs_inp) * 0.1))
    train_graphs_input = graphs_inp[num_validation:]
    train_graphs_target = graphs_tgt[num_validation:]
    valid_graphs_input = graphs_inp[:num_validation]
    valid_graphs_target = graphs_tgt[:num_validation]

    pyg = constants.pyg
    batch_size = constants.batch_size

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
        ic ("training start")
        #args.num_node_features = datasets["train"][0]["graph_input"]["nodes"].shape[-1]
        #args.num_edge_features = datasets["train"][0]["graph_input"]["edges"].shape[-1]
        #args.num_global_features = datasets["train"][0]["graph_input"]["globals"].shape[-1]

        args.decoder_layers = constants.gru_layers

        representation_size = constants.representation_size
        gnn_rounds = constants.gnn_rounds

        _model = GNN_GRU(
            n_features=args.num_node_features,
            n_edge_features=args.num_edge_features,
            n_global_features = args.num_global_features,
            n_hidden=representation_size,
            gnn_rounds= gnn_rounds,
            num_decoder_layers = args.decoder_layers ,
            dropout = 0,
            action_space= training_data[3],
            batch_size=batch_size
        )


        continue_training = constants.continue_training
        train_env_name = args.domain
        save_model_prefix=os.path.join(
            model_dir, "bce10_model_seed{}".format(constants.seed)),
        dataset_size = len(training_data[0])

        model_outfile, message_string,save_folder = get_filenames(dataset_size,train_env_name,
                                                        args.epochs,args.model_version,
                                                        representation_size,
                                                        save_model_prefix,constants.seed)
        

        if not os.path.exists(model_outfile) or continue_training == True:
            optimizer = torch.optim.Adam(_model.parameters(),lr=constants.learning_rate) 
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
            model_dict = train_model_graphnetwork_ltp_batch(_model, 
                                    datasets,
                                    #dataloaders,
                                    criterion=criterion, optimizer=optimizer,
                                    use_gpu=constants.use_gpu,
                                    starting_epoch = constants.starting_epoch,
                                    save_folder = save_folder,
                                    final_epoch= args.epochs,
                                    train_env_name=train_env_name,seed=constants.seed,
                                    message_string=message_string)
            #torch.save(model_dict, model_outfile)
            state_save = {'state_dict': model_dict,
            'optimizer': optimizer.state_dict(),
            'epochs': args.epochs}
            torch.save(state_save,model_outfile)
            _model_state = torch.load(model_outfile)
            _model.load_state_dict(_model_state['state_dict'])
            #self._model_state.load_state_dict(model_dict)
            #self._model = self._model_state['state_dict']
            print("Saved model to {}.".format(model_outfile))
        else:
            #ic (model_outfile)
            if constants.server == False:
                _model_state = torch.load(model_outfile,map_location="cpu")
            else :
                _model_state = torch.load(model_outfile,map_location="cuda:0")

            _model.load_state_dict(_model_state['state_dict'])
            #print("Loaded saved model from {}.".format(model_outfile))

        #run_planner_with_gnn(_model, args.domain, args.num_test_problems, args.timeout)

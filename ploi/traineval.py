import copy
import os
import time
import warnings
import wandb

import numpy as np
from torch._C import device
import pddlgym
import torch
import torch.nn as nn

import ploi.constants as constants
import matplotlib.pyplot as plt
from icecream import ic
from torchviz import make_dot

import os
import sys

from .planning import PlanningFailure, PlanningTimeout, validate_strips_plan

def save_model_graphnetwork(model, save_folder, epoch, optimizer,train_env_name,
                             seed, message_string, best_seen_running_validation_loss,
                             running_loss, best_seen_model_weights, best_validation_loss_epoch,
                             time_taken_for_save_iter):
    save_path = os.path.join(save_folder, str(train_env_name) + "_seed" + str(seed) + "_model" + str(
        epoch) + "_" + message_string + ".pt")
    state_save = {'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    'epochs':epoch}
    torch.save(state_save,save_path)
    if running_loss['val'] < best_seen_running_validation_loss:
        best_seen_running_validation_loss = running_loss['val']
        best_seen_model_weights = model.state_dict()
        #print("Found new best model with validation loss {} at epoch {}".format(
        #    best_seen_running_validation_loss, epoch), flush=True)
        best_validation_loss_epoch = epoch
        #print ()
        print("Saved model checkpoint {}, Time : {} , (New Best)".format(save_path,time.time()-time_taken_for_save_iter))
    else :
        print("Saved model checkpoint {}, Time : {}".format(save_path,time.time()-time_taken_for_save_iter))

    return best_seen_running_validation_loss,best_validation_loss_epoch, best_seen_model_weights

def train_model_graphnetwork_ltp_batch(model, datasets,
                                 #dataloaders,
                                   criterion, optimizer, use_gpu, print_iter=10, 
                save_iter=100, save_folder='/tmp',starting_epoch=0, final_epoch=1000, global_criterion=None,
                return_last_model_weights=True,dagger_train=False,train_env_name=None,seed=None,
                message_string='',
                log_wandb=False,
                chpkt_manager=None):

    since = time.time()
    min_save_epoch = 0
    print_iter = 10
    save_iter = print_iter
    if use_gpu:
        model = model.cuda()
        device = "cuda:0"
        if criterion is not None:
            criterion = criterion.cuda()
    else:
        device = "cpu"

    epochs = []
    train_loss_values = []
    val_loss_values = []
    time_taken_for_save_iter = time.time()
    for epoch in range(starting_epoch,final_epoch+1):
        if epoch % print_iter == 0:
            print('Epoch {}/{}'.format(epoch, final_epoch), flush=True)
            print('-' * 10, flush=True)
        # Each epoch has a training and validation phase
        running_num_samples = 0
        if epoch % print_iter == 0 :
            phases = ['train','val']
        else:
            phases = ['train']

        running_loss = {'train':0.0,'val':0.0}

        for phase in phases:
            if phase == 'train':
                # Set model to training mode
                model.train()  
            else:
                # Set model to evaluate mode
                model.eval()

            for i,batch_data in enumerate(datasets[phase]):
                optimizer.zero_grad()
                batch_data = batch_data.to(device)
                action_scores, action_object_scores = model(batch_data, beam_search=False)
                tgt_action_scores = batch_data['target_action_scores'].x
                tgt_action_object_scores = batch_data['target_action_object_scores'].x
                tgt_params = batch_data['target_n_parameters'].x
                loss = 0.
                curr_param_counter = 0
                required_action_object_scores = []
                total_number_params = 0

                for idx,n_params in enumerate(tgt_params):
                    #ic (output['action_object_scores'][0:2])
                    #ic (curr_param_counter)
                    n_params = int(n_params)
                    #ic (n_params)
                    for correct_index in range(curr_param_counter,curr_param_counter+n_params):
                        required_action_object_scores.append(correct_index)
                    #loss += criterion(output['action_object_scores'][curr_param_counter:curr_param_counter+n_params], targets['action_object_scores'][curr_param_counter:curr_param_counter+n_params])
                    #TODO ADD an assert here to check if any of the elements in target are all zeroes
                    #ic (output['action_object_scores'][curr_param_counter:curr_param_counter+n_params])
                    #ic (targets['action_object_scores'][curr_param_counter:curr_param_counter+n_params])
                    #curr_param_counter += 2
                    curr_param_counter += model.max_number_action_parameters
                    total_number_params += n_params

                required_action_object_scores = torch.tensor(required_action_object_scores)
                target_indices = tgt_action_scores.argmax(dim=1)
                target_indices_2 = tgt_action_object_scores[required_action_object_scores].argmax(dim=1)
                tgt_action_scores = tgt_action_scores.squeeze(0)

                m = torch.nn.ConstantPad2d((0,tgt_action_object_scores.shape[1]-action_object_scores.shape[1]\
                                            ,0,0),0)
                
                action_object_scores = m(action_object_scores)
                #loss += criterion(output['action_object_scores'][required_action_object_scores],targets['action_object_scores'][required_action_object_scores])/division_coeff
                loss += criterion(action_scores,target_indices)
                loss += criterion(action_object_scores[required_action_object_scores],target_indices_2)

                if phase == 'train':
                    backward_time = time.time()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    #ic ("backprop time",time.time()-backward_time)

                # statistics
                running_loss[phase] += loss.item()
                running_num_samples += 1
            if log_wandb:
                wandb.log({f"loss_{phase}": running_loss[phase]})

        if epoch % print_iter == 0:
            print("running_loss:", running_loss, flush=True)
            epochs.append(epoch)
            train_loss_values.append(running_loss['train'])
            val_loss_values.append(running_loss['val'])
    
        if epoch % save_iter == 0 and epoch >= min_save_epoch:
            chpkt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_env_name=train_env_name,
                seed=42,
                losses={'train': running_loss["train"], 'val': running_loss["val"]},
            )
            print ("Time taken for {} epochs : {}".format(save_iter, time.time() - time_taken_for_save_iter))
            time_taken_for_save_iter = time.time()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), flush=True)

def train_model_graphnetwork(
    model,
    datasets,
    criterion,
    optimizer,
    use_gpu=False,
    print_every=10,
    save_every=100,
    save_folder="/tmp",
    epochs=1000,
    global_criterion=None,
    return_last_model_weights=True,
):
    since = time.time()
    best_seen_model_weights = None  # as measured over the validation set
    best_seen_running_validation_loss = np.inf

    trainset, validset = datasets["train"], datasets["val"]

    if use_gpu:
        device = "cuda:0"
    else:
        device = "cpu"

    for e in range(epochs):

        running_loss = 0.0
        running_num_samples = 0

        model.train()

        for idx in range(len(trainset)):
            g_inp = trainset[idx]["graph_input"]
            g_tgt = trainset[idx]["graph_target"]
            nfeat = torch.from_numpy(g_inp["nodes"]).float().to(device)
            efeat = torch.from_numpy(g_inp["edges"]).float().to(device)
            senders = torch.from_numpy(g_inp["senders"]).long().to(device)
            receivers = torch.from_numpy(g_inp["receivers"]).long().to(device)
            tgt = torch.from_numpy(g_tgt["nodes"]).float().to(device)
            edge_indices = torch.stack((senders, receivers))
            preds = model(nfeat, edge_indices, efeat)
            loss = criterion(preds, tgt)

            running_loss += loss.item()
            running_num_samples += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(
            f"== [EPOCH {e:03d} / {epochs}] Train loss: {(running_loss / running_num_samples):03.5f}"
        )

        if e % 100 == 0:

            model.eval()

            if e % save_every == 0:
                savefile = os.path.join(save_folder, f"model_{e:04d}.pt")
                torch.save(model.state_dict(), savefile)
                print(f"Saved model checkpoint {savefile}")

            running_loss = 0.0
            running_num_samples = 0

            for idx in range(len(validset)):
                g_inp = validset[idx]["graph_input"]
                g_tgt = validset[idx]["graph_target"]
                nfeat = torch.from_numpy(g_inp["nodes"]).float().to(device)
                efeat = torch.from_numpy(g_inp["edges"]).float().to(device)
                senders = torch.from_numpy(g_inp["senders"]).long().to(device)
                receivers = torch.from_numpy(g_inp["receivers"]).long().to(device)
                tgt = torch.from_numpy(g_tgt["nodes"]).float().to(device)
                edge_indices = torch.stack((senders, receivers))
                preds = model(nfeat, edge_indices, efeat)
                loss = criterion(preds, tgt)

                running_loss += loss.item()
                running_num_samples += 1

            print(
                f"===== [EPOCH {e:03d} / {epochs}] Val loss: {(running_loss / running_num_samples):03.5f}"
            )

            val_loss = running_loss / running_num_samples
            if val_loss < best_seen_running_validation_loss:
                best_seen_running_validation_loss = copy.deepcopy(val_loss)
                best_seen_model_weights = model.state_dict()
                savefile = os.path.join(save_folder, "best.pt")
                torch.save(best_seen_model_weights, savefile)
                print(
                    f"Found new best model with val loss {best_seen_running_validation_loss} at epoch {e}. Saved!"
                )

    time_elapsed = time.time() - since
    print(
        f"Training complete in {(time_elapsed // 60):.0f} m {(time_elapsed % 60):.0f} sec"
    )

    return best_seen_model_weights

def predict_graph_with_graphnetwork(model, input_graph):
    """Predict the target graph given the input graph"""
    model.eval()
    nfeat = torch.from_numpy(input_graph["nodes"]).float()
    efeat = torch.from_numpy(input_graph["edges"]).float()
    senders = torch.from_numpy(input_graph["senders"]).long()
    receivers = torch.from_numpy(input_graph["receivers"]).long()
    edge_indices = torch.stack((senders, receivers))
    scores = model(nfeat, edge_indices, efeat)
    scores = torch.sigmoid(scores)
    input_graph["nodes"] = scores.detach().cpu().numpy()
    return input_graph

def test_planner(
    planner, domain_name, num_problems, timeout, debug_mode=False, all_problems=False
):
    print("Running testing...")
    # In debug mode, use train problems for testing too (False by default)
    env = pddlgym.make("PDDLEnv{}Test-v0".format(domain_name))
    if debug_mode:
        warnings.warn(
            "WARNING: Running in debug mode (i.e., testing on train problems)"
        )
        env = pddlgym.make("PDDLEnv{}-v0".format(domain_name))
    num_problems = min(num_problems, len(env.problems))
    # If `all_problems` is set to True, override num_problems
    if all_problems:
        num_problems = len(env.problems)
    stats_to_log = ["num_node_expansions", "plan_length", "search_time", "total_time"]
    num_timeouts = 0
    num_failures = 0
    num_invalidated_plans = 0
    run_stats = []
    for problem_idx in range(num_problems):
        print(
            "\tTesting problem {} of {}".format(problem_idx + 1, num_problems),
            flush=True,
        )
        env.fix_problem_index(problem_idx)
        state, _ = env.reset()
        start = time.time()
        try:
            plan, planner_stats = planner(
                env.domain, state, timeout=timeout, 
                #domain_file_global=None# 
                domain_file_global=env._domain_file
            )
        except PlanningFailure as e:
            num_failures += 1
            print("\t\tPlanning failed with error: {}".format(e), flush=True)
            continue
        except PlanningTimeout as e:
            num_timeouts += 1
            print("\t\tPlanning failed with error: {}".format(e), flush=True)
            continue
        # Validate plan on the full test problem.
        if plan is None:
            num_failures += 1
            continue
        if not validate_strips_plan(
            domain_file=env.domain.domain_fname,
            problem_file=env.problems[problem_idx].problem_fname,
            plan=plan,
        ):
            print("\t\tPlanning returned an invalid plan")
            num_invalidated_plans += 1
            continue
        wall_time = time.time() - start
        print(
            "\t\tSuccess, got plan of length {} in {:.5f} seconds".format(
                len(plan), wall_time
            ),
            flush=True,
        )
        planner_stats["wall_time"] = wall_time
        run_stats.append(planner_stats)

    global_stats = dict()
    stats_to_track = {
        "num_node_expansions",
        "plan_length",
        "search_time",
        "total_time",
        "objects_used",
        "objects_total",
        "neural_net_time",
        "wall_time",
    }
    num_stats = len(run_stats)
    for stat in stats_to_track:
        if stat not in global_stats:
            global_stats[stat] = np.zeros(num_stats)
        for i, run in enumerate(run_stats):
            global_stats[stat][i] = run[stat]
    for stat in stats_to_track:
        stat_mean = float(global_stats[stat].mean().item())
        stat_std = float(global_stats[stat].std().item())
        global_stats[stat] = stat_mean
        global_stats[f"{stat}_std"] = stat_std
    global_stats["success_rate"] = float(num_stats / num_problems)
    global_stats["timeout_rate"] = float(num_timeouts / num_problems)
    global_stats["failure_rate"] = float(num_failures / num_problems)
    global_stats["invalid_rate"] = float(num_invalidated_plans / num_problems)

    global_stats["num_timeouts"] = num_timeouts
    global_stats["num_failures"] = num_failures
    global_stats["num_invalidated_plans"] = num_invalidated_plans
    global_stats["num_timeouts"] = num_timeouts
    return run_stats, global_stats

def train_model_hierarchical(
    model,
    datasets,
    criterion,
    optimizer,
    use_gpu=False,
    print_every=10,
    save_every=100,
    save_folder="/tmp",
    epochs=1000,
    global_criterion=None,
    return_last_model_weights=True,
    model_type="room",
    eval_every=100,
):
    if model_type not in ["room", "object"]:
        raise ValueError(
            f"Unknown model type {model_type}. Valid model types are 'room', 'object'."
        )

    if use_gpu:
        device = "cuda:0"
    else:
        device = "cpu"

    def unpack_item(item):
        if model_type == "object":
            _input_graph = item["graph_input"]
            _target_graph = item["graph_target"]
            _nfeat = _input_graph["nodes"].float().to(device)
            _efeat = _input_graph["edges"].float().to(device)
            _senders = _input_graph["senders"].long().to(device)
            _receivers = _input_graph["receivers"].long().to(device)
            _tgt = _target_graph["nodes"].float().to(device)
            _edge_indices = torch.stack((_senders, _receivers))
            return _nfeat, _edge_indices, _efeat, _tgt
        elif model_type == "room":
            _input_graph = item["graph_input"]
            _target_graph = item["graph_target"]
            _nfeat = _input_graph["room_graph"]["nodes"].float().to(device)
            _efeat = _input_graph["room_graph"]["edges"].float().to(device)
            _senders = _input_graph["room_graph"]["senders"].long().to(device)
            _receivers = _input_graph["room_graph"]["receivers"].long().to(device)
            _tgt = _target_graph["room_graph"]["nodes"].float().to(device)
            _edge_indices = torch.stack((_senders, _receivers))
            return _nfeat, _edge_indices, _efeat, _tgt

    since = time.time()
    best_seen_model_weights = None  # as measured over the validation set
    best_seen_running_validation_loss = np.inf

    trainset, validset = datasets["train"], datasets["val"]

    for e in range(epochs):

        running_loss = 0.0
        running_num_samples = 0

        model.train()

        permuted_indx = torch.randperm(len(trainset))

        for idx in range(len(trainset)):
            i = permuted_indx[idx]
            nfeat, edge_indices, efeat, tgt = unpack_item(trainset[i])
            preds = model(nfeat, edge_indices, efeat)
            loss = criterion(preds, tgt)

            running_loss += loss.item()
            running_num_samples += 1

            loss.backward()

            if idx % 20 == 0 or idx == len(trainset) - 1:
                optimizer.step()
                optimizer.zero_grad()

        print(
            f"== [Model: {model_type}] [EPOCH {e:03d} / {epochs}] Train loss: {(running_loss / running_num_samples):03.5f}"
        )

        if e % eval_every == 0:

            model.eval()

            if e % save_every == 0:
                savefile = os.path.join(save_folder, f"{model_type}_model_{e:04d}.pt")
                torch.save(model.state_dict(), savefile)
                print(f"Saved model checkpoint {savefile}")

            running_loss = 0.0
            running_num_samples = 0

            for idx in range(len(validset)):
                nfeat, edge_indices, efeat, tgt = unpack_item(validset[idx])
                preds = model(nfeat, edge_indices, efeat)
                loss = criterion(preds, tgt)

                running_loss += loss.item()
                running_num_samples += 1

            print(
                f"===== [Model: {model_type}] [EPOCH {e:03d} / {epochs}] Val loss: {(running_loss / running_num_samples):03.5f}"
            )

            val_loss = running_loss / running_num_samples
            if val_loss < best_seen_running_validation_loss:
                best_seen_running_validation_loss = copy.deepcopy(val_loss)
                best_seen_model_weights = model.state_dict()
                savefile = os.path.join(save_folder, f"{model_type}_best.pt")
                torch.save(best_seen_model_weights, savefile)
                print(
                    f"Found new best model with val loss {best_seen_running_validation_loss} at epoch {e}. Saved!"
                )

    time_elapsed = time.time() - since
    print(
        f"Training complete in {(time_elapsed // 60):.0f} m {(time_elapsed % 60):.0f} sec"
    )

    return best_seen_model_weights


def predict_graph_with_graphnetwork_hierarchical(room_model, object_model, input_graph):
    """Predict scores across both levels of the hierarchy. """
    room_model.eval()
    object_model.eval()

    # Get room scores
    nfeat = torch.from_numpy(input_graph["room_graph"]["nodes"]).float()
    efeat = torch.from_numpy(input_graph["room_graph"]["edges"]).float()
    senders = torch.from_numpy(input_graph["room_graph"]["senders"]).long()
    receivers = torch.from_numpy(input_graph["room_graph"]["receivers"]).long()
    edge_indices = torch.stack((senders, receivers))
    room_scores = room_model(nfeat, edge_indices, efeat)
    room_scores = torch.sigmoid(room_scores)

    nfeat = torch.from_numpy(input_graph["nodes"]).float()
    efeat = torch.from_numpy(input_graph["edges"]).float()
    senders = torch.from_numpy(input_graph["senders"]).long()
    receivers = torch.from_numpy(input_graph["receivers"]).long()
    edge_indices = torch.stack((senders, receivers))
    object_scores = object_model(nfeat, edge_indices, efeat)
    object_scores = torch.sigmoid(object_scores)

    return room_scores, object_scores

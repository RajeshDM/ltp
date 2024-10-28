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

def train_model_graphnetwork_ltp_batch(model, datasets,
                                 #dataloaders,
                                   criterion, optimizer, use_gpu, print_iter=10, 
                save_iter=100, save_folder='/tmp',starting_epoch=0, final_epoch=1000, global_criterion=None,
                return_last_model_weights=True,dagger_train=False,train_env_name=None,seed=None,
                message_string='',
                log_wandb=False):

    since = time.time()
    min_save_epoch = 0
    print_iter = 10
    save_iter = print_iter
    best_seen_model_weights = None # as measured on validation loss
    best_seen_running_validation_loss = np.inf
    best_validation_loss_epoch = 0
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
    for epoch in range(starting_epoch,final_epoch):
        if epoch % print_iter == 0:
            print('Epoch {}/{}'.format(epoch, final_epoch - 1), flush=True)
            print('-' * 10, flush=True)
        # Each epoch has a training and validation phase
        running_num_samples = 0
        if epoch % print_iter == 0 :
            phases = ['train','val']
            #phases = ['train']
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

            for batch_data in datasets[phase]:
                batch_data = batch_data.to(device)
                action_scores, action_object_scores = model(batch_data)
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
                #division_coeff = curr_param_counter /total_number_params
                #ic (division_coeff)
                required_action_object_scores = torch.tensor(required_action_object_scores)
                target_indices = tgt_action_scores.argmax(dim=1)
                target_indices_2 = tgt_action_object_scores[required_action_object_scores].argmax(dim=1)
                tgt_action_scores = tgt_action_scores.squeeze(0)

                #ic (required_action_object_scores)

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
                    optimizer.zero_grad()
                    #ic ("backprop time",time.time()-backward_time)

                # statistics
                running_loss[phase] += loss.item()
                running_num_samples += 1
            if log_wandb:
                wandb.log({f"loss_{phase}": running_loss[phase]})

        if epoch % print_iter == 0:
            print("running_loss:", running_loss, flush=True)
            #ic (phase)
            #ic (action_scores,target_indices)
            #ic (action_object_scores, target_indices_2, required_action_object_scores)
            #ic (loss)
            #exit()
            epochs.append(epoch)
            train_loss_values.append(running_loss['train'])
            val_loss_values.append(running_loss['val'])
    
        if epoch % save_iter == 0 and epoch >= min_save_epoch:
            best_seen_running_validation_loss, best_validation_loss_epoch, best_seen_model_weights =  save_model_graphnetwork(model, save_folder, epoch, optimizer,train_env_name,seed,message_string,
                                    best_seen_running_validation_loss,running_loss,best_seen_model_weights,best_validation_loss_epoch, time_taken_for_save_iter)
            time_taken_for_save_iter = time.time()

    plt.plot(epochs, train_loss_values, label="Training")
    plt.plot(epochs, val_loss_values, label="Val")
    plt.legend()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), flush=True)
    return best_seen_model_weights

def train_model_graphnetwork_ltp(model, datasets,
                                 #dataloaders,
                                   criterion, optimizer, use_gpu, print_iter=10, 
                save_iter=100, save_folder='/tmp',starting_epoch=0, final_epoch=1000, global_criterion=None,
                return_last_model_weights=True,dagger_train=False,train_env_name=None,seed=None,
                message_string=''):
    """Optimize the model and save checkpoints
    """
    since = time.time()
    save_iter = 10
    min_save_epoch = 0
    print_iter = 10
    #num_epochs = 160
    #torch.manual_seed(seed)

    best_seen_model_weights = None # as measured on validation loss
    best_seen_running_validation_loss = np.inf
    best_validation_loss_epoch = 0

    if use_gpu:
        model = model.cuda()
        if criterion is not None:
            criterion = criterion.cuda()
        if global_criterion is not None:
            global_criterion = global_criterion.cuda()

    if use_gpu:
        device = "cuda:0"
    else:
        device = "cpu"

    #epochs = [i for i in range(num_epochs)]
    epochs = []
    train_loss_values = []
    val_loss_values = []

    time_taken_for_save_iter = time.time()
    #for name, param in model.named_parameters():
    #    ic(name, param)

    for epoch in range(starting_epoch,final_epoch):
        if epoch % print_iter == 0:
            print('Epoch {}/{}'.format(epoch, final_epoch - 1), flush=True)
            print('-' * 10, flush=True)
        # Each epoch has a training and validation phase
        #if epoch % print_iter == 0 and dagger_train == False:
        running_num_samples = 0
        if epoch % print_iter == 0 :
            phases = ['train','val']
        else:
            phases = ['train']
        
        running_loss = {'train':0.0,'val':0.0}

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                #model.train(False)  # Set model to evaluate mode
                model.eval()

            for data in datasets[phase]:
                optimizer.zero_grad()
                #g_inp = trainset[idx]["graph_input"]
                #g_tgt = trainset[idx]["graph_target"]
                g_inp = data['graph_input']
                g_tgt = data['graph_target']
                nfeat = torch.from_numpy(g_inp["nodes"]).float().to(device)
                efeat = torch.from_numpy(g_inp["edges"]).float().to(device)
                u     = torch.from_numpy(g_inp["globals"]).float().to(device)
                senders = torch.from_numpy(g_inp["senders"]).long().to(device)
                receivers = torch.from_numpy(g_inp["receivers"]).long().to(device)
                tgt_action_scores = torch.from_numpy(g_tgt["action_scores"]).float().to(device)
                tgt_action_object_scores = torch.from_numpy(g_tgt["action_object_scores"]).float().to(device)
                tgt_params = torch.from_numpy(g_tgt["n_parameters"]).float().to(device)
                edge_indices = torch.stack((senders, receivers))
                a_scores = torch.from_numpy(g_inp["action_scores"]).long().to(device)
                ao_scores = torch.from_numpy(g_inp["action_object_scores"]).long().to(device)
                action_scores, action_object_scores = model(nfeat, edge_indices, efeat,u,a_scores,ao_scores)
                #loss = criterion(preds, tgt)
                #running_loss += loss.item()
                #running_num_samples += 1
                #loss.backward()
                #optimizer.step()
                #optimizer.zero_grad()
                #output = outputs[0][0]
                #ic (preds)
                #output = preds[0]
                loss = 0.

                #ic (output['edges'].shape)
                #ic (output['action_object_scores'].shape)
                #ic (targets['action_object_scores'].shape)

                if criterion is not None:
                    #ic (type(output['action_scores']))
                    #ic (output)
                    #ic (targets)
                    #ic (targets['action_scores'].shape)
                    #ic (targets['action_scores'])
                    #loss += criterion(output['edges'], targets['edges'])
                    #loss += criterion(output['globals'], targets['globals'])
                    #loss += criterion(output['action_object_scores'], targets['action_object_scores'])
                    #best_target_action_locations = get_best_target_locations(outputs['action_scores'],targets['action_scores'])
                    #extract_action_scores = extract_scores_from_location(best_target_action_locations,output['action_scores'],targets['action_scores'])
                    #extracted_action_scores,extracted_action_target_scores = get_best_target_based_scores(output['action_scores'],targets['action_scores'])
                    #loss += criterion(extracted_action_scores,extracted_action_target_scores)
                    #extracted_object_scores,extracted_object_target_scores = get_best_target_based_scores(output['action_object_scores'],targets['action_object_scores'])
                    #loss += criterion(extracted_object_scores,extracted_object_target_scores)
                    #loss += criterion(output['action_object_scores'], targets['action_object_scores'])
                    #a = torch.tensor([1,2,3])
                    #ic (torch.index_select(output['action_object_scores'],0,a))
                    #ic (output['action_object_scores'])
                    #ic (output['action_object_scores'][a])

                    #exit()
                    #loss += criterion(output['action_scores'], targets['action_scores'])
                    curr_param_counter = 0
                    required_action_object_scores = []
                    total_number_params = 0
                    #ic (output['action_object_scores'])
                    #ic (targets['action_object_scores'])
                    #for n_params in tgt['n_parameters']:
                    for idx,n_params in enumerate(tgt_params):
                        #ic (output['action_object_scores'][0:2])
                        #ic (curr_param_counter)
                        n_params = int(n_params.item())
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
                    #division_coeff = curr_param_counter /total_number_params
                    #ic (division_coeff)
                    required_action_object_scores = torch.tensor(required_action_object_scores)
                    #loss += criterion(output['action_object_scores'][required_action_object_scores],targets['action_object_scores'][required_action_object_scores])/division_coeff
                    #loss += criterion(output['action_object_scores'][required_action_object_scores],targets['action_object_scores'][required_action_object_scores])#/division_coeff
                    #exit()

                    #target_indices = tgt['action_scores'].argmax(dim=1)
                    #target_indices_2 = tgt['action_object_scores'][required_action_object_scores].argmax(dim=1)
                    target_indices = tgt_action_scores.argmax(dim=1)
                    target_indices_2 = tgt_action_object_scores[required_action_object_scores].argmax(dim=1)
                    #ic (targets['action_scores'])
                    #ic (target_indices)
                    #ic (target_indices_2)
                    #ic (output['action_scores'])
                    #ic (targets['action_object_scores'][required_action_object_scores])
                    #ic (output['action_object_scores'][required_action_object_scores])
                    #loss += criterion(output['action_scores'], target_indices)
                    #loss += criterion(output['action_object_scores'][required_action_object_scores],target_indices_2)

                    tgt_action_scores = tgt_action_scores.squeeze(0)

                    #ic (required_action_object_scores)

                    m = torch.nn.ConstantPad2d((0,tgt_action_object_scores.shape[1]-action_object_scores.shape[1]\
                                                ,0,0),0)
                    
                    action_object_scores = m(action_object_scores)

                    #ic (action_object_scores[required_action_object_scores].shape)
                    #ic (tgt_action_object_scores[required_action_object_scores].shape)
                    #tgt_action_object_scores = tgt_action_object_scores.squeeze(0)
                    #ic (tgt_action_object_scores[required_action_object_scores].shape)
                    #ic (tgt_action_object_scores)

                    #ic (action_scores)
                    #ic (action_object_scores)
                    #loss += criterion(action_scores,tgt_action_scores)
                    loss += criterion(action_scores.unsqueeze(0),target_indices)
                    loss += criterion(action_object_scores[required_action_object_scores],target_indices_2)
                    #loss += criterion(action_object_scores[required_action_object_scores],tgt_action_object_scores[required_action_object_scores])
                    #ic (loss)

                    #ic (loss)
                    #exit()


                if phase == 'train':
                    backward_time = time.time()
                    loss.backward()
                    optimizer.step()
                    #ic ("backprop time",time.time()-backward_time)

                # statistics
                running_loss[phase] += loss.item()
                running_num_samples += 1
                #ic (loss.item())
                #current_running_loss = loss.item()

        if epoch % print_iter == 0:
            print("running_loss:", running_loss, flush=True)
            #exit()
            epochs.append(epoch)
            train_loss_values.append(running_loss['train'])
            val_loss_values.append(running_loss['val'])
            #for name, param in model.named_parameters():
            #    ic(name, param)
            #exit()

        if epoch % save_iter == 0 and epoch >= min_save_epoch:

            save_model_graphnetwork(model, save_folder, epoch, optimizer,train_env_name,seed,message_string,
                                    best_seen_running_validation_loss,running_loss,best_seen_model_weights,best_validation_loss_epoch)
            #save_path = os.path.join(save_folder,str(train_env_name)+ "_seed"+ str(seed) + "_model" + str(epoch) + ".pt")
            #save_path = os.path.join(save_folder,str(train_env_name)+ "_seed"+ str(seed) + "_model" + str(epoch) + "_rounds" + str(gnn_rounds) +"_"+message_string+ ".pt")
            '''
            save_path = os.path.join(save_folder, str(train_env_name) + "_seed" + str(seed) + "_model" + str(
                epoch) + "_" + message_string + ".pt")
            #print("Time taken for {} epochs : {}".format(save_iter, time.time() - time_taken_for_save_iter))
            state_save = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
            'epochs':epoch}
            #torch.save(model.state_dict(), save_path)
            torch.save(state_save,save_path)
            '''
            time_taken_for_save_iter = time.time()

    #ic (train_loss_values,val_loss_values)
    plt.plot(epochs, train_loss_values, label="Training")
    plt.plot(epochs, val_loss_values, label="Val")
    plt.legend()
    #plt.show()
    #plt.savefig(str(epochs))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), flush=True)

    if return_last_model_weights:
        return model.state_dict()

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

def get_single_model_multiple_prediction(model,single_input):
    model.train(False)
    model.eval()
    inputs = create_super_graph([single_input])
    if constants.use_gpu:
        for key in inputs.keys():
            if inputs[key] == None or key == 'prev_graph' :
                continue
            inputs[key] = inputs[key].cuda()
    outputs = model(inputs.copy())
    return outputs


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


def test_planner_ltp(planner, domain_name, num_problems, timeout,current_problems_to_solve=None,ensemble=False,train_planner=None,epoch_number=0,debug_level=constants.max_debug_level-1):
    if debug_level < constants.max_debug_level :
        print("Running testing...")
    #ic (domain_name)
    #exit()
    env = pddlgym.make("PDDLEnv{}-v0".format(domain_name))
    env_2 = pddlgym.make("PDDLEnv{}-v0".format(domain_name))
    #ic (domain_name)
    #exit()
    j = 0
    failed_plans = 0
    failed_actions = 0
    succesful_actions = 0
    action_in_top_3 = 0
    #plan_lengths = []
    succesful_plan_lengths = []
    object_selection_issue = 0
    action_selection_issue = 0
    output_plan_lengths = []
    num_problems = min(num_problems, len(env.problems))
    plan_lengths ={}
    correct_actions = []
    planner_plan_lengths = []
    #for problem_idx in reversed(range(num_problems)):
    number_rejected_actions = 0
    number_impossible_actions = 0
    correct_plans = 0
    problem_number = constants.problem_number
    number_problems_each_division = constants.number_problems_each_division
    max_plan_length_permitted = constants.max_plan_length_permitted
    opt_planner = constants.opt_planner
    non_opt_planner = constants.non_opt_planner
    external_monitor_bool = constants.external_monitor_bool
    heuristic_planner = constants.heuristic_planner
    plot_aggregates = constants.plot_aggregates
    max_debug_level = constants.max_debug_level

    correct_plan_lengths_system = 0
    correct_plan_lengths_planner = 0
    total_correct_time_system = 0
    total_correct_time_planner = 0
    failed_plans_locations = []
    #for problem_idx in (range(num_problems)):
    #number_divisions = max (int((num_problems-problem_number)/number_problems_each_division),1)
    number_divisions = max (int((num_problems)/number_problems_each_division),1)
    failure_dict = {i:[] for i in range(int(number_divisions) )}
    planner_failure_dict = {i:[] for i in range(int(number_divisions) )}
    opt_planner_failure_dict = {i:[] for i in range(int(number_divisions) )}
    planner_timer = []
    planner_plan_len = []
    opt_planner_timer = []
    opt_planner_plan_len = []
    learned_system_timer = []
    learned_system_plan_len = []
    average_success_time_system = {i:[] for i in range(int(number_divisions) )}
    average_success_time_planner = {i:[] for i in range(int(number_divisions))}
    average_success_time_opt_planner = {i:[] for i in range(int(number_divisions))}
    average_plan_len_system = {i:[] for i in range(int(number_divisions) )}
    average_plan_len_planner = {i:[] for i in range(int(number_divisions))}
    average_plan_len_opt_planner = {i:[] for i in range(int(number_divisions))}

    #average_success_time = {}

    if current_problems_to_solve == None :
        problems_to_solve = range(num_problems)
    else :
        problems_to_solve = current_problems_to_solve[:]

    if ensemble :
        models_specifications = []
        ensemble_models_dir = '/Users/rajesh/Rajesh/Subjects/Research/affordance_learning/PLOI/model/ensemble_models'
        ensemble_models = [ensemble_models_dir+"/" + f for f in listdir(ensemble_models_dir) if isfile(join(ensemble_models_dir, f)) and not f.startswith('.')]
        for location in ensemble_models:
            model_specification = 6,False,64,64,location
            models_specifications.append(model_specification)
        planner._guidance.load_ensemble_models(models_specifications, domain_name)

    if constants.use_gpu == True :
        planner._guidance._model = planner._guidance._model.cuda()
    #exit()
    if len(problems_to_solve) == 0 :
        ic ("exiting from 0 oproblems to solve")
        exit()
    not_in_grounding = 0

    for problem_idx in problems_to_solve :
    #for problem_idx in range(1):
        curr_plan_states = []
        #for problem_idx in range(num_problems):
        if debug_level < max_debug_level :
            print  ("#############################################")
            print("   Testing problem {} of {}, scene {}".format(problem_idx+1+problem_number, num_problems+problem_number,env.problems[problem_idx+problem_number].problem_fname),
                  flush=True)
        env.fix_problem_index(problem_idx+problem_number)
        env_2.fix_problem_index(problem_idx+problem_number)
        #ic (env.problems[problem_idx+problem_number].problem_fname)
        current_succesful_plan_length = 0
        #start_state, _ = env.reset()
        #state = start_state
        state,_ = env.reset()
        #ic (type(state))
        #ic ()
        #ic(type(pddlgym.structs))
        #ic (pddlgym.structs)

        state_2, _ = env_2.reset()
        #planning_time_start = time.time()
        action_space = env.action_space
        plan = None
        new_plan = []
        j+= 1
        num_actions = 0
        plan_lengths[problem_idx] = []
        opt_plan = None
        non_opt_plan = None
        non_opt_time = None
        if non_opt_planner:
            non_opt_plan, non_opt_time = run_non_opt_planner(env,state,action_space._action_predicate_to_operators,timeout,planner)
        if opt_planner:
            opt_plan, opt_time_taken = run_opt_planner(env, state, action_space._action_predicate_to_operators, timeout, train_planner)


        #planner_time = time.time() - planning_time_start
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        env.domain.write(dom_file)

        if heuristic_planner == True :
            correct_plans = run_heuristic_planner(env,state,planner,action_space,domain_name,correct_plans)
            continue

        #exit()
        start_time = time.time()
        correct_actions_current = 0
        prev_state = state[0]
        prev_graph = None

        while True :
            single_action_time = time.time()
            #decoded_action, decoded_action_parameters = planner._guidance.get_action_object_scores(state,action_space._action_predicate_to_operators)
            #problem = parse_dom_problem_for_grounding(dom_file, prob_file)
            #pyperplan_task = grounding.ground(problem)
            groundings = env.action_space.all_ground_literals(state)
            prev_actions = None
            if len(new_plan) == 1 :
                prev_actions = [new_plan[0]]
            #if len(new_plan) >= 2 :
            #    prev_actions = [new_plan[-1],new_plan[-2]]
            prev_actions= None
            prev_graph = None
            #action_param_list = planner._guidance.get_action_object_scores(state,action_space._action_predicate_to_operators,prev_actions)
            #action_param_list = planner._guidance.get_action_object_scores(state,action_space._action_predicate_to_operators,None)
            #action_param_list,prev_graph = planner._guidance.get_action_object_scores(state,action_space._action_predicate_to_operators,prev_actions,prev_graph)
            #action_param_list,prev_graph = planner._guidance.get_action_object_scores(state,action_space._action_predicate_to_operators,prev_actions,prev_graph)
            #action_param_list,prev_graph = planner._guidance.get_action_object_scores_ensemble(state,action_space._action_predicate_to_operators,pyperplan_task, prev_actions,prev_graph,ensemble=ensemble)
            inference_start_time = time.time()
            action_param_list,prev_graph = planner._guidance.get_action_object_scores_ensemble(state,action_space._action_predicate_to_operators,groundings, prev_actions,prev_graph,ensemble=ensemble)
            inference_end_time = time.time()
            #ic ("Inference time", inference_end_time-inference_start_time)
            groundings = list(groundings)
            groundings_list = []
            for grounding in groundings:
                grounding_action = grounding.predicate
                objects = grounding.variables
                groundings_list.append(pddlgym.structs.Literal(grounding_action, objects))
            #ic (groundings_list)
            #ic (action_param_list)
            '''
            action_data = action_param_list[0]
            decoded_action, decoded_action_parameters = action_data[0], action_data[1]
            new_action = pddlgym.structs.Literal(decoded_action, decoded_action_parameters)
            '''
            action_selection_time = time.time()
            for action_data in action_param_list :

                in_grounding = False

                #decoded_action, decoded_action_parameters = action_param_list[0][0],action_param_list[0][1]
                decoded_action,decoded_action_parameters = action_data[0],action_data[1]
                #ic (decoded_action)
                #ic (type(decoded_action))
                #ic (type(decoded_action_parameters[0]))
                #exit()
                new_action = pddlgym.structs.Literal(decoded_action,decoded_action_parameters)
                #ic (decoded_action)
                #ic (decoded_action_parameters)

                #if new_action in groundings_list :
                for grounded_action in groundings_list:
                    in_grounding_temp = True
                    if new_action.predicate == grounded_action.predicate :
                        for grounded_var,action_var in zip(grounded_action.variables,new_action.variables):
                            if grounded_var != action_var:
                                in_grounding_temp = False
                                break
                        if in_grounding_temp == True :
                            in_grounding = True
                            break
                    #else :
                    #    in_grounding = False
                    #ic (grounded_action.__dict__)
                    #ic (new_action.__dict__)

                if in_grounding == False :
                    number_impossible_actions += 1
                    continue
                '''
                '''
                step_taking_time = time.time()
                #ic (new_action,action_data[2])
                #exit()
                state = env.step(new_action)
                step_taking_end_time = time.time()
                #ic (state.__dict__)
                #state = env.step(action)
                #ic ("Step taking time", step_taking_end_time-step_taking_time)
                state=state[0]
                #continue
                break
                #ic (state[0])
                #ic (prev_state)
                #ic (new_action.__dict__)
                if state[0] != prev_state:
                    num_actions += 1
                    prev_state = state[0]
                    #curr_pl#an_states.append(state[0])
                    if external_monitor_bool == True :
                        if state[0] in curr_plan_states:
                            #ic ("Current action took back to old state")
                            #ic (new_action)
                            #ic ("##")
                            num_actions = 0
                            new_start_state, _ = env.reset()
                            for actions_taken in new_plan:
                                new_state = env.step(actions_taken)
                                num_actions += 1
                            state = new_state[0]
                            prev_state = new_state[0][0]
                            number_rejected_actions += 1
                            continue
                    break
                else :
                    #ic (prev_state)
                    #ic (state[0])
                    #ic ("rejected action" , new_action)
                    number_impossible_actions += 1
                    #new_action =
                prev_state = state[0]
            #ic (new_action)
            #state = env.step(new_action)
            #state = state[0]
            #ic (new_action)
            #ic ("action selection time", time.time()-action_selection_time)
            #ic ("single action time ", time.time()-single_action_time)

            curr_plan_states.append(state[0])
            new_plan.append(new_action)
            #ic (len(new_plan))
            #break
            #for action_loop in new_plan :
            #    print (action_loop.__dict__)
            #ic (state[0])

            plan_val_time = time.time()
            plan_found = True
            for goal in state.goal.literals:
                if goal not in list(state.literals):
                    plan_found = False

            if plan_found == True:
            #    #break
            #if validate_strips_plan(
            #        domain_file=env.domain.domain_fname,
            #        problem_file=env.problems[problem_idx+problem_number].problem_fname,
            #        plan=new_plan):
                end_time = time.time()
                total_correct_time_system += end_time-start_time
                output_plan_lengths.append((len(new_plan), "success"))
                correct_plan_lengths_system += len(new_plan)
                #ic (output_plan_lengths[-1])
                #correct_plan_lengths_planner += output_plan_lengths[-1][0]
                if plan != None:
                    correct_plan_lengths_planner += len(plan)
                plan_lengths[problem_idx] += output_plan_lengths[-1]
                plan_lengths[problem_idx].append(end_time - start_time)
                learned_system_timer.append(end_time-start_time)
                learned_system_plan_len.append(len(new_plan))
                #average_success_time = failure_dict
                curr_div = int(problem_idx / number_problems_each_division)
                start_point = curr_div * number_problems_each_division
                end_point = min(start_point+number_problems_each_division, len(learned_system_timer))
                average_success_time_system[curr_div] = np.nanmean(np.array(learned_system_timer[start_point:end_point]))
                #average_plan_len_system[curr_div] = np.nanmean(np.array(learned_system_plan_len[start_point:end_point]))

                #total_correct_time_planner += time_taken
                if debug_level < max_debug_level :
                    print ("Valid plan")
                break
            else :
                pass
                #continue
            #ic ("plan validation time", time.time()-plan_val_time)

            #if num_actions > 40 :
            #if len(new_plan) > 8:
            if len(new_plan) > max_plan_length_permitted:
                failed_plans += 1
                #output_plan_lengths.append(len(new_plan))
                output_plan_lengths.append((len(new_plan), "fail"))
                plan_lengths[problem_idx] += output_plan_lengths[-1]
                end_time = time.time()
                plan_lengths[problem_idx].append(end_time - start_time)
                failed_plans_locations.append(problem_number+problem_idx)
                failure_dict[int(problem_idx / number_problems_each_division)].append(problem_idx)
                learned_system_timer.append(np.nan)
                learned_system_plan_len.append(np.nan)
                break
        if non_opt_plan != None:
            ic(non_opt_plan)
            plan = non_opt_plan
            planner_plan_len.append(len(plan))
            planner_plan_lengths.append(len(plan))
            plan_lengths[problem_idx].append(len(plan))
            plan_lengths[problem_idx].append(non_opt_time)
            plan_lengths[problem_idx].append(correct_actions_current)
            correct_actions.append(correct_actions_current)
            planner_timer.append(non_opt_time)
            #curr_div = int((problem_idx-problem_number) / number_problems_each_division)
            curr_div = int((problem_idx) / number_problems_each_division)
            #end_point = min(curr_div + number_problems_each_division, len(learned_system_timer))
            start_point = curr_div*number_problems_each_division
            end_point = min(start_point + number_problems_each_division, len(planner_timer))
            average_success_time_planner[curr_div] = np.nanmean(np.array(planner_timer[start_point:end_point]))
            average_plan_len_planner[curr_div] = np.nanmean(np.array(planner_plan_len[start_point:end_point]))
        else:
            plan_lengths[problem_idx].append((0,correct_actions_current))
            planner_failure_dict[int(problem_idx / number_problems_each_division)].append(problem_idx)
            planner_timer.append(np.nan)
            planner_plan_len.append(np.nan)

        if opt_plan != None:
            opt_planner_timer.append(opt_time_taken)
            opt_planner_plan_len.append(len(opt_plan))
            curr_div = int(problem_idx / number_problems_each_division)
            # end_point = min(curr_div + number_problems_each_division, len(learned_system_timer))
            start_point = curr_div * number_problems_each_division
            end_point = min(start_point + number_problems_each_division, len(opt_planner_timer))
            average_success_time_opt_planner[curr_div] = np.nanmean(np.array(opt_planner_timer[start_point:end_point]))
            #average_plan_len_opt_planner[curr_div] = np.nanmean(np.array(opt_planner_plan_len[start_point:end_point]))
            ic (opt_plan)
        else:
            #plan_lengths[problem_idx].append((0, correct_actions_current))
            opt_planner_timer.append(np.nan)
            opt_planner_plan_len.append(np.nan)
            opt_planner_failure_dict[int(problem_idx / number_problems_each_division)].append(problem_idx)

        if opt_plan != None and plan_found == True :
            #start_point = curr_div * number_problems_each_division
            #end_point = min(start_point + number_problems_each_division, len(opt_planner_timer))
            average_plan_len_opt_planner[curr_div] = np.nanmean(np.array(opt_planner_plan_len[start_point:end_point]))
            #curr_div = int(problem_idx / number_problems_each_division)
            #start_point = curr_div * number_problems_each_division
            #end_point = min(start_point + number_problems_each_division, len(learned_system_timer))
            average_plan_len_system[curr_div] = np.nanmean(np.array(learned_system_plan_len[start_point:end_point]))

        #ic (average_plan_len_opt_planner)
        #ic (average_success_time_system)
        #ic (average_plan_len_system)
        start_state,_ = env.reset()

        discrepancy_search_bool = False
        #discrepancy_search_bool = True
        if discrepancy_search_bool == True :
            if len(new_plan) > max_plan_length_permitted:
                succesful_plans = discrepancy_search(planner, env, start_state, action_space, ensemble,new_plan)
                #for plan in succesful_plans:
                if len(succesful_plans) != 0 :
                    new_plan = succesful_plans[-1]
                    if new_plan != None :
                        #ic ("Valid plan")
                        if debug_level < max_debug_level :
                            print ("valid plan")
                        failed_plans -= 1
                        #new_plan = plan
                        #break
        #ic(succesful_plans)
        #exit()
        #correct_actions.append(correct_actions_current)
        if debug_level < max_debug_level - 1:
            for action_loop in new_plan :
                print (action_loop.__dict__)
    #ic(plan_lengths)
    # ic ("Plan success rate ", (1-failed_plans/num_problems))
    #ic("Plan success rate ", (1 - failed_plans / j))
    if debug_level <= max_debug_level :
        print ("Total Plan successes {}/{} , succes rate {}".format(j-failed_plans,j,1-(failed_plans/j) ))

    if debug_level == max_debug_level -1 :
        ic (debug_level,max_debug_level)
        ic (external_monitor_bool)
        #ic("Plan success rate ", correct_plans / j)
        ic (sum(planner_plan_lengths))#,sum(correct_actions))
        #ic (number_rejected_actions)
        ic (number_impossible_actions)
        ic (correct_plan_lengths_system)
        ic (correct_plan_lengths_planner)
        ic ("Average Time taken by system  :", total_correct_time_system/j)
        ic ("Average Time taken by planner :", total_correct_time_planner/j)
        ic (failure_dict)

    if heuristic_planner == False and plot_aggregates == True:

        #if plot_common == True :
        plt.clf()
        plt.plot(range(len(problems_to_solve)), learned_system_timer, label="Learned System")
        plt.plot(range(len(problems_to_solve)), planner_timer, label="Planner")
        if opt_plan != None:
            plt.plot(range(len(problems_to_solve)), opt_planner_timer, label="Opt Planner")
        #plt.xticks(range(len(problems_to_solve)), range(len(problems_to_solve)))
        plt.xlabel("Problem number")
        plt.ylabel("Time taken (seconds)")
        plt.legend()
        plt.savefig("system_vs_planner_success_runtimes"+ str(num_problems)+ ".png")
        #plt.show()

        plt.show()
        plt.clf()
        plt.plot(range(6,number_divisions+6), average_success_time_system.values(), label="Learned System")
        plt.plot(range(6,number_divisions+6), average_success_time_planner.values(), label="Planner")
        if opt_plan != None:
            plt.plot(range(6,number_divisions+6), average_success_time_opt_planner.values(), label="Opt Planner")
        plt.xticks(range(6,number_divisions+6), range(6,number_divisions+6))
        plt.xlabel("Problem Size")
        plt.ylabel("Time taken (seconds)")
        plt.legend()
        plt.savefig("system_vs_planner_avg_success_runtimes_" + str(num_problems) + ".png")
        #plt.show()

        plt.clf()
        plt.plot(range(6,number_divisions+6), average_plan_len_system.values(), label="Learned System")
        plt.plot(range(6,number_divisions+6), average_plan_len_planner.values(), label="Planner")
        if opt_plan != None:
            plt.plot(range(6,number_divisions+6), average_plan_len_opt_planner.values(), label="Opt Planner")
        plt.xticks(range(6,number_divisions+6), range(6,number_divisions+6))
        plt.xlabel("Problem Size")
        plt.ylabel("Average Plan lengths")
        plt.legend()
        plt.savefig("system_vs_planner_avg_success_plan_len_" + str(num_problems) + ".png")

    elif heuristic_planner == True:

        plt.plot(range(len(learned_gbf_expansions)), learned_gbf_expansions, label="GBF learned")
        plt.plot(range(len(gbf_lmcut_expansions)), gbf_lmcut_expansions, label="gbf_lmcut")
        plt.plot(range(len(a_star_h_add_expansions)), a_star_h_add_expansions, label="a_star h_add")
        # plt.plot(range(len(problems_to_solve)), opt_planner_timer, label="Opt Planner")
        # plt.xticks(range(len(problems_to_solve)), range(len(problems_to_solve)))
        plt.xlabel("Problem number")
        plt.ylabel("Node Expansions")
        plt.legend()
        plt.savefig("Node_expansions_comparisons.png")
        #plt.show()

    #ic (planner_failure_dict)
    #ic (opt_planner_failure_dict)
    #ic (average_success_time_system)
    #ic (average_success_time_planner)
    #ic (average_success_time_opt_planner)
    return failed_plans_locations



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

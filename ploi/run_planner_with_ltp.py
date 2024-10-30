import pddlgym

import os
import torch
import random
seed = 10
import numpy as np
import warnings
import time
import argparse
import pddlgym
#from planning import PlanningTimeout, PlanningFailure, FD, \
#    validate_strips_plan, IncrementalPlanner
from ploi.planning import FD 

from icecream import ic
import tempfile
from pddlgym.parser import PDDLProblemParser
from os import listdir
from os.path import isfile, join
from pyperplan import grounding
import matplotlib.pyplot as plt
from ploi.datautils_ltp import _state_to_graph_ltp
import ploi.constants as constants
from ploi.datautils_ltp import state_to_graph_wrapper
from ploi.datautils_ltp import (
    graph_dataset_to_pyg_dataset,
)
from torch_geometric.loader import DataLoader as pyg_dataloader
from dataclasses import dataclass

@dataclass
class ModelMetrics:
    """Data class to store metrics for a single model run"""
    number_impossible_actions: int
    correct_plan_lengths_system: int
    time_taken_system: float
    plan_success_rate : float
    total_plan_successes: float


'''
def run_planner_with_gnn(planner, domain_name, num_problems, timeout,current_problems_to_solve=None,ensemble=False,
                         train_planner=None,epoch_number=0,debug_level=constants.max_debug_level-1,
                         model=None):
    if debug_level < constants.max_debug_level :
        print("Running testing...")
    env = pddlgym.make("PDDLEnv{}-v0".format(domain_name))
    num_problems = min(num_problems, len(env.problems))

    problem_number = constants.problem_number
    number_problems_each_division = constants.number_problems_each_division
    max_plan_length_permitted = constants.max_plan_length_permitted
    opt_planner = constants.opt_planner
    non_opt_planner = constants.non_opt_planner
    external_monitor_bool = constants.external_monitor_bool
    heuristic_planner = constants.heuristic_planner
    plot_aggregates = constants.plot_aggregates
    max_debug_level = constants.max_debug_level

    if current_problems_to_solve == None :
        problems_to_solve = range(num_problems)
    else :
        problems_to_solve = current_problems_to_solve[:]

    for problem_idx in problems_to_solve :
        curr_plan_states = []
        new_plan = []
        if debug_level < max_debug_level :
            print  ("#############################################")
            print("   Testing problem {} of {}, scene {}".format(problem_idx+1+problem_number, num_problems+problem_number,env.problems[problem_idx+problem_number].problem_fname),
                  flush=True)
        env.fix_problem_index(problem_idx+problem_number)
        state,_ = env.reset()
        action_space = env.action_space
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        env.domain.write(dom_file)
        while True :
            groundings = env.action_space.all_ground_literals(state)
            groundings = list(groundings)
            groundings_list = []
            for grounding in groundings:
                grounding_action = grounding.predicate
                objects = grounding.variables
                groundings_list.append(pddlgym.structs.Literal(grounding_action, objects))

            action_param_list = get_action_object_scores_ensemble(state,action_space._action_predicate_to_operators,groundings,
                                                                  model,prev_actions=None,
                                                                  prev_state=None,ensemble=ensemble)

            for action_data in action_param_list :
                in_grounding = False
                decoded_action,decoded_action_parameters = action_data[0],action_data[1]
                new_action = pddlgym.structs.Literal(decoded_action,decoded_action_parameters)
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
                if in_grounding == False :
                    number_impossible_actions += 1
                    continue
                state = env.step(new_action)
                state=state[0]
                break
            curr_plan_states.append(state[0])
            new_plan.append(new_action)
            plan_found = True
            for goal in state.goal.literals:
                if goal not in list(state.literals):
                    plan_found = False
            if plan_found == True:
                if debug_level < max_debug_level :
                    print ("Valid plan")
                break
            if len(new_plan) > max_plan_length_permitted:
                if debug_level < max_debug_level :
                    print ("InValid plan")
                break
        if debug_level < max_debug_level - 1:
            for action_loop in new_plan :
                print (action_loop.__dict__)

    return
'''

def get_action_object_scores_ensemble(state, action_space,pyperplan_task ,model,
                                          prev_actions=None,prev_state=None
                                          ,correct_action_object_tuple=None,ensemble=False):

        graph, node_to_objects = _state_to_graph_ltp(state,action_space,pyperplan_task,
                                                      prev_actions,prev_state,test=True)
        object_idxs = []
        action_idxs = []
        action_param_tuples = []
        all_model_outputs = []

        for i,elem in enumerate(graph['nodes']):
            if elem[0] == 1 :
                action_idxs.append(i)
            elif elem[0] == 0 :
                object_idxs.append(i)

        number_objects = len(object_idxs)

        predictions = _predict_multiple_graph_from_single_output_with_model(model,graph)
        output = get_action_param_list_from_predictions(predictions, action_space, node_to_objects, number_objects)
        all_model_outputs.append(output)

        all_possible_actions = {}

        for output in all_model_outputs:
            current_graph_keys = []
            for action_data in output :
                key = str(action_data[0]) + str(action_data[1])
                if key in current_graph_keys :
                    continue
                current_graph_keys.append(key)
                if key not in all_possible_actions.keys():
                    all_possible_actions[key] = [action_data,action_data[2]]
                else :
                    current_score = all_possible_actions[key][1]
                    all_possible_actions[key] = [action_data,current_score+action_data[2]]

        for key,value in all_possible_actions.items():
            action_param_tuples.append((value[0][0],value[0][1],value[1]))

        #output = sorted(action_param_tuples, key=lambda x: x[-1])
        #return output[::-1],graph
        return action_param_tuples


def _predict_multiple_graph_from_single_output_with_model(self,model,input_graph):
    assert self._model is not None, "Must train before calling predict"
    predictions = get_single_model_multiple_prediction(model, input_graph)
    sigmoid = lambda x: 1/(1 + np.exp(-x))
    for i,prediction in enumerate(predictions):
        predictions[i][0]['action_scores'][:] = torch.nn.functional.softmax(prediction[0]['action_scores'], dim=1)[:]
        predictions[i][0]['action_object_scores'][:] = torch.nn.functional.softmax(prediction[0]['action_object_scores'], dim=1)[:]

    return predictions

def get_single_model_multiple_prediction(model,single_input):
    model.train(False)
    model.eval()
    inputs = create_super_graph([single_input])
    if  constants.use_gpu:
        for key in inputs.keys():
            if inputs[key] == None or key == 'prev_graph' :
                continue
            inputs[key] = inputs[key].cuda()
    outputs = model(inputs.copy())
    return outputs

def get_action_param_list_from_predictions(predictions,action_space, node_to_objects,number_objects):

    action_param_tuples = []
    #ic (len(predictions))
    max_action_arity = 0
    for action in action_space:
        if action.arity > max_action_arity:
            max_action_arity = action.arity

    for prediction in predictions:
        obj_idxs = []
        action_idx = int(prediction[1][0][0])
        for arity in range(max_action_arity):
            obj_idxs.append(int(prediction[1][arity+1][arity]))
        tuple_score = prediction[3][0][0].item()
        action = node_to_objects[action_idx+number_objects]
        number_parameters = len(action_space[action].params)
        action_parameters = []
        tuple_object_score = 0
        divide_score_by = 1
        objects_score = 0
        for i in range(number_parameters):
            object = node_to_objects[obj_idxs[i]]
            action_parameters.append(object)
            objects_score += prediction[3][i+1][i].item()
            divide_score_by += 1
        action_param_tuples.append((action,action_parameters,tuple_score +objects_score/divide_score_by))

    return action_param_tuples

def run_non_opt_planner(env,state,action_space,timeout,planner):
    try:
        plan, time_taken = planner(env.domain, state, action_space, timeout=timeout)
        return plan, time_taken
    except Exception as e:
        print("\t\tPlanning failed with error: {}".format(e), flush=True)
        return None,None

def run_opt_planner(env,state,action_space,timeout,train_planner):
    try:
        opt_start_time = time.time()
        opt_plan = train_planner(env.domain, state, timeout=timeout)
        opt_time_taken = time.time() - opt_start_time
    except Exception as e:
        print("\t\tPlanning failed with error: {}".format(e), flush=True)
        return None,None

def compare_actions(action1,action2):
    if action1.predicate != action2.predicate:
        return False
    if len(action1.variables) != len(action2.variables):
        return False
    for obj1,obj2 in zip(action1.variables,action2.variables):
        if obj1 != obj2:
            return False
    return True

def _create_planner(planner_name):
    if planner_name == "fd-lama-first":
        return FD(alias_flag="--alias lama-first")
    if planner_name == "fd-opt-lmcut":
        return FD(alias_flag="--alias seq-opt-lmcut")
    raise ValueError(f"Uncrecognized planner name {planner_name}")

def convert_state_and_run_model(model, state, action_space , device, groundings, 
                                graph_metadata,cheating_input=None):
    g_inp , _, node_to_objects = state_to_graph_wrapper(state,action_space,groundings,
                                                    prev_actions=None,prev_state=None,
                                                    graph_metadata=graph_metadata,
                                                    curr_action=None,objects=None,goal_state=state.goal,
                                                    cheating_input=cheating_input)

    all_actions = [k for k, v in action_space.items()]
    num_actions =len(all_actions)
    num_non_action_nodes = len(node_to_objects) - (num_actions) 
                                    
    model_input = convert_graph_to_model_input_v2(g_inp,device)
    #nfeat, edge_indices, efeat, u, a_scores, ao_scores = model_input
    #results = model(nfeat, edge_indices, efeat,u,a_scores,ao_scores)
    results = model(model_input, beam_search=True)
    action_param_list = []

    for action_data in results : 
        action_idx = int(action_data[1][0])
        number_parameters = len(action_data[1]) - 1
        decoded_action = node_to_objects[action_idx+num_non_action_nodes]
        decoded_action_parameters = []
        for i in range(number_parameters):
            obj_idx = int(action_data[1][i+1])
            obj = node_to_objects[obj_idx]
            decoded_action_parameters.append(obj)

        #ic (decoded_action)
        #ic (type(decoded_action))
        #ic (type(decoded_action_parameters[0]))
        new_action = pddlgym.structs.Literal(decoded_action,decoded_action_parameters)
        action_param_list.append(new_action)

    return action_param_list 

def convert_graph_to_model_input_v1(g_inp, device):
    nfeat = torch.from_numpy(g_inp["nodes"]).float().to(device)
    efeat = torch.from_numpy(g_inp["edges"]).float().to(device)
    u     = torch.from_numpy(g_inp["globals"]).float().to(device)
    senders = torch.from_numpy(g_inp["senders"]).long().to(device)
    receivers = torch.from_numpy(g_inp["receivers"]).long().to(device)
    edge_indices = torch.stack((senders, receivers))
    a_scores = torch.from_numpy(g_inp["action_scores"]).long().to(device)
    ao_scores = torch.from_numpy(g_inp["action_object_scores"]).long().to(device)
    return nfeat, edge_indices, efeat, u, a_scores, ao_scores

def convert_graph_to_model_input_v2(g_inp, device):
    hetero_graphs = graph_dataset_to_pyg_dataset([g_inp])
    hetero_dataset = pyg_dataloader(hetero_graphs, batch_size=1)
    return next(iter(hetero_dataset)).to(device) 

def discrepancy_get_best_action(feasible_action_list,action_loc,discrepancy_loc):
    if action_loc == discrepancy_loc and len(feasible_action_list) > 1:
        new_action = feasible_action_list[1]
    else:
        new_action = feasible_action_list[0]
    return new_action

def discrepancy_search(planner,env,state,action_space,plan):
    #ic (plan)
    discrepancy_locations = np.arange(0,len(plan))
    succesful_plans = []
    action_selection_function = discrepancy_get_best_action
    for discrepancy_loc in discrepancy_locations:
        action_loc = 0
        new_plan = []
        plan_found = False
        _, _ = env.reset()
        while len(new_plan) < len(plan):
            if action_loc < discrepancy_loc :
                state = env.step(plan[action_loc])
                state = state[0]
                new_plan.append(plan[action_loc])
                action_loc += 1
                continue
            feasible_action_list = None 
            #feasible_action_list = planner._guidance.get_feasible_action_param_list(env,state,action_space,ensemble)
            #if len(feasible_action_list) == [] :
            if feasible_action_list[-1] == -1:
                plan_found =False
                #succesful_plans.append(None)
                break

            new_action = action_selection_function(feasible_action_list,action_loc,discrepancy_loc)

            state = env.step(new_action[0])
            state = state[0]
            new_plan.append(new_action[0])
            plan_found = True
            for goal in state.goal.literals:
                if goal not in list(state.literals):
                    plan_found = False
            action_loc += 1
            if plan_found == True :
                succesful_plans.append(new_plan)
                break
        #ic ("plan 1 done")
        #ic (new_plan)
        if plan_found != True:
            succesful_plans.append(None)
        else :
            #ic (plan)
            #ic (new_plan)
            print ("Found new plan in discrepancy search")
            break

    return succesful_plans


def _test_planner(planner, domain_name, num_problems, 
                  timeout,current_problems_to_solve=None,
                  train_planner=None,
                  epoch_number=0,debug_level=constants.max_debug_level-1,
                  model=None,
                  graph_metadata=None,):
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

    #TODO - ADD CHECK HERE ABOUT WHETHER TO USE GPU
    device = "cuda:0"

    #average_success_time = {}

    if current_problems_to_solve == None :
        problems_to_solve = range(num_problems)
    else :
        problems_to_solve = current_problems_to_solve[:]

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

        state_2, _ = env_2.reset()
        #planning_time_start = time.time()
        action_space = env.action_space
        action_space = action_space._action_predicate_to_operators
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
            #pyperplan_task = grounding.ground(problem)
            groundings = env.action_space.all_ground_literals(state)
            prev_actions = None
            if len(new_plan) == 1 :
                prev_actions = [new_plan[0]]
            #if len(new_plan) >= 2 :
            #    prev_actions = [new_plan[-1],new_plan[-2]]
            prev_actions= None
            prev_graph = None
            inference_start_time = time.time()
            action_param_list = convert_state_and_run_model(model, state, action_space, device, groundings, graph_metadata)
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
            #for action_data in action_param_list :
            for new_action in action_param_list : 
                in_grounding = False

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

                if number_impossible_actions > max_plan_length_permitted :
                    break

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
                #succesful_plans = discrepancy_search(planner, env, start_state, action_space, ensemble,new_plan)
                succesful_plans = discrepancy_search(planner, env, start_state, action_space, new_plan)
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
        #if debug_level < max_debug_level - 1:
        #    for action_loop in new_plan :
        #        print (action_loop.__dict__)
    #ic(plan_lengths)
    # ic ("Plan success rate ", (1-failed_plans/num_problems))
    #ic("Plan success rate ", (1 - failed_plans / j))
    if debug_level <= max_debug_level :
        print ("Total Plan successes {}/{} , succes rate {}".format(j-failed_plans,j,1-(failed_plans/j) ))

    if True or debug_level == max_debug_level -1 :
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
        metrics = ModelMetrics(
                number_impossible_actions=number_impossible_actions,
                correct_plan_lengths_system=correct_plan_lengths_system,
                time_taken_system=total_correct_time_system/j,
                plan_success_rate = 1-(failed_plans/j),
                total_plan_successes=j-failed_plans,
            )
        return metrics

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
    #return failed_plans_locations


def run_planner_with_gnn_ltp(test_planner,train_planner, args,
                              model, graph_metadata):
    metrics = _test_planner(test_planner, args.domain+"Test",
                                            num_problems=args.num_test_problems, timeout=args.timeout,
                                            train_planner=train_planner,debug_level=args.debug_level,
                                            graph_metadata=graph_metadata,model=model)
    return metrics

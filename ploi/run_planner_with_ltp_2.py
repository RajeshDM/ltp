import pddlgym

import os
import torch
import random
seed = 10
import numpy as np
import warnings
import time
import argparse
from ploi.planning import FD 
from icecream import ic
import tempfile
import ploi.constants as constants
from ploi.datautils_ltp import state_to_graph_wrapper
from ploi.datautils_ltp import (
    graph_dataset_to_pyg_dataset,
)
from torch_geometric.loader import DataLoader as pyg_dataloader
from dataclasses import dataclass
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
import pddlgym
import tempfile
import time
from enum import Enum, auto

class PlannerType(Enum):
    LEARNED_MODEL = auto()
    NON_OPTIMAL = auto()
    OPTIMAL = auto()

@dataclass
class PlannerConfig:
    planner_types: List[PlannerType]
    domain_name: str
    num_problems: int
    timeout: float
    max_plan_length: int = 40
    problems_per_division: int = 10
    device: str = "cuda:0"
    debug_level: int = 0
    enable_state_monitor: bool = False

@dataclass
class PlannerMetrics:
    success_rate: float
    avg_plan_length: float
    avg_time_taken: float
    impossible_actions: int
    failures: Dict[int, List[int]]
    repeated_states: int = 0

class PlanningResult:
    def __init__(self):
        self.plan = []
        self.time_taken = 0
        self.success = False
        self.plan_length = 0
        self.repeated_states = 0 

# New class to add
class StateMonitor:
    def __init__(self):
        self.visited_states = set()
        
    def _state_to_hashable(self, state) -> tuple:
        return tuple(sorted(str(lit) for lit in state.literals))
        
    def has_visited(self, state) -> bool:
        return self._state_to_hashable(state) in self.visited_states
        
    def add_state(self, state):
        state_hash = self._state_to_hashable(state)
        self.visited_states.add(state_hash)

@dataclass
class ModelMetrics:
    """Data class to store metrics for a single model run"""
    number_impossible_actions: int
    correct_plan_lengths_system: int
    time_taken_system: float
    plan_success_rate : float
    total_plan_successes: float

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
        return opt_plan, opt_time_taken
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
            feasible_action_list = [] 
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

class PlannerTester:
    def __init__(self, config: PlannerConfig):
        self.config = config
        self.env = pddlgym.make(f"PDDLEnv{config.domain_name}-v0")
        self.metrics = {}

    def _is_valid_action(self, action, groundings) -> bool:
        for grounded_action in groundings:
            if (action.predicate == grounded_action.predicate and 
                all(v1 == v2 for v1, v2 in zip(action.variables, grounded_action.variables))):
                return True
        return False

    def _run_learned_model(self, problem_idx: int, action_space: Any, model: Any, 
                        graph_metadata: Any, use_monitor: bool = False) -> PlanningResult:
        result = PlanningResult()
        start_time = time.time()
        monitor = StateMonitor() if use_monitor else None
        
        # Initialize state
        self.env.fix_problem_index(problem_idx)
        state, _ = self.env.reset()
        
        if monitor:
            monitor.add_state(state)
        
        while True:
            groundings = list(self.env.action_space.all_ground_literals(state))
            action_param_list = convert_state_and_run_model(
                model, state, action_space, self.config.device, groundings, graph_metadata
            )
            
            valid_actions = [action for action in action_param_list 
                        if self._is_valid_action(action, groundings)]
            
            # If no valid actions at all, exit
            if not valid_actions:
                result.time_taken = time.time() - start_time
                return result
                
            # Try each valid action until we find one that doesn't lead to a repeated state
            action_taken = False
            for new_action in valid_actions:
                next_state = self.env.step(new_action)[0]
                
                if monitor and monitor.has_visited(next_state):
                    # Found a repeated state - try next action
                    result.repeated_states += 1
                    # Reset to current state to try next action
                    self.env.fix_problem_index(problem_idx)
                    state, _ = self.env.reset()
                    for action in result.plan:
                        state = self.env.step(action)[0]
                    continue
                
                # Found a non-repeating state
                if monitor:
                    monitor.add_state(next_state)
                state = next_state
                result.plan.append(new_action)
                action_taken = True
                
                if self._check_goal_reached(state):
                    result.success = True
                    result.time_taken = time.time() - start_time
                    result.plan_length = len(result.plan)
                    return result
                    
                break  # Break to get new action predictions for new state
            
            # If all actions led to repeated states, just take the first valid action and continue
            if not action_taken and valid_actions:
                new_action = valid_actions[0]
                state = self.env.step(new_action)[0]
                if monitor:
                    monitor.add_state(state)
                result.plan.append(new_action)
            
            # Check if plan is too long
            if len(result.plan) > self.config.max_plan_length:
                result.time_taken = time.time() - start_time
                return result

    def _check_goal_reached(self, state) -> bool:
        return all(goal in list(state.literals) for goal in state.goal.literals)

    def _run_external_planner(self, env, problem_idx, action_space, timeout, planner, optimal=False):
        result = PlanningResult()
        self.env.fix_problem_index(problem_idx)
        state, _ = self.env.reset()
        if optimal:
            plan, time_taken = run_opt_planner(env, state, action_space, timeout, planner)
        else:
            plan, time_taken = run_non_opt_planner(env, state, action_space, timeout, planner)

        if plan:
            result.plan = plan
            result.success = True
            result.plan_length = len(plan)
        result.time_taken = time_taken
        
        return result

    def test_planners(self, problems_to_solve: Optional[List[int]] = None,
                     model=None, graph_metadata=None) -> Dict[PlannerType, PlannerMetrics]:
        if problems_to_solve is None:
            problems_to_solve = range(min(self.config.num_problems, len(self.env.problems)))
            
        results = {planner_type: [] for planner_type in self.config.planner_types}
        
        for problem_idx in problems_to_solve:
            action_space = self.env.action_space._action_predicate_to_operators
            result = None
            
            for planner_type in self.config.planner_types:
                if planner_type == PlannerType.LEARNED_MODEL and model:
                    result = self._run_learned_model(problem_idx, action_space, model, graph_metadata)
                elif planner_type == PlannerType.NON_OPTIMAL:
                    #result = self._run_non_optimal_planner(problem_idx, None)  # Replace None with actual planner
                    result = self._run_external_planner(self.env, problem_idx, action_space, self.config.timeout, planner=None, optimal=False)
                elif planner_type == PlannerType.OPTIMAL:
                    result = self._run_external_planner(self.env, problem_idx, action_space, self.config.timeout, planner=None, optimal=True)
                    #result = self._run_optimal_planner(problem_idx, None)  # Replace None with actual planner
                    
                results[planner_type].append(result)

        return self._compute_metrics(results)

    def _compute_metrics(self, results: Dict[PlannerType, List[PlanningResult]]) -> Dict[PlannerType, PlannerMetrics]:
        metrics = {}
        
        for planner_type, planner_results in results.items():
            successful_results = [r for r in planner_results if r.success]
            num_problems = len(planner_results)
            
            if successful_results:
                avg_plan_length = np.mean([r.plan_length for r in successful_results])
                avg_time = np.mean([r.time_taken for r in successful_results])
            else:
                avg_plan_length = 0
                avg_time = 0
                
            total_repeated_states = sum(r.repeated_states for r in planner_results)
            
            metrics[planner_type] = PlannerMetrics(
                success_rate=len(successful_results) / num_problems,
                avg_plan_length=avg_plan_length,
                avg_time_taken=avg_time,
                repeated_states=total_repeated_states,
                impossible_actions=sum(1 for r in planner_results if not r.success),
                failures=self._compute_failures(planner_results)
            )
        
        return metrics

    def _compute_failures(self, planner_results: List[PlanningResult]) -> Dict[int, List[int]]:
        """
        Compute failures by division.
        
        Args:
            planner_results: List of PlanningResult objects
            
        Returns:
            Dictionary mapping division index to list of problem indices that failed in that division
        """
        failures = {}
        
        for i, result in enumerate(planner_results):
            if not result.success:
                div_idx = i // self.config.problems_per_division
                if div_idx not in failures:
                    failures[div_idx] = []
                failures[div_idx].append(i)
                
        return failures
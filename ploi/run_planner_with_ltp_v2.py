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
from torch_geometric.data import Batch
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
import pddlgym
import tempfile
import time
from tqdm import tqdm
from ploi.test_utils import (
    PlannerConfig, PlannerType, PlanningResult, PlannerMetrics,
    LearnedSearchStrat,
    compute_metrics,
    validate_strips_plan,
    learned_planner_types,
    baselines
)
import copy
from ploi.run_planner_with_ltp_v1 import (
    _create_planner,
)
import sys
import json
from io import StringIO
from functools import wraps

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from ploi.baselines.exp_1.plan import _plan as exp_1_learned_planner
from ploi.baselines.exp_2.train import _plan_exp_2 as exp_2_learned_planner
from ploi.baselines.exp_3.plan import _plan_exp_3 as exp_3_learned_planner
import pymimir as mm

# New class to add
class StateMonitor:
    def __init__(self):
        self.visited_states = set()

    def _state_to_hashable(self, state):
        # Old version (str-converts and sorts every literal, every state):
        #return tuple(sorted(str(lit) for lit in state.literals))
        # state.literals is already a frozenset of hashable Literals
        # (frozenset() on a frozenset returns the same object - no copy)
        return frozenset(state.literals)
        
    def has_visited(self, state) -> bool:
        return self._state_to_hashable(state) in self.visited_states
        
    def add_state(self, state):
        state_hash = self._state_to_hashable(state)
        self.visited_states.add(state_hash)

    def remove_state(self, state):
        """
        Remove a state from visited set (when backtracking)
        """
        state_hash = self._state_to_hashable(state)
        if state_hash in self.visited_states:
            self.visited_states.remove(state_hash)

@dataclass
class ModelMetrics:
    """Data class to store metrics for a single model run"""
    number_impossible_actions: int
    correct_plan_lengths_system: int
    time_taken_system: float
    plan_success_rate : float
    total_plan_successes: float

def silence_prints(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Store the original stdout
        original_stdout = sys.stdout
        # Redirect stdout to a null stream
        sys.stdout = StringIO()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore stdout
            sys.stdout = original_stdout
    return wrapper

def silence_all_output(func):
    """
    Decorator that suppresses ALL output including C/C++ prints
    by redirecting file descriptors
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get file descriptors for stdout and stderr
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        
        # Save copies of the file descriptors
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)
        
        try:
            # Open null device
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            
            # Replace file descriptors with null device
            os.dup2(devnull_fd, stdout_fd)
            os.dup2(devnull_fd, stderr_fd)
            
            result = func(*args, **kwargs)
            return result
            
        finally:
            # Restore original file descriptors
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            
            # Clean up
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(devnull_fd)
            
    return wrapper

def run_non_opt_planner(env,state,action_space,timeout,planner):
    try:
        #plan, time_taken = planner(env.domain, state, action_space, timeout=timeout)
        #plan, time_taken = planner(env.domain, state, timeout=timeout)
        start_time = time.time()
        plan = planner(env.domain, state, timeout=timeout)
        time_taken = time.time() - start_time
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

# State for GABAR_CHECK_PARALLEL_BEAM=1: caches the previous state's graph and
# v2 results so consecutive rollout states can be batched together through
# beam_search_parallel and compared per-graph against their v2 outputs.
# Exercises real mixed-size (and mixed-arity) batches, not just duplicates.
_beam_check = {"n": 0, "mismatches": 0, "g": None, "res": None, "model_id": None}

def _compare_beam_results(v2_results, par_results, tag):
    """Compare beam_search_v2 output against one batch entry of beam_search_parallel."""
    if len(v2_results) != len(par_results):
        print(f"[parallel-beam-check] {tag}: beam count differs "
              f"(v2={len(v2_results)}, parallel={len(par_results)})")
        return False
    ok = True
    for rank, ((s2, b2), (sp, bp)) in enumerate(zip(v2_results, par_results)):
        seq2 = [int(t) for t in b2]
        seqp = [int(t) for t in bp]
        score_close = abs(float(s2) - float(sp)) <= 1e-4
        if seq2 != seqp or not score_close:
            note = " (equal scores - tie reorder, likely benign)" \
                if score_close and seq2 != seqp else ""
            print(f"[parallel-beam-check] {tag}: rank {rank} differs: "
                  f"v2=({float(s2):.6f}, {seq2}) "
                  f"parallel=({float(sp):.6f}, {seqp}){note}")
            ok = False
            break
    return ok

def _check_parallel_beam(model, g_inp, v2_results, device):
    if _beam_check["model_id"] != id(model):
        _beam_check.update(g=None, res=None, model_id=id(model))
    if _beam_check["g"] is not None:
        _beam_check["n"] += 1
        tag = f"pair {_beam_check['n']}"
        try:
            model_input = convert_graph_to_model_input_v2(
                [_beam_check["g"], g_inp], device)
            with torch.inference_mode():
                par_results = model.forward_with_parallel_beam_search(model_input)
            ok = _compare_beam_results(_beam_check["res"], par_results[0], f"{tag} prev-state") \
                 and _compare_beam_results(v2_results, par_results[1], f"{tag} curr-state")
        except Exception as e:
            print(f"[parallel-beam-check] {tag}: parallel decode raised: {type(e).__name__}: {e}")
            ok = False
        if not ok:
            _beam_check["mismatches"] += 1
        if _beam_check["n"] % 100 == 0:
            print(f"[parallel-beam-check] {_beam_check['n']} pairs checked, "
                  f"{_beam_check['mismatches']} mismatches")
    _beam_check["g"] = g_inp
    _beam_check["res"] = v2_results

# Score tracing (GABAR_TRACE_SCORES=<path>): writes one JSON line per model
# call with every candidate's (score, token sequence), keyed by
# (epoch, problem, step). Compare two runs with compare_score_traces.py.
# Temporary verification apparatus for the batched-eval migration.
_score_trace = {"file": None, "epoch": None, "problem": None, "step": 0}

def _trace_scores(epoch, problem_idx, step_idx, results):
    path = os.environ.get("GABAR_TRACE_SCORES", "")
    if not path:
        return
    if _score_trace["file"] is None:
        _score_trace["file"] = open(path, "w")
    entry = {"epoch": epoch, "problem": int(problem_idx), "step": int(step_idx),
             "candidates": [[round(float(s), 6), [int(t) for t in seq]]
                            for s, seq in results]}
    _score_trace["file"].write(json.dumps(entry) + "\n")
    _score_trace["file"].flush()

def decode_beam_results(results, node_to_objects, action_space):
    """Turn one graph's sorted beam results into ranked pddlgym Literals.
    Shared by the sequential path and the batched harness."""
    all_actions = [k for k, v in action_space.items()]
    num_actions = len(all_actions)
    num_non_action_nodes = len(node_to_objects) - (num_actions)

    action_param_list = []
    for action_data in results:
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
                                    
    model_input = convert_graph_to_model_input_v2([g_inp],device)
    # Old version:
    #with torch.no_grad() :
    with torch.inference_mode() :
        #results = model(model_input, beam_search=True)
        #results = model.forward_with_parallel_beam_search(model_input, beam_search=True)
        #results = results[0]
        # Old (single-graph) decoder call:
        #results = model.forward_beam_decode(model_input)
        # GABAR_USE_PARALLEL_DECODER=1 routes even single-state calls through
        # the batched decoder (A/B test of beam_search_parallel; don't combine
        # with GABAR_CHECK_PARALLEL_BEAM, which assumes v2 results here)
        if os.environ.get("GABAR_USE_PARALLEL_DECODER", "") == "1":
            results = model.forward_with_parallel_beam_search(model_input)[0]
        else:
            results = model.forward_beam_decode(model_input)

    # Set GABAR_CHECK_PARALLEL_BEAM=1 to validate beam_search_parallel against
    # beam_search_v2 on real rollout state pairs (Tier 3 groundwork)
    if os.environ.get("GABAR_CHECK_PARALLEL_BEAM", "") == "1":
        _check_parallel_beam(model, g_inp, results, device)

    if os.environ.get("GABAR_TRACE_SCORES", ""):
        _trace_scores(_score_trace["epoch"], _score_trace["problem"],
                      _score_trace["step"], results)
        _score_trace["step"] += 1

    #print (results)
    # Old inline decode loop moved verbatim into decode_beam_results (shared
    # with the batched harness)
    return decode_beam_results(results, node_to_objects, action_space)

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

def _pad_variable_width_attrs(graphs):
    """Pad action_object_scores to uniform width across graphs so
    Batch.from_data_list can stack them. Training does this via
    _expand_graph_to_max_size_features; this is the test-time equivalent."""
    if not graphs:
        return graphs
    max_width = max(g['action_object_scores'].shape[1] for g in graphs)
    for g in graphs:
        ao = g['action_object_scores']
        if ao.shape[1] < max_width:
            padded = np.zeros((ao.shape[0], max_width), dtype=ao.dtype)
            padded[:, :ao.shape[1]] = ao
            g['action_object_scores'] = padded
    return graphs


def convert_graph_to_model_input_v2(g_inp, device):
    _pad_variable_width_attrs(g_inp)
    hetero_graphs = graph_dataset_to_pyg_dataset(g_inp, batch_wise=False)
    return Batch.from_data_list(hetero_graphs).to(device)

def convert_state_and_run_model_val(model, states, action_space ,
                                     device, groundings, 
                                graph_metadata,cheating_input=None):
    graph_inputs = []
    for state in states :
        g_inp , _, _= state_to_graph_wrapper(state,action_space,groundings,
                                                        prev_actions=None,prev_state=None,
                                                        graph_metadata=graph_metadata,
                                                        curr_action=None,objects=None,goal_state=state.goal,
                                                        cheating_input=cheating_input)
        graph_inputs.append(g_inp)

    model_input = convert_graph_to_model_input_v2(graph_inputs,device)
    # Old version:
    #with torch.no_grad() :
    with torch.inference_mode() :
        results = model(model_input).squeeze(1)
    return results


class PlannerTester:
    def __init__(self, config: PlannerConfig):
        self.config = config
        self.env = pddlgym.make(f"PDDLEnv{config.domain_name}Test-v0")
        #self.non_optimal_planner_data = {}
        #self.optimal_planner_data = {}
        # Initialize dictionary to store data for all planner types
        self.planner_data = {planner_type: {} for planner_type in self.config.planner_types}
        self.learned_search_strat = config.learned_search_strat

        self.load_planner_data()
        self.opt_planner = _create_planner(config.train_planner_name)
        self.non_opt_planner = _create_planner(config.eval_planner_name)
        self.metrics = {}

    def load_planner_data(self):
        """
        Load data for all planner types specified in config.
        """
        for planner_type in self.config.planner_types:
            filename = self.get_planner_filename(planner_type)
            self.planner_data[planner_type] = self.load_planner_data_from_file(filename)

    def get_planner_filename_old(self, planner_type: PlannerType) -> str:
        """
        Generate filename for storing planner data based on domain and planner type.
        Args:
            planner_type: Type of the planner
        Returns:
            str: Full path to the planner data file
        """
        domain_name = self.config.domain_name
        base_dir = "cache/results/planner_data"
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        if planner_type == PlannerType.LEARNED_MODEL:
            # Generate filename based on domain name
            #filename = f"{base_dir}/{domain_name}_"#learned_model.json"
            #return filename
            param_strs = []
            filename = ""
            for k,v in sorted(self.config.model_hyperparameters.items()):
                if isinstance(v, float):
                    # Format floating point numbers nicely
                    param_str = f"{k}{v:.0e}" if v < 0.01 else f"{k}{v}"
                else:
                    if k == 'g_node' : 
                        if v is True :
                            continue
                    param_str = f"{k}{v}"
                param_strs.append(param_str)
            
            if param_strs:
                filename += "_" + "_".join(param_strs)
            
            # Clean up any characters that might cause issues
            #folder_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in folder_name)
            filename = f"{base_dir}/{domain_name}_{filename}.json"
            return filename 
        
        # Generate filename based on planner type
        filename = f"{base_dir}/{domain_name}_{planner_type.name.lower()}.json"
        return filename

    def get_planner_filename(self, planner_type: PlannerType ) -> str:
        """
        Generate filename for storing planner data based on domain and planner type.
        
        Args:
            planner_type: Type of the planner
            ignore_defaults: Dictionary of param_name: default_value pairs to ignore when at default
            
        Returns:
            str: Full path to the planner data file
        """
        # Default to empty dict if None
        
        domain_name = self.config.domain_name
        base_dir = "cache/results/planner_data"
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        ignore_defaults =self.config.ignore_defaults
        
        if planner_type in learned_planner_types :# == PlannerType.LEARNED_MODEL:
            # Generate filename based on model hyperparameters
            param_strs = []
            
            for k, v in sorted(self.config.model_hyperparameters.items()):
                # Skip if this parameter is at its default value
                if k in ignore_defaults and v == ignore_defaults[k]:
                    continue
                    
                # Format the parameter value
                if isinstance(v, float):
                    # Format floating point numbers nicely
                    param_str = f"{k}{v:.0e}" if v < 0.01 else f"{k}{v}"
                else:
                    param_str = f"{k}{v}"
                    
                param_strs.append(param_str)

            training_param_part = "_" + "_".join(param_strs) if param_strs else ""
            testing_param_strs = []
            
            for k, v in sorted(self.config.testing_hyperparameters.items()):
                testing_param_strs.append(f'{v}')

            testing_param_part = "_".join(testing_param_strs) if testing_param_strs else "" + "_"
            
            #filename = f"{base_dir}/{domain_name}{param_part}.json"
            filename = f"{base_dir}/{testing_param_part}{training_param_part}.json"
            
            return filename
        
        # Generate filename based on planner type for non-learned models
        filename = f"{base_dir}/{domain_name}_{planner_type.name.lower()}.json"
        return filename

    def load_planner_data_from_file(self, filename: str) -> Dict:
        """
        Load planner data from a JSON file if it exists.
        Args:
            filename: Path to the JSON file
        Returns:
            dict: Loaded data or empty dict if file doesn't exist
        """
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return json.load(f)
        else :
            return {}

    def save_planner_data(self):
        """
        Save data for all planner types to their respective files.
        """
        for planner_type in self.config.planner_types:
            filename = self.get_planner_filename(planner_type )
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Save data to file
            with open(filename, "w") as f:
                json.dump(self.planner_data[planner_type], f, indent=4)

    '''
    def save_planner_data_old(self):
        opt_filename, non_opt_filename = self.get_planner_filename_old()

        if PlannerType.NON_OPTIMAL in self.config.planner_types:
            with open(non_opt_filename, "w") as f:
                json.dump(self.non_optimal_planner_data, f)
            
        if PlannerType.OPTIMAL in self.config.planner_types:
            with open(opt_filename, "w") as f:
                json.dump(self.optimal_planner_data, f)

    def get_planner_filename_old(self) :
        domain_name = self.config.domain_name
        base_dir = "cache/results/planner_data"
        non_opt_filename = f"{base_dir}/" + domain_name + "_non_opt.json" 
        opt_filename = f"{base_dir}/" + domain_name + "_opt.json"

        return opt_filename, non_opt_filename

    def load_planner_data_from_file_old(self, filename)  :
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return json.load(f)
        else :
            return {}

    def load_planner_data_old(self)  :
        opt_filename, non_opt_filename = self.get_planner_filename_old() 

        if PlannerType.NON_OPTIMAL in self.config.planner_types:
            self.non_optimal_planner_data = self.load_planner_data_from_file(non_opt_filename)

        if PlannerType.OPTIMAL in self.config.planner_types:
            self.optimal_planner_data = self.load_planner_data_from_file(opt_filename)
    '''

    def _is_valid_action(self, action, groundings) -> bool:
        for grounded_action in groundings:
            if (action.predicate == grounded_action.predicate and
                all(v1 == v2 for v1, v2 in zip(action.variables, grounded_action.variables))):
                return True
        return False

    @staticmethod
    def _make_grounding_set(groundings):
        """Precompute a hashable set of groundings for O(1) validity checks."""
        return {(g.predicate, tuple(g.variables)) for g in groundings}

    @staticmethod
    def _is_valid_action_fast(action, grounding_set) -> bool:
        return (action.predicate, tuple(action.variables)) in grounding_set

    def _run_learned_model(self, problem_idx: int, action_space: Any, model_epoch: Any, 
                        graph_metadata: Any, use_monitor: bool = False):
                        #strategy : LearnedSearchStrat= LearnedSearchStrat.GREEDY) -> PlanningResult:
        result = PlanningResult()
        result.problem_idx = problem_idx
        start_time = time.time()
        monitor = StateMonitor() if use_monitor else None
        model, epoch = model_epoch[0], model_epoch[1]
        
        # Initialize state
        self.env.fix_problem_index(problem_idx)
        state, _ = self.env.reset()
        groundings = list(self.env.action_space.all_ground_literals(state, reground=True))
        fname = self.env.problems[problem_idx].problem_fname
        fname = "/".join(fname.split("/")[-2:]) 
        fname = fname + "_" + str(epoch)

        planner_data = self.planner_data[PlannerType.LEARNED_MODEL]

        if fname in planner_data and False:
            result.success = True
            result.plan_length, result.time_taken = planner_data[fname]
            return result

        if monitor:
            monitor.add_state(state)

        # Context for GABAR_TRACE_SCORES (score-level verification)
        _score_trace["epoch"] = epoch
        _score_trace["problem"] = problem_idx
        _score_trace["step"] = 0

        # CURRENLY SINGLE SEARCH TYPE ALLOWED - CAN EXTEND TO ALL LATER IF NEEDED
        strategy = self.config.learned_search_strat[0]
        # Choose strategy based on input
        if strategy == LearnedSearchStrat.GREEDY:
            return self._run_greedy_search(state, model, action_space, graph_metadata, monitor, 
                                           result, start_time, fname, planner_data)
        elif strategy == LearnedSearchStrat.DFS:
            # DFS Prune Factor
            dfs_width_prune_factor = 2
            return self._run_iterative_deepening_search(state, model, action_space, graph_metadata, monitor, 
                                           result, start_time, fname, planner_data,
                                           max_depth=self.config.max_plan_length ,
                                           prune_factor=dfs_width_prune_factor)
        elif strategy == LearnedSearchStrat.BFS:
            # Placeholder for your DFS implementation
            # return self._run_dfs_search(state, model, action_space, graph_metadata, monitor, result, start_time, fname)
            raise NotImplementedError("BFS strategy not implemented yet")
        elif strategy == LearnedSearchStrat.MCTS:
            # Placeholder for your DFS implementation
            # return self._run_dfs_search(state, model, action_space, graph_metadata, monitor, result, start_time, fname)
            raise NotImplementedError("BFS strategy not implemented yet")
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")

    def _run_greedy_search(self, state, model, action_space, graph_metadata,monitor,
                            result,start_time, fname, planner_data):
        while True:
            groundings = list(self.env.action_space.all_ground_literals(state))
            action_param_list = convert_state_and_run_model(
                model, state, action_space, self.config.device, groundings, graph_metadata
            )

            # Old version (linear scan over all groundings per candidate):
            #valid_actions = [action for action in action_param_list
            #            if self._is_valid_action(action, groundings)]
            grounding_set = self._make_grounding_set(groundings)
            valid_actions = [action for action in action_param_list
                        if self._is_valid_action_fast(action, grounding_set)]
            
            # If no valid actions at all, exit
            if not valid_actions:
                result.time_taken = time.time() - start_time

                groundings = list(self.env.action_space.all_ground_literals(state))
                return result
                
            # Try each valid action until we find one that doesn't lead to a repeated state
            action_taken = False
            for new_action in valid_actions:
                next_state = self.env.step(new_action)[0]
                
                if monitor and monitor.has_visited(next_state):
                    # Found a repeated state - try next action
                    result.repeated_states += 1
                    # Reset to current state to try next action
                    self.env.set_state(state)
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
                    planner_data[fname] = (result.plan_length, result.time_taken)
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

        return result

    def _greedy_step_single(self, st, action_param_list, groundings, planner_data, start_time):
        """One greedy-search round for one problem. Mirrors the body of the
        while-loop in _run_greedy_search exactly (same candidate order, same
        monitor/fallback/goal/plan-length logic). Returns True when the
        problem is finished (success or failure)."""
        env, state, monitor, result = st["env"], st["state"], st["monitor"], st["result"]

        grounding_set = self._make_grounding_set(groundings)
        valid_actions = [action for action in action_param_list
                    if self._is_valid_action_fast(action, grounding_set)]

        if not valid_actions:
            result.time_taken = time.time() - start_time
            return True

        action_taken = False
        for new_action in valid_actions:
            next_state = env.step(new_action)[0]

            if monitor and monitor.has_visited(next_state):
                result.repeated_states += 1
                env.set_state(state)
                continue

            if monitor:
                monitor.add_state(next_state)
            st["state"] = next_state
            result.plan.append(new_action)
            action_taken = True

            if self._check_goal_reached(next_state):
                result.success = True
                result.time_taken = time.time() - start_time
                result.plan_length = len(result.plan)
                planner_data[st["fname"]] = (result.plan_length, result.time_taken)
                return True

            break

        if not action_taken and valid_actions:
            new_action = valid_actions[0]
            st["state"] = env.step(new_action)[0]
            if monitor:
                monitor.add_state(st["state"])
            result.plan.append(new_action)

        if len(result.plan) > self.config.max_plan_length:
            result.time_taken = time.time() - start_time
            return True

        return False

    def _run_learned_model_batch(self, problem_idxs, action_space, model_epoch,
                                 graph_metadata, use_monitor=False):
        """Greedy search over many problems in lockstep: one batched
        encoder+decoder call per round via forward_with_parallel_beam_search.
        Per problem, trajectories match _run_greedy_search exactly when the
        parallel decoder's scores match beam_search_v2 (verify with
        GABAR_TRACE_SCORES + compare_score_traces.py).

        Note: result.time_taken is wall-clock from batch start to that
        problem's finish - not comparable with sequential per-problem times.
        """
        model, epoch = model_epoch[0], model_epoch[1]
        strategy = self.config.learned_search_strat[0]
        if strategy != LearnedSearchStrat.GREEDY:
            raise NotImplementedError("GABAR_BATCH_EVAL currently supports greedy search only")

        planner_data = self.planner_data[PlannerType.LEARNED_MODEL]
        start_time = time.time()

        # One env per problem: identical per-problem semantics to the
        # sequential path (which fixes one shared env per problem in turn)
        active, results_by_problem = {}, {}
        for p in problem_idxs:
            env = pddlgym.make(f"PDDLEnv{self.config.domain_name}Test-v0")
            env.fix_problem_index(p)
            state, _ = env.reset()
            # prime the action-space grounding cache, as _run_learned_model does
            list(env.action_space.all_ground_literals(state, reground=True))
            result = PlanningResult()
            result.problem_idx = p
            monitor = StateMonitor() if use_monitor else None
            if monitor:
                monitor.add_state(state)
            fname = env.problems[p].problem_fname
            fname = "/".join(fname.split("/")[-2:]) + "_" + str(epoch)
            active[p] = {"env": env, "state": state, "monitor": monitor,
                         "result": result, "fname": fname, "step": 0}
            results_by_problem[p] = result

        round_num = 0
        while active:
            round_num += 1
            batch_order = sorted(active.keys())

            graphs, groundings_by_problem = [], {}
            for p in batch_order:
                st = active[p]
                groundings = list(st["env"].action_space.all_ground_literals(st["state"]))
                groundings_by_problem[p] = groundings
                g_inp, _, node_to_objects = state_to_graph_wrapper(
                    st["state"], action_space, groundings,
                    prev_actions=None, prev_state=None,
                    graph_metadata=graph_metadata,
                    curr_action=None, objects=None,
                    goal_state=st["state"].goal, cheating_input=None)
                st["node_to_objects"] = node_to_objects
                graphs.append(g_inp)

            model_input = convert_graph_to_model_input_v2(graphs, self.config.device)
            with torch.inference_mode():
                batch_results = model.forward_with_parallel_beam_search(model_input)

            finished = []
            for i, p in enumerate(batch_order):
                st = active[p]
                if os.environ.get("GABAR_TRACE_SCORES", ""):
                    _trace_scores(epoch, p, st["step"], batch_results[i])
                st["step"] += 1
                action_param_list = decode_beam_results(
                    batch_results[i], st["node_to_objects"], action_space)
                if self._greedy_step_single(st, action_param_list,
                                            groundings_by_problem[p],
                                            planner_data, start_time):
                    finished.append(p)
            for p in finished:
                del active[p]

            if round_num % 50 == 0:
                print(f"[batch-eval] round {round_num}: {len(active)} problems still active", flush=True)

        n_success = sum(1 for r in results_by_problem.values() if r.success)
        print(f"[batch-eval] done: {len(problem_idxs)} problems, {n_success} solved, "
              f"{round_num} rounds, {time.time() - start_time:.1f}s", flush=True)
        return [results_by_problem[p] for p in problem_idxs]

    def _run_iterative_deepening_search(self, state, model, action_space, graph_metadata, monitor,
                                        result, start_time, fname, planner_data, max_depth, prune_factor=3,
                                        pure_dfs=True):
        """
        Run iterative deepening search with cycle avoidance
        """
        # Initialize the initial state in the monitor
        if monitor:
            monitor.add_state(state)

        min_depth = 1
        if pure_dfs == True :
            min_depth = max_depth
        
        for depth_limit in range(min_depth, max_depth + 1):
            #print(f"Searching with depth limit: {depth_limit}")
            
            # Create a copy of the current state to reset between iterations
            current_state = self.env.get_state()
            
            # Reset result plan for this depth-limited search
            temp_plan = []
            
            # Run depth-limited search
            success = self._depth_limited_dfs( state,model,action_space, graph_metadata, monitor,
                depth_limit,prune_factor, temp_plan, result,current_state
            )
            
            if success:
                # Update the result with the successful plan
                result.plan = temp_plan
                result.success = True
                result.time_taken = time.time() - start_time
                result.plan_length = len(result.plan)
                planner_data[fname] = (result.plan_length, result.time_taken)
                return result
            
            # Reset environment to initial state for next iteration
            self.env.set_state(state)
        
        # If no solution found at any depth
        result.time_taken = time.time() - start_time
        return result

    def _depth_limited_dfs(self, state, model, action_space, graph_metadata, monitor, 
                        depth_limit, prune_factor, current_plan, result, original_state):
        """
        Depth-limited DFS with cycle detection
        """
        # Check if goal is reached
        if self._check_goal_reached(state):
            return True
            
        # If depth limit is reached, return failure
        if depth_limit <= 0:
            result.cutoffs += 1
            return False
        
        # Check if plan is too long
        if len(current_plan) > self.config.max_plan_length:
            return False
        
        # Get and rank actions
        groundings = list(self.env.action_space.all_ground_literals(state))
        action_param_list = convert_state_and_run_model(
            model, state, action_space, self.config.device, groundings, graph_metadata
        )

        # Filter valid actions
        # Old version (linear scan over all groundings per candidate):
        #valid_actions = [action for action in action_param_list
        #                if self._is_valid_action(action, groundings)]
        grounding_set = self._make_grounding_set(groundings)
        valid_actions = [action for action in action_param_list
                        if self._is_valid_action_fast(action, grounding_set)]
        
        # If no valid actions, return failure
        if not valid_actions:
            result.deadends += 1
            return False
        
        # Prune to top actions if needed
        if prune_factor < len(valid_actions):
            valid_actions = valid_actions[:prune_factor]
        
        # Try each valid action in order of neural network ranking
        for new_action in valid_actions:
            result.nodes_expanded += 1
            
            # Apply the action to get the next state
            next_state = self.env.step(new_action)[0]
            
            # Check if this state has been visited before (cycle detection)
            if monitor and monitor.has_visited(next_state):
                # Found a repeated state - try next action
                result.repeated_states += 1
                # Reset to current state to try next action
                self.env.set_state(state)
                continue
            
            # Mark the new state as visited
            if monitor:
                monitor.add_state(next_state)
            
            # Add action to current plan
            current_plan.append(new_action)
            
            # Recursively search from the new state
            if self._depth_limited_dfs(
                next_state,
                model,
                action_space,
                graph_metadata,
                monitor,
                depth_limit - 1,
                prune_factor,
                current_plan,
                result,
                original_state
            ):
                return True
            
            # If search failed, backtrack
            current_plan.pop()  # Remove the last action
            self.env.set_state(state)
            
            if monitor:
                monitor.remove_state(next_state)
        
        # If all actions failed, return failure
        return False

    def _get_successor_states(self,state, applicable_actions):
        return [ (action, self._apply_action(state, action)) for action in applicable_actions ]

    def _apply_action(self,state,action):
        new_state = self.env.step(action)[0] 
        self.env.set_state(state)
        return new_state

    def _run_learned_model_val(self, problem_idx: int, action_space: Any, model_epoch: Any, 
                        graph_metadata: Any, use_monitor: bool = False) -> PlanningResult:

        planner_type = PlannerType.LEARNED_MODEL_VAL
        result = PlanningResult()
        result.problem_idx = problem_idx
        start_time = time.time()
        monitor = StateMonitor() if use_monitor else None
        model, epoch = model_epoch[0], model_epoch[1]
        
        # Initialize state
        self.env.fix_problem_index(problem_idx)
        state, _ = self.env.reset()
        groundings = list(self.env.action_space.all_ground_literals(state, reground=True))
        fname = self.env.problems[problem_idx].problem_fname
        fname = "/".join(fname.split("/")[-2:]) 
        fname = fname + "_" + str(epoch)

        planner_data = self.planner_data[planner_type]

        if fname in planner_data and False:
            result.success = True
            result.plan_length, result.time_taken = planner_data[fname]
            return result 
        
        if monitor:
            monitor.add_state(state)

        while True:
            groundings = list(self.env.action_space.all_ground_literals(state))
            # explore current state (avoid loops by removing already visited successors)
            successor_candidates = [ transition for transition in self._get_successor_states(state, groundings) ]

            #If no successor exists, we are at a dead end - then we return 
            if len(successor_candidates) == 0:
                return result

            if monitor :
                successor_candidates_refined = [ transition for transition in successor_candidates
                                         if not monitor.has_visited(transition[1]) ]

                #If at least one non-visited successor exists, then that list is the possible next states
                if len(successor_candidates_refined) != 0:
                    successor_candidates = successor_candidates_refined[:]

            successor_actions = [ candidate[0] for candidate in successor_candidates ]
            successor_states = [ candidate[1] for candidate in successor_candidates ]
            #collated_input, encoded_states = _to_input(successor_states, goal_denotation, obj_encoding, augment_fn, language, device, logger)
            #output_values, output_solvables = model(collated_input)
            output_values = convert_state_and_run_model_val(model, successor_states,
                                                            action_space,self.config.device,
                                                            groundings, graph_metadata)
            best_successor_index = torch.argmin(output_values)
            state = successor_states[best_successor_index]
            new_action = successor_actions[best_successor_index]
            self.env.step(new_action)

            if monitor :
                monitor.add_state(state)

            result.plan.append(new_action)
            action_taken = True
            
            if self._check_goal_reached(state):
                result.success = True
                result.time_taken = time.time() - start_time
                result.plan_length = len(result.plan)
                planner_data[fname] = (result.plan_length, result.time_taken)
                return result

            # Check if plan is too long
            if len(result.plan) > self.config.max_plan_length:
                result.time_taken = time.time() - start_time
                return result


    def _check_goal_reached(self, state) -> bool:
        # Old version (rebuilds a list of all state literals per goal literal,
        # then does O(n) membership on it):
        #return all(goal in list(state.literals) for goal in state.goal.literals)
        # state.literals is a frozenset - O(1) membership, no copy
        return all(goal in state.literals for goal in state.goal.literals)

    def _run_external_planner(self, env, problem_idx, action_space, timeout, optimal=False):
        result = PlanningResult()
        result.problem_idx = problem_idx
        self.env.fix_problem_index(problem_idx)
        state, _ = self.env.reset()
        fname = env.problems[problem_idx].problem_fname
        fname = "/".join(fname.split("/")[-2:])

        plan_len = -1 
        plan = False

        if not optimal :
            planner_to_send = self.non_opt_planner
            function_to_run = run_non_opt_planner
            planner_data = self.planner_data[PlannerType.NON_OPTIMAL]#self.non_optimal_planner_data
        else :
            planner_to_send = self.opt_planner
            function_to_run = run_opt_planner
            planner_data = self.planner_data[PlannerType.OPTIMAL]#self.optimal_planner_data

        if fname in planner_data:
            plan_len, time_taken = planner_data[fname]
            if plan_len != -1:
                plan = True
        else :
            plan, time_taken = function_to_run(env, state, action_space, timeout,planner_to_send)
            if plan is not None :
                plan_len = len(plan)
            planner_data[fname] = (plan_len,time_taken)

        if plan:
            result.success = True
            result.plan_length = plan_len
        result.time_taken = time_taken
        return result

    @silence_all_output
    def create_baseline_parser(self, domain_file, problem_file):
        parser = mm.PDDLParser(domain_file, problem_file)
        factories = parser.get_pddl_repositories()
        problem = parser.get_problem()
        return problem, factories

    def _run_exp_baseline_3(self, env, problem_idx, model, 
                            plan_function, planner_type,
                            hyperparams):
        result = PlanningResult()
        result.problem_idx = problem_idx

        domain_file, fname, problem_file = self.get_domain_name_fname_from_env(env, problem_idx)
        planner_data = self.planner_data[planner_type]

        fname = fname + "_" + str(hyperparams['aggregation']) + "_" + str(hyperparams['loss_fn'])

        '''
        if fname in planner_data:
            result.plan_length, result.time_taken = planner_data[fname]
            result.success = True
            return result 
        '''

        start_time = time.time()
        solution = plan_function(domain_file, problem_file, model, self.config.device, self.config.max_plan_length)
        time_taken = time.time() - start_time

        if solution is not None:
            planner_data[fname] = (len(solution), time_taken)
            result.success = True
            result.plan_length =len(solution) 

        result.time_taken = time_taken
        return result


    def get_domain_name_fname_from_env(self, env, problem_idx):
        problem_file = env.problems[problem_idx].problem_fname
        domain_file = env.domain.domain_fname
        fname = env.problems[problem_idx].problem_fname
        fname = "/".join(fname.split("/")[-2:])
        return domain_file, fname, problem_file 

    def _run_exp_baseline(self, env, problem_idx, model, plan_function, planner_type):
        result = PlanningResult()
        result.problem_idx = problem_idx

        domain_file, fname, problem_file = self.get_domain_name_fname_from_env(env, problem_idx)
        planner_data = self.planner_data[planner_type]

        if fname in planner_data:
            result.plan_length, result.time_taken = planner_data[fname]
            result.success = True
            return result 

        problem, factories = self.create_baseline_parser(domain_file, problem_file)
        start_time = time.time()
        solution = plan_function(problem, factories, model, self.config.device, self.config.max_plan_length)
        time_taken = time.time() - start_time

        if solution is not None:
            planner_data[fname] = (len(solution), time_taken)
            result.success = True
            result.plan_length =len(solution) 

        result.time_taken = time_taken
        return result

    def test_planners(self, problems_to_solve: Optional[List[int]] = None,
                     models=None, graph_metadata=None) -> Dict[PlannerType, PlannerMetrics]:
        if problems_to_solve is None:
            problems_to_solve = range(min(self.config.num_problems, len(self.env.problems)))
            
        #results = {planner_type: [] for planner_type in self.config.planner_types}
        results = {}
        number_divisions = max(int((max(problems_to_solve)) / self.config.problems_per_division), 1) + 1
        self.failure_dict = {i:[] for i in range(int(number_divisions) )}
        success_until_now_for_learned = 0
        progress_bar = tqdm(problems_to_solve)

        # Set GABAR_PROFILE_EVAL=1 to cProfile the first learned-model problem
        # (output goes to stdout, so it lands in the run log too)
        profile_eval = os.environ.get("GABAR_PROFILE_EVAL", "") == "1"
        profile_done = False

        # Set GABAR_BATCH_EVAL=1 to run all learned-model problems in lockstep
        # with one batched model call per round (greedy search only)
        batch_learned_results = None
        if (os.environ.get("GABAR_BATCH_EVAL", "") == "1"
                and models is not None
                and PlannerType.LEARNED_MODEL in self.config.planner_types
                and PlannerType.LEARNED_MODEL in models):
            batch_results_list = self._run_learned_model_batch(
                list(problems_to_solve),
                self.env.action_space._action_predicate_to_operators,
                models[PlannerType.LEARNED_MODEL],
                graph_metadata,
                use_monitor=self.config.enable_state_monitor)
            batch_learned_results = {r.problem_idx: r for r in batch_results_list}

        for problem_idx in progress_bar:
            action_space = self.env.action_space._action_predicate_to_operators
            result = None

            for planner_type in self.config.planner_types:
                if planner_type == PlannerType.LEARNED_MODEL and PlannerType.LEARNED_MODEL in models:
                    if batch_learned_results is not None:
                        result = batch_learned_results[problem_idx]
                    elif profile_eval and not profile_done:
                        import cProfile, pstats
                        profiler = cProfile.Profile()
                        profiler.enable()
                        result = self._run_learned_model(problem_idx, action_space, models[planner_type], graph_metadata, use_monitor=self.config.enable_state_monitor)
                        profiler.disable()
                        profile_done = True
                        print(f"\n=== cProfile: problem {problem_idx} (learned model) ===")
                        stats = pstats.Stats(profiler, stream=sys.stdout)
                        stats.sort_stats('cumulative').print_stats(30)
                        print("=== end cProfile ===\n")
                    else:
                        result = self._run_learned_model(problem_idx, action_space, models[planner_type], graph_metadata, use_monitor=self.config.enable_state_monitor)
                    if result.success :
                        success_until_now_for_learned  += 1
                    #tqdm.set_postfix(success=f"{success_until_now_for_learned}")
                    progress_bar.set_postfix(success=f"{success_until_now_for_learned}")

                if planner_type == PlannerType.LEARNED_MODEL_VAL and PlannerType.LEARNED_MODEL_VAL in models:
                    result = self._run_learned_model_val(problem_idx, action_space, models[planner_type], graph_metadata, use_monitor=self.config.enable_state_monitor)
                
                elif planner_type == PlannerType.EXP_BASELINE and PlannerType.EXP_BASELINE in models:
                    result = self._run_exp_baseline(self.env, problem_idx, models[planner_type][0], exp_1_learned_planner, planner_type)

                elif planner_type == PlannerType.EXP_BASELINE_2 and PlannerType.EXP_BASELINE_2 in models:
                    result = self._run_exp_baseline(self.env, problem_idx, models[planner_type][0], exp_2_learned_planner, planner_type)

                elif planner_type == PlannerType.EXP_BASELINE_2 and PlannerType.EXP_BASELINE_2 in models:
                    result = self._run_exp_baseline(self.env, problem_idx, models[planner_type][0], exp_2_learned_planner, planner_type)

                elif planner_type == PlannerType.EXP_BASELINE_3 and PlannerType.EXP_BASELINE_3 in models:
                    result = self._run_exp_baseline_3(self.env, problem_idx, models[planner_type][0], exp_3_learned_planner, planner_type, models[planner_type][2])

                elif planner_type == PlannerType.NON_OPTIMAL:
                    result = self._run_external_planner(self.env, problem_idx, action_space, self.config.timeout, optimal=False)

                elif planner_type == PlannerType.OPTIMAL:
                    result = self._run_external_planner(self.env, problem_idx, action_space, self.config.timeout, optimal=True)
                    
                if result is not None : 
                    if planner_type not in results :
                        results[planner_type] = []
                    results[planner_type].append(result)

        self.save_planner_data()

        #return self._compute_metrics(results)
        return results, compute_metrics(self.config.problems_per_division , results , self.failure_dict )
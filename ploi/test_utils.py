#from ploi.run_planner_with_ltp_2 import PlannerTester, PlannerConfig, PlannerType
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Any 
import wandb
import numpy as np
import tempfile
import os
import subprocess

class LearnedSearchStrat(Enum):
    """Defines search strategies for learned models."""
    GREEDY = auto()
    DFS = auto()
    BFS = auto()
    MCTS = auto()
    # Add more search strategies as needed

class PlannerType(Enum):
    """
    Main planner types.
    For LEARNED_MODEL and LEARNED_MODEL_VAL, use with_strategy() to specify the search strategy.
    """
    LEARNED_MODEL = auto()
    LEARNED_MODEL_VAL = auto()
    NON_OPTIMAL = auto()
    OPTIMAL = auto()
    EXP_BASELINE = auto()
    EXP_BASELINE_2 = auto()
    EXP_BASELINE_3 = auto()
    
    def with_strategy(self, strategy: LearnedSearchStrat) -> 'PlannerTypeWithStrategy':
        """
        Attach a search strategy to this planner type.
        Only valid for LEARNED_MODEL and LEARNED_MODEL_VAL.
        
        Args:
            strategy: The search strategy to use
            
        Returns:
            A PlannerTypeWithStrategy combining this planner type with the given strategy
        
        Raises:
            ValueError: If attaching a strategy to a non-learned model planner type
        """
        if self not in [PlannerType.LEARNED_MODEL, PlannerType.LEARNED_MODEL_VAL]:
            raise ValueError(f"Cannot attach search strategy to {self.name}")
        return PlannerTypeWithStrategy(self, strategy)

class PlannerTypeWithStrategy:
    """
    Combined planner type and search strategy.
    Used for learned model planners that require a specific search strategy.
    """
    def __init__(self, planner_type: PlannerType, strategy: LearnedSearchStrat):
        if planner_type not in [PlannerType.LEARNED_MODEL, PlannerType.LEARNED_MODEL_VAL]:
            raise ValueError(f"Cannot attach search strategy to {planner_type.name}")
        self.planner_type = planner_type
        self.strategy = strategy
    
    def __eq__(self, other):
        if isinstance(other, PlannerTypeWithStrategy):
            return self.planner_type == other.planner_type and self.strategy == other.strategy
        return False
    
    def __hash__(self):
        return hash((self.planner_type, self.strategy))
    
    def __str__(self):
        return f"{self.planner_type.name}_{self.strategy.name}"
    
    def __repr__(self):
        return f"PlannerTypeWithStrategy({self.planner_type.name}, {self.strategy.name})"

@dataclass
class PlannerConfig:
    planner_types: List[PlannerType]
    domain_name: str
    num_problems: int
    timeout: float
    learned_search_strat: List[LearnedSearchStrat] 
    max_plan_length: int = 40
    problems_per_division: int = 10
    device: str = "cuda:0"
    debug_level: int = 0
    enable_state_monitor: bool = False
    eval_planner_name :str = ""
    train_planner_name :str = "" 
    model_hyperparameters: Dict[str, float] = None
    ignore_defaults : Dict[str, Any] = None
    testing_hyperparameters: Dict[str, Any] = None

@dataclass
class PlannerMetrics:
    failures: Dict[int, List[int]]
    success_rate_with_monitor : float = 0
    success_rate_without_monitor : float = 0
    avg_plan_length: float = 0
    avg_time_taken: float = 0
    impossible_actions: int = 0
    repeated_states: int = 0
    max_plan_length: int = 0
    max_time_taken: float = 0
    plan_quality: float = 0
    median_plan_length: float = 0

class PlanningResult:
    def __init__(self):
        self.plan = []
        self.time_taken = 0
        self.success = False
        self.plan_length = 0
        self.repeated_states = 0 
        self.problem_idx = -1
        self.nodes_expanded = 0
        self.cutoffs = 0

learned_planner_types = [
    PlannerType.LEARNED_MODEL,
    PlannerType.LEARNED_MODEL_VAL
]

baselines = [PlannerType.EXP_BASELINE, 
             PlannerType.EXP_BASELINE_2, 
             PlannerType.EXP_BASELINE_3] 

def compute_metrics(problems_per_division, 
                    results: Dict[PlannerType, List[PlanningResult]],
                    failure_dict) -> Dict[PlannerType, PlannerMetrics]:
    metrics = {}
    
    for planner_type, planner_results in results.items():
        successful_results_with_monitor = [r for r in planner_results if r.success]
        successful_results_without_monitor = [r for r in planner_results if r.success and r.repeated_states == 0]
        num_problems = len(planner_results)
        
        
        if successful_results_with_monitor:
            avg_plan_length = np.mean([r.plan_length for r in successful_results_with_monitor])
            avg_time = np.mean([r.time_taken for r in successful_results_with_monitor])
            max_plan_length = max(r.plan_length for r in planner_results if r.success)
            max_time_taken = max(r.time_taken for r in planner_results if r.success)
            median_plan_length = np.median([r.plan_length for r in successful_results_with_monitor])
        else:
            avg_plan_length = 0
            avg_time = 0
            max_plan_length = 0
            max_time_taken = 0
            median_plan_length = 0
            
        total_repeated_states = sum(r.repeated_states for r in planner_results)
        
        metrics[planner_type] = PlannerMetrics(
            success_rate_with_monitor=len(successful_results_with_monitor) / num_problems,
            success_rate_without_monitor=len(successful_results_without_monitor) / num_problems,
            avg_plan_length=avg_plan_length,
            avg_time_taken=avg_time,
            repeated_states=total_repeated_states,
            impossible_actions=sum(1 for r in planner_results if not r.success),
            failures=compute_failures(problems_per_division,planner_results, failure_dict),
            max_plan_length=max_plan_length,
            max_time_taken=max_time_taken,
            median_plan_length=median_plan_length
        )
    
    return metrics

def compute_combined_metrics(results, curr_learned_model):

    non_optimal_results = results[PlannerType.NON_OPTIMAL]
    learned_model_results = results[curr_learned_model]

    num_problems = len(non_optimal_results)

    successful_results_with_monitor = [(non_opt,leared) for non_opt, leared in zip(non_optimal_results, learned_model_results) 
                                       if non_opt.success and leared.success]

    if successful_results_with_monitor:
        total_plan_len_learned = np.sum([r[1].plan_length for r in successful_results_with_monitor])
        total_plan_len_non_opt = np.sum([r[0].plan_length for r in successful_results_with_monitor])
        avg_plan_length = np.mean([r[1].plan_length for r in successful_results_with_monitor])
        avg_time = np.mean([r[1].time_taken for r in successful_results_with_monitor])
        max_plan_length = max(r[1].plan_length for r in successful_results_with_monitor)
        max_time_taken = max(r[1].time_taken for r in successful_results_with_monitor)
        #plan_quality = total_plan_len_learned / total_plan_len_non_opt
        plan_quality = total_plan_len_non_opt / total_plan_len_learned

    else:
        avg_plan_length = 0
        avg_time = 0
        max_plan_length = 0
        max_time_taken = 0
        plan_quality = 0

    total_repeated_states = sum(r[1].repeated_states for r in successful_results_with_monitor)
    
    combined_metrics = PlannerMetrics(
        success_rate_with_monitor=len(successful_results_with_monitor) / num_problems,
        avg_plan_length=avg_plan_length,
        avg_time_taken=avg_time,
        repeated_states=total_repeated_states,
        failures={},
        max_plan_length=max_plan_length,
        max_time_taken=max_time_taken,
        plan_quality=plan_quality
    )
    return combined_metrics 

def compute_failures(problems_per_division, 
                     planner_results: List[PlanningResult],
                     failure_dict) -> Dict[int, List[int]]:
    """
    Compute failures by division.
    
    Args:
        planner_results: List of PlanningResult objects
        
    Returns:
        Dictionary mapping division index to list of problem indices that failed in that division
    """
    for i, result in enumerate(planner_results):
        if not result.success:
            #div_idx = i // problems_per_division
            #if div_idx not in failures:
            #    failures[div_idx] = []
            #failures[div_idx].append(i)
            problem_idx = result.problem_idx
            failure_dict[int(problem_idx / problems_per_division)].append(problem_idx) 
            
    return failure_dict

def format_metrics(metrics, epoch=None):
    """
    Formats and displays model metrics with clean formatting.
    
    Args:
        result (dict): Dictionary containing model results and metrics
        
    Returns:
        dict: Formatted metrics for potential further use
    """
    # Extract metrics for cleaner access
    #metrics = result['test_results'][PlannerType.LEARNED_MODEL]
    
    '''
    # Format numeric values
    formatted_metrics = {
        'Success': f"{metrics.success_rate_with_monitor :.2%}",
        'Success No/M': f"{metrics.success_rate_without_monitor :.2%}",
        'Plan Length': f"{metrics.avg_plan_length:.1f}",
        'Time (sec)': f"{metrics.avg_time_taken:.2f}",
        'MAx PL': f"{metrics.max_plan_length}",
        'Max Time (sec)': f"{metrics.max_time_taken:.2f}",
        'Median PL': f"{metrics.median_plan_length:.1f}",
        #'Impossible Actions': f"{metrics.impossible_actions:,d}"
    }
    
    # Print header
    if epoch is not None:
        print(f"\nModel Metrics - Epoch {epoch}")
    '''

    formatted_metrics = {
        'Succ': f"{metrics.success_rate_with_monitor:.2%}/{metrics.success_rate_without_monitor:.2%}",
        'Plan': f"{metrics.avg_plan_length:.1f}/{metrics.median_plan_length:.1f}/{metrics.max_plan_length}",
        'Time': f"{metrics.avg_time_taken:.2f}/{metrics.max_time_taken:.2f}"
    }

    # Print header and metrics in one line
    if epoch is not None:
        print(f"E{epoch} Metrics: " + " | ".join([f"{k}: {v}" for k, v in formatted_metrics.items()]))
    print("-" * 30)
    
    # Print metrics in aligned format
    #for key, value in formatted_metrics.items():
    #    print(f"{key:<25} {value:>8}")
    
    return formatted_metrics

def format_metrics_non_opt(metrics, epoch=None):
    """
    Formats and displays model metrics with clean formatting.
    
    Args:
        result (dict): Dictionary containing model results and metrics
        
    Returns:
        dict: Formatted metrics for potential further use
    """
    # Extract metrics for cleaner access
    #metrics = result['test_results'][PlannerType.LEARNED_MODEL]
    
    # Format numeric values
    formatted_metrics = {
        'Success ': f"{metrics.success_rate_with_monitor :.2%}",
        #'Success without monitor': f"{metrics.success_rate_without_monitor :.2%}",
        'Plan Length': f"{metrics.avg_plan_length:.1f}",
        'Time (sec)': f"{metrics.avg_time_taken:.2f}",
        'MAx Plan Length': f"{metrics.max_plan_length}",
        'Max Time (sec)': f"{metrics.max_time_taken:.2f}",
        'Median Plan Length': f"{metrics.median_plan_length:.1f}",
        #'Impossible Actions': f"{metrics.impossible_actions:,d}"
    }
    

'''
def log_model_metrics(all_results_dict, args):
    """
    Logs metrics to wandb and returns best model info.
    
    Args:
        all_results_dict (dict): Dictionary with model types as keys and their results as values
        args: Arguments containing wandb configuration
    
    Returns:
        tuple: (best_model_type, best_epoch, best_success_rate)
    """
    #if not args.wandb:
    #    return None, None, None

    best_success_rate = 0
    best_model_type = None
    best_epoch = None

    # Log metrics for each model and epoch
    for model_type, results in all_results_dict.items():
        for result in results:
            metrics = result['test_results'][PlannerType.LEARNED_MODEL]
            epoch = result['epoch']
            
            # Log to wandb
            if args.wandb:
                wandb.log({
                    f"{model_type}/success_rate_monitor": metrics.success_rate_with_monitor,
                    f"{model_type}/success_rate_no_monitor": metrics.success_rate_without_monitor,
                    f"{model_type}/plan_length": metrics.avg_plan_length,
                },step=epoch)
            
            # Track best model
            if metrics.success_rate_with_monitor > best_success_rate:
                best_success_rate = metrics.success_rate_with_monitor
                best_model_type = model_type
                best_epoch = epoch

    # Log best model info
    if args.wandb:
        wandb.log({
            "best_model/type": best_model_type,
            "best_model/epoch": best_epoch,
            "best_model/success_rate": best_success_rate
        })

    return best_model_type, best_epoch, best_success_rate
'''

def log_model_metrics(all_results_dict, args):
    """
    Logs metrics to wandb and returns best model info for each planner type.
    
    Args:
        all_results_dict (dict): Dictionary with model types as keys and their results as values
        args: Arguments containing wandb configuration
        learned_planner_types (list): List of planner types to track
        
    Returns:
        dict: Dictionary with planner types as keys and tuples of (best_model_type, best_epoch, best_success_rate) as values
    """
        
    # Dictionary to track best results for each planner type
    best_results = {planner_type: {"success_rate": -1, "model_type": None, "epoch": None} 
                    for planner_type in learned_planner_types}
    
    # Log metrics for each model and epoch
    for model_type, results in all_results_dict.items():
        for result in results:
            epoch = result['epoch']
            
            # Log metrics for each planner type
            for planner_type in learned_planner_types:
                if planner_type in result['test_results']:
                    metrics = result['test_results'][planner_type]
                    
                    # Log to wandb
                    if args.wandb:
                        wandb.log({
                            f"{model_type}/{planner_type}/success_rate_monitor": metrics.success_rate_with_monitor,
                            f"{model_type}/{planner_type}/success_rate_no_monitor": metrics.success_rate_without_monitor,
                            f"{model_type}/{planner_type}/plan_length": metrics.avg_plan_length,
                        }, step=epoch)
                    
                    # Track best model for this planner type
                    if metrics.success_rate_with_monitor > best_results[planner_type]["success_rate"]:
                        best_results[planner_type]["success_rate"] = metrics.success_rate_with_monitor
                        best_results[planner_type]["model_type"] = model_type
                        best_results[planner_type]["epoch"] = epoch
    
    # Log best model info for each planner type
    if args.wandb:
        for planner_type in learned_planner_types:
            if best_results[planner_type]["model_type"]:
                wandb.log({
                    f"best_model/{planner_type}/type": best_results[planner_type]["model_type"],
                    f"best_model/{planner_type}/epoch": best_results[planner_type]["epoch"],
                    f"best_model/{planner_type}/success_rate": best_results[planner_type]["success_rate"]
                })
    
    # Print best model info for each planner type
    for planner_type in learned_planner_types:
        if best_results[planner_type]["model_type"] is not None:
            print(f"\nBest Model for {planner_type} (With Monitor):")
            print(f"Type: {best_results[planner_type]['model_type']}")
            print(f"Epoch: {best_results[planner_type]['epoch']}")
            print(f"Success Rate: {best_results[planner_type]['success_rate']:.2%}")
    
    # Return results in the format (best_model_type, best_epoch, best_success_rate) for each planner type
    return {planner_type: (best_results[planner_type]["model_type"], 
                           best_results[planner_type]["epoch"], 
                           best_results[planner_type]["success_rate"])
            for planner_type in learned_planner_types}


"""Validate plans.
"""


def validate_strips_plan(domain_file, problem_file, plan):
    """Return True for a successfully validated plan, False otherwise.
    """
    plan_str = ""
    #ic (domain_file)
    #ic (problem_file)
    for t, action in enumerate(plan):
        plan_str += "{}: {}\n".format(t, action.pddl_str())
    planfile = tempfile.NamedTemporaryFile(delete=False).name
    with open(planfile, "w") as f:
        f.write(plan_str)
    cmd_str = "validate -v {} {} {}".format(domain_file, problem_file, planfile)
    output = subprocess.getoutput(cmd_str)
    #ic (output)
    os.remove(planfile)
    if "Plan valid" in output:
        return True
    return False

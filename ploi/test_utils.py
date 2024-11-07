#from ploi.run_planner_with_ltp_2 import PlannerTester, PlannerConfig, PlannerType
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import wandb
import numpy as np

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
    eval_planner_name :str = ""
    train_planner_name :str = "" 

@dataclass
class PlannerMetrics:
    success_rate_with_monitor : float
    success_rate_without_monitor : float
    avg_plan_length: float
    avg_time_taken: float
    impossible_actions: int
    failures: Dict[int, List[int]]
    repeated_states: int = 0
    max_plan_length: int = 0
    max_time_taken: float = 0

class PlanningResult:
    def __init__(self):
        self.plan = []
        self.time_taken = 0
        self.success = False
        self.plan_length = 0
        self.repeated_states = 0 
        self.problem_idx = -1

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
        else:
            avg_plan_length = 0
            avg_time = 0
            max_plan_length = 0
            max_time_taken = 0
            
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
        )
    
    return metrics

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

def format_metrics(result):
    """
    Formats and displays model metrics with clean formatting.
    
    Args:
        result (dict): Dictionary containing model results and metrics
        
    Returns:
        dict: Formatted metrics for potential further use
    """
    # Extract metrics for cleaner access
    metrics = result['test_results'][PlannerType.LEARNED_MODEL]
    epoch = result['epoch']
    
    # Format numeric values
    formatted_metrics = {
        'Success with monitor': f"{metrics.success_rate_with_monitor :.2%}",
        'Success without monitor': f"{metrics.success_rate_without_monitor :.2%}",
        'Plan Length': f"{metrics.avg_plan_length:.1f}",
        'Time (sec)': f"{metrics.avg_time_taken:.2f}",
        'MAx Plan Length': f"{metrics.max_plan_length}",
        'Max Time (sec)': f"{metrics.max_time_taken:.2f}",
        #'Impossible Actions': f"{metrics.impossible_actions:,d}"
    }
    
    # Print header
    print(f"\nModel Metrics - Epoch {epoch}")
    print("-" * 30)
    
    # Print metrics in aligned format
    for key, value in formatted_metrics.items():
        print(f"{key:<25} {value:>8}")
    
    return formatted_metrics

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
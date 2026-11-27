#!/usr/bin/env python3
"""Run LAMA (fd-lama-first) on every candidate of the new splits (CPU only).

    python tools/solve_split_candidates.py --build /nfs/hpc/share/mangannr/di_ltp/splits_build \\
        --domain spanner [--timeout 600] [--workers 15]

For each <build>/<domain>/<split>/ written by tools/build_splits.py, plans
every candidate with the same planner and alias the training data uses, and
writes <split>/solve_results.json:
    {problem file: {"solved": bool, "plan_length": int|None, "seconds": float,
                    "plan": [action strings] | None, "error": str|None}}
A candidate counts as a usable problem only if solved. Already-recorded
results are kept, so a rerun after an interrupted job continues.
Runs inside a SLURM CPU allocation (never the login node).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PDDL = "/nfs/hpc/share/mangannr/PLOI_full_repos/pddlgym/pddlgym/pddl"
DOMAIN_FILES = {"manyblocks": "manyblocks_ipcc_big", "gripper": "gripper_ipcc", "miconic": "miconic_ipcc",
                "visitall": "visitall_ipcc", "grid": "grid_ipcc", "logistics": "logistics_ipcc",
                "spanner": "spanner_ipcc", "rovers": "rovers_ipcc"}

_ENV = None
_PLANNER = None


def _init(domain_file, split_dir, timeout):
    global _ENV, _PLANNER, _TIMEOUT
    os.chdir(tempfile.mkdtemp(prefix="fdwork_"))      # FD drops sas_plan in the cwd
    from pddlgym.core import PDDLEnv
    from ploi.run_planner_with_ltp_v1 import _create_planner
    _ENV = PDDLEnv(domain_file, split_dir, operators_as_actions=True, dynamic_action_space=True)
    _PLANNER = _create_planner("fd-lama-first")
    _TIMEOUT = timeout


def _solve(index):
    from ploi.planning import PlanningFailure, PlanningTimeout
    name = os.path.basename(_ENV.problems[index].problem_fname)
    t0 = time.time()
    try:
        _ENV.fix_problem_index(index)
        state, _ = _ENV.reset()
        plan = _PLANNER(_ENV.domain, state, timeout=_TIMEOUT)
        ok = bool(plan)
        return name, dict(solved=ok, plan_length=len(plan) if ok else None,
                          seconds=round(time.time() - t0, 1),
                          plan=[str(a) for a in plan] if ok else None, error=None)
    except (PlanningTimeout, PlanningFailure) as e:
        return name, dict(solved=False, plan_length=None, seconds=round(time.time() - t0, 1),
                          plan=None, error=type(e).__name__)
    except Exception as e:  # noqa: BLE001 - a broken candidate must not stop the job
        return name, dict(solved=False, plan_length=None, seconds=round(time.time() - t0, 1),
                          plan=None, error=f"{type(e).__name__}: {str(e)[:200]}")


def solve_split(domain, split_dir, timeout, workers):
    out = os.path.join(split_dir, "solve_results.json")
    results = json.load(open(out)) if os.path.exists(out) else {}
    from pddlgym.core import PDDLEnv
    domain_file = os.path.join(PDDL, DOMAIN_FILES[domain] + ".pddl")
    env = PDDLEnv(domain_file, split_dir, operators_as_actions=True, dynamic_action_space=True)
    todo = [i for i, p in enumerate(env.problems)
            if os.path.basename(p.problem_fname) not in results]
    print(f"{domain}/{os.path.basename(split_dir)}: {len(env.problems)} candidates, {len(todo)} to solve",
          flush=True)
    if todo:
        with mp.Pool(workers, initializer=_init, initargs=(domain_file, split_dir, timeout)) as pool:
            for k, (name, res) in enumerate(pool.imap_unordered(_solve, todo), 1):
                results[name] = res
                if k % 10 == 0 or k == len(todo):
                    with open(out + ".tmp", "w") as f:
                        json.dump(results, f)
                    os.replace(out + ".tmp", out)
    solved = sum(r["solved"] for r in results.values())
    print(f"  solved {solved}/{len(results)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--splits", default="", help="comma-separated; default all of the domain's")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per problem")
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("refusing to plan outside a SLURM allocation (login node)")
    base = os.path.join(args.build, args.domain)
    splits = [s for s in args.splits.split(",") if s] or sorted(os.listdir(base))
    for s in splits:
        if os.path.isdir(os.path.join(base, s)):
            solve_split(args.domain, os.path.join(base, s), args.timeout, args.workers)


if __name__ == "__main__":
    main()

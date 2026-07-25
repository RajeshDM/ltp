"""Random-applicable-action baseline (the zero-shot floor).

At each step, sample uniformly among the applicable ground actions and
execute, under the same execution bound as the learned policy. No training,
no model: any coverage this achieves is attributable to blind applicability
alone, so it is the floor every ladder rung is read against (the per-domain
GABAR reference is the ceiling).

Usage (same domain/split syntax as --test-domains in configs):
  python tools/random_policy_baseline.py \
      --domains manyblocks_ipcc_big:200,gripper_ipcc:173,rovers_ipcc@train:312 \
      --max-plan-length 500 --rollouts 3

Sequential over problems; run several domains in parallel shells if needed.
Results land in cache/results/results_random_policy.json (merged across
invocations, keyed by domain@split).
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pddlgym


def parse_domains(spec):
    """'name[:count]' or 'name@train[:count]' -> (Name, split, count)."""
    entries = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        count = 0
        if ":" in raw:
            raw, count_str = raw.rsplit(":", 1)
            count = int(count_str)
        split = "test"
        if "@" in raw:
            raw, split = raw.split("@", 1)
            if split != "train":
                raise ValueError(f"Unknown split '{split}' (only @train)")
        entries.append((raw.capitalize(), split, count))
    return entries


def run_problem(env, idx, max_len, rollouts, seed):
    """Random rollouts on problem idx. Returns (success_rate, lengths, secs)."""
    successes, lengths = 0, []
    start = time.time()
    for r in range(rollouts):
        rng = random.Random(seed + 1000 * idx + r)
        env.fix_problem_index(idx)
        state, _ = env.reset()
        # Reground ONCE per problem (builds pddlgym's ground-action cache for
        # the new instance), then use the cached path per step - same as the
        # learned tester's greedy loop. Regrounding every step is O(|O|^arity)
        # work per step and dominated everything (~0.4s/step on hard Visitall).
        env.action_space.all_ground_literals(state, reground=True)
        for step in range(max_len):
            groundings = list(env.action_space.all_ground_literals(state))
            if not groundings:
                break  # dead end
            action = rng.choice(sorted(groundings))
            state, _, done, _ = env.step(action)
            if done:
                successes += 1
                lengths.append(step + 1)
                break
    return successes / rollouts, lengths, time.time() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", required=True,
                        help="Comma-separated 'name[:count]' or "
                             "'name@train[:count]' (config test-domains syntax)")
    parser.add_argument("--max-plan-length", type=int, default=500)
    parser.add_argument("--rollouts", type=int, default=3,
                        help="Rollouts per problem; coverage averages over them")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str,
                        default="cache/results/results_random_policy.json")
    args = parser.parse_args()

    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)

    for name, split, count in parse_domains(args.domains):
        suffix = "Test" if split == "test" else ""
        env = pddlgym.make(f"PDDLEnv{name}{suffix}-v0")
        n = len(env.problems) if count <= 0 else min(count, len(env.problems))
        key = f"{name}@{split}"
        print(f"=== {key}: {n} problems, {args.rollouts} rollout(s), "
              f"cap {args.max_plan_length} ===")

        per_problem = []
        for idx in range(n):
            rate, lengths, secs = run_problem(
                env, idx, args.max_plan_length, args.rollouts, args.seed)
            per_problem.append({"idx": idx, "success_rate": rate,
                                "plan_lengths": lengths,
                                "time": round(secs, 2)})
            cov = 100 * sum(p["success_rate"] for p in per_problem) / len(per_problem)
            print(f"  [{idx + 1}/{n}] {secs:6.1f}s  running coverage {cov:.1f}%",
                  flush=True)

        coverage = 100 * sum(p["success_rate"] for p in per_problem) / n
        all_lengths = [l for p in per_problem for l in p["plan_lengths"]]
        avg_len = sum(all_lengths) / len(all_lengths) if all_lengths else None
        results[key] = {
            "policy": "random_applicable",
            "split": split, "problems": n, "rollouts": args.rollouts,
            "max_plan_length": args.max_plan_length, "seed": args.seed,
            "coverage": round(coverage, 2),
            "avg_plan_length": round(avg_len, 1) if avg_len else None,
            "per_problem": per_problem,
        }
        print(f"  {key}: coverage {coverage:.2f}%"
              + (f", avg length {avg_len:.1f}" if avg_len else ""))

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, args.output)

    print(f"\nSaved to {args.output}")
    for key in sorted(results):
        print(f"  {key:45s} {results[key]['coverage']:6.2f}%")


if __name__ == "__main__":
    main()

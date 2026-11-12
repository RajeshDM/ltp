"""Non-learned baselines under the learned policy's executor.

--policy random (default, the zero-shot floor): at each step, sample
uniformly among the applicable ground actions and execute, under the same
execution bound as the learned policy. No training, no model: any coverage
this achieves is attributable to blind applicability alone, so it is the
floor every ladder rung is read against (the per-domain GABAR reference is
the ceiling).

--policy achieve (the achievement oracle as a policy): prefer an applicable
action that adds an unsatisfied goal atom; otherwise random. What a
hand-coded terminal trigger gets with nothing behind it.

--policy features (C2 control): rank applicable actions by the hand-coded
goal information GADAR's chain layer puts in the graph, and nothing learned.
Lexicographic, best first:
  1. adds more unsatisfied goal atoms            (achieves_goal edges)
  2. deletes fewer currently-satisfied goal atoms (deletes_true, on goals)
  3. schema closer to the goal                   (gdist bucket, BFS over enables)
  4. more goal-relevant objects among its arguments (goal_relevant_obj)
ties broken at random. If this matches GADAR's zero-shot coverage, the
network adds nothing to the features; the gap between them is what was
learned from the training domains.

Usage (same domain/split syntax as --test-domains in configs):
  python tools/random_policy_baseline.py \
      --domains manyblocks_ipcc_big:200,gripper_ipcc:173,rovers_ipcc@train:312 \
      --max-plan-length 500 --rollouts 3

Sequential over problems; run several domains in parallel shells if needed.
Results land in cache/results/results_<policy>_policy.json (merged across
invocations, keyed by domain@split).
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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


def _atom(literal, binding=None):
    """(positive predicate name, argument tuple) of a ground or lifted literal."""
    from ploi.lifted_layer import _positive_predicate
    args = tuple((binding or {}).get(v, v) for v in literal.variables)
    return (_positive_predicate(literal).name, args)


def make_scorer(policy, env, state):
    """-> score(grounding, state) for one problem, higher = better; None for
    the random policy. Everything goal-dependent but state-independent (the
    goal atoms, the per-schema goal distance) is computed once here."""
    if policy == "random":
        return None
    from ploi.lifted_layer import (build_lifted_spec, build_chain_spec,
                                   schema_goal_distances)
    ops = env.action_space._action_predicate_to_operators
    goal = {_atom(g) for g in state.goal.literals}
    gdist = {}
    if policy == "features":
        spec = build_lifted_spec(ops)
        gdist = schema_goal_distances(spec, build_chain_spec(spec),
                                      {name for name, _ in goal})

    def score(grounding, state):
        op = ops[grounding.predicate]
        binding = dict(zip(op.params, grounding.variables))
        true_now = {_atom(l) for l in state.literals}
        open_goal = goal - true_now
        achieves = deletes = 0
        for lit in op.effects.literals:
            atom = _atom(lit, binding)
            if getattr(lit, "is_anti", False):
                deletes += atom in goal and atom in true_now
            else:
                achieves += atom in open_goal
        if policy == "achieve":
            return (achieves,)
        relevant = {o for _, args in open_goal for o in args}
        return (achieves, -deletes,
                -gdist.get(grounding.predicate.name, 99),
                sum(o in relevant for o in grounding.variables))
    return score


def run_problem(env, idx, max_len, rollouts, seed, proposals=8, monitor=True,
                policy="random"):
    """Rollouts on problem idx. Returns (success_rate, lengths, secs).

    Protocol parity with the learned tester (_run_greedy_search): the only
    difference is WHERE the ranked proposal list comes from - a random sample
    of applicable actions instead of the model's beam. Same list size, same
    revisit monitor, same 'all candidates revisit -> take a random one'
    fallback, same step bound. proposals=0 means rank all applicable actions.
    """
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
        score = make_scorer(policy, env, state)
        visited = {frozenset(state.literals)} if monitor else None
        for step in range(max_len):
            groundings = sorted(env.action_space.all_ground_literals(state))
            if not groundings:
                break  # dead end
            ranked = groundings[:]
            rng.shuffle(ranked)
            if score is not None:
                # Stable sort over the shuffle: ties stay in random order.
                ranked.sort(key=lambda g: score(g, state), reverse=True)
            if proposals > 0:
                ranked = ranked[:proposals]

            chosen, next_state, done = None, None, False
            for cand in ranked:
                ns, _, d, _ = env.step(cand)
                if monitor and not d and frozenset(ns.literals) in visited:
                    env.set_state(state)   # revert, try next candidate
                    continue
                chosen, next_state, done = cand, ns, d
                break
            if chosen is None:      # every candidate revisits: take one anyway
                chosen = ranked[0]
                next_state, _, done, _ = env.step(chosen)

            state = next_state
            if monitor:
                visited.add(frozenset(state.literals))
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
    parser.add_argument("--proposals", type=int, default=8,
                        help="Size of the random 'ranked proposal' list per "
                             "step, matching the learned beam's candidate "
                             "count (action_options x object_options). "
                             "0 = rank every applicable action.")
    parser.add_argument("--no-monitor", dest="monitor", action="store_false",
                        help="Disable the revisited-state monitor. Default ON "
                             "for parity with the learned tester.")
    parser.add_argument("--policy", default="random",
                        choices=["random", "achieve", "features"],
                        help="random = uniform over applicable actions (the "
                             "floor); achieve = prefer actions adding an open "
                             "goal atom; features = rank by GADAR's hand-coded "
                             "goal features alone (see module docstring).")
    parser.add_argument("--output", type=str, default=None,
                        help="default cache/results/results_<policy>_policy.json")
    parser.add_argument("--summary", type=str, default=None,
                        help="Fixed-width table of all results so far; "
                             "rewritten after each domain. Default "
                             "cache/results/<policy>_floor_summary.txt")
    args = parser.parse_args()
    # Imported here so the scorers above stay importable (and unit-testable)
    # without pddlgym.
    import pddlgym
    from tqdm import tqdm
    if args.output is None:
        args.output = f"cache/results/results_{args.policy}_policy.json"
    if args.summary is None:
        args.summary = f"cache/results/{args.policy}_floor_summary.txt"

    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)

    for name, split, count in parse_domains(args.domains):
        suffix = "Test" if split == "test" else ""
        env = pddlgym.make(f"PDDLEnv{name}{suffix}-v0")
        n = len(env.problems) if count <= 0 else min(count, len(env.problems))
        key = f"{name}@{split}"
        print(f"=== {key} [{args.policy}]: {n} problems, {args.rollouts} rollout(s), "
              f"cap {args.max_plan_length} steps, proposals="
              f"{args.proposals or 'all'}, monitor={args.monitor} ===")

        per_problem = []
        bar = tqdm(range(n), desc=f"{key:32s}", unit="prob")
        for idx in bar:
            rate, lengths, secs = run_problem(
                env, idx, args.max_plan_length, args.rollouts, args.seed,
                proposals=args.proposals, monitor=args.monitor,
                policy=args.policy)
            per_problem.append({"idx": idx, "success_rate": rate,
                                "plan_lengths": lengths,
                                "time": round(secs, 2)})
            cov = 100 * sum(p["success_rate"] for p in per_problem) / len(per_problem)
            bar.set_postfix(cov=f"{cov:.1f}%")
        bar.close()

        coverage = 100 * sum(p["success_rate"] for p in per_problem) / n
        all_lengths = [l for p in per_problem for l in p["plan_lengths"]]
        avg_len = sum(all_lengths) / len(all_lengths) if all_lengths else None
        results[key] = {
            "policy": ("random_applicable" if args.policy == "random"
                       else args.policy),
            "proposals": args.proposals, "monitor": args.monitor,
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
        write_summary(results, args.summary)   # after every domain: partials readable

    print(f"\nSaved to {args.output}\nSummary: {args.summary}")
    print(open(args.summary).read())


def write_summary(results, path):
    """Paper-ready fixed-width table (cat this straight into the draft)."""
    policies = sorted({r.get("policy", "?") for r in results.values()})
    lines = [f"# Non-learned baseline: {', '.join(policies)} (same executor,",
             "# monitor and step bound as the learned policy)",
             "#",
             f"# {'domain':28s} {'split':6s} {'probs':>5s} {'roll':>4s} "
             f"{'cov%':>7s} {'avg_len':>8s}"]
    for key in sorted(results):
        r = results[key]
        dom, split = key.rsplit("@", 1)
        avg = f"{r['avg_plan_length']:.1f}" if r.get("avg_plan_length") else "-"
        lines.append(f"  {dom:28s} {split:6s} {r['problems']:5d} "
                     f"{r['rollouts']:4d} {r['coverage']:7.2f} {avg:>8s}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

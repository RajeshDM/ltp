#!/usr/bin/env python3
"""Verify the C2 component ablations on REAL states (needs pddlgym).

    python tools/check_ablation_modes.py --domain Manyblocks_ipcc_big
    python tools/check_ablation_modes.py --domain Logistics_ipcc --problems 3 --steps 25

The unit tests cover the mode registry and the lifted-layer edges; the
'bind', 'eff' and 'goal' switches live in _state_to_graph_ltp, which needs
real pddlgym states. For each ablation mode this featurizes the same seeded
rollout states as 'joint_chain' and checks, column group by column group:

  - a REMOVED component's feature columns are all zero;
  - every KEPT column group has exactly the counts 'joint_chain' has.

So each ablation changes its own component and nothing else. Also prints the
per-mode counts, which is the quickest way to see what each mode contains.
Run it once before launching ablation training; exit code 1 on any mismatch.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def column_groups(md):
    """{group: ('nodes'|'edges', [column indices])} for the four components
    plus 'rest' (everything the ablations must leave alone)."""
    N, E = md["node_feature_to_index"], md["edge_feature_to_index"]
    groups = {
        "bind": ("edges", [i for k, i in E.items() if str(k).startswith("bind_slot_")]),
        "chain": ("edges", [E["enables"], E["threatens"]]),
        "eff": ("edges", [E["achieves_goal"], E["deletes_true"]]),
        "goal": ("nodes", [N[k] for k in ("adds_goal_pred", "goal_relevant_obj",
                                          "gdist_0", "gdist_1", "gdist_2",
                                          "gdist_3plus")]),
    }
    used_e = {i for kind, cols in groups.values() if kind == "edges" for i in cols}
    used_n = {i for kind, cols in groups.values() if kind == "nodes" for i in cols}
    groups["rest_edges"] = ("edges", [i for i in range(md["num_edge_features"])
                                      if i not in used_e])
    groups["rest_nodes"] = ("nodes", [i for i in range(md["num_node_features"])
                                      if i not in used_n])
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, help="pddlgym name, e.g. Visitall_ipcc")
    ap.add_argument("--problems", type=int, default=2)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kp", type=int, default=3)
    ap.add_argument("--ka", type=int, default=6)
    args = ap.parse_args()

    import pddlgym
    from ploi.datautils_ltp import state_to_graph_wrapper
    from ploi.lifted_layer import (ABLATION_MODES, build_lifted_metadata,
                                   mode_components)

    env = pddlgym.make(f"PDDLEnv{args.domain}-v0")
    action_space = env.action_space._action_predicate_to_operators
    modes = ("joint_chain",) + ABLATION_MODES + ("joint",)
    mds = {m: build_lifted_metadata(action_space, args.kp, args.ka, m) for m in modes}
    groups = column_groups(mds["joint_chain"])
    totals = {m: {g: 0 for g in groups} for m in modes if m != "joint"}

    rng = random.Random(args.seed)
    n_states = 0
    for problem_idx in range(min(args.problems, len(env.problems))):
        env.fix_problem_index(problem_idx)
        state, _ = env.reset()
        for _ in range(args.steps):
            # Sorted: all_ground_literals returns a set, so the seeded choice
            # below would otherwise follow PYTHONHASHSEED.
            groundings = sorted(env.action_space.all_ground_literals(state), key=repr)
            if not groundings:
                break
            n_states += 1
            for m in totals:
                graph, _, _ = state_to_graph_wrapper(
                    state, action_space, groundings, prev_actions=None,
                    prev_state=None, graph_metadata=mds[m], curr_action=None,
                    objects=None, goal_state=state.goal)
                for g, (kind, cols) in groups.items():
                    totals[m][g] += int(graph[kind][:, cols].sum())
            state, _, done, _ = env.step(rng.choice(groundings))
            if done:
                break

    names = list(groups)
    print(f"{args.domain}: {n_states} states, kp={args.kp} ka={args.ka}\n")
    print(f"{'mode':<15}" + "".join(f"{g:>12}" for g in names))
    full = totals["joint_chain"]
    bad = []
    for m in totals:
        print(f"{m:<15}" + "".join(f"{totals[m][g]:>12}" for g in names))
        comps = mode_components(m)
        for g in names:
            want = full[g] if (g.startswith("rest") or g in comps) else 0
            if totals[m][g] != want:
                bad.append(f"{m}: {g} = {totals[m][g]}, expected {want}")
    for g in ("bind", "chain", "eff", "goal"):
        if full[g] == 0:
            bad.append(f"joint_chain has no '{g}' features on these states - "
                       f"the check is vacuous for it; use more --steps/--problems")
    print()
    if bad:
        print("MISMATCH:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("ok: every ablation zeroes exactly its own component and matches "
          "joint_chain everywhere else")


if __name__ == "__main__":
    main()

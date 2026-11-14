"""Time state featurization on real states, with no model and no training.

Featurization (`state_to_graph_wrapper` -> `_state_to_graph_ltp`) runs once per
state of every rollout at test time, so it is the part of inference that scales
with instance size rather than with the network. This script isolates it: walk a
rollout of applicable actions and time only the graph build, plus the PyG
conversion it feeds.

Usage
-----
    python tools/bench_featurize.py \\
        --domain Visitall_ipcc \\
        --graphs-pickle cache/results/Visitall_ipcc_graphs_0_125_joint_chain_kp3_ka6_ty2.pkl \\
        --problems 3 --steps 40

`--graphs-pickle` is a per-(domain, tag) sidecar; it is read only for the
`graph_metadata` it carries, so the numbers reflect the featurization that
config actually uses. Use `--split test` for the large instances, which is
where the cost matters.

To compare before and after a change, run it on both revisions with the same
arguments and the same pickle:

    git stash && python tools/bench_featurize.py ... | tee /tmp/before.txt
    git stash pop && python tools/bench_featurize.py ... | tee /tmp/after.txt

The rollout is seeded, so both runs visit the same states.
"""

import argparse
import gc
import os
import pickle
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_metadata(path):
    with open(path, 'rb') as fh:
        payload = pickle.load(fh)
    if isinstance(payload, tuple) and len(payload) == 2:
        _graphs, metadata = payload
        return metadata
    raise ValueError(
        f"{path} is not a (graphs, metadata) sidecar; pass a "
        f"cache/results/<Domain>_graphs_*.pkl file")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domain', required=True,
                        help="pddlgym domain name, e.g. Visitall_ipcc")
    parser.add_argument('--graphs-pickle', required=True,
                        help="sidecar to read graph_metadata from")
    parser.add_argument('--split', default='test', choices=['test', 'train'],
                        help="test uses the large instances (default)")
    parser.add_argument('--problems', type=int, default=3)
    parser.add_argument('--steps', type=int, default=40,
                        help="states to featurize per problem")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--convert', action='store_true',
                        help="also time graph_to_pyg_data on each graph")
    args = parser.parse_args()

    import pddlgym  # noqa: F401  (registers the environments)
    from ploi.datautils_ltp import state_to_graph_wrapper

    metadata = _load_metadata(args.graphs_pickle)
    print(f"metadata: featurization={metadata.get('featurization', 'per_domain')} "
          f"node_features={metadata['num_node_features']} "
          f"edge_features={metadata['num_edge_features']}")

    suffix = 'Test' if args.split == 'test' else ''
    env = pddlgym.make(f"PDDLEnv{args.domain}{suffix}-v0")
    action_space = None  # predicate-keyed map; set after first grounding
    rng = random.Random(args.seed)

    build_times, convert_times, sizes, edge_counts = [], [], [], []

    for problem_idx in range(min(args.problems, len(env.problems))):
        env.fix_problem_index(problem_idx)
        state, _ = env.reset()

        for _ in range(args.steps):
            groundings = list(env.action_space.all_ground_literals(state))
            if not groundings:
                break
            if action_space is None:
                action_space = env.action_space._action_predicate_to_operators

            gc.collect()
            t0 = time.perf_counter()
            graph, _, _ = state_to_graph_wrapper(
                state, action_space, groundings,
                prev_actions=None, prev_state=None,
                graph_metadata=metadata, curr_action=None, objects=None,
                goal_state=state.goal)
            build_times.append(time.perf_counter() - t0)

            sizes.append(graph['nodes'].shape[0])
            edge_counts.append(graph['edges'].shape[0])

            if args.convert:
                from ploi.datautils_ltp import graph_to_pyg_data
                t0 = time.perf_counter()
                graph_to_pyg_data(graph)
                convert_times.append(time.perf_counter() - t0)

            state, _, done, _ = env.step(rng.choice(groundings))
            if done:
                break

    if not build_times:
        print("no states featurized; check --domain and --split")
        return

    def report(label, values):
        print(f"{label:<16} n={len(values):<5} "
              f"total={sum(values):8.3f}s  "
              f"mean={statistics.mean(values)*1000:8.2f}ms  "
              f"median={statistics.median(values)*1000:8.2f}ms  "
              f"max={max(values)*1000:8.2f}ms")

    print()
    report("graph build", build_times)
    if convert_times:
        report("pyg convert", convert_times)
    print(f"{'graph size':<16} nodes: mean={statistics.mean(sizes):.0f} "
          f"max={max(sizes)}   edges: mean={statistics.mean(edge_counts):.0f} "
          f"max={max(edge_counts)}")
    # What the dense (arity, N, N, k) array used to cost at the largest state
    # seen, as float64. Kept as the reference point for the sparse rewrite.
    worst = max(sizes)
    dense_mb = (6 * worst * worst * metadata['num_edge_features'] * 8) / 1e6
    print(f"{'reference':<16} a dense arity-6 edge array at {worst} nodes "
          f"would be {dense_mb:.0f} MB per state")


if __name__ == '__main__':
    main()

"""Attribute featurization cost to exact call sites.

Runs state_to_graph_wrapper over real rollout states twice: once unprofiled
(the honest ms/state) and once under cProfile, then prints WHO CALLS the hot
primitives (Literal.__eq__/__lt__, repr, list.index, sorted, dict lookups do
not appear by name but their __eq__ traffic does). This answers "which of our
loops issues N tiny operations per state" - the thing a flat profile cannot,
because cProfile's per-call overhead (~0.3us) inflates functions that are
called millions of times with nanoseconds of real work each.

Usage (same arguments as bench_featurize):
    python tools/profile_featurize.py \\
        --domain Visitall_ipcc \\
        --graphs-pickle cache/results/Visitall_ipcc_graphs_0_125_joint_chain_kp2_ka2_ty2.pkl \\
        --split test --problems 2 --steps 25

Read the output as: honest ms/state from the first block; RATIOS and CALLER
NAMES from the profile blocks. Do not read profiled seconds as wall time.
"""

import argparse
import cProfile
import io
import os
import pickle
import pstats
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_metadata(path):
    with open(path, 'rb') as fh:
        payload = pickle.load(fh)
    if isinstance(payload, tuple) and len(payload) == 2:
        return payload[1]
    raise ValueError(f"{path} is not a (graphs, metadata) sidecar")


def _collect_states(domain, split, problems, steps, seed):
    """Walk seeded rollouts, returning (state, groundings) pairs to featurize."""
    import pddlgym
    suffix = 'Test' if split == 'test' else ''
    env = pddlgym.make(f"PDDLEnv{domain}{suffix}-v0")
    action_space = env.domain.operators
    rng = random.Random(seed)
    pairs = []
    for idx in range(min(problems, len(env.problems))):
        env.fix_problem_index(idx)
        state, _ = env.reset()
        for _ in range(steps):
            groundings = list(env.action_space.all_ground_literals(state))
            if not groundings:
                break
            pairs.append((state, groundings))
            state, _, done, _ = env.step(rng.choice(groundings))
            if done:
                break
    return pairs, action_space


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--graphs-pickle', required=True)
    parser.add_argument('--split', default='test', choices=['test', 'train'])
    parser.add_argument('--problems', type=int, default=2)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    from ploi.datautils_ltp import state_to_graph_wrapper
    metadata = _load_metadata(args.graphs_pickle)
    pairs, action_space = _collect_states(
        args.domain, args.split, args.problems, args.steps, args.seed)
    print(f"{len(pairs)} states collected; featurization="
          f"{metadata.get('featurization', 'per_domain')}")

    def featurize_all():
        for state, groundings in pairs:
            state_to_graph_wrapper(
                state, action_space, groundings,
                prev_actions=None, prev_state=None, graph_metadata=metadata,
                curr_action=None, objects=None, goal_state=state.goal)

    # Pass 1: honest wall time (no profiler).
    times = []
    for state, groundings in pairs:
        t0 = time.perf_counter()
        state_to_graph_wrapper(
            state, action_space, groundings,
            prev_actions=None, prev_state=None, graph_metadata=metadata,
            curr_action=None, objects=None, goal_state=state.goal)
        times.append(time.perf_counter() - t0)
    print(f"\nHONEST TIMING (no profiler): "
          f"mean={statistics.mean(times)*1000:.1f}ms  "
          f"median={statistics.median(times)*1000:.1f}ms  "
          f"max={max(times)*1000:.1f}ms per state")

    # Pass 2: profile for attribution only.
    prof = cProfile.Profile()
    prof.enable()
    featurize_all()
    prof.disable()

    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf)

    buf.write("\n===== top 25 by tottime (profiled - use for RANKING only) =====\n")
    stats.sort_stats('tottime').print_stats(25)

    for label, pattern in [
            ("who calls Literal.__eq__", r"structs.py.*__eq__"),
            ("who calls Literal.__lt__", r"structs.py.*__lt__"),
            ("who calls repr",           r"\{built-in method builtins.repr\}"),
            ("who calls sorted",         r"\{built-in method builtins.sorted\}"),
            ("who calls list.index",     r"method 'index' of 'list'"),
            ("who calls str",            r"\{built-in method builtins.str\}"),
    ]:
        buf.write(f"\n===== {label} =====\n")
        stats.print_callers(pattern)

    out = buf.getvalue()
    # print_callers lines are wide; keep them intact.
    print(out)


if __name__ == '__main__':
    main()

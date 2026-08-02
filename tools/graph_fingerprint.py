"""Fingerprint featurized graphs so a refactor can be proved byte-identical.

`_state_to_graph_ltp` is the core of the pipeline and every result depends on
it producing exactly the tensors it produced before. This walks seeded
rollouts, featurizes each state, and hashes every array in the resulting graph
dict (plus its shape and dtype). Two revisions that produce the same
fingerprint file are byte-identical on those states; a mismatch names the
state and the key.

Record a baseline, change the code, record again, compare:

    python tools/graph_fingerprint.py --domains visitall_ipcc,grid_ipcc \\
        --tag joint_chain_kp2_ka2_ty2 --out /tmp/before.json
    # ... edit _state_to_graph_ltp ...
    python tools/graph_fingerprint.py --domains visitall_ipcc,grid_ipcc \\
        --tag joint_chain_kp2_ka2_ty2 --out /tmp/after.json
    python tools/graph_fingerprint.py --compare /tmp/before.json /tmp/after.json

The rollout is seeded per problem, so both runs visit identical states. Use
several domains: arity, typing convention and goal size all change which
branches of the featurizer run.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_sidecar(domain, tag):
    pattern = os.path.join('cache', 'results', f'{domain}_graphs_0_*{tag}.pkl')
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"no sidecar matching {pattern}; pass the right --tag "
            f"(e.g. joint_chain_kp3_ka6_ty2)")
    return hits[-1]


def _load_metadata(path):
    import pickle
    with open(path, 'rb') as fh:
        payload = pickle.load(fh)
    if isinstance(payload, tuple) and len(payload) == 2:
        return payload[1]
    raise ValueError(f"{path} is not a (graphs, metadata) sidecar")


def _hash_array(value):
    """Stable digest of one graph entry: dtype, shape and raw bytes."""
    import numpy as np
    if hasattr(value, 'detach'):          # torch tensor
        value = value.detach().cpu().numpy()
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return f"{arr.dtype}:{arr.shape}:{h.hexdigest()[:16]}"


def fingerprint_domain(domain, tag, split, problems, steps, seed):
    import pddlgym
    from ploi.datautils_ltp import state_to_graph_wrapper

    metadata = _load_metadata(_find_sidecar(domain, tag))
    suffix = 'Test' if split == 'test' else ''
    env = pddlgym.make(f"PDDLEnv{domain}{suffix}-v0")
    action_space = None
    states = []

    for idx in range(min(problems, len(env.problems))):
        env.fix_problem_index(idx)
        state, _ = env.reset()
        # Seed per problem so interleaving cannot change the walk.
        rng = random.Random(seed + idx)
        for step in range(steps):
            groundings = list(env.action_space.all_ground_literals(state))
            if not groundings:
                break
            if action_space is None:
                action_space = env.action_space._action_predicate_to_operators
            graph, _, _ = state_to_graph_wrapper(
                state, action_space, groundings,
                prev_actions=None, prev_state=None, graph_metadata=metadata,
                curr_action=None, objects=None, goal_state=state.goal)
            states.append({
                'problem': idx,
                'step': step,
                'keys': {k: _hash_array(v) for k, v in sorted(graph.items())},
            })
            state, _, done, _ = env.step(rng.choice(groundings))
            if done:
                break
    return states


def compare(path_a, path_b):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    domains = sorted(set(a) | set(b))
    total, mismatches = 0, 0
    for dom in domains:
        sa, sb = a.get(dom), b.get(dom)
        if sa is None or sb is None:
            print(f"{dom}: MISSING from {'A' if sa is None else 'B'}")
            mismatches += 1
            continue
        if len(sa) != len(sb):
            print(f"{dom}: DIFFERENT STATE COUNT {len(sa)} vs {len(sb)} "
                  f"(the rollouts diverged, not just the tensors)")
            mismatches += 1
            continue
        bad = []
        for ra, rb in zip(sa, sb):
            total += 1
            keys = sorted(set(ra['keys']) | set(rb['keys']))
            for k in keys:
                va, vb = ra['keys'].get(k), rb['keys'].get(k)
                if va != vb:
                    bad.append((ra['problem'], ra['step'], k, va, vb))
        if bad:
            mismatches += len(bad)
            print(f"{dom}: {len(bad)} differing entries; first 5:")
            for problem, step, k, va, vb in bad[:5]:
                print(f"    p{problem} step{step} '{k}':")
                print(f"        A {va}")
                print(f"        B {vb}")
        else:
            print(f"{dom}: identical ({len(sa)} states)")

    print(f"\n{total} states compared, {mismatches} mismatches")
    return 1 if mismatches else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domains', help="comma-separated pddlgym domain names")
    parser.add_argument('--tag', default='joint_chain_kp2_ka2_ty2',
                        help="sidecar tag suffix to locate graph_metadata")
    parser.add_argument('--split', default='test', choices=['test', 'train'])
    parser.add_argument('--problems', type=int, default=2)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', help="write fingerprints here")
    parser.add_argument('--compare', nargs=2, metavar=('A', 'B'),
                        help="compare two fingerprint files and exit")
    args = parser.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))

    if not args.domains or not args.out:
        parser.error("--domains and --out are required unless --compare is used")

    result = {}
    for domain in [d.strip() for d in args.domains.split(',') if d.strip()]:
        states = fingerprint_domain(domain, args.tag, args.split,
                                    args.problems, args.steps, args.seed)
        result[domain] = states
        n_keys = len(states[0]['keys']) if states else 0
        print(f"{domain}: {len(states)} states, {n_keys} keys each")

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()

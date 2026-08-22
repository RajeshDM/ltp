"""Can the decoder's dot product even SELECT the expert's object?

The decoder scores a candidate object as `q . e_o`: a query vector against the
object's embedding, where `e_o` was fixed by the encoder before decoding began.
Geometrically that means "choose a direction, take the object furthest along
it". An object buried inside the cloud of candidate embeddings is furthest
along NO direction, so no query - however well trained, however cleverly
conditioned on what was already committed - can ever select it.

This script measures how often that happens, teacher-forced, on expert plans.
It needs no rollouts (so no divergence from the expert), no decoder, and no
training: only the encoder's embeddings and the expert's own choices.

Read the output like this:

  reachable ~100% everywhere
      The geometry is not the constraint. Wrong arguments are a LEARNING
      problem - the right query exists and the GRU is not producing it - so a
      module that changes how the query is computed is the right fix.

  reachable much lower in the weak domains (Logistics, Rovers) than the
  strong ones (Gripper, Miconic)
      The geometry IS the constraint in exactly the domains that fail. No
      query-side change can help there; you need a wider embedding (raises the
      ceiling directly) or a scorer that is NON-LINEAR in e_o. Note that FiLM
      and bilinear scoring are both still linear in e_o and do not qualify.

  norm spread (printed first) very tight, e.g. rel-std < 0.05
      All embeddings sit on a sphere, where every point is on the hull, so
      reachability is ~100% by construction and this test cannot discriminate.
      Report it and fall back to the learnability question.

Usage
-----
    python tools/reachability_probe.py \\
        --config configs/all8_joint_chain.yaml \\
        --checkpoint models/<dir>/model_e450_<ts>.pt \\
        --domain Manyblocks_ipcc_big \\
        --problems 20

CPU is fine and preferred - this competes with nothing.
"""

import argparse
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _canonical(domain):
    return domain[:1].upper() + domain[1:]


def _find_one(pattern, what):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {what} matching {pattern}")
    return hits[-1]


def _load_sidecar_metadata(domain, tag):
    path = _find_one(os.path.join('cache', 'results',
                                  f'{domain}_graphs_0_*{tag}*.pkl'), 'sidecar')
    with open(path, 'rb') as fh:
        payload = pickle.load(fh)
    if not (isinstance(payload, tuple) and len(payload) == 2):
        raise ValueError(f"{path} is not a (graphs, metadata) sidecar")
    return payload[1], path


def _load_problems(domain, limit):
    """Raw per-problem 5-tuples from the unified cache: no featurization, no
    graphs - just the states, the expert plan and the groundings."""
    path = _find_one(os.path.join('cache', 'results',
                                  f'{domain}_unified_cache_0_*.pkl'),
                     'unified cache')
    with open(path, 'rb') as fh:
        cache = pickle.load(fh)
    problems = cache.get('problems', {})
    out = []
    for key in sorted(problems)[:limit]:
        td = problems[key].get('training_data')
        # Loaders reject short tuples loudly (CLAUDE.md 1.4 contract 5).
        if td is None or len(td) < 5:
            continue
        out.append(td)
    if not out:
        raise ValueError(f"{path} holds no usable per-problem tuples")
    return out, path


def reachable(E, target, iters, device):
    """Is there a direction q with q.E[target] > q.E[j] for every other j?

    Maximise the smallest margin with q constrained to the unit sphere. If the
    best achievable smallest margin is positive, some query selects the target;
    if it converges to <= 0, none does. Softplus rather than a hard min so the
    gradient is informative when several candidates are close.
    """
    import torch
    n = E.size(0)
    if n < 2:
        return True, float('inf'), False

    diffs = E[target].unsqueeze(0) - E                      # [n, d]
    diffs = torch.cat([diffs[:target], diffs[target + 1:]])  # drop self
    # An exact duplicate embedding can never be strictly beaten: report it
    # separately rather than counting it as unreachable, since it is a
    # collision in the encoder, not a hull property.
    dup = bool((diffs.norm(dim=1) < 1e-9).any())

    q = torch.randn(E.size(1), device=device)
    q = (q / q.norm()).requires_grad_(True)
    opt = torch.optim.Adam([q], lr=0.05)
    for _ in range(iters):
        opt.zero_grad()
        torch.nn.functional.softplus(-(diffs @ q), beta=20.0).mean().backward()
        opt.step()
        with torch.no_grad():
            q.div_(q.norm() + 1e-12)          # keep the scale from running away
    margin = float((diffs @ q.detach()).min())
    return margin > 1e-6, margin, dup


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help="the config the model was trained with")
    ap.add_argument('--checkpoint', required=True, help="a .pt under models/")
    ap.add_argument('--domain', required=True, help="pddlgym domain, e.g. Manyblocks_ipcc_big")
    ap.add_argument('--problems', type=int, default=20)
    ap.add_argument('--states-per-problem', type=int, default=15,
                    help="decision points sampled per problem (evenly spaced)")
    ap.add_argument('--iters', type=int, default=300, help="optimisation steps per test")
    ap.add_argument('--device', default='cpu')
    args_cli = ap.parse_args()

    import torch
    import pddlgym
    from ploi.argparsers import get_ploi_argument_parser, apply_config_defaults
    from ploi.datautils_ltp import state_to_graph_wrapper, _get_action_object_from_action
    from ploi.modelutils_ltp import GNN_GRU
    from ploi.run_planner_with_ltp_v2 import convert_graph_to_model_input_v2

    # Config -> the hyperparameters the checkpoint was built with.
    sys.argv = ['probe', '--config', args_cli.config]
    parser = get_ploi_argument_parser()
    parser.add_argument("--all-problems", type=lambda v: str(v).lower() in ('true', '1', 'yes'),
                        default=False)
    apply_config_defaults(parser)
    args, _ = parser.parse_known_args([])

    domain = _canonical(args_cli.domain)
    tag = args.featurization
    metadata, sidecar = _load_sidecar_metadata(domain, tag)
    print(f"metadata from {sidecar}")

    env = pddlgym.make(f"PDDLEnv{domain}-v0")
    action_space = env.action_space._action_predicate_to_operators

    nf = int(metadata['num_node_features'])
    ef = int(metadata['num_edge_features'])
    model = GNN_GRU(
        n_features=nf, n_edge_features=ef,
        n_global_features=nf,          # globals row is built at node width
        n_hidden=args.representation_size, gnn_rounds=args.gnn_rounds,
        num_decoder_layers=args.gru_layers, dropout=0.0, attn_dropout=0.0,
        action_space=action_space, batch_size=1, n_heads=args.n_heads,
        g_node=args.use_global_node, num_mlp_layers_gnn=args.num_mlp_layers_gnn,
        device=args_cli.device, action_options=args.action_options,
        object_options=args.object_options, ablation=args.ablation,
    ).to(args_cli.device)
    blob = torch.load(args_cli.checkpoint, map_location=args_cli.device)
    model.load_state_dict(blob['state_dict'] if 'state_dict' in blob else blob)
    model.eval()
    print(f"loaded {args_cli.checkpoint}  (nf={nf} ef={ef} d={args.representation_size})")

    problems, cache_path = _load_problems(domain, args_cli.problems)
    print(f"{len(problems)} problems from {cache_path}\n")

    norms, per_slot, dups, skipped = [], {}, 0, 0
    checked_alignment = False

    # Each cache entry is ONE problem: (states, objects, plan, groundings, goal_dists).
    for states, _objects, plan, grnd, _gd in problems:
        n = min(len(states), len(plan))
        if n == 0:
            continue
        step = max(1, n // max(args_cli.states_per_problem, 1))
        for j in range(0, n, step):
            state, action = states[j], plan[j]
            try:
                g, _, node_to_objects = state_to_graph_wrapper(
                    state, action_space, grnd, prev_actions=None, prev_state=None,
                    graph_metadata=metadata, curr_action=None, objects=None,
                    goal_state=state.goal)
                with torch.inference_mode():
                    enc = model.extract_data_and_run_encoder(
                        convert_graph_to_model_input_v2([g], args_cli.device))
                x, n_objects, object_idxs = enc[0], enc[7], enc[8]
                n_obj = int(n_objects[0])
                E = x[object_idxs][:n_obj].float()
            except Exception as exc:      # one bad state must not kill the run
                skipped += 1
                if skipped <= 3:
                    print(f"  skipped a state: {type(exc).__name__}: {exc}")
                continue

            node_of = {v: k for k, v in node_to_objects.items()}
            _pred, objs = _get_action_object_from_action(action)

            # The one assumption worth checking out loud: that row j of E is the
            # object at node_to_objects[j], i.e. objects occupy the first n_obj
            # node slots. If that is wrong every lookup lands out of range and
            # the run would silently report nothing.
            if not checked_alignment:
                checked_alignment = True
                idx = [node_of.get(o) for o in objs]
                print(f"alignment check: n_objects={n_obj}, E={tuple(E.shape)}, "
                      f"expert arg rows={idx}")
                if any(i is None or i >= n_obj for i in idx):
                    print("  WARNING: an expert argument does not map into the object rows.")
                    print("  Row order is not node_to_objects order - fix the mapping before")
                    print("  trusting any number below.")

            norms.extend(E.norm(dim=1).tolist())
            for slot, o in enumerate(objs):
                t = node_of.get(o)
                if t is None or t >= E.size(0):
                    skipped += 1
                    continue
                ok, _margin, dup = reachable(E, t, args_cli.iters, args_cli.device)
                dups += int(dup)
                rec = per_slot.setdefault(slot, {'n': 0, 'ok': 0, 'cands': 0})
                rec['n'] += 1
                rec['ok'] += int(ok)
                rec['cands'] += E.size(0)

    # ---- report --------------------------------------------------------------
    print(f"=== {domain} ===")
    if norms:
        import statistics
        mu, sd = statistics.mean(norms), statistics.pstdev(norms)
        print(f"embedding norms: mean {mu:.3f}, std {sd:.3f}, rel-std {sd / max(mu, 1e-9):.3f}")
        if sd / max(mu, 1e-9) < 0.05:
            print("  WARNING: norms are nearly equal - the embeddings lie on a sphere,")
            print("  where every point is on the hull. Reachability will read ~100% and")
            print("  this test cannot discriminate. The question is learnability instead.")
    print()
    print(f"{'slot':<6}{'decisions':<12}{'reachable':<12}{'avg candidates':<16}")
    tot_n = tot_ok = 0
    for slot in sorted(per_slot):
        r = per_slot[slot]
        tot_n += r['n']; tot_ok += r['ok']
        print(f"{slot:<6}{r['n']:<12}{100.0 * r['ok'] / max(r['n'], 1):>8.1f}%   "
              f"{r['cands'] / max(r['n'], 1):>10.1f}")
    if tot_n:
        print(f"\noverall: {100.0 * tot_ok / tot_n:.1f}% reachable over {tot_n} decisions")
    print(f"duplicate-embedding collisions: {dups}")
    if skipped:
        print(f"skipped: {skipped}")
    print("\nCompare across domains. Uniformly high = learnability question;")
    print("low in the weak domains = the geometry is the constraint there.")


if __name__ == '__main__':
    main()

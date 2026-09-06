"""Cheating-input ceiling: teacher-forced accuracy on ANY domain.

Companion to `--cheating-input True` training. That run injects the expert
answer as node features (is_correct_action / is_correct_obj_j) and drives
training loss to ~0. Rollout evaluation of such a model is meaningless -
at rollout there is no expert action, so the columns are all zero and the
model sees no hint. This script measures the thing that IS meaningful:
teacher-forced top-1 accuracy on expert-plan states, WITH the hint set, on
domains the model never trained on.

~100% on held-out domains = the encoder/decoder can carry an explicitly
marked, symbol-free node feature to both output heads in a domain-agnostic
way, i.e. the architecture can exploit good features across domains (the
ceiling for any computed-guidance feature). Well below 100% = the
bottleneck is architectural reach, not the features.

Usage (same config as the cheating training run; --eval-domains defaults
to the config's test_domains that are NOT in domains):

  python tools/cheat_ceiling_eval.py --config configs/ho4_joint.yaml \\
      --cheating-input True --num-train-problems 10 --seed 11 \\
      --device cuda:0 --eval-domains miconic_ipcc,grid_ipcc,spanner_ipcc

Prints a per-domain table; also writes cache/results/cheat_ceiling.txt.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from ploi.argparsers import get_ploi_argument_parser, apply_config_defaults
from ploi.datautils_ltp import (load_domain_metadata, process_pddl_to_graphs,
                                _create_graph_dataset_ltp,
                                graph_dataset_to_pyg_dataset)
from ploi.lifted_layer import build_lifted_metadata
from ploi.model_checkpointing import ModelManager
from ploi.modelutils_ltp import GNN_GRU
from ploi.multidomain import parse_domain_arg
from ploi.run_planner_with_ltp_v1 import _create_planner
from ploi.structural import build_structural_metadata


def build_args():
    parser = get_ploi_argument_parser()
    parser.add_argument("--all-problems", action="store_true")
    parser.add_argument("--eval-domains", type=str, default="",
                        help="Comma-separated domains to evaluate (default: "
                             "config test_domains not present in domains).")
    parser.add_argument("--eval-problems", type=int, default=10,
                        help="Problems per evaluated domain (expert plans).")
    parser.add_argument("--metric", type=str, default="combined")
    apply_config_defaults(parser)
    args = parser.parse_args()
    args.domain = args.domain.capitalize()
    return args


def domain_metadata(args, planner, names):
    """(name -> (metadata, action_space)) via the per-domain collection pass."""
    out = {}
    for name in names:
        md, aspace = load_domain_metadata(name, planner, args.eval_problems,
                                          args, _create_graph_dataset_ltp)
        out[name] = (md, aspace)
    return out


@torch.no_grad()
def accuracy(model, loader, device):
    """Teacher-forced top-1: schema, per-parameter object, and whole action."""
    n_sch = n_sch_ok = n_obj = n_obj_ok = n_act = n_act_ok = 0
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        action_scores, ao_scores = model(batch, beam_search=False)

        tgt_a = batch['target_action_scores'].x
        tgt_ao = batch['target_action_object_scores'].x
        tgt_p = batch['target_n_parameters'].x

        # Same index construction as the training loop: the rows of
        # action_object_scores that belong to real parameter slots.
        counts = tgt_p.long().flatten()
        starts = (torch.arange(counts.numel(), device=counts.device)
                  * model.max_number_action_parameters)
        within = (torch.arange(int(counts.sum()), device=counts.device)
                  - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts))
        rows = torch.repeat_interleave(starts, counts) + within

        pad = torch.nn.ConstantPad2d(
            (0, tgt_ao.shape[1] - ao_scores.shape[1], 0, 0), 0)
        ao_scores = pad(ao_scores)

        sch_ok = action_scores.argmax(1) == tgt_a.argmax(1)
        obj_ok = ao_scores[rows].argmax(1) == tgt_ao[rows].argmax(1)

        n_sch += sch_ok.numel(); n_sch_ok += int(sch_ok.sum())
        n_obj += obj_ok.numel(); n_obj_ok += int(obj_ok.sum())

        # Whole action = schema right AND every parameter right.
        per_sample = torch.split(obj_ok, counts.tolist())
        all_objs_ok = torch.tensor([bool(p.all()) for p in per_sample],
                                   device=sch_ok.device)
        n_act += sch_ok.numel(); n_act_ok += int((sch_ok & all_objs_ok).sum())

    pct = lambda a, b: 100.0 * a / b if b else float('nan')
    return pct(n_sch_ok, n_sch), pct(n_obj_ok, n_obj), pct(n_act_ok, n_act), n_act


def main():
    args = build_args()
    if not args.cheating_input:
        print("WARNING: --cheating-input is False; this measures the ceiling "
              "of a model trained WITHOUT the hint columns.")
    planner = _create_planner(args.train_planner_name)

    train_domains = [n for n, _, _ in parse_domain_arg(
        args.domains, args.heldout_domains, args.num_train_problems)]
    if args.domains:
        # Checkpoints are keyed by the composite multi-domain env name.
        args.domain = "MULTI-" + "-".join(train_domains)
    if args.eval_domains:
        eval_names = [d.split(":")[0].split("@")[0].strip().capitalize()
                      for d in args.eval_domains.split(",") if d.strip()]
    else:
        eval_names = [d.split(":")[0].split("@")[0].strip().capitalize()
                      for d in args.test_domains.split(",") if d.strip()]
    eval_names = list(dict.fromkeys(eval_names))

    # Canonical arities of the TRAINING set: the trained feature widths.
    train_meta = domain_metadata(args, planner, train_domains)
    kp = max(p.arity for (md, _) in train_meta.values()
             for p in md['all_predicates'])
    ka = max(len(op.params) for (_, a) in train_meta.values()
             for op in a.values())
    if args.max_pred_arity > 0:
        kp = max(kp, args.max_pred_arity)
    if args.max_action_arity > 0:
        ka = max(ka, args.max_action_arity)
    tag = f"_{args.featurization}_kp{kp}_ka{ka}"
    if args.cheating_input:
        tag += "_cheat"

    def metadata_for(aspace):
        if args.featurization == 'structural':
            return build_structural_metadata(aspace, kp, ka,
                                             cheating=args.cheating_input)
        return build_lifted_metadata(aspace, kp, ka, args.featurization,
                                     cheating=args.cheating_input)

    # Model input widths come from the TRAINING metadata (any training
    # domain: structural/joint widths depend only on kp/ka).
    ref_md = metadata_for(next(iter(train_meta.values()))[1])
    args.num_node_features = ref_md['num_node_features']
    args.num_edge_features = ref_md['num_edge_features']
    args.num_global_features = ref_md.get('num_global_features', 1)

    from ploi.multidomain import merge_action_spaces
    action_space = merge_action_spaces([a for (_, a) in train_meta.values()])

    device = args.device if args.use_gpu else "cpu"
    model = GNN_GRU(
        n_features=args.num_node_features,
        n_edge_features=args.num_edge_features,
        n_global_features=args.num_global_features,
        n_hidden=args.representation_size, gnn_rounds=args.gnn_rounds,
        num_decoder_layers=args.gru_layers, dropout=args.dropout,
        attn_dropout=args.attention_dropout, action_space=action_space,
        batch_size=args.batch_size, n_heads=args.n_heads,
        g_node=args.use_global_node,
        num_mlp_layers_gnn=args.num_mlp_layers_gnn, device=device,
        action_options=args.action_options, object_options=args.object_options,
        ablation=args.ablation)

    hyperparameters = {'lr': args.lr, 'gnn_rounds': args.gnn_rounds,
                       'd': args.num_train_problems,
                       'ad': args.attention_dropout, 'wd': args.dropout,
                       'heads': args.n_heads, 'g_node': args.use_global_node,
                       'abl_': args.ablation,
                       'mlp_layers': args.num_mlp_layers_gnn}
    manager = ModelManager(os.path.join(os.getcwd(), "models"),
                           hyperparameters=hyperparameters,
                           train_env_name=args.domain, seed=args.seed,
                           ignore_defaults={'mlp_layers': 2})
    best = manager.load_best_models(train_env_name=args.domain, seed=args.seed,
                                    hyperparameters=hyperparameters,
                                    metric=args.metric,
                                    ignore_defaults={'mlp_layers': 2})
    if not best:
        raise SystemExit(f"No checkpoint for {args.domain} seed {args.seed} "
                         f"metric {args.metric}. Same --expid/--seed as training?")
    info = best[-1]
    model.load_state_dict(info['state_dict'])
    model.to(device)
    print(f"Loaded epoch {info['epoch']} ({args.metric}); widths "
          f"n={args.num_node_features} e={args.num_edge_features}, kp{kp} ka{ka}")

    rows = []
    for name in eval_names:
        md, aspace = load_domain_metadata(name, planner, args.eval_problems,
                                          args, _create_graph_dataset_ltp)
        graphs, _, _ = process_pddl_to_graphs(
            name, planner, args.eval_problems, args, _create_graph_dataset_ltp,
            metadata_override=metadata_for(aspace), cache_tag=tag)
        if not graphs:
            print(f"  {name}: no graphs collected, skipping")
            continue
        loader = graph_dataset_to_pyg_dataset(
            graphs, batch_wise=True, batch_size=args.batch_size,
            shuffle=False, num_workers=0)
        sch, obj, act, n = accuracy(model, loader, device)
        seen = "train" if name in train_domains else "HELD-OUT"
        rows.append((name, seen, n, sch, obj, act))
        print(f"  {name:24s} [{seen:8s}] states={n:5d}  schema={sch:6.2f}%  "
              f"object={obj:6.2f}%  action={act:6.2f}%")

    lines = ["# Cheating-input ceiling: teacher-forced top-1 accuracy with the",
             "# expert answer injected as node features. ~100% on HELD-OUT =",
             "# the architecture carries explicit symbol-free guidance across",
             "# domains; the ceiling for any computed-guidance feature.",
             f"# checkpoint: {args.domain} seed{args.seed} epoch{info['epoch']} "
             f"({args.metric})",
             f"# {'domain':24s} {'seen':9s} {'states':>6s} {'schema%':>8s} "
             f"{'object%':>8s} {'action%':>8s}"]
    for name, seen, n, sch, obj, act in rows:
        lines.append(f"  {name:24s} {seen:9s} {n:6d} {sch:8.2f} {obj:8.2f} "
                     f"{act:8.2f}")
    out = "cache/results/cheat_ceiling.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

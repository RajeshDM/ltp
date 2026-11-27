#!/usr/bin/env python3
"""Write the configs of the 2026-11 protocol (configs/p2_*.yaml) (C1, C2).

    python tools/make_protocol_configs.py

One seed, every run evaluated with constrained decoding on the new splits
(tools/install_splits.py). Paper table each feeds:
  p2_sd_{jc,gabar}_<d>          8 + 8   Table 2 cols 2, 1   (in-domain)
  p2_typed_{jc,gabar}_<d>       2 + 2   Appendix A5          (type spelling)
  p2_all8_jc                    1       Table 2 col 3; Table 1 Organic Synthesis
  p2_loo_{jc,union}_no_<d>      8 + 8   Table 1              (zero-shot)
  p2_lad_<rung>_no_<d>          7 x 3   Table 3              (ablation ladder)
Ladder folds (held out) are Blocks, Logistics, Visitall; their configs (and the
jc/union LOO runs of the same folds) also test the 7 training domains' hard
band for Table 3's in-domain column.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs")
DOMAINS = ["manyblocks_ipcc_big", "gripper_ipcc", "miconic_ipcc", "visitall_ipcc",
           "grid_ipcc", "logistics_ipcc", "spanner_ipcc", "rovers_ipcc"]
SHORT = {d: d.split("_")[0] for d in DOMAINS}
LADDER_FOLDS = ["manyblocks_ipcc_big", "logistics_ipcc", "visitall_ipcc"]
LADDER = ["structural", "joint_lite", "joint", "jc_no_bind", "jc_no_chain", "jc_no_eff", "jc_no_goal"]
LIFTED_OR_STRUCT = set(LADDER) | {"joint_chain"}

BASE = """# 2026-11 evaluation protocol (ICLR27/PAPER_PLAN.html, "Train / validation /
# test"). Delta over _common.yaml; every p2_* config points here.
#   validation : loss on UNSEEN problems (<dom>_val_ood = 'val', <dom>_val_id
#                logged beside it), not 10% of training states
#   snapshots  : every 50 epochs (checkpoints are written every 10, so 50 is
#                the finest interval that is not silently thinned), all kept
#   selection  : best Val-OOD coverage over those snapshots, ties -> earlier
#                (tools/select_checkpoint.py; campaign/jobs/protocol_eval.sh)
#   decoding   : constrained (GABAR_CONSTRAINED_DECODE=1, set by the job)
description: 2026-11 protocol base
base: _common.yaml
val_protocol: split
checkpoint_every: 50
max_periodic: 10
test_model_metrics: periodic
num_models_to_test: 1
"""


def bands(d):
    return [f"{d}@easy", f"{d}@medium", f"{d}@hard"]


def write(name, desc, domains, feat, tests, extra=None):
    lines = [f"description: {desc}", "base: _protocol.yaml", f"expid: {name}",
             f"domains: {','.join(domains)}", f"featurization: {feat}",
             f"test_domains: {','.join(tests)}"]
    for k, v in (extra or {}).items():
        lines.append(f"{k}: {v}")
    with open(os.path.join(CFG, name + ".yaml"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return name


def main():
    with open(os.path.join(CFG, "_protocol.yaml"), "w") as f:
        f.write(BASE)
    out = []
    for d in DOMAINS:
        s = SHORT[d]
        out.append(write(f"p2_sd_jc_{s}", f"GADAR (joint_chain) trained on {s} alone", [d], "joint_chain", bands(d)))
        out.append(write(f"p2_sd_gabar_{s}", f"GABAR (per_domain) trained on {s}", [d], "per_domain", bands(d)))
    for d in ["grid_ipcc", "logistics_ipcc"]:
        t = d.replace("_ipcc", "_typed_ipcc")
        s = SHORT[d]
        out.append(write(f"p2_typed_jc_{s}", f"GADAR on typed {s} (E6)", [t], "joint_chain", bands(t)))
        out.append(write(f"p2_typed_gabar_{s}", f"GABAR on typed {s} (E6)", [t], "per_domain", bands(t)))
    out.append(write("p2_all8_jc", "One GADAR model on all 8 domains", DOMAINS, "joint_chain",
                     [b for d in DOMAINS for b in bands(d)]))
    for fold in DOMAINS:
        train = [d for d in DOMAINS if d != fold]
        tests = bands(fold) + ([f"{d}@hard" for d in train] if fold in LADDER_FOLDS else [])
        rungs = [("jc", "joint_chain"), ("union", "union")]
        if fold in LADDER_FOLDS:
            rungs += [(r, r) for r in LADDER]
        for short, feat in rungs:
            prefix = "p2_loo" if short in ("jc", "union") else "p2_lad"
            extra = {"max_action_arity": 6} if fold == "rovers_ipcc" and feat in LIFTED_OR_STRUCT else None
            out.append(write(f"{prefix}_{short}_no_{SHORT[fold]}", f"{feat}, {SHORT[fold]} held out",
                             train, feat, tests, extra))
    print(f"wrote configs/_protocol.yaml and {len(out)} configs")


if __name__ == "__main__":
    main()

"""The checkpoint key must separate every config, and move no existing one.

Two properties, and the second is the dangerous one:

  1. Any setting that changes the trained weights changes the model
     directory. Without this, two configs differing only in that setting
     share one directory and evict each other's loss-ranked checkpoints.
     That is not hypothetical: sweep_jc_base and sweep_jc_l2 ran
     concurrently, keyed identically because weight_decay was not in the
     key, and destroyed each other's arm.

  2. Adding a key must leave every EXISTING directory name and hash
     byte-identical, or every checkpoint in models/ becomes unfindable.
     This holds only because each new key is defaulted away at the value
     the suite actually runs at - which is not always the argparse default
     (batch_size is 16 in constants.py and 64 in _common.yaml).

The key deliberately covers only what the suite VARIES, so directory names
stay readable. What keeps that safe is main.py's
_assert_key_covers_variation, which refuses to train when a weight-changing
parameter that is not in the key is moved off the suite's value.

The reference names below were taken from real directories in models/ on
2026-09-05, before the keys were added. If a change here renames them, the
test says so with both names.

Run: python tests/test_checkpoint_key.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ploi.model_checkpointing import ModelManager

folder = ModelManager.get_readable_folder_name
config_hash = ModelManager.get_config_hash

# What main.py builds, with the suite's effective settings (_common.yaml).
SUITE = {
    'lr': 0.0005, 'gnn_rounds': 9, 'd': 0, 'ad': 0.0, 'wd': 0.0, 'heads': 1,
    'g_node': True, 'abl_': 'main', 'mlp_layers': 2,
    'feat': 'joint_chain', 'nf': 38, 'ef': 60, 'l2': 0.0,
    'bs': 64, 'ka': 0,
}

IGNORE = {'mlp_layers': 2, 'l2': 0.0, 'bs': 64, 'ka': 0}

ENV = ("MULTI-Manyblocks_ipcc_big-Gripper_ipcc-Miconic_ipcc-Visitall_ipcc-"
       "Grid_ipcc-Logistics_ipcc-Spanner_ipcc-Rovers_ipcc")

# Verbatim from `ls models/` before the keys were added.
EXPECTED = (ENV + "_seed11_abl_main_ad0e_00_d0_ef60_featjoint_chain_"
            "g_nodeTrue_gnn_rounds9_heads1_lr5e-04_nf38_wd0e_00")


def test_existing_directories_do_not_move():
    got = folder(ENV, 11, SUITE, IGNORE)
    assert got == EXPECTED, (
        "the checkpoint key renamed an existing directory - every checkpoint "
        f"under it becomes unfindable\n  was: {EXPECTED}\n  now: {got}")
    print("ok  existing model directories keep their names")


def test_every_weight_changing_setting_separates():
    """One at a time: change it, the directory must change."""
    base_dir = folder(ENV, 10, SUITE, IGNORE)
    base_hash = config_hash(ENV, 42, SUITE, IGNORE)

    # (key, a different value) for everything that changes trained weights.
    variants = [
        ('l2', 1e-4),           # sweep_jc_l2 vs sweep_jc_base
        ('bs', 16),             # legacy configs vs _common.yaml
        ('ka', 6),              # the 9 rovers configs
        ('lr', 1e-3), ('gnn_rounds', 6), ('heads', 4), ('wd', 0.1),
        ('ad', 0.1), ('feat', 'union'), ('d', 100), ('abl_', 'non_CD'),
        ('mlp_layers', 3), ('nf', 163), ('ef', 111),
    ]
    for key, other in variants:
        hp = dict(SUITE, **{key: other})
        assert folder(ENV, 10, hp, IGNORE) != base_dir, (
            f"'{key}' does not change the model directory: two configs "
            f"differing only in {key} would share one and evict each other")
        assert config_hash(ENV, 42, hp, IGNORE) != base_hash, (
            f"'{key}' does not change the config hash")
    print(f"ok  all {len(variants)} weight-changing settings separate")


def test_the_sweep_collision_is_closed():
    """The exact pair that destroyed each other."""
    base = dict(SUITE)                      # sweep_jc_base
    l2 = dict(SUITE, l2=1e-4)               # sweep_jc_l2
    assert folder(ENV, 11, base, IGNORE) != folder(ENV, 11, l2, IGNORE), \
        "sweep_jc_base and sweep_jc_l2 still resolve to one directory"
    # ...and base still has the name it had while they were colliding, so the
    # base arm's surviving checkpoints stay reachable.
    assert folder(ENV, 11, base, IGNORE) == EXPECTED
    print("ok  base/l2 separate, and base keeps its directory")


def test_seed_and_domain_set_separate():
    a = folder(ENV, 10, SUITE, IGNORE)
    assert folder(ENV, 11, SUITE, IGNORE) != a, "seed does not separate"
    assert folder(ENV.replace("-Miconic_ipcc", ""), 10, SUITE, IGNORE) != a, \
        "training-domain set does not separate"
    print("ok  seed and training-domain set separate")


if __name__ == '__main__':
    test_existing_directories_do_not_move()
    test_every_weight_changing_setting_separates()
    test_the_sweep_collision_is_closed()
    test_seed_and_domain_set_separate()
    print("\nall passed")

"""Unit tests for the multi-domain harness metadata logic (CLAUDE.md Phase 0/1, C1).

Dependency-free (no torch/pddlgym): exercises the pure-dict union-vocabulary
merge that Baseline 0 (§5.2) relies on. Run with: python -m pytest tests/ -q
(or plain `python tests/test_multidomain_metadata.py`).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ploi.multidomain import (
    DomainSpec,
    MultiDomainConfig,
    merge_feature_metadata,
    merge_action_spaces,
    parse_domain_arg,
)


def _metadata(node_symbols, edge_symbols, predicates):
    return {
        "num_node_features": len(node_symbols),
        "num_edge_features": len(edge_symbols),
        "node_feature_to_index": {s: i for i, s in enumerate(node_symbols)},
        "edge_feature_to_index": {s: i for i, s in enumerate(edge_symbols)},
        "unary_types": [],
        "unary_predicates": list(predicates),
        "binary_predicates": [],
        "all_predicates": list(predicates),
    }


BLOCKS = _metadata(
    ["action_node", "object_node", "predicate_node", "pickup", "on", "clear", "block"],
    ["action_object", "pos_0", "pos_1", "on0", "clear1", "pred_pos_0"],
    ["on", "clear"],
)
GRIPPER = _metadata(
    ["action_node", "object_node", "predicate_node", "move", "at", "free", "ball"],
    ["action_object", "pos_0", "at0", "free1", "pred_pos_0"],
    ["at", "free"],
)


def test_single_domain_merge_is_identity():
    merged = merge_feature_metadata([BLOCKS])
    assert merged == BLOCKS


def test_union_contains_all_symbols_once():
    merged = merge_feature_metadata([BLOCKS, GRIPPER])
    node_map = merged["node_feature_to_index"]
    for symbol in list(BLOCKS["node_feature_to_index"]) + list(
        GRIPPER["node_feature_to_index"]
    ):
        assert symbol in node_map
    # Shared structural keys appear once; indices are dense and unique.
    indices = sorted(node_map.values())
    assert indices == list(range(len(node_map)))
    assert merged["num_node_features"] == len(node_map)
    assert merged["num_edge_features"] == len(merged["edge_feature_to_index"])


def test_union_widths_exceed_each_domain():
    merged = merge_feature_metadata([BLOCKS, GRIPPER])
    assert merged["num_node_features"] > BLOCKS["num_node_features"]
    assert merged["num_node_features"] > GRIPPER["num_node_features"]


def test_merge_is_deterministic_and_order_stable():
    m1 = merge_feature_metadata([BLOCKS, GRIPPER])
    m2 = merge_feature_metadata([BLOCKS, GRIPPER])
    assert m1 == m2
    # First domain's symbols keep their relative order (cache stability).
    node_map = m1["node_feature_to_index"]
    blocks_order = sorted(
        BLOCKS["node_feature_to_index"],
        key=lambda s: BLOCKS["node_feature_to_index"][s],
    )
    merged_positions = [node_map[s] for s in blocks_order]
    assert merged_positions == sorted(merged_positions)


def test_predicate_lists_are_unioned():
    merged = merge_feature_metadata([BLOCKS, GRIPPER])
    assert set(merged["all_predicates"]) == {"on", "clear", "at", "free"}


def test_action_space_merge_and_collision():
    op_a, op_b = object(), object()
    merged = merge_action_spaces([{"pickup": op_a}, {"move": op_b}])
    assert merged == {"pickup": op_a, "move": op_b}
    try:
        merge_action_spaces([{"pickup": op_a}, {"pickup": op_b}])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected collision to raise ValueError")


def test_parse_domain_arg():
    specs = parse_domain_arg("blocks:100,gripper", "spanner",
                             num_train_problems=50, num_test_problems=5)
    assert [s.name for s in specs] == ["Blocks", "Gripper", "Spanner"]
    assert specs[0].num_train_problems == 100
    assert specs[1].num_train_problems == 50
    assert [s.held_out for s in specs] == [False, False, True]
    cfg = MultiDomainConfig(domains=specs)
    assert [d.name for d in cfg.train_domains] == ["Blocks", "Gripper"]
    assert [d.name for d in cfg.held_out_domains] == ["Spanner"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All multidomain metadata tests passed.")

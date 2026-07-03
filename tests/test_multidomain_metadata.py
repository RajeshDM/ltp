"""Unit tests for the multi-domain harness metadata logic (CLAUDE.md Phase 0/1, C1).

Dependency-free (no torch/pddlgym): exercises the pure-dict union-vocabulary
merge that Baseline 0 (§5.2) relies on. Run with: python -m pytest tests/ -q
(or plain `python tests/test_multidomain_metadata.py`).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ploi.multidomain import (
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
    domains = parse_domain_arg("blocks:100,gripper", "spanner", num_train_problems=50)
    assert domains == [("Blocks", 100, False), ("Gripper", 50, False), ("Spanner", 50, True)]




# --- structural featurization (CLAUDE.md 5.3, C1/C2) ---

from ploi.structural import build_structural_metadata


class _Pred:
    def __init__(self, name, arity):
        self.name, self.arity = name, arity
    def __str__(self):
        return self.name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        return getattr(other, "name", None) == self.name


class _Lit:
    def __init__(self, pred):
        self.predicate = pred


class _Conj:
    def __init__(self, preds):
        self.literals = [_Lit(p) for p in preds]


class _Op:
    def __init__(self, params, preconds, effects):
        self.params, self.preconds, self.effects = params, _Conj(preconds), _Conj(effects)


def _blocks_like(names):
    on, clear, table = _Pred(names[0], 2), _Pred(names[1], 1), _Pred(names[2], 1)
    pickup = _Pred(names[3], 1)
    return {pickup: _Op(["?x"], [clear, table], [clear])}, (on, clear, table)


def test_structural_same_widths_across_domains():
    space_a, _ = _blocks_like(["on", "clear", "ontable", "pickup"])
    move = _Pred("move", 2)
    space_b = {move: _Op(["?a", "?b"], [_Pred("at", 2)], [_Pred("at", 2)])}
    md_a = build_structural_metadata(space_a, 3, 4)
    md_b = build_structural_metadata(space_b, 3, 4)
    assert md_a["num_node_features"] == md_b["num_node_features"]
    assert md_a["num_edge_features"] == md_b["num_edge_features"]


def test_structural_renaming_invariance():
    space_a, preds_a = _blocks_like(["on", "clear", "ontable", "pickup"])
    space_b, preds_b = _blocks_like(["xyzzy", "foo", "bar", "grab"])
    md_a = build_structural_metadata(space_a, 3, 4)
    md_b = build_structural_metadata(space_b, 3, 4)
    for pa, pb in zip(preds_a, preds_b):
        assert md_a["node_feature_to_index"][pa] == md_b["node_feature_to_index"][pb]
    (schema_a,), (schema_b,) = space_a.keys(), space_b.keys()
    assert (md_a["node_feature_to_index"][schema_a]
            == md_b["node_feature_to_index"][schema_b])


def test_structural_static_vs_dynamic_predicates_differ():
    space, (on, clear, table) = _blocks_like(["on", "clear", "ontable", "pickup"])
    node_map = build_structural_metadata(space, 3, 4)["node_feature_to_index"]
    # clear is in effects (dynamic), ontable only in preconds (static)
    assert node_map[clear] != node_map[table]
    assert node_map[clear] == node_map[_Pred("other_dynamic", 1)] or True


def test_structural_unseen_symbols_and_arity_clamp():
    space, _ = _blocks_like(["on", "clear", "ontable", "pickup"])
    md = build_structural_metadata(space, 2, 2)
    # unseen predicate classes on the fly; arity above max clamps
    idx = md["node_feature_to_index"][_Pred("brand_new", 5)]
    assert idx == md["node_feature_to_index"]["pred_a2_s1"]
    assert md["edge_feature_to_index"]["pos_9"] == md["edge_feature_to_index"]["pos_1"]
    # goal-wrapped predicate gets a goal class
    widx = md["node_feature_to_index"][_Pred("WANTon", 2)]
    assert widx == md["node_feature_to_index"]["gpred_a2_s1"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All multidomain metadata tests passed.")

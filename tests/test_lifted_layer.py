"""Unit tests for the lifted domain layer + binding layer (CLAUDE.md 5.4/5.5).

Dependency-free (no torch/pddlgym): mocks mirror the pddlgym operator
surface that structural.py/lifted_layer.py touch. Run with:
python -m pytest tests/test_lifted_layer.py -q  (or plain python).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ploi.lifted_layer import (
    build_lifted_spec,
    build_lifted_metadata,
    lifted_node_keys,
    add_lifted_node_features,
    add_lifted_layer_edges,
    pred_node_key,
    occ_node_key,
)


# ─── pddlgym-shaped mocks ───────────────────────────────────────────────────

class MockPred:
    def __init__(self, name, arity):
        self.name = name
        self.arity = arity

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return getattr(other, "name", None) == self.name


class MockLiteral:
    def __init__(self, pred, variables, is_anti=False):
        self.predicate = pred
        self.variables = list(variables)
        self.is_anti = is_anti


class MockConj:
    def __init__(self, literals):
        self.literals = list(literals)


class MockOp:
    def __init__(self, params, preconds, effects):
        self.params = list(params)
        self.preconds = MockConj(preconds)
        self.effects = MockConj(effects)


def blocks_action_space():
    """4-op blocks: stack/unstack share arity AND predicates -- only their
    add/del roles differ. The case Method 0 cannot represent."""
    on = MockPred("on", 2)
    clear = MockPred("clear", 1)
    holding = MockPred("holding", 1)
    stack = MockOp(
        params=["?x", "?y"],
        preconds=[MockLiteral(holding, ["?x"]), MockLiteral(clear, ["?y"])],
        effects=[MockLiteral(on, ["?x", "?y"]),
                 MockLiteral(holding, ["?x"], is_anti=True)])
    unstack = MockOp(
        params=["?x", "?y"],
        preconds=[MockLiteral(on, ["?x", "?y"]), MockLiteral(clear, ["?x"])],
        effects=[MockLiteral(on, ["?x", "?y"], is_anti=True),
                 MockLiteral(holding, ["?x"])])
    return {MockPred("stack", 2): stack, MockPred("unstack", 2): unstack}


def renamed_blocks_action_space():
    """Same domain, every symbol renamed (renaming-invariance probe)."""
    on = MockPred("zzz_rel", 2)
    clear = MockPred("aaa_free", 1)
    holding = MockPred("mmm_grip", 1)
    stack = MockOp(
        params=["?a", "?b"],
        preconds=[MockLiteral(holding, ["?a"]), MockLiteral(clear, ["?b"])],
        effects=[MockLiteral(on, ["?a", "?b"]),
                 MockLiteral(holding, ["?a"], is_anti=True)])
    unstack = MockOp(
        params=["?a", "?b"],
        preconds=[MockLiteral(on, ["?a", "?b"]), MockLiteral(clear, ["?a"])],
        effects=[MockLiteral(on, ["?a", "?b"], is_anti=True),
                 MockLiteral(holding, ["?a"])])
    return {MockPred("put", 2): stack, MockPred("take", 2): unstack}


KP, KA = 3, 3


# ─── spec extraction ────────────────────────────────────────────────────────

def test_spec_roles_distinguish_stack_from_unstack():
    spec = build_lifted_spec(blocks_action_space())
    stack_roles = [(o["role"], o["pred"]) for o in spec["schemas"]["stack"]]
    unstack_roles = [(o["role"], o["pred"]) for o in spec["schemas"]["unstack"]]
    assert ("add", "on") in stack_roles and ("del", "on") not in stack_roles
    assert ("del", "on") in unstack_roles and ("add", "on") not in unstack_roles
    assert stack_roles != unstack_roles


def test_spec_preconditions_first_in_original_order():
    spec = build_lifted_spec(blocks_action_space())
    occs = spec["schemas"]["unstack"]
    assert occs[0]["role"] == "pre" and occs[0]["pred"] == "on"
    assert occs[1]["role"] == "pre" and occs[1]["pred"] == "clear"


def test_spec_bindings_map_pred_args_to_slots():
    spec = build_lifted_spec(blocks_action_space())
    on_pre = spec["schemas"]["unstack"][0]
    assert on_pre["bindings"] == [(0, 0), (1, 1)]
    clear_pre = spec["schemas"]["stack"][1]  # clear(?y) -> slot 1
    assert clear_pre["bindings"] == [(0, 1)]


def test_spec_static_flags():
    spec = build_lifted_spec(blocks_action_space())
    assert spec["predicates"]["on"]["static"] is False
    assert spec["predicates"]["clear"]["static"] is True  # never in effects here


# ─── metadata ───────────────────────────────────────────────────────────────

def test_metadata_widths_are_canonical_across_domains():
    md1 = build_lifted_metadata(blocks_action_space(), KP, KA, "joint")
    md2 = build_lifted_metadata(renamed_blocks_action_space(), KP, KA, "joint")
    assert md1["num_node_features"] == md2["num_node_features"]
    assert md1["num_edge_features"] == md2["num_edge_features"]


def test_joint_and_joint_lite_share_widths():
    mdj = build_lifted_metadata(blocks_action_space(), KP, KA, "joint")
    mdl = build_lifted_metadata(blocks_action_space(), KP, KA, "joint_lite")
    assert mdj["num_node_features"] == mdl["num_node_features"]
    assert mdj["num_edge_features"] == mdl["num_edge_features"]


def test_joint_has_occurrence_nodes_lite_does_not():
    mdj = build_lifted_metadata(blocks_action_space(), KP, KA, "joint")
    mdl = build_lifted_metadata(blocks_action_space(), KP, KA, "joint_lite")
    keys_j = lifted_node_keys(mdj["lifted_spec"])
    keys_l = lifted_node_keys(mdl["lifted_spec"])
    assert any(k.startswith("LIFTED_OCC::") for k in keys_j)
    assert not any(k.startswith("LIFTED_OCC::") for k in keys_l)
    assert [k for k in keys_j if k.startswith("LIFTED_PRED::")] == keys_l


def test_lifted_wider_than_structural():
    from ploi.structural import build_structural_metadata
    md_s = build_structural_metadata(blocks_action_space(), KP, KA)
    md_j = build_lifted_metadata(blocks_action_space(), KP, KA, "joint")
    assert md_j["num_node_features"] > md_s["num_node_features"]
    assert md_j["num_edge_features"] > md_s["num_edge_features"]


# ─── graph-side features and edges ──────────────────────────────────────────

def _build_edges(mode, action_space):
    md = build_lifted_metadata(action_space, KP, KA, mode)
    spec = md["lifted_spec"]
    keys = lifted_node_keys(spec)
    all_actions = list(action_space.keys())
    # tiny fake state: 2 objects, 1 literal on(a,b)
    on = next(p for p in [l.predicate for op in action_space.values()
                          for l in op.preconds.literals] if p.arity == 2)
    lit = MockLiteral(on, ["a", "b"])
    node_order = ["a", "b", lit] + keys + all_actions
    objects_to_node = {v: i for i, v in enumerate(node_order)}
    n = len(node_order)
    feats = np.zeros((n, md["num_node_features"]))
    edges = np.zeros((n, n, md["num_edge_features"]))
    add_lifted_node_features(feats, keys, spec, objects_to_node,
                             md["node_feature_to_index"], KP)
    add_lifted_layer_edges(edges, [lit], all_actions, spec, objects_to_node,
                           md["edge_feature_to_index"], KP, KA)
    return md, spec, objects_to_node, feats, edges, lit


def test_instantiation_edge_links_literal_to_symbol():
    md, spec, o2n, feats, edges, lit = _build_edges("joint_lite",
                                                    blocks_action_space())
    E = md["edge_feature_to_index"]
    li, pi = o2n[lit], o2n[pred_node_key("on")]
    assert edges[li, pi, E["instance_of"]] == 1
    assert edges[pi, li, E["instance_of"]] == 1


def test_role_edges_differ_for_stack_vs_unstack():
    md, spec, o2n, feats, edges, _ = _build_edges("joint_lite",
                                                  blocks_action_space())
    E = md["edge_feature_to_index"]
    pi = o2n[pred_node_key("on")]
    stack_i = o2n[next(a for a in o2n if getattr(a, "name", "") == "stack")]
    unstack_i = o2n[next(a for a in o2n if getattr(a, "name", "") == "unstack")]
    assert edges[pi, stack_i, E["role_add"]] == 1
    assert edges[pi, stack_i, E["role_del"]] == 0
    assert edges[pi, unstack_i, E["role_del"]] == 1
    assert edges[pi, unstack_i, E["role_add"]] == 0


def test_joint_occurrence_edges_present():
    md, spec, o2n, feats, edges, _ = _build_edges("joint",
                                                  blocks_action_space())
    E = md["edge_feature_to_index"]
    occ0 = o2n[occ_node_key("unstack", 0)]  # unstack's on(?x,?y) precond
    pi = o2n[pred_node_key("on")]
    unstack_i = o2n[next(a for a in o2n if getattr(a, "name", "") == "unstack")]
    assert edges[occ0, pi, E["occ_of"]] == 1
    assert edges[occ0, unstack_i, E["role_pre"]] == 1
    assert edges[occ0, unstack_i, E["pair_p0_s0"]] == 1
    assert edges[occ0, unstack_i, E["pair_p1_s1"]] == 1


def test_renaming_invariance_of_lifted_tensors():
    """Renamed domain must produce identical lifted node-feature tensors
    (up to the shared canonical ordering of structural classes)."""
    md1, spec1, o2n1, feats1, edges1, _ = _build_edges(
        "joint", blocks_action_space())
    md2, spec2, o2n2, feats2, edges2, _ = _build_edges(
        "joint", renamed_blocks_action_space())
    # Same number of lifted nodes and identical multisets of feature rows.
    k1 = lifted_node_keys(spec1)
    k2 = lifted_node_keys(spec2)
    assert len(k1) == len(k2)
    rows1 = sorted(tuple(feats1[o2n1[k]]) for k in k1)
    rows2 = sorted(tuple(feats2[o2n2[k]]) for k in k2)
    assert rows1 == rows2
    assert edges1.sum() == edges2.sum()


def test_metadata_pickle_roundtrip():
    """Metadata (incl. StructuralMap) is stored in the unified cache; it must
    pickle. Reloaded maps act as plain dicts (no unseen-symbol fallback)."""
    import pickle
    md = build_lifted_metadata(blocks_action_space(), KP, KA, "joint")
    blob = pickle.dumps(md)
    md2 = pickle.loads(blob)
    assert md2["num_node_features"] == md["num_node_features"]
    assert md2["edge_feature_to_index"]["instance_of"] == \
        md["edge_feature_to_index"]["instance_of"]
    assert md2["lifted_spec"]["schemas"].keys() == md["lifted_spec"]["schemas"].keys()
    try:
        md2["node_feature_to_index"]["never_seen_symbol"]
        raise RuntimeError("reloaded map should not invent classes")
    except KeyError:
        pass


def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

# --- schema chaining (#4): enables/threatens + goal distances ---

from ploi.lifted_layer import build_chain_spec, schema_goal_distances


class _CPred:
    def __init__(self, name, arity=1):
        self.name, self.arity = name, arity
class _CLit:
    def __init__(self, pred, is_anti=False):
        self.predicate, self.is_anti, self.variables = pred, is_anti, []
class _CConj:
    def __init__(self, lits):
        self.literals = lits
class _COp:
    def __init__(self, pre, add, dele):
        self.params = []
        self.preconds = _CConj([_CLit(_CPred(p)) for p in pre])
        self.effects = _CConj([_CLit(_CPred(p)) for p in add]
                              + [_CLit(_CPred(p), True) for p in dele])


def _gripper_like_space():
    return {
        "pick": _COp(pre=["at", "at-robby", "free"], add=["carry"],
                     dele=["at", "free"]),
        "drop": _COp(pre=["carry", "at-robby"], add=["at", "free"],
                     dele=["carry"]),
        "move": _COp(pre=["at-robby"], add=["at-robby"], dele=["at-robby"]),
    }


def test_chain_enables_and_threatens():
    spec = build_lifted_spec(_gripper_like_space())
    chain = build_chain_spec(spec)
    pairs = {(a, b) for (a, _, b, _) in chain["enables"]}
    assert ("pick", "drop") in pairs      # pick.add:carry -> drop.pre:carry
    assert ("drop", "pick") in pairs      # drop.add:at -> pick.pre:at
    assert ("move", "pick") in pairs      # move.add:at-robby -> pick.pre:at-robby
    tpairs = {(a, b) for (a, _, b, _) in chain["threatens"]}
    assert ("pick", "pick") in tpairs     # pick.del:at -> pick.pre:at
    assert ("drop", "drop") in tpairs     # drop.del:carry -> drop.pre:carry


def test_chain_goal_distances():
    spec = build_lifted_spec(_gripper_like_space())
    chain = build_chain_spec(spec)
    d = schema_goal_distances(spec, chain, {"at"})
    assert d["drop"] == 0                  # achieves the goal predicate
    assert d["pick"] == 1                  # enables drop (carry)
    assert d["move"] == 1                  # enables drop (at-robby)
    d2 = schema_goal_distances(spec, chain, {"carry"})
    assert d2["pick"] == 0 and d2["drop"] == 1 and d2["move"] == 1
    d3 = schema_goal_distances(spec, chain, {"unrelated"}, max_dist=3)
    assert all(v == 3 for v in d3.values())   # not goal-connected bucket


def test_chain_renaming_invariance():
    def renamed(space, mapping):
        out = {}
        for s, op in space.items():
            out[s + "_x"] = _COp(
                pre=[mapping[l.predicate.name] for l in op.preconds.literals],
                add=[mapping[l.predicate.name] for l in op.effects.literals
                     if not l.is_anti],
                dele=[mapping[l.predicate.name] for l in op.effects.literals
                      if l.is_anti])
        return out
    space = _gripper_like_space()
    mapping = {"at": "p1", "at-robby": "p2", "free": "p3", "carry": "p4"}
    c1 = build_chain_spec(build_lifted_spec(space))
    c2 = build_chain_spec(build_lifted_spec(renamed(space, mapping)))
    strip = lambda c: sorted((a.replace("_x", ""), i, b.replace("_x", ""), j)
                             for (a, i, b, j) in c["enables"])
    assert strip(c1) == strip(c2)

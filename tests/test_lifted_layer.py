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
    add_type_edges,
    pred_node_key,
    occ_node_key,
    type_pred_name,
    ABLATION_MODES,
    mode_components,
    spec_components,
    has_occurrence_nodes,
    add_grounded_effect_edges,
)
from ploi.structural import (has_bare_effects, op_literals, featurizer_version,
                             has_negative_preconditions)


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
    def __init__(self, pred, variables, is_anti=False, is_negative=False):
        self.predicate = pred
        self.variables = list(variables)
        self.is_anti = is_anti
        self.is_negative = is_negative


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


# --- 'joint_chain' featurization: metadata, graph features, edges ---

from ploi.lifted_layer import (
    add_chain_node_features,
    add_grounded_effect_edges,
)

_CHAIN_NODE_CLASSES = ("adds_goal_pred", "goal_relevant_obj",
                       "gdist_0", "gdist_1", "gdist_2", "gdist_3plus")
_CHAIN_EDGE_CLASSES = ("enables", "threatens", "achieves_goal", "deletes_true")


def test_joint_chain_metadata_has_new_classes_joint_does_not():
    mdc = build_lifted_metadata(blocks_action_space(), KP, KA, "joint_chain")
    mdj = build_lifted_metadata(blocks_action_space(), KP, KA, "joint")
    for cls in _CHAIN_NODE_CLASSES:
        assert cls in mdc["node_feature_to_index"], cls
        assert cls not in mdj["node_feature_to_index"], cls
    for cls in _CHAIN_EDGE_CLASSES:
        assert cls in mdc["edge_feature_to_index"], cls
        assert cls not in mdj["edge_feature_to_index"], cls
    # joint_chain = joint + exactly the new classes; joint indices unchanged.
    assert mdc["num_node_features"] == mdj["num_node_features"] + 6
    assert mdc["num_edge_features"] == mdj["num_edge_features"] + 4
    for cls, i in mdj["edge_feature_to_index"].items():
        assert mdc["edge_feature_to_index"][cls] == i
    assert "chain" in mdc["lifted_spec"]
    assert "chain" not in mdj["lifted_spec"]
    assert set(mdc["lifted_spec"]["chain"]) == {"enables", "threatens"}


def test_joint_chain_widths_deterministic_across_renaming():
    md1 = build_lifted_metadata(blocks_action_space(), KP, KA, "joint_chain")
    md2 = build_lifted_metadata(renamed_blocks_action_space(), KP, KA,
                                "joint_chain")
    assert md1["num_node_features"] == md2["num_node_features"]
    assert md1["num_edge_features"] == md2["num_edge_features"]
    assert md1["node_feature_to_index"]["gdist_0"] == \
        md2["node_feature_to_index"]["gdist_0"]


def test_joint_chain_has_occurrence_nodes():
    md = build_lifted_metadata(blocks_action_space(), KP, KA, "joint_chain")
    keys = lifted_node_keys(md["lifted_spec"])
    assert any(k.startswith("LIFTED_OCC::") for k in keys)


def _build_chain_graph(action_space, on_name, holding_name):
    """Tiny 'joint_chain' state: objects a,b; on(a,b) true; goal holding(a)
    unsatisfied; one applicable grounding of the schema that adds holding."""
    md = build_lifted_metadata(action_space, KP, KA, "joint_chain")
    spec = md["lifted_spec"]
    keys = lifted_node_keys(spec)
    all_actions = list(action_space.keys())
    pred_objs = {l.predicate.name: l.predicate
                 for op in action_space.values() for l in op.preconds.literals}
    state_lit = MockLiteral(pred_objs[on_name], ["a", "b"])
    goal_lit = MockLiteral(pred_objs[holding_name], ["a"])
    goal_node = "WANT_GOAL_NODE"  # stands in for the wrapped WANT literal
    node_order = ["a", "b", state_lit, goal_node] + keys + all_actions
    o2n = {v: i for i, v in enumerate(node_order)}
    n = len(node_order)
    feats = np.zeros((n, md["num_node_features"]))
    edges = np.zeros((n, n, md["num_edge_features"]))
    add_lifted_node_features(feats, keys, spec, o2n,
                             md["node_feature_to_index"], KP)
    add_chain_node_features(feats, spec, all_actions, [state_lit], [goal_lit],
                            o2n, md["node_feature_to_index"])
    add_lifted_layer_edges(edges, [state_lit], all_actions, spec, o2n,
                           md["edge_feature_to_index"], KP, KA)
    achiever = next(a for a in all_actions
                    if any(o["role"] == "add" and o["pred"] == holding_name
                           for o in spec["schemas"][a.name]))
    grounding = MockLiteral(achiever, ["a", "b"])
    unsat_goal_nodes = {(holding_name, ("a",)): o2n[goal_node]}
    true_literal_nodes = {(on_name, ("a", "b")): o2n[state_lit]}
    add_grounded_effect_edges(edges, [grounding], action_space,
                              unsat_goal_nodes, true_literal_nodes, o2n,
                              md["edge_feature_to_index"])
    return md, spec, o2n, feats, edges, state_lit, goal_node, all_actions


def test_chain_gdist_and_goal_node_features():
    md, spec, o2n, feats, edges, lit, goal_node, actions = _build_chain_graph(
        blocks_action_space(), "on", "holding")
    N = md["node_feature_to_index"]
    unstack = next(a for a in actions if a.name == "unstack")  # adds holding
    stack = next(a for a in actions if a.name == "stack")      # enables it
    assert feats[o2n[unstack], N["gdist_0"]] == 1
    assert feats[o2n[unstack], N["adds_goal_pred"]] == 1
    assert feats[o2n[stack], N["gdist_1"]] == 1
    assert feats[o2n[stack], N["adds_goal_pred"]] == 0
    # holding(a) is an unsatisfied goal: a is goal-relevant, b is not.
    assert feats[o2n["a"], N["goal_relevant_obj"]] == 1
    assert feats[o2n["b"], N["goal_relevant_obj"]] == 0


def test_chain_edges_between_occurrence_nodes():
    md, spec, o2n, feats, edges, _, _, _ = _build_chain_graph(
        blocks_action_space(), "on", "holding")
    E = md["edge_feature_to_index"]
    # stack occ 2 = add on -> unstack occ 0 = pre on: enables (both ways).
    a, b = o2n[occ_node_key("stack", 2)], o2n[occ_node_key("unstack", 0)]
    assert edges[a, b, E["enables"]] == 1 and edges[b, a, E["enables"]] == 1
    # unstack occ 2 = del on -> unstack occ 0 = pre on: threatens.
    t = o2n[occ_node_key("unstack", 2)]
    assert edges[t, b, E["threatens"]] == 1
    assert edges[a, b, E["threatens"]] == 0


def test_chain_grounded_effect_edges():
    md, spec, o2n, feats, edges, lit, goal_node, _ = _build_chain_graph(
        blocks_action_space(), "on", "holding")
    E = md["edge_feature_to_index"]
    # unstack(a,b) adds holding(a) == unsatisfied goal: a <-> goal node.
    assert edges[o2n["a"], o2n[goal_node], E["achieves_goal"]] == 1
    assert edges[o2n[goal_node], o2n["a"], E["achieves_goal"]] == 1
    assert edges[o2n["b"], o2n[goal_node], E["achieves_goal"]] == 0
    # unstack(a,b) deletes on(a,b) == true literal: a,b <-> that literal.
    assert edges[o2n["a"], o2n[lit], E["deletes_true"]] == 1
    assert edges[o2n["b"], o2n[lit], E["deletes_true"]] == 1


def test_chain_satisfied_goal_atom_is_inert():
    """A goal atom already true yields no goal_relevant_obj bit (and the
    call site excludes it from the achieves_goal lookup)."""
    space = blocks_action_space()
    md = build_lifted_metadata(space, KP, KA, "joint_chain")
    spec = md["lifted_spec"]
    keys = lifted_node_keys(spec)
    all_actions = list(space.keys())
    on = next(l.predicate for op in space.values()
              for l in op.preconds.literals if l.predicate.name == "on")
    lit = MockLiteral(on, ["a", "b"])  # same object serves as state AND goal
    node_order = ["a", "b", lit] + keys + all_actions
    o2n = {v: i for i, v in enumerate(node_order)}
    feats = np.zeros((len(node_order), md["num_node_features"]))
    add_chain_node_features(feats, spec, all_actions, [lit], [lit], o2n,
                            md["node_feature_to_index"])
    N = md["node_feature_to_index"]
    assert feats[o2n["a"], N["goal_relevant_obj"]] == 0
    assert feats[o2n["b"], N["goal_relevant_obj"]] == 0


def test_chain_renaming_invariance_of_graph_tensors():
    md1, spec1, o2n1, feats1, edges1, _, _, _ = _build_chain_graph(
        blocks_action_space(), "on", "holding")
    md2, spec2, o2n2, feats2, edges2, _, _, _ = _build_chain_graph(
        renamed_blocks_action_space(), "zzz_rel", "mmm_grip")
    rows1 = sorted(tuple(r) for r in feats1)
    rows2 = sorted(tuple(r) for r in feats2)
    assert rows1 == rows2
    assert edges1.sum() == edges2.sum()
    E1, E2 = md1["edge_feature_to_index"], md2["edge_feature_to_index"]
    for cls in _CHAIN_EDGE_CLASSES:
        assert edges1[:, :, E1[cls]].sum() == edges2[:, :, E2[cls]].sum(), cls


# --- type compilation: typed domains == predicate-typed domains ---

class MockTypedParam(str):
    """A pddlgym-shaped `?x - block`: a name that also carries a var_type."""
    def __new__(cls, name, var_type):
        obj = str.__new__(cls, name)
        obj.var_type = var_type
        return obj


class MockObj(str):
    def __new__(cls, name, var_type):
        obj = str.__new__(cls, name)
        obj.var_type = var_type
        return obj


def _typed_space():
    """move(?r - robot, ?from - room, ?to - room): types declared in PDDL."""
    at = MockPred("at", 2)
    move = MockOp(
        params=[MockTypedParam("?r", "robot"), MockTypedParam("?from", "room"),
                MockTypedParam("?to", "room")],
        preconds=[MockLiteral(at, ["?r", "?from"])],
        effects=[MockLiteral(at, ["?r", "?to"]),
                 MockLiteral(at, ["?r", "?from"], is_anti=True)])
    return {MockPred("move", 3): move}


def _predicate_typed_space():
    """The same domain written untyped, with type predicates spelled out.

    This is the form grid/logistics use and the only form the pre-compilation
    featurization could represent.
    """
    at = MockPred("at", 2)
    robot = MockPred("robot", 1)
    room = MockPred("room", 1)
    move = MockOp(
        params=["?r", "?from", "?to"],
        preconds=[MockLiteral(at, ["?r", "?from"]), MockLiteral(robot, ["?r"]),
                  MockLiteral(room, ["?from"]), MockLiteral(room, ["?to"])],
        effects=[MockLiteral(at, ["?r", "?to"]),
                 MockLiteral(at, ["?r", "?from"], is_anti=True)])
    return {MockPred("move", 3): move}


def test_declared_types_become_unary_static_predicates():
    spec = build_lifted_spec(_typed_space())
    assert spec["type_preds"] == {"robot": type_pred_name("robot"),
                                  "room": type_pred_name("room")}
    for pred_name in spec["type_preds"].values():
        assert spec["predicates"][pred_name] == {"arity": 1, "static": True}


def test_type_occurrences_do_not_shift_real_precondition_indices():
    """The binding layer maps the k-th precondition literal to occurrence k;
    type occurrences must come after every real precondition."""
    spec = build_lifted_spec(_typed_space())
    occs = spec["schemas"]["move"]
    assert occs[0]["role"] == "pre" and occs[0]["pred"] == "at"
    for slot, occ_index in spec["type_occ"]["move"].items():
        assert occ_index > 0
        assert occs[occ_index]["role"] == "pre"
        assert occs[occ_index]["bindings"] == [(0, slot)]
    # every typed slot has exactly one occurrence
    assert set(spec["type_occ"]["move"]) == {0, 1, 2}


def test_typed_and_predicate_typed_domains_agree_structurally():
    """The point of compilation: both spellings of the same domain give the
    same multiset of (role, arity, static, bindings) occurrences."""
    def signature(spec):
        preds = spec["predicates"]
        return sorted(
            (o["role"], preds[o["pred"]]["arity"], preds[o["pred"]]["static"],
             tuple(o["bindings"]))
            for occs in spec["schemas"].values() for o in occs)
    assert signature(build_lifted_spec(_typed_space())) == \
        signature(build_lifted_spec(_predicate_typed_space()))


def test_untyped_domain_compiles_nothing():
    spec = build_lifted_spec(_predicate_typed_space())
    assert spec["type_preds"] == {} and spec["type_occ"] == {}
    # 'default' is pddlgym's pseudo-type for untyped entities: not a type.
    spec2 = build_lifted_spec(
        {MockPred("a", 1): MockOp(params=[MockTypedParam("?x", "default")],
                                  preconds=[], effects=[])})
    assert spec2["type_preds"] == {}


def test_has_type_edge_links_object_to_its_type_symbol():
    md = build_lifted_metadata(_typed_space(), KP, KA, "joint_chain")
    spec = md["lifted_spec"]
    keys = lifted_node_keys(spec)
    objs = [MockObj("r1", "robot"), MockObj("rm1", "room"),
            MockObj("x", "untyped_thing")]
    node_order = list(objs) + keys + list(_typed_space())
    o2n = {v: i for i, v in enumerate(node_order)}
    edges = np.zeros((len(node_order), len(node_order), md["num_edge_features"]))
    add_type_edges(edges, objs, spec, o2n, md["edge_feature_to_index"])
    E = md["edge_feature_to_index"]["has_type"]
    robot_node = o2n[pred_node_key(type_pred_name("robot"))]
    room_node = o2n[pred_node_key(type_pred_name("room"))]
    assert edges[o2n["r1"], robot_node, E] == 1
    assert edges[robot_node, o2n["r1"], E] == 1
    assert edges[o2n["rm1"], room_node, E] == 1
    assert edges[o2n["rm1"], robot_node, E] == 0
    # a type no schema parameter declares constrains nothing -> no node, no edge
    assert edges[o2n["x"], :, E].sum() == 0


def test_has_type_edge_also_fires_for_predicate_typed_domains():
    """A true unary static literal is a hand-written type declaration; it must
    produce the same edge class the compiled types do, or a model trained on
    one family meets an unfamiliar edge on the other."""
    space = _predicate_typed_space()
    md = build_lifted_metadata(space, KP, KA, "joint_chain")
    spec = md["lifted_spec"]
    assert spec["type_preds"] == {}          # nothing to compile here
    assert spec["predicates"]["room"] == {"arity": 1, "static": True}
    keys = lifted_node_keys(spec)
    robot_p = MockPred("robot", 1)
    at = MockPred("at", 2)
    objs = ["r1", "rm1"]
    state = [MockLiteral(robot_p, ["r1"]), MockLiteral(at, ["r1", "rm1"])]
    node_order = objs + state + keys + list(space)
    o2n = {v: i for i, v in enumerate(node_order)}
    edges = np.zeros((len(node_order), len(node_order), md["num_edge_features"]))
    add_type_edges(edges, objs, spec, o2n, md["edge_feature_to_index"],
                   state_literals=state)
    E = md["edge_feature_to_index"]["has_type"]
    assert edges[o2n["r1"], o2n[pred_node_key("robot")], E] == 1
    # `at` is binary and dynamic: not a type declaration
    assert edges[o2n["r1"], o2n[pred_node_key("at")], E] == 0
    assert edges[o2n["rm1"], :, E].sum() == 0   # no room(rm1) literal in state


def test_type_compilation_is_renaming_invariant():
    """Renaming the types must not change the tensors -- types are symbols
    too, and C1 depends on none of them reaching the features."""
    def renamed():
        at = MockPred("at", 2)
        move = MockOp(
            params=[MockTypedParam("?r", "zzz"), MockTypedParam("?from", "aaa"),
                    MockTypedParam("?to", "aaa")],
            preconds=[MockLiteral(at, ["?r", "?from"])],
            effects=[MockLiteral(at, ["?r", "?to"]),
                     MockLiteral(at, ["?r", "?from"], is_anti=True)])
        return {MockPred("move", 3): move}

    def tensors(space, type_names):
        md = build_lifted_metadata(space, KP, KA, "joint_chain")
        spec = md["lifted_spec"]
        keys = lifted_node_keys(spec)
        objs = [MockObj("o1", type_names[0]), MockObj("o2", type_names[1])]
        node_order = list(objs) + keys + list(space)
        o2n = {v: i for i, v in enumerate(node_order)}
        n = len(node_order)
        feats = np.zeros((n, md["num_node_features"]))
        edges = np.zeros((n, n, md["num_edge_features"]))
        add_lifted_node_features(feats, keys, spec, o2n,
                                 md["node_feature_to_index"], KP)
        add_lifted_layer_edges(edges, [], list(space), spec, o2n,
                               md["edge_feature_to_index"], KP, KA)
        add_type_edges(edges, objs, spec, o2n, md["edge_feature_to_index"])
        return md, feats, edges

    md1, f1, e1 = tensors(_typed_space(), ["robot", "room"])
    md2, f2, e2 = tensors(renamed(), ["zzz", "aaa"])
    assert md1["num_node_features"] == md2["num_node_features"]
    assert md1["num_edge_features"] == md2["num_edge_features"]
    assert sorted(tuple(r) for r in f1) == sorted(tuple(r) for r in f2)
    assert e1.sum() == e2.sum()
    E = md1["edge_feature_to_index"]["has_type"]
    assert e1[:, :, E].sum() == e2[:, :, E].sum() == 4  # 2 objects, both ways


def test_type_compilation_does_not_disturb_chaining():
    """Nothing adds or deletes a type predicate, so enables/threatens are
    unchanged by compilation."""
    typed = build_lifted_spec(_typed_space())
    chain = build_chain_spec(typed)
    for (sa, occ_a, sb, occ_b) in chain["enables"] + chain["threatens"]:
        assert "type_slot" not in typed["schemas"][sa][occ_a]
        assert "type_slot" not in typed["schemas"][sb][occ_b]


# ─── component ablations (C2) ───────────────────────────────────────────────

def test_ablation_modes_share_joint_chain_widths():
    """Same network input for the whole study: only graph content differs."""
    space = blocks_action_space()
    full = build_lifted_metadata(space, KP, KA, "joint_chain")
    for mode in ABLATION_MODES:
        md = build_lifted_metadata(space, KP, KA, mode)
        assert md["num_node_features"] == full["num_node_features"], mode
        assert md["num_edge_features"] == full["num_edge_features"], mode
        assert md["node_feature_to_index"] == full["node_feature_to_index"], mode
        assert md["edge_feature_to_index"] == full["edge_feature_to_index"], mode
        assert md["featurization"] == mode
        assert md["lifted_spec"]["mode"] == mode


def test_ablation_components_are_leave_one_out_and_add_one():
    full = mode_components("joint_chain")
    assert full == {"bind", "chain", "eff", "goal"}
    for c in full:
        assert mode_components(f"jc_no_{c}") == full - {c}
    for c in ("chain", "eff", "goal"):
        assert mode_components(f"jc_only_{c}") == {"bind", c}
    assert mode_components("joint") == {"bind"}
    assert mode_components("joint_lite") == set()


def test_existing_modes_keep_their_components_and_legacy_specs_resolve():
    space = blocks_action_space()
    for mode in ("joint_lite", "joint", "joint_chain"):
        spec = build_lifted_metadata(space, KP, KA, mode)["lifted_spec"]
        assert spec_components(spec) == mode_components(mode)
        # A spec pickled before 'components' existed carries only its mode.
        legacy = {k: v for k, v in spec.items() if k != "components"}
        assert spec_components(legacy) == mode_components(mode)
    # joint/joint_lite keep the narrow widths: no chain classes appended.
    joint = build_lifted_metadata(space, KP, KA, "joint")
    assert "enables" not in joint["edge_feature_to_index"]
    assert "chain" not in joint["lifted_spec"]


def test_ablations_have_occurrence_nodes_lite_does_not():
    space = blocks_action_space()
    for mode in ("joint", "joint_chain") + ABLATION_MODES:
        spec = build_lifted_metadata(space, KP, KA, mode)["lifted_spec"]
        assert has_occurrence_nodes(spec), mode
        assert any(k.startswith("LIFTED_OCC::") for k in lifted_node_keys(spec))
    lite = build_lifted_metadata(space, KP, KA, "joint_lite")["lifted_spec"]
    assert not has_occurrence_nodes(lite)


def _lifted_edges(mode):
    space = blocks_action_space()
    md = build_lifted_metadata(space, KP, KA, mode)
    spec = md["lifted_spec"]
    keys = lifted_node_keys(spec)
    all_actions = list(space.keys())
    node_order = keys + all_actions
    o2n = {v: i for i, v in enumerate(node_order)}
    n = len(node_order)
    edges = np.zeros((n, n, md["num_edge_features"]))
    add_lifted_layer_edges(edges, [], all_actions, spec, o2n,
                           md["edge_feature_to_index"], KP, KA)
    return md, edges


def test_chain_edges_follow_the_chain_component():
    """enables/threatens are present exactly where 'chain' is on, and every
    other lifted-layer edge is untouched by the ablation."""
    full_md, full = _lifted_edges("joint_chain")
    E = full_md["edge_feature_to_index"]
    chain_cols = [E["enables"], E["threatens"]]
    other = [i for i in range(full.shape[2]) if i not in chain_cols]
    assert full[:, :, chain_cols].sum() > 0
    for mode in ABLATION_MODES:
        _, edges = _lifted_edges(mode)
        if "chain" in mode_components(mode):
            assert (edges[:, :, chain_cols] == full[:, :, chain_cols]).all(), mode
        else:
            assert edges[:, :, chain_cols].sum() == 0, mode
        assert (edges[:, :, other] == full[:, :, other]).all(), mode


def test_goal_distances_survive_removing_chain_edges():
    """'goal' buckets are a BFS over the enables RELATION, which is kept in
    the spec even when its EDGES are ablated."""
    space = blocks_action_space()
    spec = build_lifted_metadata(space, KP, KA, "jc_no_chain")["lifted_spec"]
    assert spec["chain"]["enables"]
    dist = schema_goal_distances(spec, spec["chain"], {"holding"})
    assert dist["unstack"] == 0 and dist["stack"] == 1


# ─── single-effect schemas (pddlgym leaves the effect a bare Literal) ───────

def _bare_effect_space():
    """pick(?x): pre clear(?x), ONE effect holding(?x) -> a bare Literal, as in
    Miconic board / Rovers calibrate. drop(?x): pre holding(?x)."""
    clear, holding = MockPred("clear", 1), MockPred("holding", 1)
    pick = MockOp(["?x"], [MockLiteral(clear, ["?x"])], [])
    pick.effects = MockLiteral(holding, ["?x"])         # bare: no .literals
    drop = MockOp(["?x"], [MockLiteral(holding, ["?x"])],
                  [MockLiteral(clear, ["?x"])])
    return {MockPred("pick", 1): pick, MockPred("drop", 1): drop}, holding


def test_op_literals_handles_conjunction_bare_literal_and_none():
    lit = MockLiteral(MockPred("p", 1), ["?x"])
    assert op_literals(None) == []
    assert op_literals(MockConj([lit, lit])) == [lit, lit]
    assert op_literals(lit) == [lit]
    assert not hasattr(lit, "literals")


def test_has_bare_effects_detects_only_unwrapped_effects():
    space, _ = _bare_effect_space()
    assert has_bare_effects(space)
    assert not has_bare_effects(blocks_action_space())


def test_single_effect_schema_keeps_its_add_occurrence_and_chain_edge():
    """Before the fix pick had NO effect occurrence, so no add(holding) and no
    enables edge pick -> drop: the board -> depart structure of the paper."""
    space, _ = _bare_effect_space()
    spec = build_lifted_spec(space)
    assert [o["role"] for o in spec["schemas"]["pick"]] == ["pre", "add"]
    assert spec["schemas"]["pick"][1]["pred"] == "holding"
    chain = build_chain_spec(spec)
    assert ("pick", 1, "drop", 0) in chain["enables"]
    assert schema_goal_distances(spec, chain, {"holding"})["pick"] == 0


def test_single_effect_schema_gets_its_achieves_goal_edge():
    space, holding = _bare_effect_space()
    md = build_lifted_metadata(space, KP, KA, "joint_chain")
    E = md["edge_feature_to_index"]
    pick_pred = next(p for p in space if p.name == "pick")
    grounding = MockLiteral(pick_pred, ["a"])
    o2n = {"a": 0, "GOAL": 1}
    edges = np.zeros((2, 2, md["num_edge_features"]))
    add_grounded_effect_edges(edges, [grounding], space, {("holding", ("a",)): 1},
                              {}, o2n, E)
    assert edges[0, 1, E["achieves_goal"]] == 1 and edges[1, 0, E["achieves_goal"]] == 1


# ─── negative preconditions ─────────────────────────────────────────────────

def _negative_precondition_space():
    """light(?x):  pre at(?x), NOT lit(?x);  add lit(?x)
    douse(?x):  pre lit(?x);                 del lit(?x)
    go(?x ?y):  pre at(?x);                  add at(?y), del at(?x)
    Lighting x makes light(x) inapplicable; dousing x makes it applicable."""
    lit, at = MockPred("lit", 1), MockPred("at", 1)
    light = MockOp(["?x"], [MockLiteral(at, ["?x"]),
                            MockLiteral(lit, ["?x"], is_negative=True)],
                   [MockLiteral(lit, ["?x"])])
    douse = MockOp(["?x"], [MockLiteral(lit, ["?x"])],
                   [MockLiteral(lit, ["?x"], is_anti=True)])
    go = MockOp(["?x", "?y"], [MockLiteral(at, ["?x"])],
                [MockLiteral(at, ["?y"]), MockLiteral(at, ["?x"], is_anti=True)])
    return {MockPred("light", 1): light, MockPred("douse", 1): douse,
            MockPred("go", 2): go}


def test_negative_precondition_keeps_its_sign_and_its_position():
    spec = build_lifted_spec(_negative_precondition_space())
    assert [(o["role"], o["pred"]) for o in spec["schemas"]["light"]] == [
        ("pre", "at"), ("npre", "lit"), ("add", "lit")]
    assert spec["schemas"]["light"][1]["bindings"] == [(0, 0)]


def test_chain_edges_through_a_negative_precondition_are_swapped():
    """Before the fix: light.add:lit ENABLED light (it disables it) and
    douse.del:lit THREATENED light (it enables it)."""
    spec = build_lifted_spec(_negative_precondition_space())
    chain = build_chain_spec(spec)
    assert ("light", 2, "light", 1) in chain["threatens"]
    assert ("light", 2, "light", 1) not in chain["enables"]
    assert ("douse", 1, "light", 1) in chain["enables"]
    assert ("douse", 1, "light", 1) not in chain["threatens"]
    # positive preconditions are unchanged
    assert ("light", 2, "douse", 0) in chain["enables"]
    assert ("douse", 1, "douse", 0) in chain["threatens"]


def test_goal_distance_follows_the_corrected_enables_relation():
    """Goal: at. go adds it (0). Nothing enables go through `lit`, so light
    and douse stay unconnected (3) - light was distance... still 3, but douse
    reaches a goal on `lit` only THROUGH enabling light when the goal is lit."""
    spec = build_lifted_spec(_negative_precondition_space())
    chain = build_chain_spec(spec)
    dist = schema_goal_distances(spec, chain, {"lit"})
    assert dist["light"] == 0
    assert dist["douse"] == 1          # douse deletes lit -> light applicable
    assert dist["go"] == 1             # go adds at -> light applicable


def test_negative_precondition_occurrence_is_pre_plus_del_at_unchanged_widths():
    space = _negative_precondition_space()
    md = build_lifted_metadata(space, KP, KA, "joint_chain")
    plain = build_lifted_metadata(blocks_action_space(), KP, KA, "joint_chain")
    assert md["num_node_features"] == plain["num_node_features"]
    assert md["num_edge_features"] == plain["num_edge_features"]
    spec, N, E = md["lifted_spec"], md["node_feature_to_index"], md["edge_feature_to_index"]
    keys = lifted_node_keys(spec)
    actions = list(space)
    o2n = {v: i for i, v in enumerate(keys + actions)}
    n = len(o2n)
    feats = np.zeros((n, md["num_node_features"]))
    edges = np.zeros((n, n, md["num_edge_features"]))
    add_lifted_node_features(feats, keys, spec, o2n, N, KP)
    add_lifted_layer_edges(edges, [], actions, spec, o2n, E, KP, KA)
    light = o2n[next(a for a in actions if a.name == "light")]
    occ_neg, occ_pos = o2n[occ_node_key("light", 1)], o2n[occ_node_key("light", 0)]
    roles = lambda i: [int(feats[i, N[f"occ_node_{r}"]]) for r in ("pre", "add", "del")]
    assert roles(occ_neg) == [1, 0, 1] and roles(occ_pos) == [1, 0, 0]
    eroles = lambda a, b: [int(edges[a, b, E[f"role_{r}"]]) for r in ("pre", "add", "del")]
    assert eroles(occ_neg, light) == [1, 0, 1] and eroles(occ_pos, light) == [1, 0, 0]
    # predicate<->schema is the union over occurrences: the negative
    # precondition gives it positions but no role, so `lit`<->light says only
    # what is true of it there - light ADDS lit.
    lit_sym = o2n[pred_node_key("lit")]
    assert eroles(lit_sym, light) == [0, 1, 0]
    assert edges[lit_sym, light, E["slot_0"]] == 1 and edges[lit_sym, light, E["parg_0"]] == 1
    # chain edges land on the right occurrence nodes
    assert edges[o2n[occ_node_key("light", 2)], occ_neg, E["threatens"]] == 1
    assert edges[o2n[occ_node_key("douse", 1)], occ_neg, E["enables"]] == 1
    assert edges[o2n[occ_node_key("light", 2)], occ_neg, E["enables"]] == 0


def test_a_domain_without_negative_preconditions_has_the_same_spec_as_before():
    """The chain lists are ordered, and the order is part of byte-identity."""
    spec = build_lifted_spec(blocks_action_space())
    assert all(o["role"] != "npre" for occs in spec["schemas"].values() for o in occs)
    chain = build_chain_spec(spec)
    adds, pres, dels = {}, {}, {}
    for schema, occs in spec["schemas"].items():
        for k, occ in enumerate(occs):
            {"add": adds, "pre": pres, "del": dels}[occ["role"]].setdefault(
                occ["pred"], []).append((schema, k))
    legacy = lambda eff: [(a, i, b, j) for p, src in eff.items() for a, i in src
                          for b, j in pres.get(p, [])]
    assert chain["enables"] == legacy(adds) and chain["threatens"] == legacy(dels)


def test_featurizer_version_marks_only_the_domains_a_fix_changed():
    bare, _ = _bare_effect_space()
    neg = _negative_precondition_space()
    assert featurizer_version(blocks_action_space()) == 0
    assert featurizer_version(bare) == 1
    assert has_negative_preconditions(neg) and featurizer_version(neg) == 2
    assert featurizer_version(neg, lifted=False) == 0   # structural: unchanged
    both = dict(bare); both.update(neg)
    assert featurizer_version(both) == 3


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

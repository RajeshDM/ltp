"""Structural (symbol-free) featurization (CLAUDE.md 5.3, Phase 2; C1/C2).

Symbols alias to structural-class feature indices: predicates are classed by
(arity, static-ness), goal predicates likewise, action schemas by arity, all
object types share one class. Two domains that are isomorphic up to renaming
get identical feature tensors by construction. Feature widths depend only on
the canonical max arities, so graphs from different domains batch together
and a held-out domain featurizes with its own metadata at the same widths.
"""

import logging

logger = logging.getLogger(__name__)


class StructuralMap(dict):
    """symbol -> structural class index; classes unseen symbols on the fly.

    Known symbols are aliased at construction. Unseen ones (held-out domain
    predicates not mentioned in its schemas, arities above the canonical max)
    are classed here: arity is clamped (CLAUDE.md 5.7), predicates outside all
    effects default to static.
    """

    def __init__(self, index_map, class_of=None):
        super().__init__(index_map)
        self._class_of = class_of

    def __missing__(self, key):
        if self._class_of is None:
            raise KeyError(key)
        index = self[self._class_of(key)]
        self[key] = index
        return index

    def __reduce__(self):
        # The class_of closure is not picklable (metadata gets stored in the
        # unified cache). A reloaded map behaves as a plain dict - fully
        # populated for its domain - losing only the unseen-symbol fallback,
        # which test-time metadata (always built fresh) retains.
        return (StructuralMap, (dict(self), None))


def _pred_arity(predicate):
    return getattr(predicate, "arity", len(getattr(predicate, "var_types", [])))


def build_structural_metadata(action_space, max_pred_arity, max_action_arity,
                              goal_prefix="WANT"):
    """Build symbol-free graph metadata for one domain from its action space.

    Same signature dicts as _create_graph_structure_ltp, so the graph code is
    unchanged; widths are fixed by (max_pred_arity, max_action_arity) alone.
    """
    kp, ka = max_pred_arity, max_action_arity

    # Structural classes, canonical order (shared across all domains).
    node_classes = ["action_node", "object_node", "predicate_node", "goal_pred",
                    "typed_object"]
    node_classes += [f"act_a{k}" for k in range(ka + 1)]
    node_classes += [f"pred_a{k}_s{s}" for k in range(kp + 1) for s in (0, 1)]
    node_classes += [f"gpred_a{k}_s{s}" for k in range(kp + 1) for s in (0, 1)]
    edge_classes = ["action_object"]
    edge_classes += [f"pos_{k}" for k in range(ka)]
    edge_classes += [f"pred_pos_{k}" for k in range(kp)]
    edge_classes += [f"precond_pos_{k}" for k in range(ka + 1)]

    node_map = {c: i for i, c in enumerate(node_classes)}
    edge_map = {c: i for i, c in enumerate(edge_classes)}

    # Domain inventory from the schemas: predicates, static-ness, types.
    predicates, effect_preds, types = {}, set(), set()
    for schema, op in action_space.items():
        for lit in getattr(op.preconds, "literals", []):
            predicates[lit.predicate.name] = lit.predicate
        for lit in getattr(getattr(op, "effects", None), "literals", []):
            predicates[lit.predicate.name] = lit.predicate
            effect_preds.add(lit.predicate.name)
        for param in getattr(op, "params", []):
            types.add(getattr(param, "var_type", None))

    def pred_class(pred, goal=False):
        arity = min(_pred_arity(pred), kp)
        static = int(pred.name.replace(goal_prefix, "", 1) not in effect_preds
                     if goal else pred.name not in effect_preds)
        return f"{'gpred' if goal else 'pred'}_a{arity}_s{static}"

    def node_class(symbol):
        # fallback classing for symbols first seen at featurization time
        if symbol in action_space:
            return f"act_a{min(len(getattr(action_space[symbol], 'params', [])), ka)}"
        if hasattr(symbol, "arity") or hasattr(symbol, "var_types"):
            goal = str(getattr(symbol, "name", symbol)).startswith(goal_prefix)
            return pred_class(symbol, goal=goal)
        return "typed_object"  # object types

    def edge_class(key):
        # 'pos_7' / 'pred_pos_7' above canonical max -> clamp; unseen
        # precondition-string keys ('<pred>3') -> positional satisfaction class
        key = str(key)
        if key.startswith("pos_"):
            return f"pos_{min(int(key[4:]), ka - 1)}"
        if key.startswith("pred_pos_"):
            return f"pred_pos_{min(int(key[9:]), kp - 1)}"
        return f"precond_pos_{min(int(key[-1]), ka)}" if key[-1].isdigit() else "action_object"

    # Alias this domain's known symbols to their classes.
    for schema, op in action_space.items():
        node_map[schema] = node_map[node_class(schema)]
        for pos, precond in enumerate(getattr(op.preconds, "literals", [])):
            for curr_pos in range(ka + 2):
                key = str(precond.predicate) + str(curr_pos)
                edge_map[key] = edge_map[f"precond_pos_{min(curr_pos, ka)}"]
    for pred in predicates.values():
        node_map[pred] = node_map[pred_class(pred)]
    for t in types:
        if t is not None:
            node_map[t] = node_map["typed_object"]

    all_predicates = sorted(predicates.values(), key=lambda p: p.name)
    return {
        "num_node_features": len(node_classes),
        "num_edge_features": len(edge_classes),
        "node_feature_to_index": StructuralMap(node_map, node_class),
        "edge_feature_to_index": StructuralMap(edge_map, edge_class),
        "all_predicates": all_predicates,
        "unary_types": sorted(t for t in types if t is not None),
        "unary_predicates": [p for p in all_predicates if _pred_arity(p) == 1],
        "binary_predicates": [p for p in all_predicates if _pred_arity(p) == 2],
        "max_pred_arity": kp,
        "max_action_arity": ka,
        "featurization": "structural",
    }

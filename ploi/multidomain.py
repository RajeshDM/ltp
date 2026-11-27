"""Multi-domain harness (CLAUDE.md Phase 0/1, claim C1)."""

import logging

logger = logging.getLogger(__name__)


def parse_domain_arg(domains_arg, heldout_arg="", num_train_problems=50):
    """Parse 'blocks:100,gripper' style args -> [(name, num_problems, held_out)]."""
    domains = []
    for arg, held_out in ((domains_arg, False), (heldout_arg, True)):
        for token in [t.strip() for t in arg.split(",") if t.strip()]:
            name, _, count = token.partition(":")
            count = int(count) if count else num_train_problems
            domains.append((name.capitalize(), count, held_out))
    names = [d[0] for d in domains]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate domains: {names}")
    return domains


def merge_feature_metadata(metadatas):
    """Union-vocab metadata across domains (Baseline 0, CLAUDE.md 5.2).

    Symbols keep first-domain relative order (deterministic, cache-stable);
    feature widths follow the merged dictionaries.
    """
    if not metadatas:
        raise ValueError("No metadata to merge")
    if len(metadatas) == 1:
        return dict(metadatas[0])

    merged = {}
    for map_key in ("node_feature_to_index", "edge_feature_to_index"):
        union_map = {}
        for md in metadatas:
            for symbol, _ in sorted(md.get(map_key, {}).items(), key=lambda kv: kv[1]):
                if symbol not in union_map:
                    union_map[symbol] = len(union_map)
        merged[map_key] = union_map

    for list_key in ("unary_types", "unary_predicates", "binary_predicates", "all_predicates"):
        seen, union_symbols = set(), []
        for md in metadatas:
            for symbol in md.get(list_key, []):
                if symbol not in seen:
                    seen.add(symbol)
                    union_symbols.append(symbol)
        merged[list_key] = union_symbols

    merged["num_node_features"] = len(merged["node_feature_to_index"])
    merged["num_edge_features"] = len(merged["edge_feature_to_index"])

    for md in metadatas:
        for key, value in md.items():
            if key in merged:
                continue
            if isinstance(value, int):
                merged[key] = max(value, *(m.get(key, value) for m in metadatas))
            else:
                merged[key] = value
    return merged


def merge_action_spaces(action_spaces):
    """Union of per-domain action spaces, used ONLY to size the model
    (schema count + per-schema/max arity). Keys are not semantic here - each
    per-state graph is built from its own domain's action space; the merged
    space just tells the decoder how many schemas and what arities to expect.

    Different domains legitimately reuse a schema NAME (e.g. 'move') with
    different preconditions/effects - that is the cross-domain aliasing the
    model must learn to handle, NOT an error. Colliding names from different
    domains are kept as DISTINCT entries under a namespaced key, so the schema
    count and arities stay correct instead of silently collapsing.
    """
    merged = {}
    for d_idx, space in enumerate(action_spaces):
        for schema, operator in space.items():
            if schema in merged and merged[schema] is not operator:
                merged[(d_idx, schema)] = operator  # same name, other domain
            else:
                merged[schema] = operator
    return merged

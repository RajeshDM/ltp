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
    """Union of per-domain action spaces; loud on cross-domain schema collisions."""
    merged = {}
    for space in action_spaces:
        for schema, operator in space.items():
            if schema in merged and merged[schema] is not operator:
                raise ValueError(f"Action schema collision across domains: {schema}")
            merged[schema] = operator
    return merged

"""Multi-domain training harness (CLAUDE.md Phase 0; serves C1 as infrastructure).

This module generalizes the per-domain GABAR data pipeline to N domains.
It deliberately does NOT change graph construction or the model: Phase 0's
gate is exact parity with single-domain GABAR when run with one domain.

Design (see CLAUDE.md §5.2, §6):
- Phase 0 ('per_domain' featurization): loop domains, reuse the existing
  `process_pddl_to_graphs` per domain, keep datasets separate. Used to prove
  the harness reproduces published GABAR numbers.
- Phase 1 ('union' featurization): build one union vocabulary over all
  training domains (union of node/edge feature dictionaries), then featurize
  every domain with that shared metadata so feature widths agree and
  mixed-domain batches are possible. This is Baseline 0, the D-in-weights
  control for claim C1.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Feature-dictionary keys that are index maps (symbol -> feature position).
_INDEX_MAP_KEYS = ("node_feature_to_index", "edge_feature_to_index")
# Metadata keys whose values are symbol collections; union = set-union.
_SYMBOL_LIST_KEYS = (
    "unary_types",
    "unary_predicates",
    "binary_predicates",
    "all_predicates",
)


@dataclass
class DomainSpec:
    """One domain in a multi-domain experiment."""

    name: str  # pddlgym env name, e.g. "Manyblocks_ipcc_big"
    num_train_problems: int = 50
    num_test_problems: int = 10
    held_out: bool = False  # True => never in training batches (zero-shot eval)


@dataclass
class DomainData:
    """Everything the trainer needs about one collected domain."""

    spec: DomainSpec
    graphs: List[Any]  # PyG HeteroData list from process_pddl_to_graphs
    graph_metadata: Dict[str, Any]
    action_space: Any


@dataclass
class MultiDomainConfig:
    """Configuration for a multi-domain run.

    featurization:
      'per_domain' — Phase 0. Each domain keeps its own feature dictionaries
                     (exact GABAR behavior; parity gate).
      'union'      — Phase 1. One shared union vocabulary across all training
                     domains (Baseline 0 / union-vocab control for C1).
    """

    domains: List[DomainSpec] = field(default_factory=list)
    featurization: str = "per_domain"

    @property
    def train_domains(self) -> List[DomainSpec]:
        return [d for d in self.domains if not d.held_out]

    @property
    def held_out_domains(self) -> List[DomainSpec]:
        return [d for d in self.domains if d.held_out]


def parse_domain_arg(domains_arg: str, heldout_arg: str = "",
                     num_train_problems: int = 50,
                     num_test_problems: int = 10) -> List[DomainSpec]:
    """Parse '--domains blocks,gripper --heldout-domains spanner' style args.

    Each entry may carry a per-domain problem count: 'blocks:100'.
    """
    specs: List[DomainSpec] = []

    def _parse_list(arg: str, held_out: bool) -> None:
        for token in [t.strip() for t in arg.split(",") if t.strip()]:
            if ":" in token:
                name, count = token.split(":", 1)
                specs.append(DomainSpec(name=name.capitalize(),
                                        num_train_problems=int(count),
                                        num_test_problems=num_test_problems,
                                        held_out=held_out))
            else:
                specs.append(DomainSpec(name=token.capitalize(),
                                        num_train_problems=num_train_problems,
                                        num_test_problems=num_test_problems,
                                        held_out=held_out))

    _parse_list(domains_arg, held_out=False)
    _parse_list(heldout_arg, held_out=True)

    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate domain names in config: {names}")
    return specs


def collect_multidomain_data(config: MultiDomainConfig, planner, args,
                             create_graph_dataset_func) -> Dict[str, DomainData]:
    """Phase 0 collection: run the existing per-domain pipeline for each domain.

    Reuses process_pddl_to_graphs unchanged (including its unified cache, which
    is already keyed by domain name), so single-domain behavior is bit-for-bit
    the published GABAR pipeline — that is the parity gate.
    """
    # Imported here so this module stays importable without torch/pddlgym
    # (the metadata-merging logic below is dependency-free and unit-tested).
    from ploi.datautils_ltp import process_pddl_to_graphs

    domain_data: Dict[str, DomainData] = {}
    for spec in config.domains:
        logger.info("Collecting domain %s (%d train problems, held_out=%s)",
                    spec.name, spec.num_train_problems, spec.held_out)
        graphs, metadata, action_space = process_pddl_to_graphs(
            spec.name, planner, spec.num_train_problems, args,
            create_graph_dataset_func,
        )
        domain_data[spec.name] = DomainData(
            spec=spec, graphs=graphs, graph_metadata=metadata,
            action_space=action_space,
        )
    return domain_data


def merge_feature_metadata(metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build union-vocabulary metadata from per-domain metadata dicts (§5.2).

    The union keeps every distinct symbol key across domains and assigns it a
    single feature index, so all domains featurize to the same widths. Symbols
    are ordered: first by the order of domains given, then by their index in
    that domain's dictionary — deterministic, so cached datasets stay valid.

    Integer-valued metadata fields (e.g. num_node_features) are recomputed
    from the merged dictionaries where possible, otherwise take the max.

    Pure dict logic — no torch/pddlgym dependency (unit-tested in
    tests/test_multidomain_metadata.py).
    """
    if not metadatas:
        raise ValueError("No metadata to merge")
    if len(metadatas) == 1:
        return dict(metadatas[0])

    merged: Dict[str, Any] = {}

    # 1) Union the symbol->index maps, reindexing densely and deterministically.
    for map_key in _INDEX_MAP_KEYS:
        union_map: Dict[Any, int] = {}
        for md in metadatas:
            index_map = md.get(map_key, {})
            for symbol, _ in sorted(index_map.items(), key=lambda kv: kv[1]):
                if symbol not in union_map:
                    union_map[symbol] = len(union_map)
        merged[map_key] = union_map

    # 2) Union the symbol collections (kept as sorted lists for determinism).
    for list_key in _SYMBOL_LIST_KEYS:
        union_symbols = []
        seen = set()
        for md in metadatas:
            for symbol in md.get(list_key, []):
                if symbol not in seen:
                    seen.add(symbol)
                    union_symbols.append(symbol)
        merged[list_key] = union_symbols

    # 3) Feature widths follow the merged dictionaries.
    merged["num_node_features"] = len(merged["node_feature_to_index"])
    merged["num_edge_features"] = len(merged["edge_feature_to_index"])

    # 4) Any remaining int fields (e.g. num_global_features): take the max.
    for md in metadatas:
        for key, value in md.items():
            if key in merged:
                continue
            if isinstance(value, int):
                merged[key] = max(value, *(m.get(key, value) for m in metadatas))
            else:
                merged[key] = value

    return merged


def merge_action_spaces(action_spaces: List[Any]) -> Dict[Any, Any]:
    """Union of per-domain action spaces (schema -> operator maps).

    Schemas are namespaced implicitly by their pddlgym predicate identity;
    collisions (same schema name in two domains) are allowed only if they
    map to the same operator object, otherwise we fail loudly — silent
    collisions would corrupt the union-vocab control (C1).
    """
    merged: Dict[Any, Any] = {}
    for space in action_spaces:
        for schema, operator in space.items():
            if schema in merged and merged[schema] is not operator:
                raise ValueError(
                    f"Action schema collision across domains: {schema}. "
                    "Namespace the domains or rename the schema."
                )
            merged[schema] = operator
    return merged


def make_mixed_dataset(domain_data: Dict[str, DomainData],
                       include_held_out: bool = False) -> List[Any]:
    """Concatenate training-domain graphs for mixed-domain batching.

    Precondition: all graphs were featurized with the SAME metadata (union or
    structural mode), so feature widths agree. Phase 0 per-domain mode must
    NOT call this across domains — widths differ by construction.
    """
    widths = set()
    graphs: List[Any] = []
    for data in domain_data.values():
        if data.spec.held_out and not include_held_out:
            continue
        widths.add((data.graph_metadata["num_node_features"],
                    data.graph_metadata["num_edge_features"]))
        graphs.extend(data.graphs)
    if len(widths) > 1:
        raise ValueError(
            f"Feature widths differ across domains: {widths}. Mixed batches "
            "require union ('union') or structural featurization, not "
            "'per_domain'."
        )
    return graphs

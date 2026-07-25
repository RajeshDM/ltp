"""Lifted domain layer + binding layer (CLAUDE.md 5.4/5.5; C1/C2).

Two featurization modes build on the structural substrate (Method 0):

- 'joint_lite' (paper: GADAR-BIND ablation): adds the lifted domain
  description to every state graph -- one symbol node per predicate,
  instantiation edges from each ground literal to its predicate's symbol
  node, and role edges between predicate symbols and schema nodes labeled
  with HOW the predicate appears (precondition / add / delete) and at
  which argument/slot positions (multi-hot).

- 'joint' (paper: GADAR, full Method B): additionally one occurrence node
  per (schema, role, k-th literal) -- so two occurrences of the SAME
  predicate in one schema are distinct -- with exact (pred-arg, slot)
  pair features, plus the BINDING layer: each grounded applicability fact
  (object o at slot j satisfies a precondition) becomes an edge from the
  object to the precondition-occurrence node it instantiates.

Everything is symbol-free (roles + positions only), so renaming
invariance and fixed cross-domain feature widths carry over from the
structural substrate. Node ordering in the state graph stays
[objects][literals][lifted nodes][schemas]: objects first and schemas
last, preserving every decoder invariant.
"""

import logging

from ploi.structural import build_structural_metadata, _pred_arity

logger = logging.getLogger(__name__)

_PRED_KEY = "LIFTED_PRED::{}"
_OCC_KEY = "LIFTED_OCC::{}::{}"


def pred_node_key(pred_name):
    return _PRED_KEY.format(pred_name)


def occ_node_key(schema_name, occ_index):
    return _OCC_KEY.format(schema_name, occ_index)


def _positive_predicate(literal):
    """The non-negated predicate of a (possibly Anti/delete-effect) literal."""
    pred = literal.predicate
    return getattr(pred, "positive", pred) or pred


def _schema_name(schema):
    return getattr(schema, "name", str(schema))


def build_lifted_spec(action_space):
    """Extract the lifted domain description from a pddlgym action space.

    Returns a plain (picklable) dict:
      predicates: {name: {'arity': int, 'static': bool}}
      pred_order: [name, ...] sorted (deterministic node order)
      schema_order: [schema_name, ...] sorted
      schemas: {schema_name: [occurrence, ...]} where occurrence =
        {'role': 'pre'|'add'|'del', 'pred': name,
         'bindings': [(pred_arg_i, schema_slot_j), ...]}
      Precondition occurrences come FIRST and keep preconds.literals
      order, so the k-th precondition of a schema is occurrence k -- the
      contract the binding layer relies on.
    """
    predicates = {}
    schemas = {}
    effect_names = set()

    for schema, op in action_space.items():
        for lit in getattr(getattr(op, "effects", None), "literals", []) or []:
            pred = _positive_predicate(lit)
            effect_names.add(pred.name)

    def record_occurrence(role, lit, params):
        pred = _positive_predicate(lit)
        name = pred.name
        predicates.setdefault(
            name, {"arity": _pred_arity(pred), "static": name not in effect_names})
        bindings = []
        for i, var in enumerate(getattr(lit, "variables", [])):
            if var in params:
                bindings.append((i, params.index(var)))
        return {"role": role, "pred": name, "bindings": bindings}

    for schema, op in sorted(action_space.items(),
                             key=lambda kv: _schema_name(kv[0])):
        params = list(getattr(op, "params", []))
        occurrences = []
        for lit in getattr(op.preconds, "literals", []):
            occurrences.append(record_occurrence("pre", lit, params))
        for lit in getattr(getattr(op, "effects", None), "literals", []) or []:
            role = "del" if getattr(lit, "is_anti", False) else "add"
            occurrences.append(record_occurrence(role, lit, params))
        schemas[_schema_name(schema)] = occurrences

    return {
        "predicates": predicates,
        "pred_order": sorted(predicates),
        "schema_order": sorted(schemas),
        "schemas": schemas,
    }


def build_chain_spec(spec):
    """Schema-chaining pairs over an existing lifted spec (#4, class-1 fix).

    enables:   (A, i, B, j) - occurrence i of schema A has role 'add' for
               predicate p, occurrence j of schema B has role 'pre' for p.
               "A produces what B needs": gripper pick.add:carry ->
               drop.pre:carry; miconic board.add:boarded -> depart.pre:boarded.
    threatens: (A, i, B, j) - same with role 'del' at A: "A destroys what
               B needs" (lifted delete reasoning).

    Pure function of the domain description; computed once, stored in
    metadata, zero per-state cost.
    """
    adds, pres, dels = {}, {}, {}
    for schema, occs in spec["schemas"].items():
        for k, occ in enumerate(occs):
            bucket = {"add": adds, "pre": pres, "del": dels}[occ["role"]]
            bucket.setdefault(occ["pred"], []).append((schema, k))
    enables = [(sa, ka_, sb, kb)
               for pred, producers in adds.items()
               for (sa, ka_) in producers
               for (sb, kb) in pres.get(pred, [])]
    threatens = [(sa, ka_, sb, kb)
                 for pred, deleters in dels.items()
                 for (sa, ka_) in deleters
                 for (sb, kb) in pres.get(pred, [])]
    return {"enables": enables, "threatens": threatens}


def schema_goal_distances(spec, chain, goal_pred_names, max_dist=3):
    """Per-schema hop count to a goal-achieving schema (#4's number, #5a at 0).

    Distance 0: schema has an add occurrence of a goal predicate. Distance
    d+1: schema enables (via any occurrence pair) a distance-d schema.
    Clamped to max_dist; schemas with no path get max_dist (a real bucket:
    'not goal-connected'). Depends only on WHICH predicates are goals - one
    computation per (domain, goal signature), not per state.
    """
    dist = {}
    for schema, occs in spec["schemas"].items():
        if any(o["role"] == "add" and o["pred"] in goal_pred_names
               for o in occs):
            dist[schema] = 0
    frontier = set(dist)
    d = 0
    while frontier and d < max_dist:
        d += 1
        nxt = set()
        for (sa, _ka, sb, _kb) in chain["enables"]:
            if sb in frontier and sa not in dist:
                dist[sa] = d
                nxt.add(sa)
        frontier = nxt
    for schema in spec["schemas"]:
        dist.setdefault(schema, max_dist)
    return dist


def build_lifted_metadata(action_space, max_pred_arity, max_action_arity, mode,
                          goal_prefix="WANT", cheating=False):
    """Structural metadata (Method 0) extended with the lifted-layer classes.

    Widths depend only on (max_pred_arity, max_action_arity, mode-independent
    class list), so graphs from different domains batch together and a
    held-out domain featurizes at the same widths -- both modes share ONE
    width so joint vs joint_lite is a pure graph-content ablation.
    """
    assert mode in ("joint", "joint_lite"), mode
    kp, ka = max_pred_arity, max_action_arity
    md = build_structural_metadata(action_space, kp, ka, goal_prefix=goal_prefix,
                                   cheating=cheating)

    node_extra = ["lifted_pred_node", "occ_node_pre", "occ_node_add",
                  "occ_node_del"]
    edge_extra = (["instance_of", "occ_of", "role_pre", "role_add", "role_del"]
                  + [f"slot_{j}" for j in range(ka)]
                  + [f"parg_{i}" for i in range(kp)]
                  + [f"pair_p{i}_s{j}" for i in range(kp) for j in range(ka)]
                  + [f"bind_slot_{j}" for j in range(ka)])

    node_map = md["node_feature_to_index"]
    next_node = md["num_node_features"]
    for cls in node_extra:
        if cls not in node_map:
            node_map[cls] = next_node
            next_node += 1
    md["num_node_features"] = next_node

    edge_map = md["edge_feature_to_index"]
    next_edge = md["num_edge_features"]
    for cls in edge_extra:
        if cls not in edge_map:
            edge_map[cls] = next_edge
            next_edge += 1
    md["num_edge_features"] = next_edge

    md["lifted_spec"] = build_lifted_spec(action_space)
    md["lifted_spec"]["mode"] = mode
    md["featurization"] = mode
    return md


def lifted_node_keys(spec):
    """Deterministic node keys for the lifted layer of one state graph.

    Predicate symbol nodes first (pred_order), then -- 'joint' mode only --
    occurrence nodes per schema in schema_order, occurrence index order.
    """
    keys = [pred_node_key(name) for name in spec["pred_order"]]
    if spec.get("mode") == "joint":
        for schema_name in spec["schema_order"]:
            for k in range(len(spec["schemas"][schema_name])):
                keys.append(occ_node_key(schema_name, k))
    return keys


def add_lifted_node_features(input_node_features, lifted_keys, spec,
                             objects_to_node, node_feature_to_index, kp):
    """Features for lifted nodes: a type bit + the predicate's structural
    class (symbol nodes) or the occurrence's role bit (occurrence nodes)."""
    for key in lifted_keys:
        idx = objects_to_node[key]
        if key.startswith("LIFTED_PRED::"):
            name = key.split("::", 1)[1]
            info = spec["predicates"][name]
            input_node_features[idx, node_feature_to_index["lifted_pred_node"]] = 1
            cls = f"pred_a{min(info['arity'], kp)}_s{int(info['static'])}"
            if cls in node_feature_to_index:
                input_node_features[idx, node_feature_to_index[cls]] = 1
        else:
            _, schema_name, occ_k = key.split("::")
            occ = spec["schemas"][schema_name][int(occ_k)]
            role_cls = f"occ_node_{occ['role']}"
            input_node_features[idx, node_feature_to_index[role_cls]] = 1


def _set_edge(all_edge_features, a, b, feature_index):
    all_edge_features[a, b, feature_index] = 1
    all_edge_features[b, a, feature_index] = 1


def add_lifted_layer_edges(all_edge_features, all_literals, all_actions,
                           spec, objects_to_node, edge_feature_to_index,
                           kp, ka, goal_prefix="WANT"):
    """Instantiation + role (+ occurrence, in 'joint' mode) edges.

    - ground literal <-> its predicate's symbol node        [instance_of]
    - predicate symbol <-> schema node                      [role_* + parg_i + slot_j]
    - 'joint': occurrence <-> predicate symbol              [occ_of]
               occurrence <-> schema node                   [role_* + pair_pi_sj]
    Binding edges (object <-> occurrence) are added where grounded
    applicability is computed; see _get_precondition_satisfaction_position.
    """
    E = edge_feature_to_index

    for lit in all_literals:
        name = lit.predicate.name
        if name.startswith(goal_prefix):
            name = name[len(goal_prefix):]
        key = pred_node_key(name)
        if key not in objects_to_node:
            continue
        _set_edge(all_edge_features, objects_to_node[lit],
                  objects_to_node[key], E["instance_of"])

    joint = spec.get("mode") == "joint"
    for action in all_actions:
        schema_name = _schema_name(action)
        if schema_name not in spec["schemas"]:
            continue
        schema_idx = objects_to_node[action]
        for k, occ in enumerate(spec["schemas"][schema_name]):
            pred_idx = objects_to_node[pred_node_key(occ["pred"])]
            role_feature = E[f"role_{occ['role']}"]
            _set_edge(all_edge_features, pred_idx, schema_idx, role_feature)
            for (i, j) in occ["bindings"]:
                ci, cj = min(i, kp - 1), min(j, ka - 1)
                _set_edge(all_edge_features, pred_idx, schema_idx, E[f"parg_{ci}"])
                _set_edge(all_edge_features, pred_idx, schema_idx, E[f"slot_{cj}"])
            if joint:
                occ_idx = objects_to_node[occ_node_key(schema_name, k)]
                _set_edge(all_edge_features, occ_idx, pred_idx, E["occ_of"])
                _set_edge(all_edge_features, occ_idx, schema_idx, role_feature)
                for (i, j) in occ["bindings"]:
                    ci, cj = min(i, kp - 1), min(j, ka - 1)
                    _set_edge(all_edge_features, occ_idx, schema_idx,
                              E[f"pair_p{ci}_s{cj}"])

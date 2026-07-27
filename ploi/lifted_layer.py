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

- 'joint_chain': everything 'joint' has, plus schema chaining and goal
  relevance. Chain edges between occurrence nodes ('enables'/'threatens',
  build_chain_spec); per-schema goal-distance buckets + 'adds_goal_pred'
  and 'goal_relevant_obj' node features (add_chain_node_features,
  schema_goal_distances); grounded effect edges 'achieves_goal'/
  'deletes_true' from applicable groundings (add_grounded_effect_edges).
  Wider metadata than 'joint' -> its own cache sidecar tag.

All three modes TYPE-COMPILE the domain first (build_lifted_spec): each
declared PDDL type becomes a synthetic unary static predicate with one
'pre' occurrence per typed schema slot, and each object gets a 'has_type'
edge to its type's symbol node (add_type_edges). Without this, half the
suite is typed (gripper, miconic, spanner, rovers, blocksworld) and half
spells types as ordinary predicates (grid, logistics) -- and only the
second half's type constraints survive the symbol-free featurization,
because structural.py collapses every declared type to one class. After
compilation both families produce the same lifted structure.

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

# Type compilation (below): a declared PDDL type becomes a synthetic unary
# static predicate, so a typed domain and a predicate-typed one present the
# same structure to the lifted + binding layers.
_TYPE_PRED = "__type_{}"
# pddlgym gives untyped entities this pseudo-type; it carries no information
# (every object has it) so compiling it would add a constant node.
_UNTYPED = {None, "default", ""}


def pred_node_key(pred_name):
    return _PRED_KEY.format(pred_name)


def occ_node_key(schema_name, occ_index):
    return _OCC_KEY.format(schema_name, occ_index)


def type_pred_name(type_name):
    return _TYPE_PRED.format(type_name)


def declared_type(entity):
    """The entity's PDDL type, or None if the domain is untyped there."""
    t = getattr(entity, "var_type", None)
    t = None if t is None else str(t)
    return None if t in _UNTYPED else t


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
      type_preds: {pddl_type_name: synthetic_predicate_name}
      type_occ:   {schema_name: {slot_j: occurrence_index}}

      Precondition occurrences come FIRST and keep preconds.literals
      order, so the k-th precondition of a schema is occurrence k -- the
      contract the binding layer relies on. TYPE-COMPILED preconditions
      (below) are appended directly after them, before any effect
      occurrence, so that contract is untouched.

    Type compilation: a schema parameter declared `?x - block` carries a
    constraint that a predicate-typed domain would spell as a precondition
    `(block ?x)`. Without compiling it, the constraint is invisible to the
    symbol-free featurizations -- structural.py collapses every declared
    type to one `typed_object` class -- so a typed domain's slots have no
    representable type at all while a predicate-typed domain's do. We
    synthesize the missing precondition: one unary static predicate per
    declared type, one 'pre' occurrence per typed slot. Both families then
    reach the binding layer through the same path.
    """
    predicates = {}
    schemas = {}
    effect_names = set()
    type_preds = {}
    type_occ = {}

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
        name = _schema_name(schema)
        occurrences = []
        for lit in getattr(op.preconds, "literals", []):
            occurrences.append(record_occurrence("pre", lit, params))
        # Type-compiled preconditions: after the real ones (preserving the
        # k-th-precondition-is-occurrence-k contract), before the effects.
        slots = {}
        for j, param in enumerate(params):
            t = declared_type(param)
            if t is None:
                continue
            pred_name = type_preds.setdefault(t, type_pred_name(t))
            predicates.setdefault(pred_name, {"arity": 1, "static": True})
            slots[j] = len(occurrences)
            occurrences.append({"role": "pre", "pred": pred_name,
                                "bindings": [(0, j)], "type_slot": j})
        if slots:
            type_occ[name] = slots
        for lit in getattr(getattr(op, "effects", None), "literals", []) or []:
            role = "del" if getattr(lit, "is_anti", False) else "add"
            occurrences.append(record_occurrence(role, lit, params))
        schemas[name] = occurrences

    return {
        "predicates": predicates,
        "pred_order": sorted(predicates),
        "schema_order": sorted(schemas),
        "schemas": schemas,
        "type_preds": type_preds,
        "type_occ": type_occ,
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
    held-out domain featurizes at the same widths. 'joint' and 'joint_lite'
    share ONE width so joint vs joint_lite is a pure graph-content ablation;
    'joint_chain' appends its extra classes after them and is strictly wider
    (it gets its own cache sidecar tag).
    """
    assert mode in ("joint", "joint_lite", "joint_chain"), mode
    kp, ka = max_pred_arity, max_action_arity
    md = build_structural_metadata(action_space, kp, ka, goal_prefix=goal_prefix,
                                   cheating=cheating)

    node_extra = ["lifted_pred_node", "occ_node_pre", "occ_node_add",
                  "occ_node_del"]
    edge_extra = (["instance_of", "occ_of", "role_pre", "role_add", "role_del",
                   "has_type"]
                  + [f"slot_{j}" for j in range(ka)]
                  + [f"parg_{i}" for i in range(kp)]
                  + [f"pair_p{i}_s{j}" for i in range(kp) for j in range(ka)]
                  + [f"bind_slot_{j}" for j in range(ka)])
    if mode == "joint_chain":
        # Appended AFTER the joint classes: joint/joint_lite indices and
        # widths stay byte-identical; joint_chain is strictly wider.
        node_extra = node_extra + ["adds_goal_pred", "goal_relevant_obj",
                                   "gdist_0", "gdist_1", "gdist_2",
                                   "gdist_3plus"]
        edge_extra = edge_extra + ["enables", "threatens", "achieves_goal",
                                   "deletes_true"]

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
    if mode == "joint_chain":
        md["lifted_spec"]["chain"] = build_chain_spec(md["lifted_spec"])
    md["lifted_spec"]["mode"] = mode
    md["featurization"] = mode
    n_types = len(md["lifted_spec"]["type_preds"])
    logger.info(
        "lifted layer (%s): %d predicates (%d type-compiled from declared "
        "PDDL types), %d schemas", mode, len(md["lifted_spec"]["predicates"]),
        n_types, len(md["lifted_spec"]["schemas"]))
    print(f"  lifted layer [{mode}]: {n_types} declared type(s) compiled to "
          f"unary static predicates"
          + (" (domain is predicate-typed or untyped)" if not n_types else ""))
    return md


def lifted_node_keys(spec):
    """Deterministic node keys for the lifted layer of one state graph.

    Predicate symbol nodes first (pred_order), then -- 'joint'/'joint_chain'
    only -- occurrence nodes per schema in schema_order, occurrence index
    order.
    """
    keys = [pred_node_key(name) for name in spec["pred_order"]]
    if spec.get("mode") in ("joint", "joint_chain"):
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
    """Instantiation + role (+ occurrence, in 'joint'/'joint_chain') edges.

    - ground literal <-> its predicate's symbol node        [instance_of]
    - predicate symbol <-> schema node                      [role_* + parg_i + slot_j]
    - 'joint'/'joint_chain':
               occurrence <-> predicate symbol              [occ_of]
               occurrence <-> schema node                   [role_* + pair_pi_sj]
    - 'joint_chain': occurrence <-> occurrence              [enables / threatens]
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

    joint = spec.get("mode") in ("joint", "joint_chain")
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

    # Chain edges ('joint_chain'): occurrence pairs from build_chain_spec.
    if spec.get("mode") == "joint_chain":
        chain = spec.get("chain")
        if chain is None:
            raise RuntimeError(
                "lifted_spec has mode 'joint_chain' but no 'chain' key -- "
                "metadata must come from build_lifted_metadata(..., "
                "'joint_chain')")
        for kind in ("enables", "threatens"):
            feat = E[kind]
            for (sa, occ_a, sb, occ_b) in chain[kind]:
                _set_edge(all_edge_features,
                          objects_to_node[occ_node_key(sa, occ_a)],
                          objects_to_node[occ_node_key(sb, occ_b)], feat)


def add_type_edges(all_edge_features, all_objects, spec, objects_to_node,
                   edge_feature_to_index, state_literals=()):
    """object <-> the symbol node of the type it belongs to  [has_type].

    Both directions of the compilation, so the edge means the same thing in
    either family of domain:

    - declared types: the compiled counterpart of the ground literal
      `(block b1)` a predicate-typed domain would carry, linking the object
      to the same symbol node the schema's type-precondition occurrence
      hangs off.
    - predicate-typed domains: a TRUE unary STATIC literal is a type
      declaration written by hand -- `(room r1)`, `(truck t1)`. Nothing in
      the domain can add or delete it, so it partitions the objects exactly
      as a declared type does. Emitting the edge here too means a model
      trained mostly on one family does not meet an unfamiliar edge class on
      the other.

    Objects whose type no schema parameter declares have no symbol node and
    are skipped -- the type constrains nothing.
    """
    feat = edge_feature_to_index["has_type"]
    predicates = spec.get("predicates", {})

    def link(obj, pred_name):
        pred_idx = objects_to_node.get(pred_node_key(pred_name))
        obj_idx = objects_to_node.get(obj)
        if pred_idx is not None and obj_idx is not None:
            _set_edge(all_edge_features, obj_idx, pred_idx, feat)

    type_preds = spec.get("type_preds") or {}
    for obj in all_objects:
        pred_name = type_preds.get(declared_type(obj))
        if pred_name is not None:
            link(obj, pred_name)

    for lit in state_literals:
        info = predicates.get(lit.predicate.name)
        if info is None or info["arity"] != 1 or not info["static"]:
            continue
        variables = getattr(lit, "variables", [])
        if variables:
            link(variables[0], lit.predicate.name)


_GDIST_MAX = 3  # buckets gdist_0..gdist_2 + gdist_3plus; fixed by metadata


def add_chain_node_features(input_node_features, spec, all_actions,
                            state_literals, goal_literals, objects_to_node,
                            node_feature_to_index):
    """Goal-conditioned node features, 'joint_chain' only (#5).

    Schema nodes get their schema_goal_distances bucket (gdist_0..gdist_3plus)
    plus 'adds_goal_pred' at distance 0. Objects appearing in an UNSATISFIED
    goal atom get 'goal_relevant_obj'. goal_literals are the UNWRAPPED goal
    atoms (no goal prefix), comparable against state_literals.
    """
    chain = spec.get("chain")
    if chain is None:
        raise RuntimeError(
            "add_chain_node_features needs spec['chain'] -- metadata must "
            "come from build_lifted_metadata(..., 'joint_chain')")
    N = node_feature_to_index
    goal_pred_names = {g.predicate.name for g in goal_literals}
    dist = schema_goal_distances(spec, chain, goal_pred_names,
                                 max_dist=_GDIST_MAX)
    for action in all_actions:
        name = _schema_name(action)
        if name not in dist:
            continue
        idx = objects_to_node[action]
        d = dist[name]
        cls = "gdist_3plus" if d >= _GDIST_MAX else f"gdist_{d}"
        input_node_features[idx, N[cls]] = 1
        if d == 0:
            input_node_features[idx, N["adds_goal_pred"]] = 1

    true_now = set(state_literals)
    for goal_lit in goal_literals:
        if goal_lit in true_now:
            continue
        for var in getattr(goal_lit, "variables", []):
            input_node_features[objects_to_node[var],
                                N["goal_relevant_obj"]] = 1


def add_grounded_effect_edges(all_edge_features, all_groundings, action_space,
                              unsat_goal_nodes, true_literal_nodes,
                              objects_to_node, edge_feature_to_index):
    """Grounded effect edges, 'joint_chain' only (#6).

    For each applicable grounding, substitute its parameter bindings into the
    schema's effects. An add effect that equals an UNSATISFIED goal atom links
    every participating object to that goal-literal node [achieves_goal]; a
    delete effect that equals a TRUE state atom links objects to that literal
    node [deletes_true]. The lookups map (pred_name, arg_tuple) -> node index;
    a computed atom without a node is expected (most effects touch neither
    goal nor a tracked literal) and skipped silently.
    """
    E = edge_feature_to_index
    achieves, deletes = E["achieves_goal"], E["deletes_true"]
    for grounding in all_groundings:
        try:
            op = action_space[grounding.predicate]
        except KeyError:
            continue
        binding = dict(zip(list(getattr(op, "params", [])),
                           grounding.variables))
        for lit in getattr(getattr(op, "effects", None), "literals", []) or []:
            # binding.get(v, v): schema params ground via the binding,
            # anything else (a constant) already is an object.
            args = tuple(binding.get(v, v)
                         for v in getattr(lit, "variables", []))
            key = (_positive_predicate(lit).name, args)
            if getattr(lit, "is_anti", False):
                lit_idx = true_literal_nodes.get(key)
                feat = deletes
            else:
                lit_idx = unsat_goal_nodes.get(key)
                feat = achieves
            if lit_idx is None:
                continue
            for obj in args:
                _set_edge(all_edge_features, objects_to_node[obj], lit_idx,
                          feat)

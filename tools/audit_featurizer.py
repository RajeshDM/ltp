#!/usr/bin/env python3
"""Audit the lifted featurization against the PDDL TEXT (needs pddlgym, no torch model).

    python tools/audit_featurizer.py                       # the eight suite domains
    python tools/audit_featurizer.py --domain Miconic_ipcc --problems 3 --steps 30
    python tools/audit_featurizer.py --domain-file d.pddl --problem-dir probs/

Why this exists. Until 2026-11 the lifted layer silently dropped the effect of
every schema with exactly one effect literal (pddlgym leaves it a bare Literal;
the code read `getattr(effects, "literals", [])`). No test caught it: the unit
tests build mock operators, and the byte-identity gate compares the code with
itself. This tool compares the featurizer with an INDEPENDENT reading of the
domain: it parses the domain file's text with its own s-expression reader and
recomputes, from that and from the pddlgym state, what the graph must contain.

  static (per domain)
    - every schema's occurrence list = its preconditions in order, one type
      occurrence per typed parameter, its effects in order with add/del roles
    - a predicate is static iff no schema's effect mentions it
    - a negative precondition `(not (p ?x))` is an 'npre' occurrence, and
      the chain edges through it are swapped (adding p threatens it,
      deleting p enables it)
    - constructs the representation cannot express are REPORTED (quantifiers,
      disjunction, conditional effects, constants, a parameter no
      precondition mentions, a type hierarchy)
  per state (seeded random rollouts, 'joint_chain' = every component on)
    - instance_of, has_type, role_*, occ_of, enables, threatens
    - bind_slot_j from the applicable groundings and the parsed preconditions
    - achieves_goal / deletes_true from the parsed effects
    - gdist buckets, adds_goal_pred, goal_relevant_obj
    - schema-object candidate edges, literal-object argument edges
    - the structural class of every literal node

Exit 1 on any MISMATCH. WARN lines are properties of the domain, not bugs: read
them before trusting a number on that domain.
"""
import argparse
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUITE = ["Manyblocks_ipcc_big", "Gripper_ipcc", "Miconic_ipcc", "Visitall_ipcc",
         "Grid_ipcc", "Logistics_ipcc", "Spanner_ipcc", "Rovers_ipcc"]


# ─── independent PDDL reader ────────────────────────────────────────────────

def read_sexpr(text):
    text = re.sub(r";[^\n]*", "", text).lower()
    tokens = re.findall(r"[()]|[^\s()]+", text)
    pos = 0

    def parse():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok != "(":
            return tok
        out = []
        while tokens[pos] != ")":
            out.append(parse())
        pos += 1
        return out
    return parse()


def typed_list(items):
    """['?a', '?b', '-', 't', '?c'] -> [('?a','t'), ('?b','t'), ('?c', None)]."""
    out, pending, i = [], [], 0
    while i < len(items):
        if items[i] == "-":
            out += [(n, items[i + 1]) for n in pending]
            pending = []
            i += 2
        else:
            pending.append(items[i])
            i += 1
    return out + [(n, None) for n in pending]


def flatten(expr, negated, out, odd, where):
    """Collect (negated, pred, args) atoms; record what is not a plain atom."""
    if not isinstance(expr, list) or not expr:
        return
    head = expr[0]
    if head == "and":
        for e in expr[1:]:
            flatten(e, negated, out, odd, where)
    elif head == "not":
        flatten(expr[1], not negated, out, odd, where)
    elif head in ("forall", "exists", "or", "imply", "when", "increase", "decrease",
                  "assign", "=", "<", ">", "<=", ">="):
        odd.add(f"{where}:{head}")
    else:
        out.append((negated, head, tuple(expr[1:])))


def read_domain(path):
    dom = read_sexpr(open(path).read())
    types, actions = {}, {}
    for sec in dom[2:]:
        if sec[0] == ":types":
            for name, parent in typed_list(sec[1:]):
                types[name] = parent
        elif sec[0] == ":action":
            body = dict(zip(sec[2::2], sec[3::2]))
            params = typed_list(body.get(":parameters", []))
            pre, eff, odd = [], [], set()
            flatten(body.get(":precondition", []), False, pre, odd, "pre")
            flatten(body.get(":effect", []), False, eff, odd, "eff")
            names = [p for p, _ in params]
            if any(a not in names for _, _, args in pre + eff for a in args):
                odd.add("constant-or-free-variable")
            actions[sec[1]] = {"params": params, "pre": pre, "eff": eff, "odd": odd}
    return types, actions


def expected_spec(actions):
    """What build_lifted_spec must return, from the text alone."""
    effect_preds = {p for a in actions.values() for _, p, _ in a["eff"]}
    schemas = {}
    for name, a in actions.items():
        names = [p for p, _ in a["params"]]

        def occ(role, pred, args):
            return (role, pred, tuple((i, names.index(v)) for i, v in enumerate(args)
                                      if v in names))
        occs = [occ("npre" if neg else "pre", p, args) for neg, p, args in a["pre"]]
        occs += [("pre", f"__type_{t}", ((0, j),))
                 for j, (_, t) in enumerate(a["params"]) if t is not None]
        occs += [occ("del" if neg else "add", p, args) for neg, p, args in a["eff"]]
        schemas[name] = occs
    return schemas, effect_preds


# ─── naming the nodes of a featurized graph ─────────────────────────────────

def obj_name(o):
    return "O:" + str(o).split(":")[0]


def lit_name(lit, goal_prefix="WANT"):
    name = lit.predicate.name
    kind = "L"
    if name.startswith(goal_prefix):
        name, kind = name[len(goal_prefix):], "G"
    return f"{kind}:{name}({','.join(str(v).split(':')[0] for v in lit.variables)})"


def node_names(node_to_objects, n_objects, n_literals, n_lifted):
    names = {}
    for i, ent in node_to_objects.items():
        if i < n_objects:
            names[i] = obj_name(ent)
        elif i < n_objects + n_literals:
            names[i] = lit_name(ent)
        elif i < n_objects + n_literals + n_lifted:
            names[i] = str(ent).replace("LIFTED_PRED::", "P:").replace("LIFTED_OCC::", "OCC:")
        else:
            names[i] = "S:" + ent.name
    return names


def both(pairs):
    return set(pairs) | {(b, a) for a, b in pairs}


# ─── the audit ──────────────────────────────────────────────────────────────

def audit_domain(env, label, n_problems, n_steps, seed, kp, ka):
    from ploi.datautils_ltp import state_to_graph_wrapper
    from ploi.lifted_layer import build_lifted_metadata

    bad, warn = [], []
    action_space = env.action_space._action_predicate_to_operators
    types, actions = read_domain(env.domain.domain_fname)
    md = build_lifted_metadata(action_space, kp, ka, "joint_chain")
    spec = md["lifted_spec"]
    want_schemas, effect_preds = expected_spec(actions)

    # static -----------------------------------------------------------------
    if set(want_schemas) != set(spec["schemas"]):
        bad.append(f"schemas differ: text {sorted(want_schemas)} vs spec {sorted(spec['schemas'])}")
    for name in sorted(set(want_schemas) & set(spec["schemas"])):
        got = [(o["role"], o["pred"], tuple(tuple(b) for b in o["bindings"]))
               for o in spec["schemas"][name]]
        if got != want_schemas[name]:
            bad.append(f"{name}: occurrences differ\n      text: {want_schemas[name]}\n      spec: {got}")
    for pred, info in spec["predicates"].items():
        want_static = pred not in effect_preds
        if info["static"] != want_static:
            bad.append(f"predicate {pred}: static={info['static']}, text says {want_static}")
    for name, a in actions.items():
        for flag in sorted(a["odd"]):
            warn.append(f"{name}: {flag} - not expressible in the lifted layer")
        mentioned = {v for _, _, args in a["pre"] for v in args}
        zero_arity = any(not args for _, _, args in a["pre"])
        for j, (p, _) in enumerate(a["params"]):
            if p not in mentioned and not zero_arity:
                warn.append(f"{name}: parameter {p} (slot {j}) is in no precondition - "
                            f"no object is ever a candidate at that slot (edge pruned)")
        n_pre = max((len(args) for _, _, args in a["pre"] + a["eff"]), default=0)
        if len(a["params"]) > ka or n_pre > kp:
            warn.append(f"{name}: arity {len(a['params'])}/pred arity {n_pre} exceeds "
                        f"ka={ka}/kp={kp} - positions are clamped and collide")
    parents = {p for p in types.values() if p not in (None, "object")}
    used = {t for a in actions.values() for _, t in a["params"] if t}
    for t in sorted(used & (parents | ({"object"} if types else set()))):
        warn.append(f"type hierarchy: a parameter is declared '{t}', a supertype - objects "
                    f"report their own type, so that slot's has_type path is dead")

    # the independent chain + goal distance ------------------------------------
    # A makes B's precondition true: A adds what B needs present, or deletes
    # what B needs absent. The other two combinations threaten it.
    enables, threatens = set(), set()
    for a, occs_a in want_schemas.items():
        for i, (role_a, pred, _) in enumerate(occs_a):
            if role_a not in ("add", "del"):
                continue
            for b, occs_b in want_schemas.items():
                for j, (role_b, pred_b, _) in enumerate(occs_b):
                    if role_b in ("pre", "npre") and pred_b == pred:
                        helps = (role_a == "add") == (role_b == "pre")
                        (enables if helps else threatens).add(
                            (f"OCC:{a}::{i}", f"OCC:{b}::{j}"))

    def goal_distances(goal_preds):
        dist = {a: 0 for a, occs in want_schemas.items()
                if any(r == "add" and p in goal_preds for r, p, _ in occs)}
        for d in (1, 2):
            layer = {s for s in want_schemas if s not in dist and any(
                any(pb == p and (r, rb) in (("add", "pre"), ("del", "npre"))
                    for rb, pb, _ in want_schemas[t])
                for r, p, _ in want_schemas[s] for t, dt in dist.items() if dt == d - 1)}
            dist.update({s: d for s in layer})
        return {s: dist.get(s, 3) for s in want_schemas}

    # per state ----------------------------------------------------------------
    N, E = md["node_feature_to_index"], md["edge_feature_to_index"]
    CHECKED = (["instance_of", "has_type", "role_pre", "role_add", "role_del", "occ_of",
                "enables", "threatens", "achieves_goal", "deletes_true"]
               + [f"bind_slot_{j}" for j in range(ka)] + [f"pos_{j}" for j in range(ka)]
               + [f"pred_pos_{i}" for i in range(kp)])
    rng = random.Random(seed)
    n_states = 0
    counts = defaultdict(int)
    for problem_idx in range(min(n_problems, len(env.problems))):
        env.fix_problem_index(problem_idx)
        state, _ = env.reset()
        for _ in range(n_steps):
            groundings = sorted(env.action_space.all_ground_literals(state), key=repr)
            if not groundings:
                break
            n_states += 1
            graph, _, n2o = state_to_graph_wrapper(
                state, action_space, groundings, None, None, md, None, None, state.goal)
            n_obj, n_act = len(state.objects), len(action_space)
            n_lit = len(state.literals) + len(state.goal.literals)
            n_lifted = len(n2o) - n_obj - n_act - n_lit
            names = node_names(n2o, n_obj, n_lit, n_lifted)
            index = {v: k for k, v in names.items()}
            if len(index) != len(names):
                bad.append(f"state {n_states}: two nodes share a name")

            got = defaultdict(set)
            cols = {f: E[f] for f in CHECKED}
            for row, (s, r) in enumerate(zip(graph["senders"], graph["receivers"])):
                for f, c in cols.items():
                    if graph["edges"][row, c]:
                        got[f].add((names[int(s)], names[int(r)]))

            truth = {(l.predicate.name, tuple(str(v).split(":")[0] for v in l.variables))
                     for l in state.literals}
            goal = {(l.predicate.name, tuple(str(v).split(":")[0] for v in l.variables))
                    for l in state.goal.literals}
            fmt = lambda kind, atom: f"{kind}:{atom[0]}({','.join(atom[1])})"
            want = defaultdict(set)

            want["instance_of"] = both([(fmt("L", a), "P:" + a[0]) for a in truth]
                                       + [(fmt("G", a), "P:" + a[0]) for a in goal])
            typed = [(obj_name(o), str(o.var_type)) for o in state.objects]
            want["has_type"] = both(
                [(o, f"P:__type_{t}") for o, t in typed if t in spec["type_preds"]]
                + [("O:" + a[1][0], "P:" + a[0]) for a in truth
                   if len(a[1]) == 1 and a[0] in spec["predicates"] and a[0] not in effect_preds])
            for a, occs in want_schemas.items():
                for k, (role, pred, _) in enumerate(occs):
                    # A negative precondition is pre+del on ITS occurrence and
                    # nothing on the predicate<->schema edge (which is a union
                    # over occurrences, where pre+del means something else).
                    for r in (("pre", "del") if role == "npre" else (role,)):
                        want[f"role_{r}"] |= both([(f"OCC:{a}::{k}", "S:" + a)])
                        if role != "npre":
                            want[f"role_{r}"] |= both([("P:" + pred, "S:" + a)])
                    want["occ_of"] |= both([(f"OCC:{a}::{k}", "P:" + pred)])
                    row = graph["nodes"][index[f"OCC:{a}::{k}"]]
                    flags = tuple(int(row[N[f"occ_node_{r}"]]) for r in ("pre", "add", "del"))
                    exp = {"pre": (1, 0, 0), "add": (0, 1, 0), "del": (0, 0, 1),
                           "npre": (1, 0, 1)}[role]
                    if flags != exp:
                        bad.append(f"state {n_states}: OCC:{a}::{k} ({role} {pred}) has role "
                                   f"flags pre/add/del {flags}, expected {exp}")
            want["enables"], want["threatens"] = both(enables), both(threatens)

            for g in groundings:
                a = actions[g.predicate.name]
                objs = [str(v).split(":")[0] for v in g.variables]
                sub = dict(zip([p for p, _ in a["params"]], objs))
                for neg, pred, args in a["pre"]:
                    if ((pred, tuple(sub[v] for v in args)) in truth) == neg:
                        bad.append(f"state {n_states}: pddlgym calls {g} applicable but "
                                   f"{'(not ' if neg else ''}{pred}{args}{')' if neg else ''} "
                                   f"does not hold")
                for j, (p, t) in enumerate(a["params"]):
                    slot = min(j, ka - 1)
                    hit = False
                    for k, (_, _, args) in enumerate(a["pre"]):
                        if p in args or not args:
                            hit = True
                            want[f"bind_slot_{slot}"] |= both(
                                [("O:" + objs[j], f"OCC:{g.predicate.name}::{k}")])
                    if t is not None:
                        k = len(a["pre"]) + sum(1 for _, tt in a["params"][:j] if tt is not None)
                        want[f"bind_slot_{slot}"] |= both(
                            [("O:" + objs[j], f"OCC:{g.predicate.name}::{k}")])
                    if hit:
                        want[f"pos_{slot}"] |= both([("S:" + g.predicate.name, "O:" + objs[j])])
                for neg, pred, args in a["eff"]:
                    atom = (pred, tuple(sub[v] for v in args))
                    if neg and atom in truth:
                        want["deletes_true"] |= both([("O:" + o, fmt("L", atom)) for o in atom[1]])
                    if not neg and atom in goal and atom not in truth:
                        want["achieves_goal"] |= both([("O:" + o, fmt("G", atom)) for o in atom[1]])
            for kind, atoms in (("L", truth), ("G", goal)):
                for atom in atoms:
                    for i, o in enumerate(atom[1]):
                        want[f"pred_pos_{min(i, kp - 1)}"] |= both([(fmt(kind, atom), "O:" + o)])

            for f in CHECKED:
                counts[f] += len(want[f])
                if got[f] != want[f]:
                    miss, extra = sorted(want[f] - got[f])[:3], sorted(got[f] - want[f])[:3]
                    bad.append(f"state {n_states} edge '{f}': {len(want[f] - got[f])} missing "
                               f"{miss}, {len(got[f] - want[f])} unexpected {extra}")

            # node features
            nodes = graph["nodes"]
            dist = goal_distances({p for p, _ in goal})
            for a, d in dist.items():
                row = nodes[index["S:" + a]]
                flags = {c: int(row[N[c]]) for c in ("gdist_0", "gdist_1", "gdist_2",
                                                     "gdist_3plus", "adds_goal_pred")}
                exp = {"gdist_0": int(d == 0), "gdist_1": int(d == 1), "gdist_2": int(d == 2),
                       "gdist_3plus": int(d >= 3), "adds_goal_pred": int(d == 0)}
                if flags != exp:
                    bad.append(f"state {n_states} schema {a}: goal features {flags}, expected {exp}")
            relevant = {"O:" + o for atom in goal - truth for o in atom[1]}
            got_rel = {names[i] for i in range(n_obj) if nodes[i, N["goal_relevant_obj"]]}
            if got_rel != relevant:
                bad.append(f"state {n_states}: goal_relevant_obj {sorted(got_rel)} vs {sorted(relevant)}")
            counts["goal_relevant_obj"] += len(relevant)
            for kind, atoms in (("L", truth), ("G", goal)):
                for atom in atoms:
                    cls = (f"{'gpred' if kind == 'G' else 'pred'}_a{min(len(atom[1]), kp)}"
                           f"_s{int(atom[0] not in effect_preds)}")
                    row = nodes[index[fmt(kind, atom)]]
                    if not row[N[cls]] or row[N["goal_pred"]] != (kind == "G"):
                        bad.append(f"state {n_states}: {fmt(kind, atom)} lacks class {cls} "
                                   f"or has the wrong goal flag")

            state, _, done, _ = env.step(rng.choice(groundings))
            if done:
                break

    from ploi.structural import featurizer_version
    version = featurizer_version(action_space)
    n_neg = sum(1 for a in actions.values() for neg, _, _ in a["pre"] if neg)
    print(f"\n== {label}: {len(actions)} schemas, {n_states} states")
    print(f"   featurizer version {version}"
          + ("  (0: no featurizer fix changes this domain's graphs)" if not version else
             "  (" + ", ".join(t for bit, t in ((1, "1 = has a single-effect schema"),
                                               (2, f"2 = has {n_neg} negative precondition(s)"))
                               if version & bit)
             + ": lifted graphs differ from those built before the fix)"))
    print("   checked edges: " + ", ".join(f"{f}={n}" for f, n in sorted(counts.items()) if n))
    empty = [f for f in ("achieves_goal", "deletes_true", "enables", "bind_slot_0")
             if not counts[f]]
    if empty and n_states:
        warn.append(f"no {empty} on these states - those checks were vacuous (more --steps)")
    for w in sorted(set(warn)):
        print(f"   WARN  {w}")
    seen = set()
    for b in bad:
        key = re.sub(r"state \d+", "state N", b)[:90]
        if key not in seen:
            seen.add(key)
            print(f"   MISMATCH  {b}")
    if not bad:
        print("   ok")
    return len(bad)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", action="append", help="pddlgym name (repeatable); default: the suite")
    ap.add_argument("--domain-file", help="a domain .pddl outside pddlgym (with --problem-dir)")
    ap.add_argument("--problem-dir")
    ap.add_argument("--problems", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kp", type=int, default=3)
    ap.add_argument("--ka", type=int, default=6)
    args = ap.parse_args()

    n_bad = 0
    if args.domain_file:
        from pddlgym.core import PDDLEnv
        env = PDDLEnv(args.domain_file, args.problem_dir, operators_as_actions=True,
                      dynamic_action_space=True)
        n_bad += audit_domain(env, args.domain_file, args.problems, args.steps, args.seed,
                              args.kp, args.ka)
    else:
        import pddlgym
        for name in args.domain or SUITE:
            try:
                env = pddlgym.make(f"PDDLEnv{name}-v0")
            except Exception as e:  # report and keep going: one missing domain must not hide the rest
                print(f"\n== {name}: could not be created: {type(e).__name__}: {e}")
                n_bad += 1
                continue
            n_bad += audit_domain(env, name, args.problems, args.steps, args.seed,
                                  args.kp, args.ka)
    print(f"\n{'FAILED' if n_bad else 'PASSED'}: {n_bad} mismatch(es)")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()

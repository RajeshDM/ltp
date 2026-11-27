#!/usr/bin/env python3
"""Typed versions of the two predicate-typed domains (Grid, Logistics).

    python tools/convert_to_typed.py --domain logistics --src <problem dir> --dst <problem dir>
    python tools/convert_to_typed.py --domain grid --write-domain <path/grid_typed_ipcc.pddl>
    python tools/convert_to_typed.py --domain logistics --check <untyped dir> <typed dir>

The experiment (RUNS_STATUS 2026-11-27): six suite domains declare PDDL types,
Grid and Logistics spell them as unary predicates. Converting these two gives
the same problems in typed spelling, so GABAR/GADAR trained on both spellings
answer whether the spelling matters. Types are FLAT on every action
parameter (no type hierarchy on a slot - GADAR's known weak spot); Logistics'
AIRPORT stays a static predicate, since it is a property of a location, and
the supertypes vehicle/physobj appear only in predicate signatures.

Problem conversion: each object takes the type of its unary type fact, those
facts leave :init, everything else is kept verbatim. --check walks seeded
random rollouts on an untyped problem and its typed twin and fails unless the
applicable ground actions are identical at every step.
"""
import argparse
import os
import random
import re
import sys

TYPE_FACTS = {"logistics": {"obj": "obj", "truck": "truck", "airplane": "airplane",
                            "location": "location", "city": "city"},
              "grid": {"place": "place", "key": "key", "shape": "shape"}}

DOMAINS = {
"logistics": """(define (domain logistics-strips)
  (:requirements :strips :typing)
  ;; Typed spelling of logistics_ipcc.pddl (tools/convert_to_typed.py). pddlgym's
  ;; parser splits parameter names on '-', so ?loc-from/?loc-to became ?from/?to. Action
  ;; parameters use flat leaf types; vehicle/physobj only type predicate arguments.
  (:types obj vehicle - physobj
          truck airplane - vehicle
          location city - object)
  (:predicates (airport ?l - location)
               (at ?o - physobj ?l - location)
               (in ?o - obj ?v - vehicle)
               (in-city ?l - location ?c - city))

(:action LOAD-TRUCK
  :parameters (?obj - obj ?truck - truck ?loc - location)
  :precondition (and (at ?truck ?loc) (at ?obj ?loc))
  :effect (and (not (at ?obj ?loc)) (in ?obj ?truck)))

(:action LOAD-AIRPLANE
  :parameters (?obj - obj ?airplane - airplane ?loc - location)
  :precondition (and (at ?obj ?loc) (at ?airplane ?loc))
  :effect (and (not (at ?obj ?loc)) (in ?obj ?airplane)))

(:action UNLOAD-TRUCK
  :parameters (?obj - obj ?truck - truck ?loc - location)
  :precondition (and (at ?truck ?loc) (in ?obj ?truck))
  :effect (and (not (in ?obj ?truck)) (at ?obj ?loc)))

(:action UNLOAD-AIRPLANE
  :parameters (?obj - obj ?airplane - airplane ?loc - location)
  :precondition (and (in ?obj ?airplane) (at ?airplane ?loc))
  :effect (and (not (in ?obj ?airplane)) (at ?obj ?loc)))

(:action DRIVE-TRUCK
  :parameters (?truck - truck ?from - location ?to - location ?city - city)
  :precondition (and (at ?truck ?from) (in-city ?from ?city) (in-city ?to ?city))
  :effect (and (not (at ?truck ?from)) (at ?truck ?to)))

(:action FLY-AIRPLANE
  :parameters (?airplane - airplane ?from - location ?to - location)
  :precondition (and (airport ?from) (airport ?to) (at ?airplane ?from))
  :effect (and (not (at ?airplane ?from)) (at ?airplane ?to)))
)
""",
"grid": """(define (domain grid)
(:requirements :strips :typing)
;; Typed spelling of grid_ipcc.pddl (tools/convert_to_typed.py); flat types.
(:types place key shape)
(:predicates (conn ?x - place ?y - place)
             (key-shape ?k - key ?s - shape)
             (lock-shape ?x - place ?s - shape)
             (at ?r - key ?x - place)
             (at-robot ?x - place)
             (locked ?x - place)
             (holding ?k - key)
             (open ?x - place)
             (arm-empty))

(:action unlock
:parameters (?curpos - place ?lockpos - place ?key - key ?shape - shape)
:precondition (and (conn ?curpos ?lockpos) (key-shape ?key ?shape)
                   (lock-shape ?lockpos ?shape) (at-robot ?curpos)
                   (locked ?lockpos) (holding ?key))
:effect (and (open ?lockpos) (not (locked ?lockpos))))

(:action move
:parameters (?curpos - place ?nextpos - place)
:precondition (and (at-robot ?curpos) (conn ?curpos ?nextpos) (open ?nextpos))
:effect (and (at-robot ?nextpos) (not (at-robot ?curpos))))

(:action pickup
:parameters (?curpos - place ?key - key)
:precondition (and (at-robot ?curpos) (at ?key ?curpos) (arm-empty))
:effect (and (holding ?key) (not (at ?key ?curpos)) (not (arm-empty))))

(:action pickup-and-loose
:parameters (?curpos - place ?newkey - key ?oldkey - key)
:precondition (and (at-robot ?curpos) (holding ?oldkey) (at ?newkey ?curpos))
:effect (and (holding ?newkey) (at ?oldkey ?curpos)
        (not (holding ?oldkey)) (not (at ?newkey ?curpos))))

(:action putdown
:parameters (?curpos - place ?key - key)
:precondition (and (at-robot ?curpos) (holding ?key))
:effect (and (arm-empty) (at ?key ?curpos) (not (holding ?key)))))
""",
}


def _balanced(text, start):
    """Index just past the parenthesised expression starting at text[start]."""
    depth = 0
    for i in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return i + 1
    raise ValueError("unbalanced parentheses")


def convert_problem(domain, text):
    """Typed spelling of one untyped problem: objects typed by their unary
    type fact, which leaves :init; everything else verbatim."""
    types = TYPE_FACTS[domain]
    low = text.lower()
    i_obj = low.index("(:objects")
    e_obj = _balanced(low, i_obj)
    i_init = low.index("(:init")
    e_init = _balanced(low, i_init)
    objects = text[i_obj + len("(:objects"):e_obj - 1].split()
    init = text[i_init:e_init]
    pattern = re.compile(r"\(\s*(" + "|".join(types) + r")\s+([^\s()]+)\s*\)", re.I)
    of_type = {}
    for t, o in pattern.findall(init):
        if o.lower() in of_type and of_type[o.lower()] != types[t.lower()]:
            raise ValueError(f"object {o} has two type facts")
        of_type[o.lower()] = types[t.lower()]
    missing = [o for o in objects if o.lower() not in of_type]
    if missing:
        raise ValueError(f"objects without a type fact: {missing[:5]}")
    by_type = {}
    for o in objects:
        by_type.setdefault(of_type[o.lower()], []).append(o)
    obj_block = "(:objects\n" + "\n".join(f"    {' '.join(v)} - {t}" for t, v in by_type.items()) + ")"
    new_init = pattern.sub("", init)
    new_init = re.sub(r"\n\s*\n", "\n", new_init)
    out = text[:i_obj] + obj_block + text[e_obj:i_init] + new_init + text[e_init:]
    out = re.sub(r"\(:domain\s+[^\s)]+\)", "(:domain " + ("logistics-strips" if domain == "logistics" else "grid") + ")",
                 out, count=1)
    return out


def convert_dir(domain, src, dst):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src)):
        if f.endswith(".pddl") and "domain" not in f:
            with open(os.path.join(src, f)) as fh:
                typed = convert_problem(domain, fh.read())
            with open(os.path.join(dst, f), "w") as fh:
                fh.write(typed)
            n += 1
    return n


def check(domain, untyped_domain_file, typed_domain_file, untyped_dir, typed_dir, problems=5, steps=40, seed=0):
    """Applicable ground actions identical at every step of seeded rollouts."""
    from pddlgym.core import PDDLEnv
    a = PDDLEnv(untyped_domain_file, untyped_dir, operators_as_actions=True, dynamic_action_space=True)
    b = PDDLEnv(typed_domain_file, typed_dir, operators_as_actions=True, dynamic_action_space=True)
    assert len(a.problems) == len(b.problems)
    name = lambda lit: (lit.predicate.name.lower(), tuple(str(v).split(":")[0] for v in lit.variables))
    rng = random.Random(seed)
    idxs = rng.sample(range(len(a.problems)), min(problems, len(a.problems)))
    for i in idxs:
        a.fix_problem_index(i)
        b.fix_problem_index(i)
        sa, _ = a.reset()
        sb, _ = b.reset()
        for step in range(steps):
            ga = {name(g): g for g in a.action_space.all_ground_literals(sa)}
            gb = {name(g): g for g in b.action_space.all_ground_literals(sb)}
            if set(ga) != set(gb):
                raise AssertionError(f"problem {i} step {step}: applicable actions differ: "
                                     f"{sorted(set(ga) ^ set(gb))[:4]}")
            if not ga:
                break
            k = rng.choice(sorted(ga))
            sa, _, da, _ = a.step(ga[k])
            sb, _, db, _ = b.step(gb[k])
            if da != db:
                raise AssertionError(f"problem {i} step {step}: goal reached in one spelling only")
            if da:
                break
    return len(idxs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    ap.add_argument("--write-domain")
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--check", nargs=4, metavar=("UNTYPED_DOMAIN", "TYPED_DOMAIN", "UNTYPED_DIR", "TYPED_DIR"))
    args = ap.parse_args()
    if args.write_domain:
        with open(args.write_domain, "w") as f:
            f.write(DOMAINS[args.domain])
        print(f"wrote {args.write_domain}")
    if args.src and args.dst:
        print(f"converted {convert_dir(args.domain, args.src, args.dst)} problems -> {args.dst}")
    if args.check:
        n = check(args.domain, *args.check)
        print(f"ok: identical applicable actions on {n} problems")


if __name__ == "__main__":
    sys.exit(main())

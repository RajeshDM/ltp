"""Dependency-free tests for the non-learned baseline scorers
(tools/random_policy_baseline.py --policy achieve|features): the ranking
uses goal achievement, goal deletion, schema goal distance and goal-relevant
arguments, in that order, and nothing else."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from random_policy_baseline import make_scorer  # noqa: E402
from test_lifted_layer import (MockLiteral, MockPred,  # noqa: E402
                               blocks_action_space)


class _Space:
    def __init__(self, ops):
        self._action_predicate_to_operators = ops


class _Env:
    def __init__(self, ops):
        self.action_space = _Space(ops)


class _Goal:
    def __init__(self, literals):
        self.literals = literals


class _State:
    def __init__(self, literals, goal):
        self.literals = literals
        self.goal = _Goal(goal)


def _setup(goal_atoms, state_atoms):
    ops = blocks_action_space()
    preds = {l.predicate.name: l.predicate
             for op in ops.values() for l in op.preconds.literals}
    lit = lambda name, args: MockLiteral(preds[name], list(args))
    state = _State([lit(n, a) for n, a in state_atoms],
                   [lit(n, a) for n, a in goal_atoms])
    schema = {p.name: p for p in ops}
    ground = lambda name, args: MockLiteral(schema[name], list(args))
    return _Env(ops), state, ground


def test_random_policy_has_no_scorer():
    env, state, _ = _setup([("on", "ab")], [])
    assert make_scorer("random", env, state) is None


def test_achieving_an_open_goal_ranks_first():
    # goal on(a,b); stack(a,b) adds it, stack(b,a) does not.
    env, state, ground = _setup([("on", "ab")], [("holding", "a"), ("clear", "b")])
    for policy in ("achieve", "features"):
        score = make_scorer(policy, env, state)
        assert score(ground("stack", "ab"), state) > score(ground("stack", "ba"), state)


def test_satisfied_goal_is_not_an_achievement_and_deleting_it_is_penalized():
    # on(a,b) is the goal and already true: stacking it again achieves nothing,
    # and unstack(a,b) deletes a satisfied goal atom.
    env, state, ground = _setup([("on", "ab")], [("on", "ab"), ("clear", "a")])
    score = make_scorer("features", env, state)
    assert score(ground("stack", "ab"), state)[0] == 0
    assert score(ground("unstack", "ab"), state)[1] == -1
    assert score(ground("unstack", "ba"), state)[1] == 0
    assert score(ground("unstack", "ba"), state) > score(ground("unstack", "ab"), state)


def test_goal_distance_then_goal_relevant_arguments_break_ties():
    # goal holding(a): unstack adds holding (distance 0), stack enables it (1).
    env, state, ground = _setup([("holding", "a")], [("on", "cd")])
    score = make_scorer("features", env, state)
    near, far = score(ground("unstack", "cd"), state), score(ground("stack", "cd"), state)
    assert near[:2] == far[:2] == (0, 0) and near[2] == 0 and far[2] == -1
    assert near > far
    # Same schema, neither achieves: the one touching the goal object wins.
    assert score(ground("stack", "ac"), state) > score(ground("stack", "cd"), state)


def test_a_schema_with_a_single_effect_is_a_bare_literal_not_a_conjunction():
    """pddlgym leaves one-effect schemas unwrapped (Miconic's board): the scorer
    crashed on `op.effects.literals` on the first real run."""
    env, state, ground = _setup([("holding", "a")], [("on", "cd")])
    ops = env.action_space._action_predicate_to_operators
    unstack = next(op for p, op in ops.items() if p.name == "unstack")
    adds_holding = next(l for l in unstack.effects.literals if l.predicate.name == "holding")
    unstack.effects = adds_holding                      # bare Literal: no .literals
    assert not hasattr(unstack.effects, "literals")
    score = make_scorer("achieve", env, state)
    assert score(ground("unstack", "ab"), state) == (1,)
    # a single-atom goal is bare too
    state.goal = state.goal.literals[0]
    assert make_scorer("achieve", env, state)(ground("unstack", "ab"), state) == (1,)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

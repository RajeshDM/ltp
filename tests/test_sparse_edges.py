"""Sparse edge accumulation must be identical to the dense array it replaced.

`_state_to_graph_ltp` used to build a dense
(max_action_arity, num_nodes, num_nodes, num_edge_features) array and scan it
with np.argwhere. That is O(arity * N^2 * k) per state, which dominated
test-time re-featurization. The replacement stores only the edges that exist.
The output must be byte-identical: same edges, same order, same values.

Dependency-free (numpy only): `_EdgeSlice` is re-declared here rather than
imported, because importing ploi.datautils_ltp pulls in torch and pddlgym.
Keep the copy in sync with the original if either changes.
"""

import random
import sys

import numpy as np


class _EdgeSlice:
    """Copy of ploi.datautils_ltp._EdgeSlice."""

    __slots__ = ('rows', 'width')

    def __init__(self, width):
        self.rows = {}
        self.width = width

    def __setitem__(self, key, value):
        sender, receiver, feature = key
        row = self.rows.get((sender, receiver))
        if row is None:
            row = self.rows[(sender, receiver)] = {}
        row[feature] = value

    def __getitem__(self, key):
        sender, receiver = key
        return self.rows.get((sender, receiver))


def _dense_emit(stack, num_edge_features):
    """The original output pass, over a dense (arity, N, N, k) array."""
    senders, receivers, edges = [], [], []
    for plane in stack:
        adjacency = np.any(plane, axis=2)
        for sender, receiver in np.argwhere(adjacency):
            senders.append(int(sender))
            receivers.append(int(receiver))
            edges.append(np.asarray(plane[sender, receiver]))
    return senders, receivers, edges


def _sparse_emit(stack, num_edge_features):
    """The replacement output pass; mirrors _state_to_graph_ltp."""
    present = []
    for slice_ in stack:
        for key in sorted(slice_.rows):
            row = slice_.rows[key]
            if any(row.values()):
                present.append((key, row))
    n_edge = len(present)
    edges_arr = np.zeros((n_edge, num_edge_features), dtype=np.uint8)
    senders, receivers = [], []
    for i, ((sender, receiver), row) in enumerate(present):
        senders.append(int(sender))
        receivers.append(int(receiver))
        for feature, value in row.items():
            edges_arr[i, feature] = value
    return senders, receivers, list(edges_arr)


def _random_writes(rng, num_nodes, arity, num_features, count):
    """Symmetric writes, the only pattern the feature writers produce."""
    writes = []
    for _ in range(count):
        position = rng.randrange(arity)
        a = rng.randrange(num_nodes)
        b = rng.randrange(num_nodes)
        if a == b:
            continue
        feature = rng.randrange(num_features)
        writes.append((position, a, b, feature))
        writes.append((position, b, a, feature))
    return writes


def test_sparse_emit_matches_dense_emit():
    rng = random.Random(0)
    for trial in range(25):
        num_nodes = rng.randint(4, 40)
        arity = rng.randint(1, 6)
        num_features = rng.randint(3, 24)
        dense = np.zeros((arity, num_nodes, num_nodes, num_features))
        sparse = [_EdgeSlice(num_features) for _ in range(arity)]

        for position, a, b, feature in _random_writes(
                rng, num_nodes, arity, num_features, rng.randint(1, 200)):
            dense[position, a, b, feature] = 1
            sparse[position][a, b, feature] = 1

        d_s, d_r, d_e = _dense_emit(dense, num_features)
        s_s, s_r, s_e = _sparse_emit(sparse, num_features)

        assert d_s == s_s, f"trial {trial}: sender order differs"
        assert d_r == s_r, f"trial {trial}: receiver order differs"
        assert len(d_e) == len(s_e), f"trial {trial}: edge count differs"
        for i, (de, se) in enumerate(zip(d_e, s_e)):
            assert np.array_equal(de, se), f"trial {trial}: edge {i} differs"
    print("PASS test_sparse_emit_matches_dense_emit")


def test_dedup_drops_the_same_edges():
    """Two flags and nothing else means "not a candidate at this slot"."""
    rng = random.Random(1)
    for trial in range(25):
        num_nodes = rng.randint(4, 30)
        arity = rng.randint(1, 4)
        num_features = rng.randint(4, 16)
        dense = np.zeros((arity, num_nodes, num_nodes, num_features))
        sparse = [_EdgeSlice(num_features) for _ in range(arity)]

        pairs = set()
        for _ in range(rng.randint(1, 30)):
            a, b = rng.randrange(num_nodes), rng.randrange(num_nodes)
            if a == b:
                continue
            pairs.add((a, b))
            # One, two or three flags: only the two-flag case is dropped.
            for feature in rng.sample(range(num_features), rng.randint(1, 3)):
                dense[:, a, b, feature] = 1
                dense[:, b, a, feature] = 1
                for position in range(arity):
                    sparse[position][a, b, feature] = 1
                    sparse[position][b, a, feature] = 1

        for a, b in sorted(pairs):
            for position in range(arity):
                assert (dense[position, a, b] == dense[position, b, a]).all()
                if sum(dense[position, a, b]) == 2:
                    dense[position, a, b] = [0] * num_features
                    dense[position, b, a] = [0] * num_features
                rows = sparse[position].rows
                fwd, rev = rows.get((a, b)), rows.get((b, a))
                assert fwd == rev
                if fwd is not None and sum(fwd.values()) == 2:
                    del rows[(a, b)]
                    del rows[(b, a)]

        d_s, d_r, d_e = _dense_emit(dense, num_features)
        s_s, s_r, s_e = _sparse_emit(sparse, num_features)
        assert d_s == s_s and d_r == s_r, f"trial {trial}: order differs after dedup"
        for i, (de, se) in enumerate(zip(d_e, s_e)):
            assert np.array_equal(de, se), f"trial {trial}: edge {i} differs after dedup"
    print("PASS test_dedup_drops_the_same_edges")


def test_repeated_writes_are_idempotent():
    """Writers set the same flag more than once; the result must not change."""
    slice_ = _EdgeSlice(8)
    slice_[1, 2, 3] = 1
    slice_[1, 2, 3] = 1
    slice_[1, 2, 5] = 1
    assert slice_[1, 2] == {3: 1, 5: 1}
    assert slice_[2, 1] is None, "writers must set both directions explicitly"
    print("PASS test_repeated_writes_are_idempotent")


def test_empty_slice_emits_nothing():
    senders, receivers, edges = _sparse_emit([_EdgeSlice(6), _EdgeSlice(6)], 6)
    assert senders == [] and receivers == [] and edges == []
    print("PASS test_empty_slice_emits_nothing")


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print("All sparse edge tests passed.")

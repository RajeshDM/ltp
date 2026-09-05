"""The batched decoder's row->graph mapping, which crashed on mixed arity.

`ao_scores` carries one row per (graph, parameter slot). The number of rows a
graph contributes is NOT uniform, so mapping a row back to its graph has to go
through `n_parameters` - which is exactly what `compute_object_scores` has
always done, and what `get_best_action_object_scores_locations` and
`beam_search_parallel`'s `parameter_locations` did not.

The failure it produced, from logs/loo8_union_no_miconic_run.log:

    get_best_object_embeddings_ltp
    RuntimeError: The size of tensor a (200) must match the size of tensor b
                  (67) at non-singleton dimension 0

200 graphs in the batch; the stride `arange(0, rows, max_number_action_
parameters)` recovered 67 of them.

Two properties are pinned here:
  1. with n_parameters, the map is exact on a mixed-arity batch;
  2. without it - and whenever rows-per-graph IS uniform - behaviour is
     unchanged, so no previously-working run moves.

Run: python tests/test_beam_parallel_indexing.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:                                    # pragma: no cover
    print("SKIP: torch not available")
    sys.exit(0)

from ploi.modelutils_ltp import GNN_GRU


class _Stub:
    """The method reads only these two attributes off self."""
    def __init__(self, max_params):
        self.max_number_action_parameters = max_params


locate = GNN_GRU.get_best_action_object_scores_locations


def _scores(rows, cols, best_col_per_row):
    """One clearly-winning column per row, so top-1 is unambiguous."""
    s = torch.full((rows, cols), -1.0)
    for r, c in enumerate(best_col_per_row):
        s[r, c] = 10.0
    return s


def test_mixed_arity_map_is_exact():
    # Three graphs contributing 2, 3 and 1 parameter rows: 6 rows total.
    n_parameters = torch.tensor([2, 3, 1])
    n_node = torch.tensor([10, 20, 30])
    n_objects = torch.tensor([4, 5, 6])
    # Row r wants column r; every column is a legal object in its graph.
    ao = _scores(6, 8, [0, 1, 2, 3, 4, 5])

    idx, _val = locate(_Stub(3), ao_scores=ao, n_node=n_node, k=1,
                       n_objects=n_objects, n_parameters=n_parameters)
    assert idx.shape[0] == 6, f"one row out per row in: {idx.shape}"
    assert [int(v) for v in idx[:, 0]] == [0, 1, 2, 3, 4, 5], idx[:, 0]

    # The map must respect PER-GRAPH object counts. Graph 0 has 4 objects, so
    # a high score in column 6 of one of its rows must not be selectable.
    ao2 = _scores(6, 8, [6, 1, 2, 3, 4, 5])          # row 0 -> illegal column
    idx2, _ = locate(_Stub(3), ao_scores=ao2, n_node=n_node, k=1,
                     n_objects=n_objects, n_parameters=n_parameters)
    assert int(idx2[0, 0]) < 4, (
        f"row 0 belongs to graph 0 (4 objects); selected {int(idx2[0,0])}")
    print("ok  mixed-arity row->graph map is exact and respects object counts")


def test_uniform_case_matches_legacy():
    # Four graphs x 3 rows each: the case the old stride handled correctly.
    n_parameters = torch.tensor([3, 3, 3, 3])
    n_node = torch.tensor([10, 10, 10, 10])
    n_objects = torch.tensor([5, 5, 5, 5])
    ao = _scores(12, 8, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1])

    legacy, _ = locate(_Stub(3), ao_scores=ao, n_node=n_node, k=2,
                       n_objects=n_objects)                     # no n_parameters
    fixed, _ = locate(_Stub(3), ao_scores=ao, n_node=n_node, k=2,
                      n_objects=n_objects, n_parameters=n_parameters)
    assert torch.equal(legacy, fixed), "the fix changed a case the old code got right"
    print("ok  uniform batches are byte-identical to the legacy path")


def test_disagreement_is_reported():
    n_parameters = torch.tensor([2, 2])          # sums to 4
    ao = _scores(5, 8, [0, 1, 2, 3, 4])          # but 5 rows
    try:
        locate(_Stub(3), ao_scores=ao, n_node=torch.tensor([9, 9]), k=1,
               n_objects=torch.tensor([4, 4]), n_parameters=n_parameters)
    except RuntimeError as exc:
        assert "n_parameters sums to" in str(exc), exc
        print("ok  a row-count disagreement raises with both numbers")
        return
    raise AssertionError("expected a RuntimeError naming the mismatch")


def test_parameter_locations_formula():
    """The slot-row selection inside beam_search_parallel.

    Mirrors the patched arithmetic. The old form was
    `arange(parameter_number, rows, max_number_action_parameters)`, which on
    the failing batch (200 graphs, one row each, cap 3) returned 67 entries
    against n_node's 200.
    """
    def locations(n_parameters, parameter_number):
        n = n_parameters.to(torch.long)
        starts = torch.cumsum(n, 0) - n
        offs = torch.minimum(torch.full_like(n, parameter_number),
                             (n - 1).clamp(min=0))
        return (starts + offs).long()

    crash = torch.ones(200, dtype=torch.long)                 # the real batch
    assert len(torch.arange(0, int(crash.sum()), 3)) == 67    # the old answer
    assert locations(crash, 0).numel() == 200                 # the new one
    assert torch.equal(locations(crash, 0), torch.arange(200))

    uniform = torch.tensor([3, 3, 3, 3])
    for p in range(3):
        assert torch.equal(locations(uniform, p),
                           torch.arange(p, 12, 3)), f"slot {p} moved"

    mixed = torch.tensor([2, 3, 1])
    assert torch.equal(locations(mixed, 0), torch.tensor([0, 2, 5]))
    # Graph 2 has one row; slots past its arity clamp into it rather than
    # running into the next graph's rows. Its score is frozen by done_prev.
    assert torch.equal(locations(mixed, 1), torch.tensor([1, 3, 5]))
    assert torch.equal(locations(mixed, 2), torch.tensor([1, 4, 5]))
    print("ok  parameter_locations: 200 rows recovered, uniform case unmoved")


if __name__ == '__main__':
    test_mixed_arity_map_is_exact()
    test_uniform_case_matches_legacy()
    test_disagreement_is_reported()
    test_parameter_locations_formula()
    print("\nall passed")

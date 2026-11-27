#!/usr/bin/env python3
"""Dependency-free tests for tools/indomain_table.py: which dump a cell is read from.

    python tests/test_indomain_table.py

A checkpoint that is the best under two metrics is tested once. Dumps written
before 2026-11-13 left the second metric's cell EMPTY, and the table then took
that cell from an older dump - another checkpoint's number. These pin that the
cell is recovered from the same dump, and that an unattributable one is
reported instead of silently replaced.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from indomain_table import collect, in_domain_cells  # noqa: E402


def _cell(cov, epoch):
    return [{"epoch": epoch, "metrics": {"PlannerType.LEARNED_MODEL":
                                         {"success_rate_with_monitor": cov}}}]


def _dump(ts, metrics, cells, exp="loo8_joint_chain_no_grid"):
    """cells: {metric: (cov, epoch) or None for an empty (skipped) cell}."""
    return {"experiment": exp, "timestamp": ts, "test_model_metrics": metrics,
            "eval_plan": [{"domain": "Miconic_ipcc", "zero_shot": False, "split": "test"},
                          {"domain": "Grid_ipcc", "zero_shot": True, "split": "test"}],
            "results": {f"Miconic_ipcc_{m}": (_cell(*v) if v else [])
                        for m, v in cells.items()}}


def _write(root, dump):
    d = os.path.join(root, dump["experiment"])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"results_x_{dump['timestamp']}.json"), "w") as f:
        json.dump(dump, f)


ORDER = ["training", "combined", "validation"]


def test_a_filled_cell_is_read_directly():
    found, amb = in_domain_cells(
        _dump("20261101_000000", ORDER, {"training": (0.9, 480), "combined": (0.8, 470),
                                         "validation": (0.7, 300)}), "combined")
    assert found == {"miconic_ipcc": (0.8, 470, "combined")} and not amb


def test_a_skipped_cell_is_recovered_from_the_metric_that_tested_it():
    found, amb = in_domain_cells(
        _dump("20261101_000000", ORDER, {"training": (0.9, 480), "combined": None,
                                         "validation": (0.7, 300)}), "combined")
    assert found == {"miconic_ipcc": (0.9, 480, "training")} and not amb


def test_a_skipped_cell_with_two_candidate_epochs_is_not_guessed():
    order = ["training", "validation", "combined"]
    found, amb = in_domain_cells(
        _dump("20261101_000000", order, {"training": (0.9, 480), "validation": (0.7, 300),
                                         "combined": None}), "combined")
    assert found == {} and amb == ["miconic_ipcc"]


def test_a_metric_the_run_did_not_request_is_simply_absent():
    found, amb = in_domain_cells(
        _dump("20261101_000000", ["validation"], {"validation": (0.7, 300)}), "combined")
    assert found == {} and not amb


def test_the_newer_dump_wins_over_an_older_one_with_a_different_checkpoint():
    """The failure this guards: an eval of a half-trained model (E200) followed
    by the final eval, whose combined checkpoint was also the training one."""
    with tempfile.TemporaryDirectory() as root:
        _write(root, _dump("20260910_000000", ORDER, {"training": (0.3, 200), "combined": (0.2, 190),
                                                      "validation": (0.1, 150)}))
        _write(root, _dump("20260920_000000", ORDER, {"training": (0.9, 480), "combined": None,
                                                      "validation": (0.7, 300)}))
        notes = []
        cells = collect(root, "combined", "20260905", notes)
        assert cells["loo8_joint_chain_no_grid"]["miconic_ipcc"] == (0.9, 480, "20260920_000000")
        assert len(notes) == 1 and "read from another metric" in notes[0]


def test_an_unattributable_newer_dump_is_reported_not_papered_over():
    order = ["training", "validation", "combined"]
    with tempfile.TemporaryDirectory() as root:
        _write(root, _dump("20260910_000000", order, {"training": (0.3, 200), "validation": (0.1, 150),
                                                      "combined": (0.2, 190)}))
        _write(root, _dump("20260920_000000", order, {"training": (0.9, 480), "validation": (0.7, 300),
                                                      "combined": None}))
        notes = []
        cells = collect(root, "combined", "20260905", notes)
        assert cells["loo8_joint_chain_no_grid"]["miconic_ipcc"][1] == 190
        assert any(n.startswith("STALE") and "20260920" in n for n in notes)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("\nall passed" if not failures else f"\n{failures} FAILED")
    sys.exit(1 if failures else 0)

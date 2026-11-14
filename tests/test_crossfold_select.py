"""Dependency-free tests: the cross-fold epoch rule never looks at the fold
it selects for (the paper's test-blindness claim), and parses real dumps."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from crossfold_select import select, zero_shot_test_coverage  # noqa: E402


def test_own_fold_never_influences_its_choice():
    folds = {"a": {0: 0.0, 10: 0.2, 20: 0.9},
             "b": {0: 0.1, 10: 0.8, 20: 0.1},
             "c": {0: 0.1, 10: 0.6, 20: 0.2}}
    before = select(folds)["a"][0]
    folds["a"] = {0: 1.0, 10: 0.0, 20: 0.0}   # change only a's own numbers
    assert select(folds)["a"][0] == before == 10


def test_ties_go_to_earliest_and_candidates_are_common_epochs():
    folds = {"a": {0: 0.0, 10: 0.0, 20: 0.0},
             "b": {10: 0.5, 20: 0.5},          # no epoch 0
             "c": {0: 0.9, 10: 0.5, 20: 0.5}}
    e_star, _, n, common = select(folds)["a"]
    assert (e_star, n, common) == (10, 2, [10, 20])


def test_reads_test_split_not_train_split():
    dump = {"eval_plan": [
                {"domain": "visitall_ipcc", "zero_shot": True, "split": "test"},
                {"domain": "visitall_ipcc@train", "zero_shot": True, "split": "train"}],
            "results": {
                "zeroshot_visitall_ipcc_periodic": [
                    {"epoch": 30, "metrics": {"PlannerType.LEARNED_MODEL":
                                              {"success_rate_with_monitor": 0.68}}}],
                "zeroshot_visitall_ipcc@train_periodic": [
                    {"epoch": 30, "metrics": {"PlannerType.LEARNED_MODEL":
                                              {"success_rate_with_monitor": 1.0}}}]}}
    assert zero_shot_test_coverage(dump) == {30: 0.68}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

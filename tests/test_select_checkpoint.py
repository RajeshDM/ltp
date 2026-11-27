"""tools/select_checkpoint.py: the reported checkpoint is test-blind.

Run: python tests/test_select_checkpoint.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from select_checkpoint import choose, val_coverage  # noqa: E402

LEARNED = "PlannerType.LEARNED_MODEL"


def dump(ts, rows):
    res = {}
    for key, epoch, cov in rows:
        res.setdefault(key, []).append({"epoch": epoch, "metrics": {LEARNED: {"success_rate_with_monitor": cov}}})
    return {"timestamp": ts, "results": res}


def test_mean_over_training_domains_ties_earlier():
    d = dump("1", [("A@val_ood_periodic", 50, 80.0), ("B@val_ood_periodic", 50, 40.0),
                   ("A@val_ood_periodic", 100, 60.0), ("B@val_ood_periodic", 100, 60.0),
                   ("A@val_ood_periodic", 150, 100.0), ("B@val_ood_periodic", 150, 20.0)])
    cov = val_coverage([d], ["A", "B"])
    assert choose(cov, ["A", "B"]) == (50, 60.0), choose(cov, ["A", "B"])
    print("ok  mean over training domains, ties go to the earlier epoch")


def test_test_splits_and_heldout_domain_never_count():
    d = dump("1", [("A@val_ood_periodic", 50, 10.0), ("A@hard_periodic", 50, 100.0),
                   ("zeroshot_H@val_ood_periodic", 100, 100.0), ("A@val_ood_periodic", 100, 20.0)])
    cov = val_coverage([d], ["A"])
    assert choose(cov, ["A"]) == (100, 20.0)
    print("ok  test bands and the held-out domain do not enter the choice")


def test_incomplete_epochs_ineligible_newest_dump_wins():
    old = dump("1", [("A@val_ood", 50, 90.0), ("B@val_ood", 50, 90.0)])
    new = dump("2", [("A@val_ood", 50, 10.0), ("A@val_ood", 100, 100.0)])
    cov = val_coverage([new, old], ["A", "B"])
    assert choose(cov, ["A", "B"]) == (50, 50.0), choose(cov, ["A", "B"])
    print("ok  an epoch missing a domain is ineligible; a newer dump replaces an older one")


if __name__ == "__main__":
    test_mean_over_training_domains_ties_earlier()
    test_test_splits_and_heldout_domain_never_count()
    test_incomplete_epochs_ineligible_newest_dump_wins()
    print("\nall passed")

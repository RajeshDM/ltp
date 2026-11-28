"""tools/select_checkpoint.py: the reported checkpoint is test-blind.

Run: python tests/test_select_checkpoint.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from select_checkpoint import choose, choose_by_loss, val_coverage  # noqa: E402

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


def test_ties_broken_by_lowest_val_loss_then_earlier():
    d = dump("1", [("A@val_ood", 50, 1.0), ("A@val_ood", 100, 1.0), ("A@val_ood", 150, 1.0),
                   ("A@val_ood", 200, 0.9)])
    cov = val_coverage([d], ["A"])
    loss = {50: 30.0, 100: 20.0, 150: 20.0, 200: 5.0}
    assert choose(cov, ["A"], loss) == (100, 1.0), choose(cov, ["A"], loss)   # 200 has worse coverage
    assert choose(cov, ["A"], {50: 1.0}) == (50, 1.0)                        # loss missing for a tie -> earliest
    print("ok  ties go to the lowest Val-OOD loss, then the earlier epoch; coverage still decides first")


def test_loss_rule_mean_over_domains_ties_earlier():
    losses = {50: {"A": 0.30, "B": 0.10}, 100: {"A": 0.10, "B": 0.20}, 150: {"A": 0.20, "B": 0.10}}
    assert choose_by_loss(losses, ["A", "B"]) == (100, 0.15000000000000002) or \
           choose_by_loss(losses, ["A", "B"])[0] == 100
    assert choose_by_loss({50: {"A": 1.0}, 100: {"A": 1.0}}, ["A"]) == (50, 1.0)
    print("ok  loss rule: lowest mean per-domain Val-OOD loss, ties go to the earlier epoch")


if __name__ == "__main__":
    test_loss_rule_mean_over_domains_ties_earlier()
    test_ties_broken_by_lowest_val_loss_then_earlier()
    test_mean_over_training_domains_ties_earlier()
    test_test_splits_and_heldout_domain_never_count()
    test_incomplete_epochs_ineligible_newest_dump_wins()
    print("\nall passed")

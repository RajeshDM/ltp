"""Compare two GABAR_TRACE_SCORES trace files for score-level equivalence.

Each trace line is one model call: {"epoch": E, "problem": P, "step": S,
"candidates": [[score, [action, obj, ...]], ...]}. Entries are aligned on
(epoch, problem, step) and every candidate's sequence and score is compared.

Usage:
    python compare_score_traces.py trace_a.jsonl trace_b.jsonl [--tol 1e-3]
        [--ignore-order]

--ignore-order compares the candidate lists as sets keyed by sequence
(score per sequence must still match) - use it when strict comparison only
shows equal-score candidates in a different order (tie reordering).

Exit code 0 = traces match, 1 = divergence found.

Temporary verification apparatus for the batched-eval migration - not part
of the normal workflow.
"""
import argparse
import json
import sys


def load_trace(path):
    entries = {}
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            key = (e.get("epoch"), e["problem"], e["step"])
            if key in entries:
                print(f"warning: duplicate key {key} in {path} (line {line_num}), keeping last")
            entries[key] = e["candidates"]
    return entries


def compare_candidates(a, b, tol, ignore_order):
    """Return None if the two candidate lists match, else a description."""
    if ignore_order:
        da = {tuple(seq): score for score, seq in a}
        db = {tuple(seq): score for score, seq in b}
        if set(da) != set(db):
            only_a = [list(s) for s in list(set(da) - set(db))[:3]]
            only_b = [list(s) for s in list(set(db) - set(da))[:3]]
            return (f"candidate sets differ (A-only: {only_a}, B-only: {only_b}, "
                    f"counts {len(da)} vs {len(db)})")
        for seq in da:
            if abs(da[seq] - db[seq]) > tol:
                return (f"score differs for {list(seq)}: "
                        f"{da[seq]:.6f} vs {db[seq]:.6f}")
        return None

    if len(a) != len(b):
        return f"candidate count differs: {len(a)} vs {len(b)}"
    for rank, ((sa, qa), (sb, qb)) in enumerate(zip(a, b)):
        if qa != qb:
            tie = " (scores equal - tie reorder? retry with --ignore-order)" \
                if abs(sa - sb) <= tol else ""
            return (f"rank {rank} sequence differs: {qa} vs {qb} "
                    f"(scores {sa:.6f} / {sb:.6f}){tie}")
        if abs(sa - sb) > tol:
            return f"rank {rank} score differs for {qa}: {sa:.6f} vs {sb:.6f}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_a")
    parser.add_argument("trace_b")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="max allowed |score difference| (default 1e-3)")
    parser.add_argument("--ignore-order", action="store_true",
                        help="compare candidates as sequence-keyed sets instead of rank order")
    args = parser.parse_args()

    trace_a = load_trace(args.trace_a)
    trace_b = load_trace(args.trace_b)

    all_keys = set(trace_a) | set(trace_b)
    problems = sorted({(k[0], k[1]) for k in all_keys},
                      key=lambda x: (str(x[0]), x[1]))

    calls_compared = 0
    diverged_problems = 0
    for epoch, problem in problems:
        steps = sorted(k[2] for k in all_keys if k[0] == epoch and k[1] == problem)
        first_divergence = None
        for s in steps:
            key = (epoch, problem, s)
            if key not in trace_a:
                first_divergence = (s, f"missing in {args.trace_a} "
                                       f"(A finished this problem earlier)")
                break
            if key not in trace_b:
                first_divergence = (s, f"missing in {args.trace_b} "
                                       f"(B finished this problem earlier)")
                break
            calls_compared += 1
            diff = compare_candidates(trace_a[key], trace_b[key],
                                      args.tol, args.ignore_order)
            if diff:
                first_divergence = (s, diff)
                break
        if first_divergence:
            diverged_problems += 1
            print(f"epoch {epoch} problem {problem}: FIRST DIVERGENCE at "
                  f"step {first_divergence[0]}: {first_divergence[1]}")
            print("  (later steps of this problem are expected to differ too - "
                  "the trajectory forks after the first divergence)")
        else:
            print(f"epoch {epoch} problem {problem}: OK "
                  f"({len(steps)} model calls, all candidates match)")

    print()
    print(f"Summary: {len(problems)} (epoch, problem) pairs, "
          f"{calls_compared} model calls compared, "
          f"{diverged_problems} diverged")
    return 1 if diverged_problems else 0


if __name__ == "__main__":
    sys.exit(main())

"""Compare GABAR_TRACE_SCORES trace files for score-level equivalence.

Each trace line is one model call: {"epoch": E, "problem": P, "step": S,
"candidates": [[score, [action, obj, ...]], ...]}. Entries are aligned on
(epoch, problem, step) and every candidate's sequence and score is compared.

Usage:
    # two files: detailed per-problem report
    python compare_score_traces.py a.jsonl b.jsonl [--tol 1e-3] [--ignore-order]

    # three or more files: pairwise match matrix + equivalence groups
    python compare_score_traces.py a.jsonl b.jsonl c.jsonl d.jsonl ...

--ignore-order compares the candidate lists as sets keyed by sequence
(score per sequence must still match) - use it when strict comparison only
shows equal-score candidates in a different order (tie reordering).

Exit code 0 = all traces match, 1 = divergence found.

Temporary verification apparatus for the batched-eval migration - not part
of the normal workflow.
"""
import argparse
import itertools
import json
import os
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


def compare_pair(trace_a, trace_b, tol, ignore_order, name_a="A", name_b="B"):
    """Compare two loaded traces.

    Returns (problems, calls_compared, divergences) where divergences is a
    list of (epoch, problem, first_divergent_step, description).
    """
    all_keys = set(trace_a) | set(trace_b)
    problems = sorted({(k[0], k[1]) for k in all_keys},
                      key=lambda x: (str(x[0]), x[1]))

    calls_compared = 0
    divergences = []
    for epoch, problem in problems:
        steps = sorted(k[2] for k in all_keys if k[0] == epoch and k[1] == problem)
        for s in steps:
            key = (epoch, problem, s)
            if key not in trace_a:
                divergences.append((epoch, problem, s,
                                    f"missing in {name_a} ({name_a} finished this problem earlier)"))
                break
            if key not in trace_b:
                divergences.append((epoch, problem, s,
                                    f"missing in {name_b} ({name_b} finished this problem earlier)"))
                break
            calls_compared += 1
            diff = compare_candidates(trace_a[key], trace_b[key], tol, ignore_order)
            if diff:
                divergences.append((epoch, problem, s, diff))
                break
    return problems, calls_compared, divergences


def report_pair_detailed(trace_a, trace_b, tol, ignore_order, name_a, name_b):
    problems, calls_compared, divergences = compare_pair(
        trace_a, trace_b, tol, ignore_order, name_a, name_b)
    diverged_keys = {(e, p) for e, p, _, _ in divergences}
    for epoch, problem in problems:
        match = next(((s, d) for e, p, s, d in divergences
                      if (e, p) == (epoch, problem)), None)
        if match:
            print(f"epoch {epoch} problem {problem}: FIRST DIVERGENCE at "
                  f"step {match[0]}: {match[1]}")
            print("  (later steps of this problem are expected to differ too - "
                  "the trajectory forks after the first divergence)")
        else:
            n_steps = len({k[2] for k in (set(trace_a) | set(trace_b))
                           if k[0] == epoch and k[1] == problem})
            print(f"epoch {epoch} problem {problem}: OK "
                  f"({n_steps} model calls, all candidates match)")
    print()
    print(f"Summary: {len(problems)} (epoch, problem) pairs, "
          f"{calls_compared} model calls compared, "
          f"{len(diverged_keys)} diverged")
    return 1 if diverged_keys else 0


def report_matrix(traces, names, tol, ignore_order):
    """Pairwise match matrix + equivalence groups for 3+ traces."""
    n = len(names)
    diverged_counts = {}
    for i, j in itertools.combinations(range(n), 2):
        problems, _, divergences = compare_pair(
            traces[i], traces[j], tol, ignore_order, names[i], names[j])
        diverged_counts[(i, j)] = (len({(e, p) for e, p, _, _ in divergences}),
                                   len(problems))

    width = max(len(nm) for nm in names) + 2
    print("Pairwise comparison (cell = diverged problems / total problems):")
    print(" " * width + "".join(nm.ljust(width) for nm in names))
    for i in range(n):
        row = names[i].ljust(width)
        for j in range(n):
            if i == j:
                cell = "-"
            else:
                d, t = diverged_counts[(min(i, j), max(i, j))]
                cell = "MATCH" if d == 0 else f"{d}/{t}"
            row += cell.ljust(width)
        print(row)

    # Equivalence groups via union-find on the "matches" relation
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (i, j), (d, _) in diverged_counts.items():
        if d == 0:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(names[i])

    print()
    print("Equivalence groups (traces that fully match each other):")
    for g_num, members in enumerate(sorted(groups.values(), key=len, reverse=True), 1):
        print(f"  Group {g_num}: {', '.join(members)}")
    print()
    print("For a per-problem divergence report, rerun with exactly the two "
          "trace files you want to inspect.")
    any_diverged = any(d for d, _ in diverged_counts.values())
    return 1 if any_diverged else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", help="two or more trace files")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="max allowed |score difference| (default 1e-3)")
    parser.add_argument("--ignore-order", action="store_true",
                        help="compare candidates as sequence-keyed sets instead of rank order")
    args = parser.parse_args()

    if len(args.traces) < 2:
        parser.error("need at least two trace files")

    names = []
    for p in args.traces:
        base = os.path.basename(p)
        names.append(base[:-6] if base.endswith(".jsonl") else base)
    loaded = [load_trace(p) for p in args.traces]

    if len(loaded) == 2:
        return report_pair_detailed(loaded[0], loaded[1], args.tol,
                                    args.ignore_order, names[0], names[1])
    return report_matrix(loaded, names, args.tol, args.ignore_order)


if __name__ == "__main__":
    sys.exit(main())

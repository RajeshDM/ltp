#!/usr/bin/env python3
"""Build candidate problems for the new validation / test splits (2026-11-27 protocol).

    python tools/build_splits.py --out /nfs/hpc/share/mangannr/di_ltp/splits_build
    python tools/build_splits.py --out ... --domains spanner,rovers --oversample 2

Training sets stay as they are (except Spanner, extended to 8 nuts). For each
domain and split this writes CANDIDATES to <out>/<domain>/<split>/, taken in
this order:
  1. existing test problems whose size falls in the band (keeps continuity
     with every number reported so far);
  2. the unloaded problem pools inside the pddlgym domain folders;
  3. the pddl-generators programs, with secondary parameters drawn from the
     ranges the existing problems use.
A candidate is skipped if its content (up to the problem name) equals any
training problem, any existing test problem not reused, or another candidate.
Nothing is planned here: tools/solve_split_candidates.py runs LAMA on the
candidates, and only solved ones are installed. The band definitions are the
contract; ICLR27/REVIEW_PLAN.md and RUNS_STATUS.md (2026-11-27) record them.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys

PDDL = "/nfs/hpc/share/mangannr/PLOI_full_repos/pddlgym/pddlgym/pddl"
GEN = "/nfs/hpc/share/mangannr/di_ltp/external/pddl-generators"

# split -> (low, high) of the domain's main size measure, inclusive; counts below.
SPLITS = {
    "manyblocks": dict(dir="manyblocks_ipcc_big", unit="blocks",
                       val_id=(6, 9), val_ood=(10, 10), easy=(11, 20), medium=(21, 30), hard=(31, 40)),
    "gripper": dict(dir="gripper_ipcc", unit="balls",
                    val_id=(5, 25), val_ood=(26, 30), easy=(31, 50), medium=(51, 75), hard=(76, 100)),
    "miconic": dict(dir="miconic_ipcc", unit="passengers",
                    val_id=(1, 12), val_ood=(13, 19), easy=(20, 40), medium=(41, 60), hard=(61, 100)),
    "visitall": dict(dir="visitall_ipcc", unit="cells",
                     val_id=(9, 49), val_ood=(50, 64), easy=(65, 144), medium=(145, 225), hard=(226, 400)),
    "grid": dict(dir="grid_ipcc", unit="cells",
                 val_id=(9, 63), val_ood=(64, 80), easy=(81, 100), medium=(101, 125), hard=(126, 154)),
    "logistics": dict(dir="logistics_ipcc", unit="cities",
                      val_id=(5, 13), val_ood=(14, 15), easy=(16, 20), medium=(21, 25), hard=(26, 30)),
    # Spanner: training is extended to 8 nuts ("train_ext"), so validation starts at 9.
    "spanner": dict(dir="spanner_ipcc", unit="nuts", train_ext=(8, 8),
                    val_id=(3, 8), val_ood=(9, 10), easy=(11, 12), medium=(13, 14), hard=(15, 16)),
    "rovers": dict(dir="rovers_ipcc", unit="waypoints",
                   val_id=(3, 12), val_ood=(13, 15), easy=(16, 30), medium=(31, 50), hard=(51, 70)),
}
COUNTS = dict(val_id=50, val_ood=50, easy=50, medium=50, hard=50, train_ext=40)
TEST_SPLITS = ("easy", "medium", "hard")
POOLS = {"manyblocks": ["manyblocks_ipcc_big_test/all_problem_data", "manyblocks_ipcc_big/all_data"],
         "gripper": ["gripper_ipcc/complex_traning_data", "gripper_ipcc_test/simple_test",
                     "gripper_ipcc_test/complex_test"],
         "logistics": ["logistics_ipcc/training_data_trial_2"]}


# ─── reading problems ────────────────────────────────────────────────────────

def _text(path):
    with open(path) as f:
        return f.read()


def content_hash(path):
    t = re.sub(r"\s+", " ", _text(path).lower())
    t = re.sub(r"\(define \(problem [^)]*\)", "", t)
    return hashlib.md5(t.encode()).hexdigest()


def typed_objects(text):
    m = re.search(r"\(:objects(.*?)\)", text.lower(), re.S)
    c = collections.Counter()
    if not m:
        return c
    toks, pend, i = m.group(1).split(), [], 0
    while i < len(toks):
        if toks[i] == "-":
            c[toks[i + 1]] += len(pend)
            pend, i = [], i + 2
        else:
            pend.append(toks[i])
            i += 1
    c["_untyped"] += len(pend)
    return c


def _unary_init(text, pred):
    init = text.lower().split(":goal")[0]
    return len(re.findall(r"\(\s*" + pred + r"\s+[^\s()]+\s*\)", init))


def size_of(domain, path):
    """The domain's main size measure, read from the problem file itself."""
    t = _text(path)
    o = typed_objects(t)
    if domain == "manyblocks":
        return o.get("block", 0) or o["_untyped"]
    if domain == "gripper":
        return o.get("object", 0)
    if domain == "miconic":
        return o.get("passenger", 0) or _unary_init(t, "passenger")
    if domain == "visitall":
        return o.get("place", 0) or len(re.findall(r"\bloc-x\d+-y\d+\b", t.split("(:init")[0]))
    if domain == "grid":
        return _unary_init(t, "place")
    if domain == "logistics":
        return _unary_init(t, "city")
    if domain == "spanner":
        return o.get("nut", 0)
    if domain == "rovers":
        return o.get("waypoint", 0)
    raise KeyError(domain)


def problems_in(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.endswith(".pddl") and "domain" not in f)


# ─── generators (secondary parameters follow the existing problems) ─────────

def _run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       universal_newlines=True, timeout=120)
    if r.returncode != 0 or "(define" not in r.stdout:
        return None
    return r.stdout


def generate(domain, split, size, rng):
    """(filename, pddl text) for one random problem of main size `size`."""
    seed = rng.randrange(1, 10 ** 6)
    test_like = split in TEST_SPLITS or split == "val_ood"
    if domain == "gripper":
        rooms = rng.randint(10, 24) if test_like else rng.randint(5, 17)
        return (f"pfile_1_{rooms}_{size}_s{seed}.pddl",
                _run([f"{GEN}/grippers/grippers", "-n", "1", "-r", str(rooms), "-o", str(size), "-s", str(seed)]))
    if domain == "miconic":
        floors = rng.randint(11, 29) if test_like else rng.randint(2, 20)
        return (f"pfile_p-{size}_f-{floors}_v{seed}.pddl",
                _run([f"{GEN}/miconic/miconic", "-f", str(floors), "-p", str(size), "-r", str(seed)]))
    if domain == "visitall":
        # x*y == size exactly is rarely possible; pick the most square factor pair
        # with x, y >= 3 among sizes near the target, then report its true size.
        pairs = [(x, size // x) for x in range(3, 21) if size % x == 0 and 3 <= size // x <= 20]
        if not pairs:
            return None, None
        x, y = rng.choice(pairs)
        return (f"pfile_x-{x}_y-{y}_r-50_s-{seed}.pddl",
                _run([f"{GEN}/visitall/grid", "-x", str(x), "-y", str(y), "-r", "0.5", "-s", str(seed)]))
    if domain == "grid":
        pairs = [(x, size // x) for x in range(3, 15) if size % x == 0 and 3 <= size // x <= 14]
        if not pairs:
            return None, None
        x, y = rng.choice(pairs)
        t = rng.randint(2, 3)
        k, l = (rng.randint(3, 5), 3) if test_like else (rng.randint(2, 5), rng.randint(2, 3))
        return (f"pfile_x-{x}_y-{y}_t-{t}_k-{k}_l-{l}_s-{seed}.pddl",
                _run([f"{GEN}/grid/grid", "-x", str(x), "-y", str(y), "-t", str(t), "-k", str(k),
                      "-l", str(l), "-p", "100", "-s", str(seed)]))
    if domain == "logistics":
        a = rng.randint(1, 2)
        p = rng.randint(9, 24) if test_like else rng.randint(3, 12)
        return (f"pfile_a-{a}_c-{size}_s-2_p-{p}_t-{size}_v{seed}.pddl",
                _run([f"{GEN}/logistics/logistics", "-a", str(a), "-c", str(size), "-s", "2",
                      "-p", str(p), "-t", str(size), "-r", str(seed)]))
    if domain == "spanner":
        spanners = rng.randint(max(size, 13), max(size, 13) + 6) if test_like else rng.randint(size, size + 11)
        locations = rng.choice([5, 25]) if test_like else rng.randint(3, 8)
        return (f"pfile_l-{spanners}_n-{size}_s_{locations}_r{seed}.pddl",
                _run([sys.executable, f"{GEN}/spanner/spanner-generator.py", str(spanners), str(size),
                      str(locations), "--seed", str(seed)]))
    if domain == "rovers":
        rovers = rng.choice([4, 6, 8]) if test_like else rng.randint(2, 6)
        objectives = rng.randint(1, 3) if test_like else rng.randint(1, 4)
        cameras = rng.randint(3, max(3, rovers))
        return (f"pfile_r-{rovers}_w-{size}_c-{objectives}_o-3_s{seed}.pddl",
                _run([f"{GEN}/rovers/rovgen", str(seed), str(rovers), str(size), str(objectives),
                      str(cameras), "3"]))
    raise KeyError(domain)


# ─── building ────────────────────────────────────────────────────────────────

def build_domain(domain, out, oversample, rng, log):
    spec = SPLITS[domain]
    train_dir = os.path.join(PDDL, spec["dir"])
    test_dir = train_dir + "_test"
    taken = {content_hash(p) for p in problems_in(train_dir)}
    existing_test = [(p, size_of(domain, p)) for p in problems_in(test_dir)]
    pool = []
    for rel in POOLS.get(domain, []):
        pool += [(p, size_of(domain, p)) for p in problems_in(os.path.join(PDDL, rel))]
    rng.shuffle(pool)
    order = ["train_ext"] if "train_ext" in spec else []
    order += ["val_id", "val_ood", "easy", "medium", "hard"]
    summary = {}
    for split in order:
        lo, hi = spec[split]
        want = COUNTS[split]
        dest = os.path.join(out, domain, split)
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.makedirs(dest)
        chosen, src = [], collections.Counter()

        def take(path_or_text, name, source):
            if path_or_text is None:
                return False
            if os.path.exists(str(path_or_text)):
                tmp = None
                path = path_or_text
            else:
                tmp = os.path.join(dest, ".tmp.pddl")
                with open(tmp, "w") as f:
                    f.write(path_or_text)
                path = tmp
            h = content_hash(path)
            if h in taken:
                if tmp:
                    os.remove(tmp)
                return False
            taken.add(h)
            target = os.path.join(dest, name)
            if tmp:
                os.replace(tmp, target)
            else:
                shutil.copy(path, target)
            chosen.append(target)
            src[source] += 1
            return True

        limit = int(want * oversample)   # room for problems LAMA cannot solve
        # 1. existing test problems in the band (test bands only), spread over sizes
        if split in TEST_SPLITS:
            for p, s in sorted(existing_test, key=lambda ps: (ps[1], ps[0])):
                if lo <= s <= hi and len(chosen) < want:
                    take(p, os.path.basename(p), "existing test")
        # 2. pools, spread evenly over the band's sizes
        by_size = collections.defaultdict(list)
        for p, s in pool:
            if lo <= s <= hi:
                by_size[s].append(p)
        while len(chosen) < limit and any(by_size.values()):
            for s in sorted(by_size):
                if by_size[s] and len(chosen) < limit:
                    p = by_size[s].pop()
                    take(p, f"pool_{os.path.basename(p)}", "pool")
        # 3. generators, cycling through the band's sizes so the band is covered
        if domain != "manyblocks":
            sizes = list(range(lo, hi + 1))
            tries = 0
            while len(chosen) < limit and tries < limit * 20:
                tries += 1
                name, text = generate(domain, split, sizes[tries % len(sizes)], rng)
                if text:
                    take(text, name, "generated")
        got = sorted(size_of(domain, p) for p in chosen)
        summary[split] = dict(band=[lo, hi], wanted=want, candidates=len(chosen), sources=dict(src),
                              size_min=got[0] if got else None, size_max=got[-1] if got else None)
        log(f"  {domain:10s} {split:9s} band {lo}-{hi} {spec['unit']}: {len(chosen)} candidates "
            f"(need {want}) {dict(src)} sizes {got[0] if got else '-'}-{got[-1] if got else '-'}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default=",".join(SPLITS))
    ap.add_argument("--oversample", type=float, default=2.0,
                    help="candidates per needed problem for validation/generated splits "
                         "(room for LAMA failures)")
    ap.add_argument("--seed", type=int, default=20261127)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for d in [x for x in args.domains.split(",") if x]:
        summary[d] = build_domain(d, args.out, args.oversample, rng, print)
    with open(os.path.join(args.out, "candidates_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {os.path.join(args.out, 'candidates_summary.json')}")


if __name__ == "__main__":
    main()

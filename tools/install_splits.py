#!/usr/bin/env python3
"""Select and install the 2026-11 evaluation splits into pddlgym (C1).

    python tools/install_splits.py --build /nfs/hpc/share/mangannr/di_ltp/splits_build [--dry-run]

Reads the candidates tools/build_splits.py wrote and the LAMA results
tools/solve_split_candidates.py recorded, picks COUNT problems per split, and
copies them to the pddlgym problem folders the fork registers
(pddlgym/__init__.py, _PROTOCOL_SPLITS):

    <dom>_val_id  <dom>_val_ood  <dom>_test_easy  <dom>_test_medium  <dom>_test_hard

Selection, deterministic:
  1. only LAMA-solved candidates (Spanner test bands excepted: LAMA times out
     there and the problems are valid by construction - the generator
     guarantees enough spanners);
  2. round-robin over sizes, so the band is covered evenly;
  3. within a size, existing test problems (original file names) first.

Spanner's train_ext problems are added to the TRAINING folder (prefix ext_),
the agreed 3-8-nut training set. That changes Spanner's problem count, so its
unified cache and checkpoint key (d = count) are new; nothing old is touched.

Typed twins of Grid and Logistics (tools/convert_to_typed.py): domain file,
training folder, old test folder and every split, then a seeded check that the
applicable actions match the untyped originals.

Writes campaign/reports/splits_manifest.json: per split the files, sizes,
source, LAMA plan length and seconds (paper Appendix A1/A8). Refuses to
overwrite an installed split folder unless --replace (the old folder is moved
aside with a timestamp, never deleted).
"""
import argparse
import collections
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from build_splits import PDDL, SPLITS, size_of  # noqa: E402

COUNT = 50
TRAIN_EXT_COUNT = 40
FOLDER = {"val_id": "val_id", "val_ood": "val_ood", "easy": "test_easy",
          "medium": "test_medium", "hard": "test_hard"}
NO_LAMA_NEEDED = {("spanner", "easy"), ("spanner", "medium"), ("spanner", "hard")}
TYPED = {"grid": "grid_ipcc", "logistics": "logistics_ipcc"}


def select(domain, split, cand_dir, count):
    files = sorted(f for f in os.listdir(cand_dir) if f.endswith(".pddl") and not f.startswith("."))
    res_path = os.path.join(cand_dir, "solve_results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if (domain, split) not in NO_LAMA_NEEDED:
        files = [f for f in files if res.get(f, {}).get("solved")]
    sized = [(size_of(domain, os.path.join(cand_dir, f)), f) for f in files]
    existing = {sf for sf in sized if not sf[1].startswith("pool_") and sf[1] in _existing_names(domain)}
    # Round-robin over sizes so the band is covered evenly; within a size,
    # existing test problems come first.
    by_size = collections.defaultdict(list)
    for s, f in sorted(sized, key=lambda sf: (sf[0], (sf[0], sf[1]) not in existing, sf[1])):
        by_size[s].append(f)
    chosen = []
    while len(chosen) < count and any(by_size.values()):
        for s in sorted(by_size):
            if by_size[s] and len(chosen) < count:
                chosen.append((s, by_size[s].pop(0)))
    if len(chosen) < count:
        raise SystemExit(f"{domain}/{split}: only {len(chosen)} usable problems, need {count}")
    rows = []
    for s, f in sorted(chosen):
        r = res.get(f, {})
        rows.append(dict(file=f, size=s, source=("existing test" if (s, f) in existing else
                                                   "pool" if f.startswith("pool_") else
                                                   "train_ext" if split == "train_ext" else "generated"),
                         lama_solved=r.get("solved"), lama_plan_length=r.get("plan_length"),
                         lama_seconds=r.get("seconds")))
    return rows


_EXISTING = {}


def _existing_names(domain):
    if domain not in _EXISTING:
        d = os.path.join(PDDL, SPLITS[domain]["dir"] + "_test")
        _EXISTING[domain] = set(os.listdir(d)) if os.path.isdir(d) else set()
    return _EXISTING[domain]


def _place(src_dir, rows, dest, replace, dry, prefix=""):
    if os.path.isdir(dest) and not prefix:
        if not replace:
            raise SystemExit(f"{dest} exists (use --replace: it is moved aside, not deleted)")
        if not dry:
            os.rename(dest, f"{dest}.old_{time.strftime('%Y%m%d_%H%M%S')}")
    if dry:
        return
    os.makedirs(dest, exist_ok=True)
    for r in rows:
        shutil.copy(os.path.join(src_dir, r["file"]), os.path.join(dest, prefix + r["file"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--domains", default=",".join(SPLITS))
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-typed", action="store_true")
    args = ap.parse_args()
    manifest_path = os.path.join(ROOT, "campaign", "reports", "splits_manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for domain in [d for d in args.domains.split(",") if d]:
        base = SPLITS[domain]["dir"]
        manifest[domain] = {}
        for split in (["train_ext"] if domain == "spanner" else []) + list(FOLDER):
            cand = os.path.join(args.build, domain, split)
            if split == "train_ext":
                rows = select(domain, split, cand, TRAIN_EXT_COUNT)
                train_dir = os.path.join(PDDL, base)
                already = [f for f in os.listdir(train_dir) if f.startswith("ext_")]
                if already:
                    print(f"  {domain} train_ext: {len(already)} ext_ problems already installed, skipping")
                else:
                    _place(cand, rows, train_dir, args.replace, args.dry_run, prefix="ext_")
            else:
                rows = select(domain, split, cand, COUNT)
                _place(cand, rows, os.path.join(PDDL, f"{base}_{FOLDER[split]}"), args.replace, args.dry_run)
            sizes = [r["size"] for r in rows]
            src = collections.Counter(r["source"] for r in rows)
            print(f"  {domain:10s} {split:9s} {len(rows)} problems, sizes {min(sizes)}-{max(sizes)} {dict(src)}")
            manifest[domain][split] = rows
    if not args.skip_typed and not args.dry_run:
        from convert_to_typed import DOMAINS as TYPED_DOMAINS, convert_dir, check
        for domain, base in TYPED.items():
            tbase = base.replace("_ipcc", "_typed_ipcc")
            with open(os.path.join(PDDL, tbase + ".pddl"), "w") as f:
                f.write(TYPED_DOMAINS[domain])
            for suffix in [""] + ["_test"] + [f"_{v}" for v in FOLDER.values()]:
                src, dst = os.path.join(PDDL, base + suffix), os.path.join(PDDL, tbase + suffix)
                if not os.path.isdir(src):
                    continue
                if os.path.isdir(dst):
                    os.rename(dst, f"{dst}.old_{time.strftime('%Y%m%d_%H%M%S')}")
                n = convert_dir(domain, src, dst)
                print(f"  typed {tbase + suffix}: {n} problems")
            for suffix in ["", "_test_hard"]:
                check(domain, os.path.join(PDDL, base + ".pddl"), os.path.join(PDDL, tbase + ".pddl"),
                      os.path.join(PDDL, base + suffix), os.path.join(PDDL, tbase + suffix))
                print(f"  typed {tbase + suffix}: applicable actions identical to the untyped problems")
    if not args.dry_run:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

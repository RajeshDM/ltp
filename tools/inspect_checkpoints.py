#!/usr/bin/env python3
"""What checkpoints does a model dir actually track, and when were they written?

Because `featurization` is not part of the ModelManager key, configs that
share a training-domain set + seed + hyperparameters share ONE directory and
ONE best-model list. Use this to confirm the checkpoint a test loaded really
came from the run you think it did.

    python tools/inspect_checkpoints.py                 # every MULTI-* dir
    python tools/inspect_checkpoints.py --grep gripper  # filter by dir name
"""
import argparse, glob, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--grep", default="")
ap.add_argument("--dir", default="models")
a = ap.parse_args()

for track in sorted(glob.glob(os.path.join(a.dir, "*", "model_tracking.json"))):
    d = os.path.dirname(track)
    if a.grep and a.grep.lower() not in d.lower():
        continue
    print(f"\n=== {d}")
    try:
        data = json.load(open(track))
    except Exception as e:
        print(f"  unreadable: {e}"); continue
    rows = []
    for cfg_hash, metrics in data.items():
        for metric, ckpts in metrics.items():
            for c in ckpts:
                p = c.get("save_path", "")
                exists = os.path.exists(p)
                mtime = (time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
                         if exists else "MISSING")
                size = f"{os.path.getsize(p)/1e6:.1f}MB" if exists else "-"
                rows.append((c.get("epoch"), metric, cfg_hash[:6], mtime, size,
                             c.get("validation_loss"), c.get("training_loss"),
                             os.path.basename(p)))
    for r in sorted(set(rows)):
        ep, metric, h, mtime, size, vl, tl, fn = r
        vl = f"{vl:.3f}" if isinstance(vl, (int, float)) else "-"
        tl = f"{tl:.3f}" if isinstance(tl, (int, float)) else "-"
        print(f"  e{ep:<4} {metric:11s} cfg{h} {mtime} {size:>8s} "
              f"val {vl:>9s} train {tl:>9s}  {fn}")
    hashes = {r[2] for r in rows}
    if len(hashes) > 1:
        print(f"  NOTE: {len(hashes)} distinct config hashes share this dir.")
    print("  Checkpoints written BEFORE your run started belong to another run.")

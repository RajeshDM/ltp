#!/usr/bin/env python3
"""Snapshot of the 2026-11 protocol campaign, embedded in the plan page (C1).

    python tools/campaign_dashboard.py [--page ICLR27/PAPER_PLAN.html] [--json out.json]

Reads only files and the scheduler (works from any node): campaign/worker.py
status, every p2 config's training log (epoch, seconds per 10 epochs), its
selected checkpoint and its final test dump, and squeue. Writes the result as
JSON between the TRACKER-DATA markers of the page, which renders the tracker
from it - so the local HTML file is always the complete, current dashboard
(republish it to any account). Notes for the page (decisions, blockers) live
in campaign/reports/dashboard_notes.json: [{"when": ..., "text": ...}].
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQUEUE = "/apps/slurm/25.11/bin/squeue"
DOMS = ["manyblocks", "gripper", "miconic", "visitall", "grid", "logistics", "spanner", "rovers"]
NICE = {"manyblocks": "Blocks", "gripper": "Gripper", "miconic": "Miconic", "visitall": "Visitall",
        "grid": "Grid", "logistics": "Logistics", "spanner": "Spanner", "rovers": "Rovers"}
RUNG = {"jc": "GADAR", "union": "UNION", "gabar": "GABAR", "structural": "Structural",
        "joint_lite": "Lifted layer", "joint": "+ binding", "jc_no_bind": "GADAR − binding",
        "jc_no_chain": "GADAR − chain", "jc_no_eff": "GADAR − effect", "jc_no_goal": "GADAR − goal"}


def classify_cfg(name):
    """(table, group, label) for a p2 config name."""
    m = re.match(r"p2_sd_(jc|gabar)_(\w+)$", name)
    if m:
        return "T2", f"In-domain: {RUNG[m[1]]} per domain", f"{RUNG[m[1]]} · {NICE[m[2]]}"
    m = re.match(r"p2_typed_(jc|gabar)_(\w+)$", name)
    if m:
        return "A5", "Typed twins (E6)", f"{RUNG[m[1]]} · typed {NICE[m[2]]}"
    if name == "p2_all8_jc":
        return "T2", "In-domain: one GADAR on all 8", "GADAR · all 8 domains"
    m = re.match(r"p2_loo_(jc|union)_no_(\w+)$", name)
    if m:
        return "T1", f"Zero-shot: {RUNG[m[1]]}", f"{RUNG[m[1]]} · {NICE[m[2]]} held out"
    m = re.match(r"p2_lad_(\w+)_no_(\w+)$", name)
    if m:
        return "T3", "Ablation ladder", f"{RUNG[m[1]]} · {NICE[m[2]]} held out"
    return "?", "other", name


def status_rows():
    out = subprocess.run(["python", "campaign/worker.py", "status", "--all", "--json"], cwd=ROOT,
                         capture_output=True, text=True, timeout=600)
    return {r["id"]: r for r in json.loads(out.stdout)}


def train_progress(cfg):
    logs = sorted(glob.glob(os.path.join(ROOT, "logs", f"{cfg}_mode_train*.log")), key=os.path.getmtime)
    if not logs:
        return None
    path = logs[-1]
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - 200_000))
        tail = f.read().decode("utf8", "replace")
    eps = re.findall(r"Epoch (\d+)/(\d+) loss: \{([^}]*)\} \| time\((\d+)\): ([\d.]+)s", tail)
    p = {"log": os.path.relpath(path, ROOT), "log_age_min": round((time.time() - os.path.getmtime(path)) / 60, 1)}
    if eps:
        e, tot, losses, n, sec = eps[-1]
        p.update(epoch=int(e), total=int(tot))
        secs = [float(x[4]) for x in eps[1:]][-5:]          # skip the epoch-0 sample
        if secs:
            p["sec_per_10"] = round(sum(secs) / len(secs), 1)
            p["eta_h"] = round((int(tot) - int(e)) / 10 * p["sec_per_10"] / 3600, 1)
        lv = dict(re.findall(r"'(\w+)': ([\d.e+-]+)", losses))
        p["loss"] = {k: round(float(v), 2) for k, v in lv.items()}
    elif "Traceback" in tail:
        p["error"] = tail.strip().splitlines()[-1][:200]
    else:
        p["phase"] = "collecting / featurizing"
    return p


def final_results(cfg):
    dumps = sorted(glob.glob(os.path.join(ROOT, "cache", "results", f"{cfg}_final", "results_*.json")),
                   key=os.path.getmtime)
    if not dumps:
        return {}
    with open(dumps[-1]) as f:
        d = json.load(f)
    res = {}
    for key, entries in (d.get("results") or {}).items():
        m = re.search(r"([A-Z][a-z_]+?)@(easy|medium|hard)", key)
        if not m or not entries:
            continue
        learned = next((v for p, v in entries[-1].get("metrics", {}).items() if "LEARNED" in p), None)
        if learned:
            dom = m[1].split("_")[0].lower()
            if "typed" in m[1].lower():
                dom = "typed_" + dom
            res[f"{dom}@{m[2]}"] = round(100 * learned["success_rate_with_monitor"], 1)   # rate -> %
    return res


def slurm_jobs():
    try:
        out = subprocess.run([SQUEUE, "-u", os.environ.get("USER", "mangannr"), "-h", "-o", "%i|%j|%P|%T|%M|%l|%R"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        i, name, part, st, el, lim, where = line.split("|")
        if name.startswith(("campaign", "p2_", "gadar")) or "camp" in name:
            rows.append(dict(id=i, name=name, partition=part, state=st, elapsed=el, limit=lim, where=where))
    return rows


def run_kind(cfg):
    m = re.match(r"p2_(sd_jc|sd_gabar|typed_jc|typed_gabar|all8_jc|loo_jc|loo_union|lad)", cfg)
    return m.group(1) if m else "other"


def load_manifest_jobs():
    import yaml
    jobs = {}
    for name in ("manifest.yaml", "manifest_second.yaml"):
        path = os.path.join(ROOT, "campaign", name)
        if os.path.exists(path):
            with open(path) as f:
                for j in (yaml.safe_load(f) or {}).get("jobs", []):
                    if name == "manifest.yaml" or j["id"] not in jobs or jobs[j["id"]].get("hold"):
                        jobs[j["id"]] = dict(j, account=2 if name != "manifest.yaml" else 1)
    return jobs


def estimate(runs, jobs, st, n_workers, collect_eta_h):
    """Expected hours to done per run, and the finish of everything, by list
    scheduling: running jobs keep their worker until their ETA; queued runs
    (train then eval on the same worker, priority order) take the next free
    worker of their account; runs that need the collection gate wait for it.
    Durations: measured median per run kind where any run of that kind has
    logged epochs, else the manifest estimate."""
    import heapq
    import statistics
    measured = {}
    for r in runs:
        t = r["train"]
        if t.get("sec_per_10") and t.get("total"):
            measured.setdefault(run_kind(r["cfg"]), []).append(t["total"] / 10 * t["sec_per_10"] / 3600 + 0.3)
    med = {k: statistics.median(v) for k, v in measured.items()}
    pools = {1: [0.0] * max(1, n_workers.get(1, 1)), 2: [0.0] * max(1, n_workers.get(2, 8))}
    for p in pools.values():
        heapq.heapify(p)
    queued = []
    for r in runs:
        cfg = r["cfg"]
        tj, ej = jobs.get(f"p2train_{cfg}", {}), jobs.get(f"p2eval_{cfg}", {})
        acct = 2 if tj.get("account") == 2 else 1
        multi = run_kind(cfg) in ("all8_jc", "loo_jc", "loo_union", "lad")
        ev = 1.0 if multi else 0.3
        tr = med.get(run_kind(cfg), float(tj.get("est_hours", 2)))
        r["est_h"] = round(tr + ev, 1)                       # full duration, for reference
        ts, es = r["train"]["state"], r["eval"]["state"]
        if es == "done":
            r["done_in_h"] = 0.0
            continue
        if ts == "running":
            left = (r["train"].get("eta_h") if r["train"].get("eta_h") is not None else tr) + ev
            r["done_in_h"] = round(left, 1)
            heapq.heapreplace(pools[acct], left) if pools[acct][0] == 0.0 else heapq.heappush(pools[acct], left)
            continue
        if es == "running":
            r["done_in_h"] = ev / 2
            continue
        if ts == "done":
            queued.append((0.1, cfg, ev, False, acct, r))     # only the eval is left
        else:
            needs = "gate_p2_collect" in (tj.get("needs") or [])
            queued.append((float(tj.get("priority", 1)), cfg, tr + ev, needs, acct, r))
    for prio, cfg, dur, needs, acct, r in sorted(queued, key=lambda q: (q[0], q[1])):
        free = heapq.heappop(pools[acct])
        start = max(free, collect_eta_h if needs else 0.0)
        heapq.heappush(pools[acct], start + dur)
        r["done_in_h"] = round(start + dur, 1)
    remaining_gpu_h = sum((r.get("done_in_h") or 0) and min(r["done_in_h"], r["est_h"]) for r in runs)
    finish = max([r.get("done_in_h") or 0 for r in runs] + [0])
    per_table = {}
    for r in runs:
        per_table[r["table"]] = max(per_table.get(r["table"], 0), r.get("done_in_h") or 0)
    return dict(finish_in_h=round(finish, 1), remaining_gpu_h=round(remaining_gpu_h),
                per_table={k: round(v, 1) for k, v in per_table.items()},
                workers=n_workers, measured_h={k: round(v, 1) for k, v in med.items()},
                collect_eta_h=collect_eta_h)


def snapshot():
    st = status_rows()
    # The second account's share (campaign/sync_second.sh): its live states
    # replace the 'held' placeholders this account's manifest shows for them.
    sec_path = os.path.join(ROOT, "campaign", "reports", "second_status.json")
    if os.path.exists(sec_path):
        for r in json.load(open(sec_path)):
            mine = st.get(r["id"])
            if mine is None or mine["state"] == "held":
                st[r["id"]] = dict(r, detail=("acct 2 · " + r.get("detail", "")).strip(" ·"))
    runs = []
    for path in sorted(glob.glob(os.path.join(ROOT, "configs", "p2_*.yaml"))):
        cfg = os.path.basename(path)[:-5]
        table, group, label = classify_cfg(cfg)
        tr, ev = st.get(f"p2train_{cfg}", {}), st.get(f"p2eval_{cfg}", {})
        sel_path = os.path.join(ROOT, "cache", "results", cfg, "selected_checkpoint.json")
        sel = json.load(open(sel_path)) if os.path.exists(sel_path) else None
        runs.append(dict(cfg=cfg, table=table, group=group, label=label,
                         train=dict(state=tr.get("state", "?"), detail=tr.get("detail", ""),
                                    est_h=tr.get("est_hours"), **(train_progress(cfg) or {})),
                         eval=dict(state=ev.get("state", "?"), detail=ev.get("detail", "")),
                         selected=((dict(epoch=sel["epoch"], val_loss=round(sel["mean_val_ood_loss"], 3))
                                    if "mean_val_ood_loss" in sel else
                                    dict(epoch=sel["epoch"], mean_val_ood=round(100 * sel["mean_val_ood"], 1)))
                                   if sel else None),
                         results=final_results(cfg)))
    gates = [dict(id=k, state=v["state"], detail=v["detail"]) for k, v in st.items() if k.startswith("gate_p2")]
    notes_path = os.path.join(ROOT, "campaign", "reports", "dashboard_notes.json")
    notes = json.load(open(notes_path)) if os.path.exists(notes_path) else []
    jobs = load_manifest_jobs()
    slurm = slurm_jobs()
    n1 = sum(1 for j in slurm if j["name"].startswith("camp-") and j["state"] == "RUNNING")
    n2 = 8   # the second account's planned workers (campaign/start_second.sh)
    gate = st.get("gate_p2_collect", {}).get("state")
    eta = estimate(runs, jobs, st, {1: n1, 2: n2}, 0.0 if gate == "done" else 1.5)
    return dict(updated=time.strftime("%Y-%m-%d %H:%M %Z"), updated_epoch=int(time.time()), eta=eta,
                runs=runs, gates=gates, slurm=slurm, notes=notes)


def embed(page, data):
    s = open(page).read()
    a, b = "<!--TRACKER-DATA-->", "<!--/TRACKER-DATA-->"
    i, j = s.index(a) + len(a), s.index(b)
    blob = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    s = s[:i] + f'<script id="tracker-data" type="application/json">{blob}</script>' + s[j:]
    tmp = page + ".tmp"
    with open(tmp, "w") as f:
        f.write(s)
    os.replace(tmp, page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", default=os.path.join(ROOT, "ICLR27", "PAPER_PLAN.html"))
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    data = snapshot()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(data, f, indent=1)
    embed(args.page, data)
    tr = sum(r["train"]["state"] == "done" for r in data["runs"])
    ev = sum(r["eval"]["state"] == "done" for r in data["runs"])
    print(f"dashboard: {len(data['runs'])} runs, {tr} trained, {ev} evaluated -> {args.page}")


if __name__ == "__main__":
    main()

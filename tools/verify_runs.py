#!/usr/bin/env python3
"""Check that every finished p2 run finished CORRECTLY, not just that it exited (C1).

    python tools/verify_runs.py [--json campaign/reports/verify.json] [--quiet]

The campaign's done-checks (tools/campaign_checks.py) only prove "rc 0 and the
epochs were reached" and "the dump is not empty". This adds what went wrong
on 2026-11-27 and could again, for every run whose training or evaluation is
done:

training  - the newest log reached its final epoch; no NaN/inf loss; a
            Val-OOD loss at every snapshot epoch (the selection rule needs
            them); no traceback after the last epoch line
evaluation- a result for every test band the config names; one tested epoch,
            equal to the epoch the CURRENT selection rule picks; the eval log
            reports the expected problem count per band (50); no traceback /
            NO MODELS; LAMA reference failures (plan-quality only) are a warning

Writes {config: {"stage", "issues": [...], "warnings": [...]}}; the dashboard
shows issues in red and warnings in amber on the run's row. Exit 1 if any
issue.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from select_checkpoint import domain_val_losses, read_config, train_domains  # noqa: E402

EXPECTED_PROBLEMS = 50


def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def check_training(cfg_name, cfg):
    issues, warns = [], []
    log = newest(os.path.join(ROOT, "logs", f"{cfg_name}_mode_train*.log"))
    if not log:
        return ["no training log"], warns
    text = open(log, errors="replace").read()
    eps = re.findall(r"Epoch (\d+)/(\d+) loss: \{([^}]*)\}", text)
    if not eps:
        return ["training log has no epoch lines"], warns
    last, total = int(eps[-1][0]), int(eps[-1][1])
    if last < total:
        issues.append(f"training stopped at epoch {last}/{total}")
    if any(re.search(r"\b(nan|inf)\b", e[2], re.I) for e in eps):
        issues.append("NaN/inf loss in the training log")
    tail = text[text.rfind(f"Epoch {last}/{total}"):]
    if "Traceback" in tail:
        issues.append("traceback after the last epoch line")
    every = int(cfg.get("checkpoint_every") or 50)
    want = {e for e in range(every, total + 1, every)}
    have = set(domain_val_losses(ROOT, cfg_name, train_domains(cfg), every))
    if want - have:
        issues.append(f"no Val-OOD loss at snapshot epochs {sorted(want - have)[:5]}")
    return issues, warns


def check_eval(cfg_name, cfg, path):
    issues, warns = [], []
    dump = newest(os.path.join(ROOT, "cache", "results", f"{cfg_name}_final", "results_*.json"))
    if not dump:
        return issues, warns
    d = json.load(open(dump))
    res = d.get("results") or {}
    bands = [t.split(":")[0] for t in str(cfg.get("test_domains", "")).split(",") if t]
    for b in bands:
        dom, split = b.rsplit("@", 1)
        if not any(f"{dom.capitalize()}@{split}" in k and v for k, v in res.items()):
            issues.append(f"no result for {b}")
    epochs = {int(e["epoch"]) for v in res.values() for e in v}
    if len(epochs) != 1:
        issues.append(f"tested epochs {sorted(epochs)} (expected one)")
    else:
        want = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "select_checkpoint.py"),
                               "--config", path, "--print-epoch"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
        if want.isdigit() and int(want) not in epochs:
            issues.append(f"tested epoch {epochs.pop()}, the selection rule picks {want}")
    log = newest(os.path.join(ROOT, "logs", "protocol", f"{cfg_name}_final*.log"))
    if log:
        text = open(log, errors="replace").read()
        counts = [int(n) for n in re.findall(r"\[batch-eval\] done: (\d+) problems", text)]
        if counts and any(c != EXPECTED_PROBLEMS for c in counts):
            issues.append(f"bands evaluated on {sorted(set(counts))} problems, expected {EXPECTED_PROBLEMS}")
        if counts and len(counts) < len(bands):
            warns.append(f"{len(counts)} band evaluations logged for {len(bands)} bands")
        if "NO MODELS" in text or "No models found" in text:
            issues.append("evaluation found no models")
        lama = text.count("returncode < 10")
        if lama:
            warns.append(f"{lama} LAMA reference runs failed (plan quality only, coverage unaffected)")
    return issues, warns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=os.path.join(ROOT, "campaign", "reports", "verify.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    st = json.loads(subprocess.run([sys.executable, "campaign/worker.py", "status", "--all", "--json"],
                                   cwd=ROOT, capture_output=True, text=True, timeout=600).stdout or "[]")
    sec = os.path.join(ROOT, "campaign", "reports", "second_status.json")
    state = {r["id"]: r["state"] for r in st}
    if os.path.exists(sec):
        for r in json.load(open(sec)):
            if state.get(r["id"]) in (None, "held"):
                state[r["id"]] = r["state"]
    out, bad = {}, 0
    for path in sorted(glob.glob(os.path.join(ROOT, "configs", "p2_*.yaml"))):
        name = os.path.basename(path)[:-5]
        trained = state.get(f"p2train_{name}") == "done"
        evaluated = state.get(f"p2eval_{name}") == "done"
        if not trained:
            continue
        cfg = read_config(path)
        issues, warns = check_training(name, cfg)
        if evaluated:
            i2, w2 = check_eval(name, cfg, path)
            issues += i2
            warns += w2
        out[name] = dict(stage="evaluated" if evaluated else "trained", issues=issues, warnings=warns)
        bad += bool(issues)
        if not args.quiet and (issues or warns):
            print(f"{name:34s} " + "; ".join([f"ISSUE {x}" for x in issues] + [f"warn {x}" for x in warns]))
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"verify: {len(out)} finished runs checked, {bad} with issues -> {os.path.relpath(args.json, ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

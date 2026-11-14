#!/usr/bin/env python3
"""Pull-based campaign worker: runs the experiments of ICLR27/REVIEW_PLAN.md.

    python campaign/worker.py run [--idle-exit-min 30]   # inside an allocation
    python campaign/worker.py status [--all] [--json]    # from anywhere
    python campaign/worker.py reset <job-id>             # forget a failed/dead job

Every allocation runs the SAME command. The worker repeatedly picks the
highest-priority READY job from campaign/manifest.yaml, claims it with an
atomic `mkdir campaign/claims/<id>` on the shared disk, runs it, and asks
again - the pattern eval_worker.sh already uses for evaluation, extended to
training and to jobs with dependencies. There is no split to plan: a late
allocation just starts taking work, and a dead one's claim goes stale (no
heartbeat for CLAIM_TTL seconds) and is taken over.

A job is DONE when its `done` command exits 0 - a check on the RESULT, never
on an exit code or a file merely existing. train_queue.sh exits 0 when the
training it ran crashed, and an evaluation whose checkpoint key did not
resolve writes an empty results file and exits 0; both have been counted as
finished before. A job without a `done` command is done when its own run
exited 0 (gates and reports only).

A job that finishes without passing its check is retried up to max_attempts
and then marked `dead`: a dead job is never retried by a worker, a human (or
the monitoring agent, once, after fixing the cause) runs `reset`.

A job with `hold: <reason>` is never run until `touch campaign/release/<id>`:
the gate for anything that needs a human decision (deleting model
directories, spending beyond the plan).

SIGTERM (SLURM's time-limit warning) stops the running job's process group,
releases the claim, and records `interrupted`, which does not count as an
attempt. `touch campaign/STOP` makes every worker finish its current job and
take no new one.
"""
# Python 3.6 compatible on purpose: a login node's default python is 3.6, and
# the monitoring agent may reach this tool from a shell with no env activated.
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIM_TTL = 300          # seconds without a heartbeat before a claim is stale
HEARTBEAT = 30
START_MARGIN_H = 0.25    # never start a job that cannot finish this far ahead of the limit
_TERM = threading.Event()

JOB_DEFAULTS = {"kind": "cpu", "min_cpus": 1, "est_hours": 1.0, "needs": [],
                "priority": 5, "max_attempts": 2}


# ─── manifest and state ─────────────────────────────────────────────────────

def load_manifest(path):
    with open(path) as f:
        m = yaml.safe_load(f)
    defaults = m.get("defaults", {})
    ids = set()
    for job in m["jobs"]:
        for key, value in JOB_DEFAULTS.items():
            job.setdefault(key, defaults.get(key, value))
        if job["id"] in ids:
            raise ValueError(f"duplicate job id {job['id']}")
        ids.add(job["id"])
        if job["kind"] not in ("gpu", "cpu"):
            raise ValueError(f"{job['id']}: kind must be gpu or cpu")
    by_id = {j["id"]: j for j in m["jobs"]}
    for job in m["jobs"]:
        for need in job["needs"]:
            if need not in by_id:
                raise ValueError(f"{job['id']} needs unknown job {need}")
    for jid in by_id:                      # cycle check
        seen, stack = set(), [jid]
        while stack:
            for need in by_id[stack.pop()]["needs"]:
                if need == jid:
                    raise ValueError(f"dependency cycle through {jid}")
                if need not in seen:
                    seen.add(need)
                    stack.append(need)
    m["by_id"] = by_id
    m.setdefault("budget", {})
    return m


def make_ctx(manifest=None, camp=None, logs=None, root=None):
    root = root or ROOT
    camp = camp or os.path.join(root, "campaign")
    return {"m": load_manifest(manifest or os.path.join(camp, "manifest.yaml")),
            "camp": camp, "logs": logs or os.path.join(root, "logs"),
            "root": root, "done_cache": {}}


def _state_path(ctx, jid):
    return os.path.join(ctx["camp"], "state", f"{jid}.json")


def read_state(ctx, jid):
    try:
        with open(_state_path(ctx, jid)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_state(ctx, jid, **fields):
    st = read_state(ctx, jid)
    st.update(fields)
    path = _state_path(ctx, jid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, path)
    return st


def is_done(ctx, job):
    """The job's RESULT check (cached within one scheduling pass)."""
    cache = ctx["done_cache"]
    if job["id"] not in cache:
        if job.get("done"):
            try:
                rc = subprocess.run(job["done"], shell=True, cwd=ctx["root"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=180).returncode
            except subprocess.TimeoutExpired:
                rc = 1
            cache[job["id"]] = rc == 0
        else:
            cache[job["id"]] = read_state(ctx, job["id"]).get("status") == "ok"
    return cache[job["id"]]


# ─── claims ─────────────────────────────────────────────────────────────────

def _claim_dir(ctx, jid):
    return os.path.join(ctx["camp"], "claims", jid)


def claim_fresh(ctx, jid):
    d = _claim_dir(ctx, jid)
    try:
        return time.time() - os.path.getmtime(os.path.join(d, "owner")) < CLAIM_TTL
    except OSError:
        try:   # created a moment ago, owner not written yet
            return time.time() - os.path.getmtime(d) < CLAIM_TTL
        except OSError:
            return False


def try_claim(ctx, jid, who):
    d = _claim_dir(ctx, jid)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    try:
        os.mkdir(d)
    except FileExistsError:
        if claim_fresh(ctx, jid):
            return False
        # Dead holder: take over. Two workers doing this in the same instant
        # can both win; the cost is one duplicated run, as in eval_worker.sh.
        shutil.rmtree(d, ignore_errors=True)
        try:
            os.mkdir(d)
        except FileExistsError:
            return False
    with open(os.path.join(d, "owner"), "w") as f:
        f.write(who)
    return True


def release(ctx, jid):
    shutil.rmtree(_claim_dir(ctx, jid), ignore_errors=True)


def _heartbeat(ctx, jid, stop):
    owner = os.path.join(_claim_dir(ctx, jid), "owner")
    while not stop.wait(HEARTBEAT):
        try:
            os.utime(owner)
        except OSError:
            pass


# ─── scheduling ─────────────────────────────────────────────────────────────

def classify(ctx, job):
    jid = job["id"]
    st = read_state(ctx, jid)
    if is_done(ctx, job):
        return "done", ""
    if st.get("status") == "dead":
        return "dead", f"{st.get('attempts', '?')} attempts, last rc={st.get('rc')}"
    if job.get("hold") and not os.path.exists(
            os.path.join(ctx["camp"], "release", jid)):
        return "held", f"{job['hold']} -- release: touch campaign/release/{jid}"
    if claim_fresh(ctx, jid):
        return "running", st.get("host", "?")
    waiting = [n for n in job["needs"] if not is_done(ctx, ctx["m"]["by_id"][n])]
    if waiting:
        return "waiting", "needs " + ",".join(waiting)
    attempts = st.get("attempts", 0)
    return "ready", f"retry {attempts}" if attempts else ""


def gpu_hours_used(ctx):
    used, now = 0.0, time.time()
    d = os.path.join(ctx["camp"], "state")
    for name in os.listdir(d) if os.path.isdir(d) else []:
        try:
            with open(os.path.join(d, name)) as f:
                st = json.load(f)
        except (OSError, ValueError):
            continue
        if st.get("kind") == "gpu" and st.get("started"):
            end = st.get("ended")
            if not end:
                # No end time: either it is running now (a live claim), or its
                # worker was KILLED outright (scancel, node failure) and never
                # wrote one. A killed job must stop at its last heartbeat:
                # counting it to `now` charged two dead jobs ~48 GPU-hours a
                # day and would have closed the budget on work nobody ran.
                jid = name[:-len(".json")] if name.endswith(".json") else name
                end = now if claim_fresh(ctx, jid) else _last_heartbeat(ctx, jid, st["started"])
            used += max(0.0, end - st["started"]) / 3600
    return used


def _last_heartbeat(ctx, jid, default):
    """When a claim was last refreshed (its `owner` file), else `default`."""
    try:
        return os.path.getmtime(os.path.join(_claim_dir(ctx, jid), "owner"))
    except OSError:
        return default


def next_job(ctx, caps, hours_left, max_priority=None):
    """The next job this worker may run. max_priority=0 restricts it to gates."""
    ctx["done_cache"].clear()
    cap = ctx["m"]["budget"].get("gpu_hours")
    used = gpu_hours_used(ctx)
    # Gates (priority <= 0) go to whichever worker is free first: they are
    # minutes long, block everything behind them, and waiting a day for a CPU
    # allocation while GPU workers sit idle costs more than the minutes.
    # Otherwise the worker's own kind comes first, THEN priority: a GPU
    # allocation trains before it does CPU work even when the CPU job has a
    # better priority, so a scarce GPU is not spent on a job that needs none.
    order = sorted(enumerate(ctx["m"]["jobs"]),
                   key=lambda iv: (0 if iv[1]["priority"] <= 0 else 1,
                                   0 if iv[1]["kind"] == caps["kind"] else 1,
                                   iv[1]["priority"], iv[0]))
    for _, job in order:
        if max_priority is not None and job["priority"] > max_priority:
            continue
        if classify(ctx, job)[0] != "ready":
            continue
        if job["kind"] == "gpu" and not caps["gpu"]:
            continue
        if job["min_cpus"] > caps["cpus"]:
            continue
        if job["est_hours"] > hours_left - START_MARGIN_H:
            continue
        if job["kind"] == "gpu" and cap is not None and used >= cap:
            continue
        return job
    return None


_STARTED = time.time()


def hours_left():
    """Wall-clock left in this SLURM allocation (inf outside SLURM).

    Order: CAMPAIGN_HOURS_LEFT (tests, manual override); squeue, the
    authoritative answer; the time limit submit.sh passed in
    CAMPAIGN_TIME_LIMIT_H minus this worker's age. The last exists because a
    batch job inherits the SUBMITTING shell's PATH: without the slurm module
    there, squeue is missing inside the job, and a guard that silently
    returned 'infinite time left' would start jobs the limit then kills.
    Inside SLURM with no way to know, assume 2 hours rather than unlimited.
    """
    env = os.environ.get("CAMPAIGN_HOURS_LEFT")
    if env:
        return float(env)
    jid = os.environ.get("SLURM_JOB_ID")
    if jid:
        try:
            out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%L"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True, timeout=30).stdout.strip()
            left = parse_slurm_time(out)
            if left != float("inf"):
                return left
        except (OSError, subprocess.TimeoutExpired):
            pass
    limit = os.environ.get("CAMPAIGN_TIME_LIMIT_H")
    if limit:
        return float(limit) - (time.time() - _STARTED) / 3600
    if jid:
        print("WARNING: inside SLURM but cannot read the time limit (no squeue, no "
              "CAMPAIGN_TIME_LIMIT_H); assuming 2h left", flush=True)
        return 2.0
    return float("inf")


def parse_slurm_time(text):
    """'1-02:03:04' | '02:03:04' | '03:04' -> hours; anything else -> inf."""
    try:
        days, rest = (text.split("-", 1) if "-" in text else ("0", text))
        parts = [int(p) for p in rest.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return int(days) * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600
    except ValueError:
        return float("inf")


def detect_caps(kind=None):
    gpu = os.environ.get("CAMPAIGN_HAS_GPU")
    if gpu is None:
        try:
            r = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True,
                               timeout=30)
            gpu = "1" if r.returncode == 0 and "GPU" in r.stdout else "0"
        except (OSError, subprocess.TimeoutExpired):
            gpu = "0"
    caps = {"gpu": gpu == "1", "cpus": len(os.sched_getaffinity(0))}
    caps["kind"] = kind or ("gpu" if caps["gpu"] else "cpu")
    return caps


# ─── running a job ──────────────────────────────────────────────────────────

def run_job(ctx, job, caps):
    """Run one claimed job to completion. Returns 'ok'|'failed'|'dead'|'interrupted'."""
    jid = job["id"]
    st = read_state(ctx, jid)
    who = f"{socket.gethostname()} slurm={os.environ.get('SLURM_JOB_ID', '-')} pid={os.getpid()}"
    write_state(ctx, jid, status="running", host=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), kind=job["kind"],
                started=time.time(), ended=None, attempts=st.get("attempts", 0))
    os.makedirs(ctx["logs"], exist_ok=True)
    log_path = os.path.join(ctx["logs"], f"campaign_{jid}.log")
    env = dict(os.environ, CAMPAIGN_JOB_ID=jid, PYTHONHASHSEED="42",
               PYTHONUNBUFFERED="1",
               CAMPAIGN_DEV="cuda:0" if caps["gpu"] else "cpu",
               CAMPAIGN_WORKERS=str(max(1, min(16, caps["cpus"] - 1))))
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(ctx, jid, stop), daemon=True).start()
    rc = None
    interrupted = False
    try:
        with open(log_path, "a") as log:
            log.write(f"\n=== campaign job {jid} on {who} at "
                      f"{time.strftime('%F %T')} ===\n$ {job['cmd']}\n")
            log.flush()
            proc = subprocess.Popen(["bash", "-c", job["cmd"]], cwd=ctx["root"],
                                    env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while proc.poll() is None:
                if _TERM.is_set():
                    interrupted = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(2)
            rc = proc.wait()
    finally:
        stop.set()
        release(ctx, jid)
    if interrupted:
        write_state(ctx, jid, status="interrupted", ended=time.time(), rc=rc)
        return "interrupted"
    ctx["done_cache"].pop(jid, None)
    write_state(ctx, jid, ended=time.time(), rc=rc, log=log_path,
                status="ok" if rc == 0 else "failed")
    if is_done(ctx, job):
        write_state(ctx, jid, status="ok")
        return "ok"
    attempts = read_state(ctx, jid).get("attempts", 0) + 1
    status = "dead" if attempts >= job["max_attempts"] else "failed"
    write_state(ctx, jid, status=status, attempts=attempts)
    return status


def run_worker(ctx, caps, idle_exit_min=30, poll_sec=60, install_handlers=False,
               only_gates=False):
    if install_handlers:
        signal.signal(signal.SIGTERM, lambda *_: _TERM.set())
        signal.signal(signal.SIGINT, lambda *_: _TERM.set())
    who = f"{socket.gethostname()} slurm={os.environ.get('SLURM_JOB_ID', '-')}"
    idle_since = None
    ran = 0
    while not _TERM.is_set():
        if os.path.exists(os.path.join(ctx["camp"], "STOP")):
            print("campaign/STOP present - taking no new job")
            break
        left = hours_left()
        job = next_job(ctx, caps, left, 0 if only_gates else None)
        if job is None:
            ctx["done_cache"].clear()
            pending = [j for j in ctx["m"]["jobs"]
                       if (not only_gates or j["priority"] <= 0)
                       and classify(ctx, j)[0] in ("ready", "waiting", "running")]
            if not pending:
                print("nothing left to run")
                break
            idle_since = idle_since or time.time()
            if time.time() - idle_since > idle_exit_min * 60:
                print(f"idle {idle_exit_min} min with {len(pending)} job(s) not runnable here - exiting")
                break
            time.sleep(poll_sec)
            continue
        idle_since = None
        if not try_claim(ctx, job["id"], who):
            continue
        ctx["done_cache"].pop(job["id"], None)
        if is_done(ctx, job):            # finished between the pass and the claim
            release(ctx, job["id"])
            continue
        print(f"[{who}] {time.strftime('%T')} start {job['id']} "
              f"(est {job['est_hours']}h, {left:.1f}h left)", flush=True)
        outcome = run_job(ctx, job, caps)
        ran += 1
        print(f"[{who}] {time.strftime('%T')} {job['id']}: {outcome}", flush=True)
    return ran


# ─── status / reset ─────────────────────────────────────────────────────────

def cmd_status(ctx, show_all=False, as_json=False):
    ctx["done_cache"].clear()
    rows = []
    for job in ctx["m"]["jobs"]:
        state, detail = classify(ctx, job)
        st = read_state(ctx, job["id"])
        age = ""
        if state == "running" and st.get("started"):
            age = f"{(time.time() - st['started']) / 3600:.1f}h"
            detail = f"{st.get('host', '?')} slurm={st.get('slurm_job')} {age}"
        rows.append({"id": job["id"], "state": state, "detail": detail,
                     "kind": job["kind"], "priority": job["priority"],
                     "est_hours": job["est_hours"]})
    if as_json:
        print(json.dumps(rows, indent=1))
        return
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    cap = ctx["m"]["budget"].get("gpu_hours")
    print(f"campaign: {len(rows)} jobs | " +
          " | ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"gpu-hours used {gpu_hours_used(ctx):.1f}" + (f" / {cap}" if cap else ""))
    for r in rows:
        if r["state"] == "done" and not show_all:
            continue
        if r["state"] == "waiting" and not show_all:
            continue
        print(f"{r['state'].upper():8s} {r['id']:44s} {r['detail']}")
    waiting = counts.get("waiting", 0)
    if waiting and not show_all:
        print(f"({waiting} waiting on dependencies; --all lists them)")
    if os.path.exists(os.path.join(ctx["camp"], "STOP")):
        print("STOP file present: workers take no new jobs")


def cmd_reset(ctx, jid):
    if jid not in ctx["m"]["by_id"]:
        raise SystemExit(f"unknown job {jid}")
    if claim_fresh(ctx, jid):
        raise SystemExit(f"{jid} is running (fresh claim); stop it first")
    release(ctx, jid)
    try:
        os.remove(_state_path(ctx, jid))
    except OSError:
        pass
    print(f"reset {jid}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None)
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True          # the required= keyword needs Python 3.7
    r = sub.add_parser("run")
    r.add_argument("--kind", choices=["gpu", "cpu"], default=None,
                   help="which kind of job to prefer (default: gpu if a GPU is visible)")
    r.add_argument("--idle-exit-min", type=float, default=30)
    r.add_argument("--poll-sec", type=float, default=60)
    r.add_argument("--only-gates", action="store_true",
                   help="run only priority-0 jobs (the gates) and exit when none is "
                        "left: for a one-off pass inside an allocation that is busy "
                        "with something else, e.g. srun --jobid=<id> --overlap")
    s = sub.add_parser("status")
    s.add_argument("--all", action="store_true")
    s.add_argument("--json", action="store_true")
    z = sub.add_parser("reset")
    z.add_argument("job")
    args = ap.parse_args()
    ctx = make_ctx(manifest=args.manifest)
    if args.cmd == "run":
        caps = detect_caps(args.kind)
        print(f"worker on {socket.gethostname()}: gpu={caps['gpu']} cpus={caps['cpus']} "
              f"prefers {caps['kind']} jobs, {hours_left():.1f}h left")
        run_worker(ctx, caps, args.idle_exit_min, args.poll_sec, install_handlers=True,
                   only_gates=args.only_gates)
    elif args.cmd == "status":
        cmd_status(ctx, args.all, args.json)
    else:
        cmd_reset(ctx, args.job)


if __name__ == "__main__":
    main()

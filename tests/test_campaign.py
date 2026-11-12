"""Tests for the campaign worker and the result checks (needs PyYAML, no torch).

What they pin: a job is done by its RESULT check, never by an exit code; a
dependency blocks until it is satisfied; failures retry and then go dead; a
claim excludes a second worker and a stale one is taken over; a job that
cannot finish in the allocation's remaining time is not started.

Run: python tests/test_campaign.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "campaign"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import worker as W  # noqa: E402
import campaign_checks as C  # noqa: E402

CAPS_CPU = {"gpu": False, "cpus": 8, "kind": "cpu"}
CAPS_GPU = {"gpu": True, "cpus": 8, "kind": "gpu"}


def _setup(jobs, budget=None):
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "campaign"))
    os.makedirs(os.path.join(tmp, "logs"))
    manifest = os.path.join(tmp, "campaign", "manifest.yaml")
    import yaml
    with open(manifest, "w") as f:
        yaml.safe_dump({"budget": budget or {}, "jobs": jobs}, f)
    return tmp, W.make_ctx(manifest=manifest, root=tmp)


def _run(ctx, caps, hours=None):
    if hours is not None:
        os.environ["CAMPAIGN_HOURS_LEFT"] = str(hours)
    try:
        return W.run_worker(ctx, caps, idle_exit_min=0, poll_sec=0)
    finally:
        os.environ.pop("CAMPAIGN_HOURS_LEFT", None)


def _states(ctx):
    return {j["id"]: W.classify(ctx, j)[0] for j in ctx["m"]["jobs"]}


def test_dependency_order_and_result_check_decide_done():
    tmp, ctx = _setup([
        {"id": "b", "needs": ["a"], "cmd": "touch b.out", "done": "test -f b.out"},
        {"id": "a", "cmd": "touch a.out", "done": "test -f a.out"},
    ])
    ran = _run(ctx, CAPS_CPU)
    assert ran == 2 and _states(ctx) == {"a": "done", "b": "done"}
    assert open(os.path.join(tmp, "logs", "campaign_a.log")).read().count("touch a.out")


def test_exit_code_zero_without_the_result_is_not_done():
    """The train_queue.sh trap: rc 0, nothing produced."""
    tmp, ctx = _setup([{"id": "liar", "cmd": "true", "done": "test -f never.out",
                        "max_attempts": 2}])
    _run(ctx, CAPS_CPU)
    st = W.read_state(ctx, "liar")
    assert st["status"] == "dead" and st["attempts"] == 2 and st["rc"] == 0
    assert _states(ctx)["liar"] == "dead"


def test_failed_job_retries_then_dies_and_reset_clears_it():
    tmp, ctx = _setup([{"id": "bad", "cmd": "false", "max_attempts": 2}])
    _run(ctx, CAPS_CPU)
    assert _states(ctx)["bad"] == "dead"
    W.cmd_reset(ctx, "bad")
    assert _states(ctx)["bad"] == "ready"


def test_job_without_done_command_is_done_when_its_run_exits_zero():
    tmp, ctx = _setup([{"id": "gate", "cmd": "true"}])
    _run(ctx, CAPS_CPU)
    assert _states(ctx)["gate"] == "done"


def test_held_job_and_blocked_dependents_are_left_alone():
    tmp, ctx = _setup([
        {"id": "h", "cmd": "touch h.out", "done": "test -f h.out", "hold": "needs a human"},
        {"id": "after", "needs": ["h"], "cmd": "touch x.out"},
    ])
    assert _run(ctx, CAPS_CPU) == 0
    assert _states(ctx) == {"h": "held", "after": "waiting"}
    assert not os.path.exists(os.path.join(tmp, "h.out"))


def test_hold_is_released_by_a_file_not_by_the_worker():
    tmp, ctx = _setup([{"id": "h", "cmd": "touch h.out", "done": "test -f h.out",
                        "hold": "delete old models first"}])
    assert _run(ctx, CAPS_CPU) == 0 and _states(ctx)["h"] == "held"
    assert "touch campaign/release/h" in W.classify(ctx, ctx["m"]["by_id"]["h"])[1]
    os.makedirs(os.path.join(tmp, "campaign", "release"))
    open(os.path.join(tmp, "campaign", "release", "h"), "w").close()
    assert _run(ctx, CAPS_CPU) == 1 and _states(ctx)["h"] == "done"


def test_a_gpu_worker_takes_gpu_jobs_before_better_priority_cpu_jobs():
    tmp, ctx = _setup([
        {"id": "cpu_first", "kind": "cpu", "priority": 1, "cmd": "true"},
        {"id": "gpu_later", "kind": "gpu", "priority": 9, "cmd": "true"}])
    assert W.next_job(ctx, CAPS_GPU, 100)["id"] == "gpu_later"
    assert W.next_job(ctx, CAPS_CPU, 100)["id"] == "cpu_first"


def test_gates_go_to_the_first_free_worker_whatever_its_kind():
    tmp, ctx = _setup([
        {"id": "train", "kind": "gpu", "priority": 3, "cmd": "true"},
        {"id": "gate", "kind": "cpu", "priority": 0, "cmd": "true"}])
    assert W.next_job(ctx, CAPS_GPU, 100)["id"] == "gate"
    assert W.next_job(ctx, CAPS_CPU, 100)["id"] == "gate"
    W.write_state(ctx, "gate", status="ok")
    assert W.next_job(ctx, CAPS_GPU, 100)["id"] == "train"


def test_only_gates_mode_runs_gates_and_nothing_else_then_exits():
    tmp, ctx = _setup([
        {"id": "gate", "priority": 0, "cmd": "touch g.out", "done": "test -f g.out"},
        {"id": "work", "priority": 3, "cmd": "touch w.out", "done": "test -f w.out"}])
    ran = W.run_worker(ctx, CAPS_CPU, idle_exit_min=0, poll_sec=0, only_gates=True)
    assert ran == 1 and _states(ctx) == {"gate": "done", "work": "ready"}
    assert not os.path.exists(os.path.join(tmp, "w.out"))


def test_gpu_job_needs_a_gpu_and_the_budget():
    jobs = [{"id": "g", "kind": "gpu", "cmd": "touch g.out", "done": "test -f g.out",
             "est_hours": 1}]
    tmp, ctx = _setup(jobs)
    assert _run(ctx, CAPS_CPU) == 0 and _states(ctx)["g"] == "ready"
    assert _run(ctx, CAPS_GPU) == 1 and _states(ctx)["g"] == "done"
    tmp, ctx = _setup(jobs, budget={"gpu_hours": 5})
    W.write_state(ctx, "other", kind="gpu", started=time.time() - 6 * 3600,
                  ended=time.time())
    assert W.gpu_hours_used(ctx) > 5.9
    assert _run(ctx, CAPS_GPU) == 0, "budget exhausted: no new GPU job"


def test_job_that_cannot_finish_before_the_time_limit_is_not_started():
    tmp, ctx = _setup([{"id": "long", "cmd": "touch l.out", "done": "test -f l.out",
                        "est_hours": 10}])
    assert _run(ctx, CAPS_CPU, hours=5) == 0
    assert _run(ctx, CAPS_CPU, hours=11) == 1


def test_claim_excludes_a_second_worker_and_stale_claim_is_taken_over():
    tmp, ctx = _setup([{"id": "x", "cmd": "true"}])
    assert W.try_claim(ctx, "x", "w1")
    assert not W.try_claim(ctx, "x", "w2"), "a fresh claim must exclude"
    owner = os.path.join(W._claim_dir(ctx, "x"), "owner")
    old = time.time() - W.CLAIM_TTL - 60
    os.utime(owner, (old, old))
    assert W.try_claim(ctx, "x", "w2"), "a stale claim is taken over"
    W.release(ctx, "x")
    assert W.try_claim(ctx, "x", "w3")


def test_stop_file_takes_no_new_job():
    tmp, ctx = _setup([{"id": "s", "cmd": "touch s.out", "done": "test -f s.out"}])
    open(os.path.join(tmp, "campaign", "STOP"), "w").close()
    assert _run(ctx, CAPS_CPU) == 0 and _states(ctx)["s"] == "ready"


def test_manifest_rejects_unknown_dependency_cycle_and_duplicate():
    for jobs in ([{"id": "a", "needs": ["zzz"], "cmd": "true"}],
                 [{"id": "a", "needs": ["b"], "cmd": "true"},
                  {"id": "b", "needs": ["a"], "cmd": "true"}],
                 [{"id": "a", "cmd": "true"}, {"id": "a", "cmd": "true"}]):
        try:
            _setup(jobs)
        except ValueError:
            continue
        raise AssertionError(f"manifest accepted {jobs}")


def test_time_left_falls_back_to_the_submitted_limit_when_squeue_is_missing():
    keep = {k: os.environ.get(k) for k in ("CAMPAIGN_HOURS_LEFT", "SLURM_JOB_ID",
                                           "CAMPAIGN_TIME_LIMIT_H", "PATH")}
    try:
        os.environ.pop("CAMPAIGN_HOURS_LEFT", None)
        os.environ["SLURM_JOB_ID"] = "999999999"
        os.environ["PATH"] = "/nonexistent"          # no squeue
        os.environ["CAMPAIGN_TIME_LIMIT_H"] = "24"
        left = W.hours_left()
        assert 23.9 < left <= 24, left
        os.environ.pop("CAMPAIGN_TIME_LIMIT_H")
        assert W.hours_left() == 2.0, "inside SLURM and blind: assume 2h, not infinity"
        os.environ.pop("SLURM_JOB_ID")
        assert W.hours_left() == float("inf"), "outside SLURM there is no limit"
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_slurm_time_parsing():
    assert W.parse_slurm_time("1-02:00:00") == 26
    assert abs(W.parse_slurm_time("00:30:00") - 0.5) < 1e-9
    assert abs(W.parse_slurm_time("45:00") - 0.75) < 1e-9
    assert W.parse_slurm_time("UNLIMITED") == float("inf")


# ─── result checks ──────────────────────────────────────────────────────────

def test_trained_check_reads_the_exit_file_and_the_epochs():
    tmp = tempfile.mkdtemp()
    logs = os.path.join(tmp, "logs")
    os.makedirs(logs)
    stem = os.path.join(logs, "cfg_mode_train")
    assert not C.check_trained("configs/cfg.yaml", "", 0.9, logs)[0]       # nothing yet
    open(stem + ".exit", "w").write("1\n")
    open(stem + ".log", "w").write("Epoch 500/500 time(10): 1.0s\n")
    assert not C.check_trained("cfg", "", 0.9, logs)[0], "crashed run is not trained"
    open(stem + ".exit", "w").write("0\n")
    assert C.check_trained("cfg", "", 0.9, logs)[0]
    open(stem + ".log", "w").write("Epoch 120/500 time(10): 1.0s\n")
    assert not C.check_trained("cfg", "", 0.9, logs)[0], "rc 0 at epoch 120 is not trained"
    open(stem + ".log", "w").write("Epoch 460/500\n")
    assert C.check_trained("cfg", "", 0.9, logs)[0]
    assert not C.check_trained("cfg", "29990101", 0.9, logs)[0], "older than --since"


def test_evaluated_check_rejects_empty_dumps_and_missing_metrics():
    tmp = tempfile.mkdtemp()
    res = os.path.join(tmp, "cache", "results")

    def dump(expid, results, metrics, ts="20261112_010101"):
        d = os.path.join(res, expid)
        os.makedirs(d, exist_ok=True)
        json.dump({"timestamp": ts, "results": results, "test_model_metrics": metrics},
                  open(os.path.join(d, "results_x_%s.json" % ts), "w"))

    dump("e_empty", {"cellA": [], "cellB": []}, ["combined"])
    assert not C.check_evaluated("e_empty", "", ["combined"], "", 1, res)[0]
    dump("e_ok", {"cellA": [{"epoch": 1}], "cellB": []}, ["combined"])
    assert C.check_evaluated("e_ok", "", ["combined"], "", 1, res)[0]
    assert not C.check_evaluated("e_ok", "", ["periodic"], "", 1, res)[0]
    assert not C.check_evaluated("e_ok", "", ["combined"], "", 2, res)[0]
    assert not C.check_evaluated("e_ok", "20270101", ["combined"], "", 1, res)[0]
    assert not C.check_evaluated("nothing", "", ["combined"], "", 1, res)[0]
    dump("e_sw_epochs", {"c": [{"epoch": 50}]}, ["periodic"])
    assert C.check_evaluated("e_sw", "", ["periodic"], "_epochs", 1, res)[0]


def test_globcount_requires_enough_keys_per_file():
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "out")
    os.makedirs(d)
    json.dump({"a": 1}, open(os.path.join(d, "x_1.json"), "w"))
    json.dump({"a": 1, "b": 2}, open(os.path.join(d, "x_2.json"), "w"))
    assert C.check_globcount("out/x_*.json", 1, "", 2, tmp)[0]
    assert not C.check_globcount("out/x_*.json", 2, "", 2, tmp)[0]


def test_real_manifest_loads_and_every_config_it_names_exists():
    import re
    path = os.path.join(ROOT, "campaign", "manifest.yaml")
    if not os.path.exists(path):
        return
    m = W.load_manifest(path)
    for job in m["jobs"]:
        for cfg in re.findall(r"configs/[A-Za-z0-9_./-]+\.yaml", job["cmd"] + job.get("done", "")):
            assert os.path.exists(os.path.join(ROOT, cfg)), f"{job['id']} names missing {cfg}"
        for script in re.findall(r"campaign/jobs/[A-Za-z0-9_.-]+\.sh", job["cmd"]):
            assert os.path.exists(os.path.join(ROOT, script)), f"{job['id']} names missing {script}"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if failed else 0)

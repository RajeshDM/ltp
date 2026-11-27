#!/usr/bin/env python3
"""Write campaign/manifest.yaml from ICLR27/REVIEW_PLAN.md.

    python campaign/make_manifest.py            # rewrites campaign/manifest.yaml

The manifest is generated so that 40-odd jobs stay consistent (a config
name, its done-check and its eval job must agree); edit THIS file and
regenerate rather than hand-editing the YAML. Priorities follow the plan's
cut order: gates, E0, E1, the two ablations that must not be cut, the rest.
"""
import os

import yaml

SINCE = "20261101"            # results older than this never satisfy a job (the cluster clock is US-Pacific:
                              # a date taken from elsewhere once rejected a result written minutes earlier)
HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = ["manyblocks", "visitall", "grid"]
# The four ablations + `joint`, AND the full model and the lifted-layer-only
# model retrained as the reference: the featurizer changed (single-effect
# schemas are kept now), so the September joint_chain/joint_lite models are not
# comparable to anything trained from now on.
E2_MODES = ["joint_chain", "jc_no_bind", "jc_no_goal", "jc_no_chain", "jc_no_eff",
            "joint", "joint_lite"]
E2_CORE = {"joint_chain", "jc_no_bind", "jc_no_goal"}    # never cut (plan, section 6)

# E1: GABAR through our pipeline. Miconic already reproduced (99.2%).
GABAR = {"logistics": 6, "rovers": 6, "visitall": 5, "spanner": 5,
         "blocks": 6, "gripper": 5, "grid": 5}
# Left behind by an interrupted September run: train_then_eval.sh would
# EVALUATE the half-trained checkpoint and call it done.
INTERRUPTED = {"blocks", "grid", "gripper"}
HOLD_INTERRUPTED = ("an interrupted September run left a half-trained model for this "
                    "key and train_then_eval.sh would evaluate it; archive "
                    "models/*bs16* for this domain first (REVIEW_PLAN E1)")


# Decision 2026-11-27: the published GABAR numbers are taken as given and the
# pipeline-reproduction question is not pursued, so the ablations run and no
# further GABAR job does.
HOLD_AFTER_E4 = ("waits for the E4 reading (REVIEW_PLAN E4): ablate the model E4 settles on "
                 "(batch size, width, features), not the one it may replace")
HOLD_NO_GABAR = ("decision 2026-11-27: published GABAR numbers are used as given; no "
                 "further GABAR reproduction runs (release only to revisit)")
# GABAR on optimal Logistics plans already got 0%, so GADAR's run answers
# nothing new, and it spent 14 h collecting lmcut plans before timing out twice.
HOLD_LOGI_OPT = ("GABAR on optimal Logistics plans got 0%, so this answers nothing new; it "
                 "spends ~14 h on lmcut plans per attempt and was cut off twice")


def job(id_, cmd, **kw):
    return {"id": id_, "cmd": cmd, **kw}


def main():
    jobs = []
    # gates ---------------------------------------------------------------
    # check_ablation_modes exits 1 on a mismatch, so a clean run IS its result
    # (a job with no `done` command is done when its own run exits 0).
    jobs += [
        job("gate_unit_tests", "campaign/jobs/gate_unit_tests.sh", priority=0, est_hours=0.3),
        job("gate_ablation_blocks",
            "python tools/check_ablation_modes.py --domain Manyblocks_ipcc_big --steps 25",
            needs=["gate_unit_tests"], priority=0, est_hours=0.5),
        job("gate_ablation_miconic",
            "python tools/check_ablation_modes.py --domain Miconic_ipcc --steps 25",
            needs=["gate_unit_tests"], priority=0, est_hours=0.5),
        job("gate_ablation_logistics",
            "python tools/check_ablation_modes.py --domain Logistics_ipcc --steps 25",
            needs=["gate_unit_tests"], priority=0, est_hours=0.5),
        # The featurizer against an independent reading of the PDDL text, on all
        # eight domains. The gate that would have caught the dropped effects.
        job("gate_audit_featurizer", "python tools/audit_featurizer.py --problems 2 --steps 20",
            needs=["gate_unit_tests"], priority=0, est_hours=0.5),
        job("gate_fingerprint", "campaign/jobs/gate_fingerprint.sh",
            needs=["gate_unit_tests"], priority=0, est_hours=1.0, min_cpus=4,
            done=f"python tools/campaign_checks.py file campaign/reports/gate_fingerprint.txt --since {SINCE}"),
    ]
    # E0: rule baselines ---------------------------------------------------
    for pol in ("features", "achieve"):
        jobs.append(job(f"e0_{pol}", f"campaign/jobs/e0.sh {pol}", priority=1,
                        est_hours=6, min_cpus=8, needs=["gate_unit_tests"],
                        done=f"python tools/campaign_checks.py globcount "
                             f"'cache/results/e0/{pol}_*.json' 8 --min-keys 2 --since {SINCE}"))
    jobs.append(job("report_e0", "campaign/jobs/report.sh e0", priority=1, est_hours=0.1,
                    needs=["e0_features", "e0_achieve"],
                    done=f"python tools/campaign_checks.py file campaign/reports/e0.txt --since {SINCE}"))
    # E1: GABAR through our pipeline, and the optimal-plan question ---------
    e1 = []
    for dom, est in GABAR.items():
        j = job(f"e1_gabar_{dom}",
                f"WORKERS=$CAMPAIGN_WORKERS ./train_test_scripts/train_then_eval.sh configs/baseline_{dom}.yaml",
                kind="gpu", priority=2, est_hours=est, needs=["gate_unit_tests"],
                done=f"python tools/campaign_checks.py evaluated baseline_{dom} --since {SINCE} --metrics combined")
        if dom in INTERRUPTED:
            j["hold"] = HOLD_INTERRUPTED     # wanted again for E4 (2026-11-27)
        jobs.append(j)
        e1.append(j["id"])
    for cfg, expid in [("baseline_logistics_opt", "baseline_logistics_opt"),
                       ("baseline_rovers_opt", "baseline_rovers_opt"),
                       ("sd_joint_chain_logistics_opt", "sd_joint_chain_logistics_opt"),
                       ("sd_joint_chain_rovers_opt", "sd_joint_chain_rovers_opt")]:
        jobs.append(job(f"e1_{cfg}",
                        f"WORKERS=$CAMPAIGN_WORKERS ./train_test_scripts/train_then_eval.sh configs/{cfg}.yaml",
                        kind="gpu", priority=2, est_hours=5, needs=["gate_unit_tests"],
                        done=f"python tools/campaign_checks.py evaluated {expid} --since {SINCE} --metrics combined"))
        if cfg == "sd_joint_chain_logistics_opt":
            jobs[-1]["hold"] = HOLD_LOGI_OPT
        else:
            e1.append(f"e1_{cfg}")
    # The E1 debug step: published-GABAR code (before any GADAR change) on the
    # two domains our pipeline fails. Decides regression vs. setup.
    for dom, ntest in (("logistics_ipcc", 96), ("rovers_ipcc", 54)):
        rep = f"campaign/reports/e1_gabar_{dom}_prejuly.txt"
        jobs.append(job(f"e1_gabar_{dom.split('_')[0]}_prejuly",
                        f"campaign/jobs/gabar_prejuly.sh {dom} {ntest}",
                        kind="gpu", priority=1, est_hours=12, hold=HOLD_NO_GABAR,
                        done=f"python tools/campaign_checks.py file {rep} --since {SINCE} && "
                             f"grep -qi 'testing model from epoch' {rep}"))
    # The cleanest test of the featurizer fix: single-domain GADAR on exactly the
    # two domains whose schemas were affected, LAMA plans, compared with the
    # September numbers (Miconic 41.2, Rovers 7.4) and with GABAR (99.2 / 82).
    for dom in ("miconic", "rovers"):
        jobs.append(job(f"e1_sd_{dom}_fixed",
                        f"WORKERS=$CAMPAIGN_WORKERS ./train_test_scripts/train_then_eval.sh configs/sd_joint_chain_{dom}.yaml",
                        kind="gpu", priority=2, est_hours=4,
                        needs=["gate_unit_tests", "gate_ablation_miconic", "gate_audit_featurizer"],
                        done=f"python tools/campaign_checks.py evaluated sd_joint_chain_{dom} --since {SINCE} --metrics combined"))
        e1.append(f"e1_sd_{dom}_fixed")
    # E4: why single-domain GADAR trails GABAR in the same pipeline (Miconic
    # 44.5 vs 99.2). A ladder from GABAR's features to GADAR's, at GABAR's
    # batch size; REVIEW_PLAN E4 has the readings. Before E2: ablating a model
    # that E4 may change would measure the wrong model.
    e4 = []
    for cfg in ("sd_union_miconic_bs16", "sd_structural_miconic_bs16",
                "sd_joint_lite_miconic_bs16", "sd_joint_chain_miconic_bs16",
                "sd_joint_chain_miconic_bs16_rs128"):
        jobs.append(job(f"e4_{cfg}",
                        f"WORKERS=$CAMPAIGN_WORKERS ./train_test_scripts/train_then_eval.sh configs/{cfg}.yaml",
                        kind="gpu", priority=1, est_hours=5,
                        done=f"python tools/campaign_checks.py evaluated {cfg} --since {SINCE} --metrics combined"))
        e4.append(jobs[-1]["id"])
    jobs.append(job("report_e4", "campaign/jobs/report.sh e4", priority=1, est_hours=0.1,
                    needs=e4,
                    done=f"python tools/campaign_checks.py file campaign/reports/e4.txt --since {SINCE}"))
    jobs.append(job("report_e1", "campaign/jobs/report.sh e1", priority=2, est_hours=0.1,
                    needs=[j for j in e1 if "gabar_blocks" not in j and "gabar_grid" not in j
                           and "gabar_gripper" not in j],
                    done=f"python tools/campaign_checks.py file campaign/reports/e1.txt --since {SINCE}"))
    # E2: component ablations ------------------------------------------------
    gates = ["gate_unit_tests", "gate_ablation_blocks", "gate_ablation_miconic",
             "gate_ablation_logistics", "gate_audit_featurizer", "gate_fingerprint"]
    evals = []
    for mode in E2_MODES:
        for fold in FOLDS:
            cfg = f"loo8_{mode}_no_{fold}"
            prio = 3 if mode in E2_CORE else 4
            jobs.append(job(f"e2_train_{mode}_{fold}",
                            f"./train_test_scripts/train_queue.sh cuda:0 configs/{cfg}.yaml",
                            kind="gpu", priority=prio, est_hours=16, needs=gates, hold=HOLD_AFTER_E4,
                            done=f"python tools/campaign_checks.py trained {cfg} --since {SINCE}"))
            jobs.append(job(f"e2_eval_{mode}_{fold}", f"campaign/jobs/eval_ablation.sh configs/{cfg}.yaml",
                            kind="cpu", priority=prio - 0.5, est_hours=6, min_cpus=16,
                            needs=[f"e2_train_{mode}_{fold}"],
                            done=f"python tools/campaign_checks.py evaluated {cfg} --since {SINCE} --metrics combined && "
                                 f"python tools/campaign_checks.py evaluated {cfg} --since {SINCE} --metrics periodic --suffix _epochs"))
            evals.append(f"e2_eval_{mode}_{fold}")
    jobs.append(job("report_e2", "campaign/jobs/report.sh e2", priority=4, est_hours=0.1,
                    needs=evals,
                    done=f"python tools/campaign_checks.py file campaign/reports/e2.txt --since {SINCE}"))

    manifest = {
        "defaults": {"max_attempts": 2},
        # Stops new GPU jobs once this many GPU-hours have been spent (running
        # jobs count). The plan needs ~400 (21 ablation/reference
        # trainings x 16 h + E1); the rest is headroom for retries.
        "budget": {"gpu_hours": 520},
        "jobs": jobs,
    }
    path = os.path.join(HERE, "manifest.yaml")
    with open(path, "w") as f:
        f.write("# GENERATED by campaign/make_manifest.py from ICLR27/REVIEW_PLAN.md - edit the\n"
                "# generator, not this file. Read by campaign/worker.py.\n")
        yaml.safe_dump(manifest, f, sort_keys=False, width=200)
    print(f"wrote {path}: {len(jobs)} jobs")


if __name__ == "__main__":
    main()

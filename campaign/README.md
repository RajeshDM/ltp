# campaign/ — running the experiments without a person at the keyboard

What this is: the experiments of `ICLR27/REVIEW_PLAN.md` as a **manifest of
jobs** (`manifest.yaml`), a **pull-based worker** that every SLURM allocation
runs (`worker.py`), a **submit script** sized to the cluster's quotas
(`submit.sh`), and **rules plus a loop prompt** for a Claude Code session that
monitors it all (`RULES.md`, `LOOP_PROMPT.md`).

```
you ──tmux──> submit-a: claude (monitor loop)         <- may die; nothing breaks
                 │  reads files, runs worker.py status, calls submit.sh
                 ▼
              sbatch ──> GPU/CPU allocations, each running:  worker.py run
                              │ claim next READY job (mkdir on the shared disk)
                              │ run it, then check its RESULT (not its exit code)
                              ▼
                  logs/, models/, cache/results/   <- the shared disk is the state
```

The design point: **the experiments never depend on the agent being alive.**
Workers are batch jobs; state is files on the shared disk; the agent only
decides when to add capacity, triages failures and reads results. If its
session dies, you restart it and it reads the same files.

## 0. Status of this setup

Written and unit-tested off-cluster (`tests/test_campaign.py`: dependencies,
retries, claims, budgets, result checks). **Not yet run on SLURM.** Section 3
is the smoke test that proves each piece before any GPU-day is spent. Anything
marked UNVERIFIED below is an assumption about your cluster.

## 1. One-time setup on the cluster

### 1.1 Where the agent runs

Recommended: Claude Code on a **login node** (`submit-a`), inside **tmux on
that node**. tmux on `flip1` works too, but then an ssh hop to `submit-a` sits
between tmux and the agent and its death kills the agent. (Experiments are
unaffected either way.) Remember which `submit-*` node holds the tmux session.

```bash
# from your computer:
ssh flip1                       # as you do today
ssh mangannr@submit-a.hpc.engr.oregonstate.edu     # alias: submit_a
tmux new -s campaign            # later: tmux attach -t campaign
```

The agent on the login node only launches, monitors and reads: a mostly idle
process, no compute. (If the cluster's login-node policy disallows long-lived
processes, UNVERIFIED, run it in a small `srun --pty` allocation instead.)

### 1.2 Install Claude Code and check it can reach the API  (UNVERIFIED)

Use the installer from Claude Code's own documentation (a user-level install
that needs no root; your home is on NFS so it is visible from every node).
Then, **before anything else**:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" https://api.anthropic.com    # any HTTP code = reachable
claude --version
```

If the login node cannot reach the API, the fallback is to run Claude Code on
`flip1` and let it run commands through `ssh submit-a '<command>'`, which needs
a passwordless ssh key from flip1 to submit-a (your home is shared, so
`ssh-keygen` + `cat id_ed25519.pub >> ~/.ssh/authorized_keys`). Tell me which
case you are in and I will adapt `RULES.md`.

Log in once, interactively (`claude`, then follow the prompt; a headless login
prints a URL you open on your computer and paste a code back).

### 1.3 The repo, a branch of its own, the environment

```bash
di_ltp                                   # cd to the repo, conda activate di_ltp_1
git fetch origin && git checkout claude/wizardly-rubin-hqMug && git pull
git checkout -b auto/campaign            # the agent commits ONLY here; you merge
python -c "import torch, yaml; print('torch', torch.__version__)"
```

The agent's commits go on `auto/campaign`, not on the branch Claude's cloud
session pushes to, so the two never conflict. Large outputs (`cache/`,
`models/`, `logs/`) are untracked; the agent commits only small text under
`campaign/reports/` and ledger entries.

### 1.4 Permissions for the agent

```bash
mkdir -p .claude && cp campaign/settings.example.json .claude/settings.local.json
```

`settings.example.json` allows the repo's scripts, the status tools, `git
add/commit/push` to `auto/campaign`, and read-only SLURM commands; it denies
editing code, the paper, and `rm -rf`. Look at it with `/permissions` inside
Claude Code and tighten it if you like. `scancel` is not allowed
unattended: the agent asks, or you do it.

## 2. What the manifest contains

`python campaign/worker.py status --all` lists every job. 50 jobs:

| Group | Jobs | Needs | Notes |
|---|---|---|---|
| gates | unit tests, ablation-mode check on two domains, byte-identity fingerprint | nothing | cheap; **everything else waits for them** |
| E0 | rule baselines `features`, `achieve` (CPU) | unit tests | answers "is anything learned beyond the hand-coded features" |
| E1 | GABAR through our pipeline (7 domains); Logistics/Rovers on optimal plans for GABAR and single-domain GADAR | unit tests | **3 jobs are `held`**: an interrupted September run left half-trained models; see below |
| E2 | 15 trainings (4 ablations + `joint`, 3 folds) and 15 evaluations | gates | the two that must not be cut (`jc_no_bind`, `jc_no_goal`) run first |
| reports | `campaign/reports/{e0,e1,e2}.txt` | their experiment | what the agent reads |

GPU budget in the manifest: 400 GPU-hours (the plan needs about 230); workers
stop taking GPU jobs past it.

**Releasing a held job** is a deliberate human act. The three held E1 jobs
(Blocks, Grid, Gripper) need you to decide about the September leftovers first:

```bash
ls -d models/*bs16* | grep -E "Manyblocks_ipcc_big|Grid_ipcc|Gripper_ipcc"   # look
mkdir -p models_archive && mv <those three> models_archive/                  # archive, do not delete
touch campaign/release/e1_gabar_blocks campaign/release/e1_gabar_grid campaign/release/e1_gabar_gripper
```

## 3. Smoke test (about 30 minutes, almost no GPU)

Each step proves one piece; stop at the first one that fails.

```bash
# 3.1 the manifest and tools load; nothing is claimable that should not be
python tests/test_campaign.py
python campaign/worker.py status

# 3.2 the submit script, without submitting
campaign/submit.sh --list
campaign/submit.sh gpu-dgxh --dry-run

# 3.3 one CPU worker runs the gates, as a batch job (needs SLURM; ~minutes)
CAMPAIGN_MAX_WORKERS=1 campaign/submit.sh cpu-share
squeue -u $USER -n camp-cpu-share
tail -f logs/campaign_slurm_<jobid>.out          # 'REFUSING: import torch' = conda path, see worker_entry.sh
python campaign/worker.py status                  # gates should turn DONE
```

If 3.3 prints `REFUSING: 'import torch' fails`, set `CONDA_SH` to your
`etc/profile.d/conda.sh` (the default guesses `/nfs/hpc/share/mangannr/anaconda3`)
and resubmit. If a gate fails, **stop**: a gate failing means the code change
is wrong, which is exactly what the gates are for. Paste the output.

## 4. Launching the unattended run

```bash
# inside tmux on submit-a, in the repo, on branch auto/campaign:
claude
```

then, at Claude Code's prompt:

```
/loop Read campaign/RULES.md, then follow campaign/LOOP_PROMPT.md.
```

(no interval: the agent paces itself, waking every 20-60 minutes). It will
start workers within the quotas, triage failures by the rules, apply the
pre-registered readings of `ICLR27/REVIEW_PLAN.md` to finished experiments,
append findings to `train_test_scripts/RUNS_STATUS.md`, commit to
`auto/campaign`, and stop when every job is done or dead, leaving
`campaign/FINAL_REPORT.md`.

**Kill switches**, in order of gentleness:

| Do | Effect |
|---|---|
| `touch campaign/STOP` | workers finish the job they are on and take no new one; `submit.sh` refuses; the agent stops |
| `scancel -n camp-gpu-dgxh` (per preset name) | ends those workers now; their claims go stale and are retaken later |
| Ctrl-C in the Claude session | stops the agent only; workers keep running |

## 5. Coming back to the results

```bash
cd <repo> && git log --oneline auto/campaign | head      # what it did
cat campaign/FINAL_REPORT.md                              # the summary, readings applied
cat campaign/reports/e0.txt campaign/reports/e1.txt campaign/reports/e2.txt
python campaign/worker.py status --all                    # per-job state
```

Nothing the agent concludes goes into the paper. It drafts to
`campaign/FINAL_REPORT.md`; the paper stays yours (and is denied to the agent).

## 6. Cost of the loop itself

Each wake reads one compact status screen (a few thousand tokens), so a
30-60 minute cadence for a week is cheap next to the GPU time. It gets
expensive only when the agent reads big logs or reasons at length, which
`RULES.md` tells it not to do outside failures and finished experiments. Use
a smaller model for the loop and a larger one only for analysis if you want to
save usage.

## 7. Editing the plan

Edit `make_manifest.py`, run `python campaign/make_manifest.py`, run
`python tests/test_campaign.py` (it checks every config and script the
manifest names exists). Adding a job never disturbs running ones; changing the
`done` command of a finished job re-opens it.

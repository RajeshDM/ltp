# CLUSTER.md — the OSU CoE HPC cluster, as this project uses it

Read this before launching anything on the cluster. Facts here came from the
cluster's own documentation and from this project's September 2026 campaign;
anything marked UNVERIFIED has not been tried.

## 1. Where things run

| Place | What it is | Rule |
|---|---|---|
| `flip1` | the user's ssh jump host (`ssh flip1`, then `ProxyJump flip1` to `submit-a`, then `ProxyJump submit-a` to any `cn*`/`dgx*` node: the user's `~/.ssh/config`); tmux lives here today | not a compute host |
| `submit-a` … `submit-d` | **login nodes**; same NFS disk, same SLURM | **never run compute here** (training, evaluation, rollouts, anything over a few seconds). Launching, `squeue`, reading files and `git` are what they are for |
| compute nodes | `dgx2-*`, `dgxh-*`, `cn-gpu*`, `cn-r-*`, `cn-s-*` … | reached only through SLURM |
| shared disk | `/nfs/hpc/share/mangannr/di_ltp/1_di_ltp/ltp` | every node sees the same files; `logs/`, `models/`, `cache/` are shared by every machine |

Environment: `conda activate di_ltp_1` (alias `di_ltp` also `cd`s to the repo).
The aliases `ltp_3`, `ltp_5`, `ltp`, `ploi`, `eval` point at OLDER repos and
environments; this repo is `di_ltp`. Always `--device cuda:0`: an allocation
shows exactly one GPU, and `cuda:1` selects a device that is not there.

**SLURM commands need `module load slurm`.** The cluster docs say it loads at
login; in practice a shell can lack it (`sbatch: command not found`), which is
why the `sq_r` alias starts with it. Run `module load slurm` in any shell
before starting the monitoring agent: it inherits that PATH, and so does every
batch job it submits. `campaign/submit.sh` also tries the module itself, and
workers carry the time limit in `CAMPAIGN_TIME_LIMIT_H` so their
finish-before-the-limit guard does not depend on `squeue`.

A batch shell has not read `~/.bashrc`, so aliases and `conda activate` do not
exist inside `sbatch` unless sourced; `campaign/worker_entry.sh` does it.

## 2. Partitions and the per-user caps that decide how to size a request

From the cluster's own documentation (full text supplied by the user,
2026-11-27). Every cap is "the amount you can have running at one time", i.e.
**resource × time limit, summed over your running jobs, interactive
allocations included**. Requesting less time is how you fit more concurrent
GPUs - and an interactive session (this project's VS Code allocation: 1 GPU ×
4 days on `gpu`) spends the same cap as the workers.

Partitions this user can use (account `-A eecs`; every other partition in the
cluster list - athena, sail, hw-grp, sy-grp, nacse, tiamat, ... - belongs to a
research group we are not in):

| Partition | GPU (memory) | Max time | GPU cap | CPU cap | RAM cap |
|---|---|---|---|---|---|
| `share` | a40 (48 GB), m60; most nodes CPU-only | 14 d | 2 GPU-days | 1024 CPU-days | 3072 GB-days, 750 GB |
| `dgxh` | h100 (40/80 GB), h200 (140 GB) | 2 d | 2 GPU-days, ≤ 8 GPUs | 64 CPU-days, ≤ 224 | 500 GB-days, ≤ 2000 GB |
| `dgx2` | v100 (32 GB) | 7 d | 8 GPU-days, ≤ 16 GPUs | 48 CPU-days, ≤ 96 | 750 GB-days, ≤ 1500 GB |
| `gpu` | rtx8000 (48 GB) | 7 d | 8 GPU-days, ≤ 8 GPUs | 48 CPU-days, ≤ 40 | 750 GB-days |
| `ampere` | a40 (48 GB) | 2 d | 2 GPU-days | 64 CPU-days | 250 GB-days |
| `eecs` | rtx2080 (**11 GB**), old Broadwell CPUs | 7 d | 8 GPU-days | 48 CPU-days, ≤ 40 | 250 GB-days |
| `preempt` | everything | 7 d | none | none | none - **can be killed at any time** |

The same cap at different time limits (concurrent GPUs):

| Partition | 6 h | 12 h | 1 d | 2 d | 4 d | 7 d |
|---|---|---|---|---|---|---|
| `dgx2` | 16 | 16 | 8 | 4 | 2 | 1 |
| `gpu` | 8 | 8 | 8 | 4 | 2 | 1 |
| `dgxh` | 8 | 4 | 2 | 1 | - | - |
| `share` | 2* | 2* | 2 | 1 | - | - |
| `ampere` | 2* | 2* | 2 | 1 | - | - |

(* other caps bind first: few GPU nodes / at most 2 GPUs.) CPU and RAM caps
bind too: at 12 h, 16 dgx2 workers × 4 CPUs × 32 GB fit (96 CPUs, 1500 GB).

**What this means for the campaign.** Long runs (seven-domain training,
16-27 h) need 1-2 day allocations; everything else (single-domain training
~1-2.5 h, every evaluation) fits 12 h allocations, which buy 2-4× the GPUs.
The worker never starts a job whose `est_hours` exceeds its allocation's time
left, so short and long workers can share one manifest. `eecs` GPUs are too
small for batch 64 on Rovers or any multi-domain run (OOM, 2026-11-27) - its
preset was removed.

Also: at most 1000 submitted and 400 running jobs per user. Node features for
`--constraint`: el8 el9 | avx avx2 avx512 | haswell broadwell skylake
cascadelake epyc sapphire | m60 t4 rtx6000 rtx8000 v100 a40 h100 h200 l40s
vram40g vram80g vram140g. A constraint does not request the GPU,
`--gres=gpu:1` does. Useful commands: `squ` / `sqe -u <onid>` (own jobs),
`showjob <id>`, `nodestat <partition>` (free CPU/GPU/RAM per node),
`#SBATCH --mail-type=FAIL,TIMELIMIT_90`.

The interactive aliases in `~/.bashrc` that exceed a cap simply pend: for
example `get_share` asks for 1 GPU × 7 days on `share` = 7 GPU-days against a
2 GPU-day cap.

## 3. Interactive vs batch, and why the campaign uses batch

| | `srun --pty bash` (what `get_*` aliases do) | `sbatch` |
|---|---|---|
| ends when | the terminal session ends or the time limit hits; losing ssh/VPN kills it | the time limit hits, or `scancel` |
| needs a person | yes, one session per GPU | no |
| used for | debugging, a quick look | everything unattended |

The campaign (section 5, `campaign/`) is `sbatch`-only. The agent's own
session can die without stopping a single experiment.

The September campaign was run from interactive sessions, which is why it kept
losing work: an allocation ending kills its cgroup, and `nohup` does not
survive that. A batch job is not tied to a terminal, but it is still killed at
its time limit, so a worker never starts a job that cannot finish before then.

## 4. Commands that are safe, and the traps

- Count your jobs: `squeue -u $USER` (alias `sq_r`); one job in detail:
  `showjob <id>`; a node's free resources: `nodestat <partition>`.
- Look inside someone's own running allocation from a login node:
  `srun --jobid=<id> --overlap -n1 <command>` (this is how we found duplicate
  evaluations on one box).
- `scancel <jobid>` for your jobs only. Never `scancel -u $USER` while a
  campaign is running.
- **Never `rm -f logs/*.running`**: `logs/` is shared, and the glob deletes the
  live markers of runs on other machines. Delete by exact name.
- **Never edit a script that is executing.** Bash reads scripts lazily; saving
  over a running `train_queue.sh` can corrupt that run. `git pull` is safe
  (git replaces the file; the running bash keeps the old one).
- DataLoader and rollout workers are forks that inherit the command line:
  `pgrep -f main.py` counts them. Count distinct configs
  (`pgrep -af main.py | grep -oE "configs/[^ ]+"| sort -u`).
- `--mode test` on a config whose checkpoint key does not resolve logs "No
  models found", writes an **empty** results file and **exits 0**. Never trust
  an exit code or a file's existence; use `tools/campaign_checks.py`.
- `train_queue.sh` exits 0 even when the run it launched crashed.
- Stopping a run: `kill -- -$(cat logs/<name>.running)` (the `-` is the
  process GROUP; without it python keeps running and holds the GPU).
- The login node's python has no torch. Anything importing `ploi` or running
  `main.py` needs `conda activate di_ltp_1`.

## 5. The user's usual allocations (from `~/.bashrc`), for sizing

| alias | partition | resources | time |
|---|---|---|---|
| `get_dgxh2` | dgxh | 1 GPU, 32 cpu, 256 GB, vram80g | 2 d |
| `get_dgxh1` | dgxh | 1 GPU, 64 GB, vram80g | 2 d |
| `get_dgx_2` | dgx2 | 1 GPU, 4 cpu, 8 GB | 7 d |
| `get_gpu` / `get_gpu_8` | gpu | 1 GPU, 64 GB / 8 cpu 128 GB | 4 d |
| `get_dgx_a40_32` | share | 1 a40, 32 cpu, 128 GB | 2 d |
| `get_dgx_cpu_big` | preempt | 1 a40, 32 cpu, 32 GB | 7 d |
| `get_prempt` | preempt | 1 GPU | 7 d |

What a 32-core a40 box does well: evaluation (host-CPU bound; 16 featurizer
workers is the measured plateau). What a 4-8 core GPU box does well: training.
Training one seven-domain LOO model took 7-14 h depending on the GPU.

## 6. Second account (`p_submit_a`, user `dugarp`)

A teammate's account. **The campaign does not use it** and no agent should log
into it: it is another person's credentials, and its quotas are theirs. If the
teammate agrees, the supported way to use extra capacity is for *them* to run
`bash campaign/worker_entry.sh` from an allocation of theirs, on a path they
can read and write; the claims on the shared disk make that safe. Whether
their account can see this directory is UNVERIFIED.

## 7. Where the experiment machinery lives

`campaign/README.md` (setting up an unattended run), `campaign/RULES.md` (what
an agent may and may not do), `campaign/manifest.yaml` (the jobs, generated by
`campaign/make_manifest.py`), `ICLR27/REVIEW_PLAN.md` (why each job exists and
how to read its result), `train_test_scripts/` (the attended scripts the jobs
call). `CLAUDE.md` is the research north star.

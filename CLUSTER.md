# CLUSTER.md — the OSU CoE HPC cluster, as this project uses it

Read this before launching anything on the cluster. Facts here came from the
cluster's own documentation and from this project's September 2026 campaign;
anything marked UNVERIFIED has not been tried.

## 1. Where things run

| Place | What it is | Rule |
|---|---|---|
| `flip1` | the user's ssh jump host; tmux lives here today | not a compute host |
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

From the cluster documentation. The CPU/GPU/GB-time limits are "the amount you
can have running at one time", i.e. **resource × time limit, summed over your
running jobs** (interactive allocations included). Requesting less time is how
you fit more concurrent GPUs.

| Partition | Max time | CPU-time cap | GPU-time cap | Fits, at the stated time |
|---|---|---|---|---|
| `share` | 14 d | 1024 CPU-days | 2 GPU-days | 2 GPUs × 1 d, or 1 GPU × 2 d |
| `dgxh` | 2 d | 64 CPU-days | 2 GPU-days | 2 GPUs × 1 d, 4 × 12 h |
| `dgx2` | 7 d | 48 CPU-days | 8 GPU-days | 4 GPUs × 2 d, 1 × 7 d |
| `gpu` | 7 d | 48 CPU-days | 8 GPU-days | 4 GPUs × 2 d |
| `eecs` | 7 d | 48 CPU-days | 8 GPU-days | as `gpu` |
| `ampere` | 2 d | 64 CPU-days | 2 GPU-days | 1 GPU × 2 d |
| `preempt` | 7 d | none | none | **can be killed at any time** |

Also: at most 1000 submitted and 400 running jobs per user. Account is
`-A eecs` for every partition above (`-A mcs -p dgx` and class accounts exist
but are not used here). Node features: `--constraint=a40|h100|h200|v100|rtx8000|
vram80g|el8|el9|avx512 …`; a constraint does not request the GPU, `--gres=gpu:1`
does.

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

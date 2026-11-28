#!/bin/bash
# A CPU-only allocation that hosts the monitoring Claude session in tmux, so it
# survives laptop sleep and dropped connections. Login nodes cap processes per
# user (Claude Code fails there with spawn EAGAIN), and an ssh login to a node
# with a GPU worker gets adopted into that worker's job and dies with it.
#   sbatch campaign/claude_host.sh            # prints the job id
#   squeue -j <id> -o %N                      # the node, e.g. cn-e43
#   ssh -t <node> 'tmux new -A -s gadar'      # from the laptop (ProxyJump in ~/.ssh/config)
#   inside tmux: cd /nfs/hpc/share/mangannr/di_ltp/1_di_ltp/ltp && claude --resume
#   detach: Ctrl-b d      reattach: ssh -t <node> 'tmux attach -t gadar'
# Costs 2 CPUs x 7 days of the share CPU cap (1024 CPU-days); no GPU.
#SBATCH -J gadar-claude-host
#SBATCH -A eecs -p share -c 2 --mem=16G -t 7-00:00:00
#SBATCH --constraint=el9 --exclude=cn-e41
#SBATCH -o /nfs/hpc/share/mangannr/di_ltp/1_di_ltp/ltp/logs/claude_host_%j.out
echo "claude host: $(hostname) job $SLURM_JOB_ID $(date)"
tmux new-session -d -s gadar -c /nfs/hpc/share/mangannr/di_ltp/1_di_ltp/ltp
while tmux has-session -t gadar 2>/dev/null; do sleep 300; done
echo "tmux session ended $(date)"

### GABAR

```sh
pip install numpy==1.26.0
pip install protobuf==3.20.0
#pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
#pip install torch-geometric==2.3.1
#pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

#pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
#pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cu118.html

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install torch-geometric==2.6.1
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install tensorboard
pip install torchviz==0.0.2
pip install pytorch-lightning==2.0.1
pip install wandb
pip install icecream
pip install pymimir==0.9.71
pip install pyperplan==2.1
pip install pandas
#for running baselines
pip install termcolor
pip install tarski
pip install clingo
```

### Additional Requirements

For use with pddlgym, we require our fork of [pddlgym](https://anonymous.4open.science/r/pddlgym-0F03/
  ), which houses our custom domains and problems.

Download and build the plan validation tool available at https://github.com/KCL-Planning/VAL, then make a symlink called validate on your path that points to the build/Validate binary, e.g. `ln -s <path to VAL>/build/Validate /usr/local/bin/validate`. If done successfully, running validate on your command line should give an output that starts with the line: `VAL: The PDDL+ plan validation tool`.

## Running the Code

All training and testing is done through the unified script
`train_test_scripts/gabar_run.sh`, or directly via `main.py`. The script
wraps `main.py` with domain presets, run modes, and HP sweep support.

```sh
# See all available commands and options
./train_test_scripts/gabar_run.sh --help

# List predefined domains with their train/test problem counts
./train_test_scripts/gabar_run.sh list-domains
```

### Quick Reference — Single Domain (Published GABAR)

```sh
# 1. Sanity check: verify pipeline runs (CPU, 2 epochs, tiny data, ~30s)
./train_test_scripts/gabar_run.sh toy --domain blocks

# 2. Full train+test on a single domain
./train_test_scripts/gabar_run.sh train_test --domain blocks --epochs 300

# 3. Test a pre-trained model only
./train_test_scripts/gabar_run.sh test --domain blocks --test-problems 10

# 4. Train with ablations
./train_test_scripts/gabar_run.sh train_test --domain blocks --ablation no_ag_no_cd
```

### Run Modes (Cost-Saving for Rented GPUs)

Four execution modes control resource usage via `--run-mode`:

| Mode | Epochs | Data | AMP | Device | Checkpointing | Early Stop |
|------|--------|------|-----|--------|---------------|------------|
| `toy` | 2 | 50 graphs | off | CPU | existing | off |
| `sweep` | 5 | 10% | FP16 | GPU | existing | patience=2 |
| `spot` | as-is | full | FP16 | GPU | every epoch + auto-resume | patience=20 |
| `default` | as-is | full | off | as-is | existing | off |

```sh
# TOY: local dry-run to catch bugs before renting a GPU ($0)
./train_test_scripts/gabar_run.sh toy --domain blocks

# SWEEP: HP tuning on free-tier Colab/Kaggle (10% data, 5 epochs per config)
./train_test_scripts/gabar_run.sh sweep --domain blocks \
    --sweep-lr "0.0005 0.001" --sweep-heads "1 4"

# SPOT: full production on a $0.20/hr spot instance
# Auto-resumes if interrupted, shuts down when done (success or crash)
./train_test_scripts/gabar_run.sh spot --domain blocks --epochs 300 --auto-shutdown

# Override early stopping patience for any mode
./train_test_scripts/gabar_run.sh train_test --domain blocks --run-mode spot --patience 30
```

**SPOT workflow:** Spot instances save a `latest_checkpoint.pt` after every
epoch. If the instance is evicted, restart with the same command — training
resumes automatically from where it stopped. All stdout/stderr is logged to
`cache/results/<domain>/<domain>_<timestamp>_<hyperparams>_spot.txt`.

**Auto-shutdown:** `--auto-shutdown` runs `sudo shutdown -h now` on exit
(success or crash). All output is already in the log file, and your disk
persists after shutdown on most cloud providers.

### HP Sweep

The `sweep` command loops over hyperparameter combinations. Each run uses
sweep mode (10% data, 5 epochs, AMP, early stopping).

```sh
# Sweep over learning rates and attention heads (2x3 = 6 configs)
./train_test_scripts/gabar_run.sh sweep --domain blocks \
    --sweep-lr "0.0005 0.001" \
    --sweep-heads "1 2 4"

# Full sweep with all knobs
./train_test_scripts/gabar_run.sh sweep --domain blocks \
    --sweep-lr "0.0005 0.001" \
    --sweep-heads "1 4" \
    --sweep-attn-drop "0 0.1" \
    --sweep-decay "0 0.001"

# Dry-run to see what commands would execute
./train_test_scripts/gabar_run.sh sweep --domain blocks \
    --sweep-lr "0.0005 0.001" --sweep-heads "1 4" --dry-run
```

### Batched Evaluation and Verification (temporary apparatus)

Environment variables controlling the evaluation path (all default off):

- `GABAR_BATCH_EVAL=1` - run all learned-model test problems in lockstep with
  one batched model call per round (greedy search only). Per-problem
  `time_taken` becomes wall-clock from batch start; success/plan metrics are
  unchanged.
- `GABAR_USE_PARALLEL_DECODER=1` - route single-state calls through the
  batched decoder (`beam_search_parallel`) instead of `beam_search_v2`.
- `GABAR_TRACE_SCORES=<file>.jsonl` - log every model call's full candidate
  list (scores + action/object token sequences) keyed by (epoch, problem,
  step). Compare two runs with
  `python compare_score_traces.py a.jsonl b.jsonl [--tol 1e-3] [--ignore-order]`.
- `GABAR_CHECK_PARALLEL_BEAM=1` - per rollout state, batch consecutive state
  pairs through the parallel decoder and compare against v2 output inline.
- `GABAR_PROFILE_EVAL=1` - cProfile the first learned-model problem.

### Analysing Results

Log files and results live in `cache/results/<domain>/`. Parse them into a
CSV for comparison:

```sh
python analyse_domain_results.py cache/results/Manyblocks_ipcc_big/ -o results.csv
```

### Direct main.py Usage

The script is a convenience wrapper. You can always call `main.py` directly:

```sh
python main.py --method ltp --domain manyblocks_ipcc_big --all-problems \
    --lr 0.0005 --n-heads 1 --attention-dropout 0 --dropout 0 \
    --weight-decay 0 --epochs 300 --gnn-rounds 9 --batch-size 16 \
    --mode train_test --max-plan-length 500 --run-learned-model True \
    --run-non-optimal True --use-global-node True --run-mode default
```

### Key Arguments

- `--method`: Planning method (ltp, ltp_no_cd, ltp_no_ag, ltp_val, etc.)
- `--mode`: train, test, train_test, visualize
- `--domain`: pddlgym domain name
- `--run-mode`: default, toy, sweep, spot
- `--epochs`, `--lr`, `--n-heads`, `--gnn-rounds`, `--batch-size`: training HPs
- `--dropout`, `--attention-dropout`, `--weight-decay`: regularization
- `--ablation`: main, main_val, no_ag_no_cd
- `--auto-shutdown`: shut down instance on exit
- `--early-stopping-patience N`: stop after N val-loss checks with no improvement

### Jupyter / Interactive Use

```python
from ploi.run_modes import notebook_setup

# Load data and metadata (uses cached plans if available)
graphs, metadata, action_space, args = notebook_setup("manyblocks_ipcc_big")

# Inspect a single graph
sample = graphs[0]
print(sample)
print(f"Node features: {metadata['num_node_features']}")
print(f"Edge features: {metadata['num_edge_features']}")

# Initialize and load a trained model
from main import initialize_model
from ploi.modelutils_ltp import GNN_GRU
import torch

args.num_node_features = metadata['num_node_features']
args.num_edge_features = metadata['num_edge_features']
model = initialize_model(GNN_GRU, args, action_space)

checkpoint = torch.load("models/.../model_e200_....pt", map_location="cpu")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
```

## Multi-Domain Training (Domain-Agnostic Extension)

Extension of GABAR from per-domain to domain-agnostic policies (see CLAUDE.md
for the research plan; claims C1-C5 referenced below). Run the steps in order -
each is a go/no-go gate.

### Step 0 - Unit tests (no GPU / no pddlgym needed)

```sh
python tests/test_multidomain_metadata.py
```

Expect: all tests PASS. Covers the union-vocabulary merge, action-space
collision detection, and domain-arg parsing.

### Step 1 - Parity gate (Phase 0)

The multi-domain harness must reproduce published single-domain GABAR exactly:

```sh
# Via script:
./train_test_scripts/gabar_run.sh multi --domains "manyblocks_ipcc_big" \
    --featurization per_domain --epochs 300 --train-problems 50 --test-problems 5

# Or direct:
python main.py --method ltp --domains manyblocks_ipcc_big --featurization per_domain \
    --all-problems --lr 0.0005 --epochs 300 --gnn-rounds 9 --batch-size 16 \
    --mode train_test --num-train-problems 50 --num-test-problems 5 \
    --max-plan-length 500 --run-learned-model True --use-global-node True
```

Expect: coverage and plan quality identical to the same command with
`--domain manyblocks_ipcc_big` (the single-domain path is untouched code; the
harness only loops it). Any difference is a harness bug - stop and fix before
proceeding. Checkpoints/caches are keyed `MULTI-<domains>`; first run is slow
(plan collection), later runs hit `cache/results/<domain>_unified_cache_*.pkl`.

### Step 2 - Union-vocab baseline training (Phase 1, C1 control)

```sh
# Via script:
./train_test_scripts/gabar_run.sh multi \
    --domains "manyblocks_ipcc_big,gripper_ipcc" \
    --heldout "spanner_learning" --featurization union --epochs 300

# Or direct:
python main.py --method ltp --domains manyblocks_ipcc_big,gripper_ipcc \
    --heldout-domains spanner_learning --featurization union --mode train_test [flags as above]
```

What happens: pass 1 collects each training domain (cached), pass 2
re-featurizes everything with the merged union vocabulary (cache key suffix
`_union`), action spaces are merged (loud error on schema-name collisions),
graphs are shuffled (seeded) before the train/val split.

Expect: healthy training-domain performance (multi-task learning works);
first run does two featurization passes.

### Step 3 - Zero-shot evaluation on held-out domains (C1)

Runs automatically after testing when `--heldout-domains` is given. Each
held-out domain is evaluated with the trained model; its unseen symbols get
no feature slot (`allow_unknown_symbols`), so evaluation runs instead of
crashing. Results are logged under `heldout_<domain>` keys.

Expect for the union baseline: near-zero coverage on held-out domains. That
is the point - this number is the control that every domain-agnostic method
(structural featurization onward) must beat by a clear margin.

### Step 4 - Structural featurization (Phase 2, first real zero-shot signal)

```sh
# Via script:
./train_test_scripts/gabar_run.sh multi \
    --domains "manyblocks_ipcc_big,gripper_ipcc" \
    --heldout "spanner_learning" --featurization structural --epochs 300

# Or direct:
python main.py --method ltp --domains manyblocks_ipcc_big,gripper_ipcc \
    --heldout-domains spanner_learning --featurization structural --mode train_test [flags]
```

Symbols are classed by structural role (predicate arity + static-ness, schema
arity) instead of identity, so a held-out domain's symbols map to trained
feature slots. Expect: training-domain performance may drop somewhat vs union
(the "expressiveness tax" - record it either way, CLAUDE.md rule 4); zero-shot
coverage on held-out domains must beat Step 3's control for claim C1 to be
alive (Phase 2 gate).

### Troubleshooting

- `Action schema collision across domains` - two domains share a schema name;
  rename in the PDDL or drop one domain.
- `per_domain featurization cannot mix domains` - use `--featurization union`
  or `structural` for >1 training domain.
- Stale/corrupt cache: delete `cache/results/<Domain>_unified_cache_*.pkl`
  (plan collection re-runs; slow).
- KeyError on a symbol during training featurization: real bug (training
  domains must know all their symbols) - do not enable tolerant lookups to
  paper over it.
- `RuntimeError: Not compiled with CUDA support` in `scatter_softmax`:
  `torch-scatter` was installed without CUDA. Reinstall matching your torch
  and CUDA versions:
  `pip install torch-scatter -f https://data.pyg.org/whl/torch-<VER>+<CUDA>.html`
  (e.g. `torch-2.6.0+cu126`). Run `python -c "import torch; print(torch.version.cuda)"`
  to check your CUDA version.

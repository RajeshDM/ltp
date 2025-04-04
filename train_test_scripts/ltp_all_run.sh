#!/bin/bash

# Unified Domain Runner Script
# Display usage information
show_usage() {
    echo "Usage: $0 <domain> [options]"
    echo ""
    echo "Arguments:"
    echo "  <domain>             Domain name (e.g., 'klondike_solitaire', 'blocks', etc.)"
    echo ""
    echo "Options:"
    echo "  --method <method>    Method to use (default: 'ltp')"
    echo "  --ablation <ablation>Ablation to run (default: 'main')"
    echo "  --heads <num>        Number of attention heads (default: 4)"
    echo "  --lr <rate>          Learning rate (default: 0.0005)"
    echo "  --decay <rate>       Weight decay rate (default: 0.000)"
    echo "  --attn-drop <rate>   Attention dropout rate (default: 0.05)"
    echo "  --drop <rate>        Regular dropout rate (default: 0)"
    echo "  --gnn <rounds>       Number of GNN rounds (default: 9)"
    echo "  --epochs <num>       Maximum number of epochs (default: 750)"
    echo "  --train <num>        Number of training problems (overrides domain default)"
    echo "  --test <num>         Number of test problems (overrides domain default)"
    echo "  --plan-len <num>     Maximum plan length (default: 500)"
    echo "  --mode <mode>        Mode: 'train_test' or 'test' (default: 'train_test')"
    echo "  --wandb <bool>       Enable/disable wandb (default: 'True' for train_test, 'False' for test)"
    echo "  --global-node <bool> Use global node (default: 'True')"
    echo "  --non-opt <bool>     Run non-optimal (default: 'True')"
    echo ""
    echo "Available predefined domains: blocks, gripper, miconic, visitall, grid, logistics, spanner, klondike_solitaire"
    exit 1
}

# Check for minimum required arguments
if [ "$#" -lt 1 ]; then
    show_usage
fi

# Domain configurations
declare -A domain_configs=(
    ["blocks"]="name=manyblocks_ipcc_big;train=200;test=200"
    ["gripper"]="name=gripper_ipcc;train=147;test=173"
    ["miconic"]="name=miconic_ipcc;train=228;test=119"
    ["visitall"]="name=visitall_ipcc;train=125;test=50"
    ["grid"]="name=grid_ipcc;train=192;test=48"
    ["logistics"]="name=logistics_ipcc;train=156;test=96"
    ["spanner"]="name=spanner_ipcc;train=234;test=96"
    ["rovers"]="name=spanner_ipcc;train=312;test=58"
    ["klondike_solitaire"]="name=klondike_solitaire;train=150;test=99"
)

# Parse positional arguments
DOMAIN=$1
shift 1

# Default values (from klondike_solitaire.sh)
METHOD="ltp"
heads=(2 4)
lrs=(0.0005)
decays=(0.000)
other_drops=(0)
gnn_rounds=(9)
epochs=750
max_plan_length=500
mode="train_test"
wandb="True"
g_node="True"
non_opt="True"
ablation="main"

# Custom train/test numbers (will override domain defaults if set)
CUSTOM_TRAIN=""
CUSTOM_TEST=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --ablation)
            ablation="$2"
            shift 2
            ;;
        --heads)
            heads=($2)
            shift 2
            ;;
        --lr)
            lrs=($2)
            shift 2
            ;;
        --decay)
            decays=($2)
            shift 2
            ;;
        --attn-drop)
            attn_drops=($2)
            shift 2
            ;;
        --attn-drop-list)
            # For comma-separated list
            IFS=',' read -r -a CUSTOM_ATTN_DROP_LIST <<< "$2"
            CUSTOM_ATTN_DROP="${CUSTOM_ATTN_DROP_LIST[@]}"  # Convert to space-separated
            shift 2
            ;;
        --drop)
            other_drops=($2)
            shift 2
            ;;
        --gnn)
            gnn_rounds=($2)
            shift 2
            ;;
        --epochs)
            epochs=$2
            shift 2
            ;;
        --train)
            CUSTOM_TRAIN="$2"
            shift 2
            ;;
        --test)
            CUSTOM_TEST="$2"
            shift 2
            ;;
        --plan-len)
            max_plan_length=$2
            shift 2
            ;;
        --mode)
            mode=$2
            if [ "$mode" = "test" ]; then
                wandb="False"
            fi
            shift 2
            ;;
        --wandb)
            wandb=$2
            shift 2
            ;;
        --global-node)
            g_node=$2
            shift 2
            ;;
        --non-opt)
            non_opt=$2
            shift 2
            ;;
        --*)
            echo "Unknown option: $1"
            show_usage
            ;;
        *)
            echo "Unknown argument: $1"
            show_usage
            ;;
    esac
done

# Set attention dropout rates based on ablation type
if [[ ! -z "$CUSTOM_ATTN_DROP" ]]; then
    # If user explicitly set attn-drop, use that value
    attn_drops=($CUSTOM_ATTN_DROP)
    echo "Using custom attention dropout rate: ${CUSTOM_ATTN_DROP}"
else
    # Otherwise set based on ablation type
    case $ablation in
        "main")
            attn_drops=(0.1 0.2)
            echo "Using attention dropout rates for 'main' ablation: 0.1 0.2"
            ;;
        *)
            # For all other ablation values
            attn_drops=(0.0 0.1)
            echo "Using attention dropout rates for non-main ablation: 0.0 0.1"
            ;;
    esac
fi

# Get domain configuration
CONFIG=${domain_configs[$DOMAIN]}
if [[ -z "$CONFIG" ]]; then
    echo "Warning: Custom domain '$DOMAIN' specified. Make sure to provide --train and --test values."
    FULL_DOMAIN_NAME=$DOMAIN
    
    # For custom domains, train and test numbers are required
    if [[ -z "$CUSTOM_TRAIN" || -z "$CUSTOM_TEST" ]]; then
        echo "Error: For custom domains, you must specify --train and --test values"
        show_usage
    fi
    
    num_train_problems=$CUSTOM_TRAIN
    num_test_problems=$CUSTOM_TEST
else
    # Parse domain configuration into separate variables
    FULL_DOMAIN_NAME=$(echo "$CONFIG" | grep -o 'name=[^;]*' | cut -d'=' -f2)
    DEFAULT_NUM_TRAIN=$(echo "$CONFIG" | grep -o 'train=[^;]*' | cut -d'=' -f2)
    DEFAULT_NUM_TEST=$(echo "$CONFIG" | grep -o 'test=[^;]*' | cut -d'=' -f2)
    
    # Use custom values if provided, otherwise use defaults
    num_train_problems=${CUSTOM_TRAIN:-$DEFAULT_NUM_TRAIN}
    num_test_problems=${CUSTOM_TEST:-$DEFAULT_NUM_TEST}
fi

# Create timestamp (from klondike_solitaire.sh)
timestamp=$(date +"%Y_%m_%d_%H_%M")

# Create cache directory if it doesn't exist
mkdir -p "cache/results/${FULL_DOMAIN_NAME}"

# Main execution loop following the structure from klondike_solitaire.sh
if [ "$mode" = "train_test" ] || [ "$mode" = "train" ]; then
    wandb_value=$wandb
    for head in ${heads[@]}; do
        for decay in ${decays[@]}; do
            for attn_drop in ${attn_drops[@]}; do
                for other_drop in ${other_drops[@]}; do
                    for lr in ${lrs[@]}; do
                        for gnn_round in ${gnn_rounds[@]}; do
                            echo "python main.py --domain ${FULL_DOMAIN_NAME} --all-problems --lr $lr --n-heads $head --attention-dropout $attn_drop --dropout $other_drop --weight-decay $decay --wandb ${wandb_value} --mode ${mode} --method ${METHOD} --gnn-rounds $gnn_round --epochs $epochs --num-test-problems $num_test_problems --num-train-problems ${num_train_problems} --max-plan-length ${max_plan_length} --use-global-node ${g_node} --run-non-optimal ${non_opt} --ablation ${ablation}"
                            
                            python main.py --domain ${FULL_DOMAIN_NAME} --all-problems --lr $lr --n-heads $head --attention-dropout $attn_drop --dropout $other_drop --weight-decay $decay --wandb ${wandb_value} --mode ${mode} --method ${METHOD} --gnn-rounds $gnn_round --epochs $epochs --num-test-problems $num_test_problems --num-train-problems ${num_train_problems} --max-plan-length ${max_plan_length} --use-global-node ${g_node} --run-non-optimal ${non_opt} --ablation ${ablation} |& tee -a "cache/results/${FULL_DOMAIN_NAME}/lr_${lr}_n_heads_${head}_attn_drop_${attn_drop}_drop_${other_drop}_decay_${decay}_g_node_${g_node}_ablation_${ablation}.txt"
                        done
                    done
                done
            done
        done
    done
fi

if [ "$mode" = "train_test" ]; then
    mode="test"
    wandb="False"
fi

if [ "$mode" = "test" ]; then
    for head in ${heads[@]}; do
        for decay in ${decays[@]}; do
            for attn_drop in ${attn_drops[@]}; do
                for other_drop in ${other_drops[@]}; do
                    for lr in ${lrs[@]}; do
                        for gnn_round in ${gnn_rounds[@]}; do
                            echo "python main.py --domain ${FULL_DOMAIN_NAME} --all-problems --lr $lr --n-heads $head --attention-dropout $attn_drop --dropout $other_drop --weight-decay $decay --wandb ${wandb} --mode ${mode} --method ${METHOD} --gnn-rounds $gnn_round --epochs $epochs --num-test-problems $num_test_problems --num-train-problems ${num_train_problems} --max-plan-length=${max_plan_length} --use-global-node ${g_node} --run-non-optimal ${non_opt} --ablation ${ablation}"
                            
                            python main.py --domain ${FULL_DOMAIN_NAME} --all-problems --lr $lr --n-heads $head --attention-dropout $attn_drop --dropout $other_drop --weight-decay $decay --wandb ${wandb} --mode ${mode} --method ${METHOD} --gnn-rounds $gnn_round --epochs $epochs --num-test-problems $num_test_problems --num-train-problems ${num_train_problems} --max-plan-length=${max_plan_length} --use-global-node ${g_node} --run-non-optimal ${non_opt} --ablation ${ablation}
                        done
                    done
                done
            done
        done
    done
fi

echo "Script execution completed for domain: ${FULL_DOMAIN_NAME}"
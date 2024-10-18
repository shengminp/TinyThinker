#!/bin/bash
echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Constants
BASE_MODEL="google-t5/t5-small"
DATA_NAME="strategyqa"
STAGE_TYPE="knowledge_reasoning"
SFT_CHECKPOINT_DIR="checkpoint-4400"
TRAINING_PER_GPU_BATCH_SIZE=32
LEARNING_RATE=0.000005
GENERATION_PER_GPU_BATCH_SIZE=256
GENERATION_TIMES=10
DPO_ITERS=(3 4 5)

# Function to extract the best checkpoint from trainer_state.json
extract_best_checkpoint() {
    local trainer_state_path=$1
    python -c "
import json
import os

def extract_best_checkpoint(trainer_state_path):
    with open(trainer_state_path, 'r') as file:
        data = json.load(file)
        best_model_checkpoint = data['best_model_checkpoint']
        last_checkpoint = os.path.basename(best_model_checkpoint)
        return last_checkpoint

trainer_state_path = '${trainer_state_path}'
print(extract_best_checkpoint(trainer_state_path))
"
}


# Function to run the generation script
run_generate() {
    local base_model=$1
    local data_name=$2
    local training_type=$3
    local stage_type=$4
    local dpo_iter=$5
    local checkpoint_dir=$6
    local per_gpu_batch_size=$7
    local generation_times=$8

    python generate.py \
        --base_model "$base_model" \
        --data_name "$data_name" \
        --training_type "$training_type" \
        --stage_type "$stage_type" \
        --dpo_iter "$dpo_iter" \
        --checkpoint_dir "$checkpoint_dir" \
        --per_gpu_batch_size "$per_gpu_batch_size" \
        --generation_type "random" \
        --generation_file "dpo" \
        --generation_times "$generation_times"
}

# Function to run the DPO script
run_dpo() {
    local base_model=$1
    local ref_model=$2
    local data_name=$3
    local stage_type=$4
    local dpo_iter=$5
    local per_gpu_batch_size=$6
    local learning_rate=$7

    python dpo.py \
        --base_model "$base_model" \
        --ref_model "$ref_model" \
        --data_name "$data_name" \
        --training_type "dpo" \
        --stage_type "$stage_type" \
        --dpo_iter "$dpo_iter" \
        --per_gpu_batch_size "$per_gpu_batch_size" \
        --learning_rate "$learning_rate"
}

# Main loop to iterate through DPO iterations
for DPO_ITER in "${DPO_ITERS[@]}"; do
    if [ $DPO_ITER -eq 1 ]; then
        CHECKPOINT_DIR=$SFT_CHECKPOINT_DIR
    else
        CHECKPOINT_DIR=$DPO_CHECKPOINT_DIR
    fi

    if [ $DPO_ITER -eq 1 ]; then
        run_generate "$BASE_MODEL" "$DATA_NAME" "sft" "$STAGE_TYPE" "$DPO_ITER" "$CHECKPOINT_DIR" "$GENERATION_PER_GPU_BATCH_SIZE" "$GENERATION_TIMES"
    else
        run_generate "$BASE_MODEL" "$DATA_NAME" "dpo" "$STAGE_TYPE" "$DPO_ITER" "$CHECKPOINT_DIR" "$GENERATION_PER_GPU_BATCH_SIZE" "$GENERATION_TIMES"
    fi

    run_dpo "$BASE_MODEL" "$CHECKPOINT_DIR" "$DATA_NAME" "$STAGE_TYPE" "$DPO_ITER" "$TRAINING_PER_GPU_BATCH_SIZE" "$LEARNING_RATE"

    TRAINER_STATE_PATH="/scratch/shengmin/models/${DATA_NAME}/${BASE_MODEL}/dpo/${STAGE_TYPE}/iter_${DPO_ITER}/trainer_state.json"
    DPO_CHECKPOINT_DIR=$(extract_best_checkpoint "$TRAINER_STATE_PATH")
done

echo "###"
echo "### END DATE=$(date)"
#!/bin/bash
echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
python generate.py \
    --base_model google-t5/t5-base \
    --data_name csqa \
    --training_type "dpo" \
    --stage_type "recall_analyze" \
    --dpo_iter 5 \
    --checkpoint_dir "checkpoint-2730"\
    --per_gpu_batch_size 128 \
    --generation_type "random" \
    --generation_file "dpo" \
    --generation_times 10

echo "###"
echo "### END DATE=$(date)"

#!/bin/bash
echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

data_names=("csqa")
stage_types=("recall_summarize" "analyze_summarize")

for data_name in "${data_names[@]}"; do
  for stage_type in "${stage_types[@]}"; do
    echo "Running finetune.py with data_name: $data_name and stage_type: $stage_type"
    
    python finetune.py \
      --base_model google-t5/t5-small \
      --data_name $data_name \
      --training_type "sft" \
      --stage_type $stage_type \
      --per_gpu_batch_size 64 \
      --num_train_epochs 10 \
      --learning_rate 0.0005 \
      --interval 100

  done
done

echo "###"
echo "### END DATE=$(date)"

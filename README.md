# :thinking: TinyThinker

Official code for "**TinyThinker: Distilling Reasoning through Coarse-to-Fine Knowledge Internalization with Self-Reflection"**

## :bookmark_tabs: Table of Contents

- :hammer_and_wrench: [Getting Started](#hammer_and_wrench-getting-started)
  - :clipboard: [Prerequisites](#clipboard-prerequisites)
  - :gear: [Installing](#gear-installing)
- :rocket: [Running TinyThinker](#rocket-running-tinythinker)
  - :memo: [Prompt Engineering](#memo-prompt-engineering)
  - :dart: [Train TinyThinker](#dart-train-tinythinker)
  - :hourglass_flowing_sand: [Inference](#hourglass_flowing_sand-inference)
- :page_facing_up: [License](#page_facing_up-license)

## :hammer_and_wrench: Getting Started

### :clipboard: Prerequisites

* Pytorch version == 2.1.0
* Python version == 3.10.x

### :gear: Installing

To set up the environment, use the following commands:

```
git clone https://github.com/shengminp/TinyThinker.git
cd TinyThinker
conda env create -f environment.yml
```

After installation, your project directory structure should look like this:
    
    .
    ├── datasets
    │   ├── csqa
    |   │   ├── final # the dataset for training TinyThinker
    |   │   ├── original # original dataset for each dataset
    │   |   └── prompt # the designed prompt for prompt engineering
    │   ├── obqa
    |   │   ├── final
    |   │   ├── original
    │   |   └── prompt
    │   |── strategyqa
    |   │   ├── final
    |   │   ├── original
    │   |   └── prompt
    │   ├── openai_request.py # Send request and collect response from OpenAI
    │   ├── Prepare Ablation.ipynb # prepare datast for ablation study
    │   ├── Prompt Engineering.ipynb # prepare dataset for training
    │   └── request.json # the instruction which will be used in prompt
    ├── models # save the checkpoint
    ├── results # save the generation result during inference
    ├── scripts
    │   ├── utils
    |   │   ├── __init__.py
    |   │   ├── config.py # the configuration file of TinyThinker
    │   |   └── trainer.py # the customized Huggingface trainer for TinyThinker
    │   ├── dpo.py # manage for running dpo
    │   ├── finetune.py # manage for training TinyThikner
    │   ├── generate.py # manage for generating data
    │   ├── run_dpo.sh # script for running iterative dpo process
    │   ├── run_finetune.sh # script for running training process
    │   └── run_generate.sh # script for runing generation process
    └── README.md

## :rocket: Running TinyThinker

### :memo: Prompt Engineering
1. Download the dataset from its official site and place it under ./datasets/DATASET/original.
2. Run ./datasets/Prompt Engineering.ipynb to prepare prompts for the dataset.
3. Execute ./datasets/openai_request.py to generate responses from OpenAI:
   - If the generated responses contain errors, repeat steps 2 and 3 as needed (see details in Prompt Engineering.ipynb).
4. Once completed, the prepared dataset will be located at ./datasets/DATASET/final.
5. Use Prepare Ablation.ipynb to prepare the ablation study dataset, with results saved at ./datasets/DATASET/final.
   
### :dart: Train TinyThinker
#### **Phase-1: Reasoning Acquisition**
In this phase, a T5 model is fine-tuned using a three-stage process. Run the following command:
```
python finetune.py \
      --base_model $model_name \
      --data_name $data_name \
      --training_type sft \
      --stage_type $stage_type \
      --per_gpu_batch_size $batch_size \
      --num_train_epochs $num_epochs \
      --learning_rate $lr \
      --interval $interval
```
- **$model_name:** Model name from Huggingface (google-t5/t5-small, google-t5/t5-base, google-t5/t5-large, google-t5/t5-3b, google-t5/t5-11b).
- **$data_name:** Dataset name (csqa, obqa, strategyqa).
- **$stage_type:** Training stage (summarize, recall_summarize, analyze_summarize, recall_analyze_summarize).
- **$interval:** Interval value between stages.

#### **Phase-2: Self-Reflection**
In this phase, we refine the reasoning through self-generated data using DPO. Run:
```
python dpo.py \
      --base_model $base_model \
      --ref_model $ref_model \
      --data_name $data_name \
      --training_type dpo \
      --stage_type $stage_type \
      --dpo_iter $dpo_iter \
      --per_gpu_batch_size $per_gpu_batch_size \
      --learning_rate $learning_rate
```
- **$base_name:** Model name (google-t5/t5-small, google-t5/t5-base, google-t5/t5-large, google-t5/t5-3b, google-t5/t5-11b).
- **$ref_model:** Path to the reference model checkpoint.
- **$data_name:** Dataset name (csqa, obqa, strategyqa).
- **$stage_type:** Training stage (recall, analyze, recall_analyze).
- **$dpo_iter:** Current iteration timestamp.

For iterative DPO, use the run_dpo.sh script.

### :hourglass_flowing_sand: Inference
Use the following command to generate inferences:
```
python generate.py \
    --base_model $base_model \
    --data_name $data_name \
    --training_type $type_name \
    --stage_type $stage_type \
    --dpo_iter $dpo_iter \
    --checkpoint_dir $checkpoint_path\
    --per_gpu_batch_size $batch_size \
    --generation_type $generation_type \
    --generation_file $generation_file \
    --generation_times $generation_times
```
- **$base_name:** Model name (google-t5/t5-small, google-t5/t5-base, google-t5/t5-large, google-t5/t5-3b, google-t5/t5-11b).
- **$data_name:** Dataset name (csqa, obqa, strategyqa).
- **$type_name:** Phase type (sft, dpo).
- **$stage_type:** Inference stage (recall, analyze, summarize, recall_summarize, analyze_summarize, recall_analyze, recall_analyze_summarize).
- **$dpo_iter:** Iteration timestamp.
- **$generation_type:** Generation type (greedy, random).
- **$generation_file:** File type (test, dpo).
- **$generation_times:** Number of generation attempts.

## :page_facing_up: License

This project is licensed under the [MIT](LICENSE) © Shengmin Piao

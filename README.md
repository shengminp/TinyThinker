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

Creating an environment with commands.

```
git clone https://github.com/shengminp/TinyThinker.git
cd TinyThinker
conda env create -f environment.yml
```

After the overall installation, make sure the directory of the project is as follows:
    
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
1. Download dataset from corresponding official site and place it at location "./datasets/DATASET/original".
2. Run "./datasets/Prompt Engineering.ipynb" to prepare the prompt for each dataset.
3. Run ".datasets/openai_request.py" to request OpenAI generate reponse for the prepared prompt.
   - Due to the reason that the generated response might have some error, continue the step 2 and 3 if you have enough budget
   - The continue step is detailed in "./datasets/Prompt Engineering.ipynb"
4. After 3, the final version for training is located at "./datasets/DATASET/final"
5. Run "./datasets/Prepare Ablation.py" to prepare dataset for ablation study, result will be located at "./datasets/DATASET/final"

### :dart: Train TinyThinker
#### **Phase-1: Reasoning Acquisition**
In this phase, we finetune a T5 model to learn how to learn reasoning by three-stage process.
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
- *$model_name:* model name from Huggingface, only support 'google-t5/t5-small', 'google-t5/t5-base', 'google-t5/t5-large', 'google-t5/t5-3b', 'google-t5/t5-11b'
- *$data_name:* dataset name for training, only support 'csqa', 'obqa', 'strategyqa'
- *$stage_type:* stage type for training, only support 'summarize', 'recall_summarize', 'analyze_summarize', 'recall_analyze_summarize'
- *$batch_size:* batch size for training
- *$num_epochs:* epochs number for training
- *$lr:* learning rate for training
- *$interval:* interval value for each stage in TinyThinker

#### **Phase-2: Self-Reflection**
In this phase, we refine the learned reasoning by self-generated data.
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
- *$base_name:* model name from Huggingface, only support 'google-t5/t5-small', 'google-t5/t5-base', 'google-t5/t5-large', 'google-t5/t5-3b', 'google-t5/t5-11b'
- *$ref_model:* checkpoint path of reference model
- *$data_name:* dataset name for training, only support 'csqa', 'obqa', 'strategyqa'
- *$stage_type:* stage type for training, only support 'recall', 'analyze', 'recall_analyze'
- *$dpo_iter:* current timestamp of iteration
- *$per_gpu_batch_size:* batch size for training
- *$learning_rate:* learning rate for training

This is the basic file for run single iteration of DPO, if you want to run iterative dpo please use "run_dpo.sh".

### :hourglass_flowing_sand: Inference
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
- *$base_name:* model name from Huggingface, only support 'google-t5/t5-small', 'google-t5/t5-base', 'google-t5/t5-large', 'google-t5/t5-3b', 'google-t5/t5-11b'
- *$data_name:* dataset name for inference, only support 'csqa', 'obqa', 'strategyqa'
- *$type_name:* phase type for inference, only support 'sft', 'dpo'
- *$stage_type:* stage type for inference, only support 'recall', 'analyze', 'summarize', 'recall_summarize', 'analyze_summarize', 'recall_analyze', 'recall_analyze_summarize'
- *$dpo_iter:* current timestamp of iteration
- *$checkpoint_path:* path of checkpoint to run inference
- *$batch_size:* batch size for inference
- *$generation_type:* generation type for inference, only support 'greedy', 'random'
- *$generation_file:* generation file for inference, only support 'test', 'dpo'
- *$generation_times:* generation times for inference


## :page_facing_up: License

[MIT](LICENSE) © Shengmin Piao

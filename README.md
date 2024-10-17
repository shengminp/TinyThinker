# :star2:TinyThinker

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
    |   │   ├── final
    |   │   ├── original
    │   |   └── prompt
    │   ├── obqa
    |   │   ├── final
    |   │   ├── original
    │   |   └── prompt
    │   |── strategyqa
    |   │   ├── final
    |   │   ├── original
    │   |   └── prompt
    │   ├── openai_request.py
    │   ├── Prepare Ablation.ipynb
    │   ├── Prompt Engineering.ipynb
    │   └── request.json
    ├── models   
    ├── results
    ├── scripts
    │   ├── utils
    |   │   ├── __init__.py
    |   │   ├── config.py
    │   |   └── trainer.py
    │   ├── dpo.py
    │   ├── finetune.py
    │   ├── generate.py
    │   ├── run_dpo.sh
    │   ├── run_finetune.sh
    │   └── run_generate.sh
    └── README.md

## :rocket: Running TinyThinker

In the following code, the values that can be used in {PROPERTY} are "drd2" and "qed".

### :memo: Prompt Engineering
```
python preprocess.py \
    --source-lang low\
    --target-lang high\
    --user-dir fairseq_mo \
    --task molecule_lev \
    --trainpref dataset/{PROPERTY}/aug_data/train\
    --validpref dataset/{PROPERTY}/aug_data/valid\
    --testpref dataset/{PROPERTY}/aug_data/test\
    --destdir dataset/{PROPERTY}/bin_data \
    --joined-dictionary\
    --workers 1\
    --padding-factor 1
```

### :dart: Train TinyThinker
```
python preprocess.py \
    --source-lang low\
    --target-lang high\
    --user-dir fairseq_mo \
    --task molecule_lev \
    --trainpref dataset/{PROPERTY}/aug_data/train\
    --validpref dataset/{PROPERTY}/aug_data/valid\
    --testpref dataset/{PROPERTY}/aug_data/test\
    --destdir dataset/{PROPERTY}/bin_data \
    --joined-dictionary\
    --workers 1\
    --padding-factor 1
```

### :hourglass_flowing_sand: Inference
```
python preprocess.py \
    --source-lang low\
    --target-lang high\
    --user-dir fairseq_mo \
    --task molecule_lev \
    --trainpref dataset/{PROPERTY}/aug_data/train\
    --validpref dataset/{PROPERTY}/aug_data/valid\
    --testpref dataset/{PROPERTY}/aug_data/test\
    --destdir dataset/{PROPERTY}/bin_data \
    --joined-dictionary\
    --workers 1\
    --padding-factor 1
```


## :page_facing_up:License

[MIT](LICENSE) © Shengmin Piao

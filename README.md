# :star2:TinyThinker

Official code for "**TinyThinker: Distilling Reasoning through Coarse-to-Fine Knowledge Internalization with Self-Reflection"**

## :bookmark_tabs: Table of Contents

- :hammer_and_wrench: [Getting Started](#getting-started)
  - :clipboard: [Prerequisites](#prerequisites)
  - :gear: [Installing](#installing)
- [:rocket: Running TinyThinker](#-running-tinythinker)
  - :memo: [Prompt Engineering](#prompt-engineering)
  - :dart: [Train TinyThinker](#train-tinythinker)
  - :hourglass_flowing_sand: [Inference](#inference)
- :page_facing_up: [License](#license)

## :hammer_and_wrench: Getting Started

### :clipboard: Prerequisites

* Pytorch version == 2.1.0
* Python version == 3.10.x

### :gear: Installing

Creating an environment with commands.

```
git clone https://github.com/sungmin630/SELF-EdiT.git
cd SELF-EdiT
conda env create -f environment.yml
```

Cloning and installing the following projects.

```
git clone https://github.com/sungmin630/fairseq.git
cd fairseq
# Make sure to set the CUDA_HOME in your environment to use the lib_nat.
pip install --editable ./
python setup.py build_ext --inplace
```

After the overall installation, make sure the directory of the project is as follows:
    
    .
    ├── checkpoints
    │   ├── drd2
    |   │   ├── mo_lev
    │   |   └── simcse
    │   └── qed
    |       ├── mo_lev
    │       └── simcse
    ├── dataset
    │   ├── drd2
    |   │   ├── aug_data
    |   │   ├── bin_data
    │   |   └── emb_data
    │   ├── qed
    |   │   ├── aug_data
    |   │   ├── bin_data
    │   │   └── emb_data
    │   └── ...    
    ├── fairseq
    ├── fairseq_mo
    ├── results
    ├── envurinment.yml
    ├── preprocess.py
    ├── train_simcse.py
    └── README.md

## :rocket: Running TinyThinker

In the following code, the values that can be used in {PROPERTY} are "drd2" and "qed".

### :file_folder: Preprocess the raw dataset to binarized dataset and generate vocabulary

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

### Run the SimCSE to get embeddings of SELFragments
First, run the code [/dataset/prepare_data_for_SimCSE.ipynb](https://github.com/sungmin630/SELF-EdiT/blob/main/dataset/prepare_data_for_SimCSE.ipynb)

Then, run the following code:
```
python train_simcse.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --tokenizer_name dataset/{PROPERTY}/emb_data/tokenizer \
    --train_file dataset/{PROPERTY}/emb_data/tokens.txt \
    --max_seq_length 50 \
    --output_dir checkpoints/{PROPERTY}/simcse
```

Finally, run the code [/dataset/extract_embedding from_SimCSE.ipynb](https://github.com/sungmin630/SELF-EdiT/blob/main/dataset/extract_embedding_from_SimCSE.ipynb)

### Train the SELF-EdiT

```
fairseq-train \
    dataset/{PROPERTY}/bin_data \
    --save-dir checkpoints/{PROPERTY}/mo_lev \
    --user-dir fairseq_mo \
    --task molecule_lev \
    --criterion nat_loss \
    --arch selfedit_transformer \
    --noise no_noise \
    --share-all-embeddings \
    --encoder-embed-dim 768 \
    --encoder-embed-path dataset/{PROPERTY}/emb_data/dict.emb \
    --decoder-embed-path dataset/{PROPERTY}/emb_data/dict.emb \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0001 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 1000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --apply-bert-init \
    --log-format 'simple' \
    --log-interval 100 \
    --log-file checkpoints/{PROPERTY}/mo_lev/logs\
    --max-tokens 8000 \
    --save-interval 10 \
    --max-update 120000\
    --disable-validation
```

### Generate the molecules by trained models

```
fairseq-generate \
    dataset/{PROPERTY}/bin_data \
    --gen-subset test \
    --user-dir fairseq_mo \
    --task molecule_lev \
    --path checkpoints/{PROPERTY}/mo_lev/{file_name} \
    --iter-decode-max-iter {ITER_NUM} \
    --results-path results/{PROPERTY}/{dir_name} \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --batch-size 400
```

## :page_facing_up:License

[MIT](LICENSE) © Shengmin Piao

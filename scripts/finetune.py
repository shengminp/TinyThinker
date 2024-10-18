import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, '../../'))
import re
import sys
import json
import logging
import datasets
import argparse
import transformers
import numpy as np
from datasets import load_dataset
from transformers import (
    GenerationConfig,
    T5Tokenizer,
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from utils import (
    SFTTrainer,
    SFTConfiguration,
    SFTTrainingArguments
)

logger = logging.getLogger(__name__)


def setup_logger(config):
    generation_configs = {
        "recall": GenerationConfig(max_length=config.max_length, do_sample=False),
        "analyze": GenerationConfig(max_length=config.max_length, do_sample=False),
        "summarize": GenerationConfig(max_length=config.max_length, do_sample=False)
    }

    training_args = SFTTrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy=config.all_strategy,
        per_device_train_batch_size=config.per_gpu_batch_size,
        per_device_eval_batch_size=config.per_gpu_batch_size * 4,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        log_level=config.log_level,
        logging_strategy=config.all_strategy,
        logging_first_step=True,
        logging_steps=config.all_strategy_steps,
        save_strategy=config.all_strategy,
        save_steps=config.all_strategy_steps,
        save_total_limit=None,
        save_only_model=True,
        eval_steps=config.all_strategy_steps,
        metric_for_best_model=config.metric_name,
        load_best_model_at_end=True,
        disable_tqdm=False,
        remove_unused_columns=True,
        predict_with_generate=True,
        # for 3 stage sft
        stage_type=config.stage_type,
        recall_interval=config.recall_interval,
        analyze_interval=config.analyze_interval,
        summarize_interval=config.summarize_interval,
        recall_generation_config=generation_configs["recall"],
        analyze_generation_config=generation_configs["analyze"],
        summarize_generation_config=generation_configs["summarize"],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    
    return training_args


def setup_model_and_tokenizer(config):
    """
    Setup model and tokenizer based on training configuration.
    """
    # Set seed before initializing model.
    set_seed(config.seed)
    if "t5" in config.base_model:
        model = T5ForConditionalGeneration.from_pretrained(config.base_model)
        tokenizer = T5Tokenizer.from_pretrained(config.base_model, legacy=True)
    '''elif "gpt2" in config.base_model:
        model = GPT2LMHeadModel.from_pretrained(config.base_model)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, padding_side='left', legacy=True)
        tokenizer.pad_token = tokenizer.eos_token'''
    return model, tokenizer


def load_and_prepare_data(config, tokenizer, training_args):
    """
    Load and prepare dataset for training.
    """
    def preprocess_function_seq2seq(dataset):
        if "analyze" in dataset.keys():
            model_inputs = tokenizer(dataset["input"], text_target=dataset["analyze"], padding=True)
        elif "recall" in dataset.keys():
            model_inputs = tokenizer(dataset["input"], text_target=dataset["recall"], padding=True)
        elif "summarize" in dataset.keys():
            model_inputs = tokenizer(dataset["input"], text_target=dataset["summarize"], padding=True)
        else:
            model_inputs = tokenizer(dataset["input"], padding=True)
        return model_inputs
    
    '''def preprocess_function_clm(dataset):
        if "analyze" in dataset.keys():
            input_texts = [f"{i} {r}{tokenizer.eos_token}" for i, r in zip(dataset["input"], dataset["analyze"])]
            model_inputs = tokenizer(input_texts)
        elif "recall" in dataset.keys():
            input_texts = [f"{i} {k}{tokenizer.eos_token}" for i, k in zip(dataset["input"], dataset["recall"])]
            model_inputs = tokenizer(input_texts)
        elif "summarize" in dataset.keys():
            input_texts = [f"{i} {s} {tokenizer.eos_token}" for i, s in zip(dataset["input"], dataset["summarize"])]
            model_inputs = tokenizer(input_texts)
        else:
            model_inputs = tokenizer(dataset["input"], padding='max_length', max_length=128)
        return model_inputs'''
    
    datasets = {split: load_dataset("json", data_files=path, split='train') for split, path in config.dataset_path.items()}

    with training_args.main_process_first():
        if "t5" in config.base_model:
            tokenized_datasets = {name: data.map(preprocess_function_seq2seq, batched=True) for name, data in datasets.items()}
        '''elif "gpt2" in config.base_model:
            tokenized_datasets = {name: data.map(preprocess_function_clm, batched=True) for name, data in datasets.items()}'''

    return tokenized_datasets

        
def run_sft(training_args, model, tokenizer, tokenized_datasets):
    """
    Main training loop.
    """
    def _compute_accuracy(total_preds):
        if hasattr(total_preds, 'predictions') and hasattr(total_preds, 'label_ids'):
            # for PredictionOutput()
            preds, ground_truth = total_preds.predictions, total_preds.label_ids
        else:
            # for EvalPrediction()
            preds, ground_truth = total_preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        pattern = re.compile(r'the answer is option ([A-Z])\.')
        predictions = [pattern.search(result).group(1) if pattern.search(result) else None for result in decoded_preds]
        correct_count = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
        accuracy = (correct_count / len(ground_truth)) * 100
        return {
            'summarize': decoded_preds,
            'prediction': predictions,
            'label': ground_truth,
            'accuracy': accuracy
        }

    if training_args.stage_type == "recall_analyze_summarize":
        train_dataset_list = [tokenized_datasets['recall'], tokenized_datasets['analyze'], tokenized_datasets['summarize']]
    elif training_args.stage_type == "summarize":
        train_dataset_list = [tokenized_datasets['summarize']]
    elif training_args.stage_type == "recall_summarize":
        train_dataset_list = [tokenized_datasets['recall'], tokenized_datasets['summarize']]
    elif training_args.stage_type == "analyze_summarize":
        train_dataset_list = [tokenized_datasets['analyze'], tokenized_datasets['summarize']]
    
    if "t5" in config.base_model:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset_list,
            eval_dataset=tokenized_datasets['valid'],
            tokenizer=tokenizer,
            compute_metrics=_compute_accuracy
        )
    '''elif "gpt2" in config.base_model:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            mlm=False
        )
        training_args.recall_generation_config.pad_token_id = tokenizer.pad_token_id
        training_args.analyze_generation_config.pad_token_id = tokenizer.pad_token_id
        training_args.summarize_generation_config.pad_token_id = tokenizer.pad_token_id

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset_list,
            eval_dataset=tokenized_datasets['valid'],
            tokenizer=tokenizer,
            compute_metrics=_compute_accuracy
        )'''

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Generation")

    parser.add_argument('--base_model', type=str, required=True, help='Base model name')
    parser.add_argument('--data_name', type=str, required=True, help='Data name')
    parser.add_argument('--training_type', type=str, required=True, help='Training type')
    parser.add_argument('--stage_type', choices=['summarize', 'recall_summarize', 'analyze_summarize', 'recall_analyze_summarize'], type=str, required=True, help='Stage type')
    parser.add_argument('--per_gpu_batch_size', type=int, required=True, help='Batch size per GPU')
    parser.add_argument('--num_train_epochs', type=float, required=True, help='Number of epoches for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005, required=True, help='Learning rate')
    parser.add_argument('--interval', type=int, default=100, help='Interval for each stage')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = SFTConfiguration(**vars(args))
    training_args = setup_logger(config)
    model, tokenizer = setup_model_and_tokenizer(config)
    data = load_and_prepare_data(config, tokenizer, training_args)
    run_sft(training_args, model, tokenizer, data)
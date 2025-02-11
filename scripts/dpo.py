import os
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
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from utils import (
    RDPOTrainer,
    RDPOConfiguration,
    RDPOTrainingArguments,
)
import torch
torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)


def setup_logger(config):
    generation_configs = {
        "recall": GenerationConfig(max_length=config.max_length, do_sample=False),
        "analyze": GenerationConfig(max_length=config.max_length, do_sample=False),
        "summarize": GenerationConfig(max_length=config.max_length, do_sample=False)
    }

    training_args = RDPOTrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy=config.all_strategy,
        per_device_train_batch_size=config.per_gpu_batch_size,
        per_device_eval_batch_size=config.per_gpu_batch_size * 2,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        log_level=config.log_level,
        logging_strategy=config.all_strategy,
        save_strategy=config.all_strategy,
        save_total_limit=None,
        save_only_model=True,
        metric_for_best_model=config.metric_name,
        load_best_model_at_end=True,
        disable_tqdm=False,
        remove_unused_columns=False,
        # for 3 stage sft
        recall_generation_config=generation_configs["recall"],
        analyze_generation_config=generation_configs["analyze"],
        summarize_generation_config=generation_configs["summarize"],
        # for dpo
        beta=config.beta,
        rpo_alpha=config.rpo_alpha,
        truncation_mode="keep_start",
        precompute_ref_log_probs=True,
        predict_with_generate=True,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        max_target_length=config.max_target_length,
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
    model = T5ForConditionalGeneration.from_pretrained(config.ref_model)
    ref_model = T5ForConditionalGeneration.from_pretrained(config.ref_model)
    tokenizer = T5Tokenizer.from_pretrained(config.ref_model, legacy=True)

    return model, ref_model, tokenizer


def load_and_prepare_data(config, tokenizer, training_args):
    """
    Load and prepare dataset for training.
    """
    def preprocess_function(dataset):
        if "analyze" in dataset.keys():
            model_inputs = tokenizer(dataset["input"], text_target=dataset["analyze"], padding=True)
        elif "recall" in dataset.keys():
            model_inputs = tokenizer(dataset["input"], text_target=dataset["recall"], padding=True)
        elif "summarize" in dataset.keys():
            model_inputs = tokenizer(dataset["input"], text_target=dataset["summarize"], padding=True)
        else:
            model_inputs = tokenizer(dataset["input"], padding=True)
        return model_inputs

    datasets = {split: load_dataset("json", data_files=path, split='train') for split, path in config.dataset_path.items()}

    with training_args.main_process_first():
        datasets['valid'] = datasets['valid'].map(preprocess_function, batched=True)

    return datasets

        
def run_dpo(training_args, model, ref_model, tokenizer, datasets):
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

    # set steps based on length of generated dpo data
    if config.stage_type == "recall_analyze":
        recall_length = len(datasets['recall_dpo'])
        analyze_length = len(datasets['analyze_dpo'])
        train_dataset_list = [datasets['recall_dpo'], datasets['analyze_dpo']]
        training_args.recall_interval = (recall_length // (training_args.per_device_train_batch_size * training_args.n_gpu)) + 1
        training_args.analyze_interval = (analyze_length // recall_length + 1) * training_args.recall_interval
    elif config.stage_type == "recall":
        recall_length = len(datasets['recall_dpo'])
        train_dataset_list = [datasets['recall_dpo']]
        training_args.recall_interval = (recall_length // (training_args.per_device_train_batch_size * training_args.n_gpu)) + 1
        training_args.analyze_interval = 0
    elif config.stage_type == "analyze":
        analyze_length = len(datasets['analyze_dpo'])
        train_dataset_list = [datasets['analyze_dpo']]
        training_args.recall_interval = 0
        training_args.analyze_interval = (analyze_length // (training_args.per_device_train_batch_size * training_args.n_gpu)) + 1
    
    strategy_interval = training_args.recall_interval + training_args.analyze_interval
    training_args.eval_steps = training_args.logging_steps = training_args.save_steps = strategy_interval
    training_args.max_steps = strategy_interval * 10

    trainer = RDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset_list,
        eval_dataset=datasets['valid'],
        tokenizer=tokenizer,
        compute_metrics=_compute_accuracy
    )
 
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

    logger.info(f'============ CHECK IMPORTANT INFO =================')
    logger.info(f">> Currently in the {config.dpo_iter} iteration <<")
    logger.info(f">> Reference model loaded from {config.ref_model} <<")
    logger.info(f">> Current learning rate is {config.learning_rate} <<")
    logger.info(f">> Beta {config.beta}, Alpha {config.rpo_alpha} <<")
    
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for RDPO training")

    parser.add_argument('--base_model', type=str, required=True, help='Base model name')
    parser.add_argument('--ref_model', type=str, default=None, help='Reference model')
    parser.add_argument('--data_name', type=str, required=True, help='Data name')
    parser.add_argument('--training_type', type=str, required=True, help='Training type')
    parser.add_argument('--stage_type', choices=['recall', 'analyze', 'recall_analyze'], type=str, required=True, help='Stage type')
    parser.add_argument('--dpo_iter', type=int, default=0, help='DPO iteration')
    parser.add_argument('--per_gpu_batch_size', type=int, required=True, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=0.0, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.5, help='DPO beta')
    parser.add_argument('--rpo_alpha', type=float, default=0.5, help='RPO alpha')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = RDPOConfiguration(**vars(args))
    training_args = setup_logger(config)
    model, ref_model, tokenizer = setup_model_and_tokenizer(config)
    data = load_and_prepare_data(config, tokenizer, training_args)
    run_dpo(training_args, model, ref_model, tokenizer, data)

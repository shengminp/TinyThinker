import os
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, '../../'))
import re
import sys
import json
import torch
import logging
import datasets
import argparse
import transformers
import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset, concatenate_datasets
from transformers import (
    GenerationConfig,
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    set_seed
)
from utils import (
    SFTTrainer,
    GenerationConfiguration,
    GenerationArguments,
)

logger = logging.getLogger(__name__)


def setup_logger(config):
    if config.generation_type == "greedy":
        generation_configs = {
            "recall": GenerationConfig(max_length=config.max_length, do_sample=False),
            "analyze": GenerationConfig(max_length=config.max_length, do_sample=False),
            "summarize": GenerationConfig(max_length=config.max_length, do_sample=False)
        }
    else:
        generation_configs = {
            "recall": GenerationConfig(
                max_length=config.max_length,
                do_sample=config.recall_sample,
                temperature=config.recall_temperature,
                top_k=config.recall_top_k
            ),
            "analyze": GenerationConfig(
                max_length=config.max_length,
                do_sample=config.analyze_sample,
                temperature=config.analyze_temperature,
                top_k=config.analyze_top_k
            ),
            "summarize": GenerationConfig(
                max_length=config.max_length,
                do_sample=False
            )
        }

    training_args = GenerationArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_gpu_batch_size,
        learning_rate=config.learning_rate,
        evaluation_strategy=config.all_strategy,
        per_device_eval_batch_size=config.per_gpu_batch_size,
        log_level=config.log_level,
        logging_strategy=config.all_strategy,
        save_strategy=config.all_strategy,
        save_total_limit=None,
        metric_for_best_model=config.metric_name,
        load_best_model_at_end=True,
        disable_tqdm=False,
        remove_unused_columns=True,
        predict_with_generate=True,
        stage_type=config.stage_type,
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
    #set_seed(config.seed)
    model = T5ForConditionalGeneration.from_pretrained(config.checkpoint_dir)
    tokenizer = T5Tokenizer.from_pretrained(config.checkpoint_dir, legacy=True)
    model.eval()
    return model, tokenizer


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
        tokenized_datasets = {name: data.map(preprocess_function, batched=True) for name, data in datasets.items()}
    
    return tokenized_datasets


def compute_accuracy(total_preds, tokenizer):
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


def self_consistency(data_list):
    answer_prediction = {}
    GT_list = []

    for item in data_list:
        question = item['input'].split("\nRecall")[0]
        if question not in answer_prediction:
            answer_prediction[question] = []
            GT_list.append(item['GT'])
        answer_prediction[question].append(item['prediction'])

    sc_list = [
        Counter(predictions).most_common(1)[0][0]
        for predictions in answer_prediction.values()
    ]

    match_count = sum(pred == gt for pred, gt in zip(sc_list, GT_list))
    accuracy = match_count / len(sc_list) * 100
    logger.info(f"Self-Consistency Accuracy: {accuracy}%")


def generate_summarize_data(trainer, tokenizer, test_dataset, dpo_type):
    logger.info(f'============ CHECK IMPORTANT INFO =================')
    logger.info(f">> Load model from {config.checkpoint_dir} <<")
    logger.info(f">> The model trained with {config.training_type} <<")
    logger.info(f">> Will generate {config.generation_file} for DPO iteration {config.dpo_iter}<<")
    logger.info(f">> Save result to {config.generate_path} <<")
    logger.info(f'>> Recall Generation Configuration: {trainer.args.recall_generation_config} <<')
    logger.info(f'>> Analyze Generation Configuration: {trainer.args.analyze_generation_config} <<')

    summarize_dataset_list = []

    for num in range(config.generation_times):
        logger.info(f"============ Generate for {num+1} time ============")
        torch.cuda.empty_cache()
        summarize_dataset, test_preds = trainer.predict(test_dataset)

        if trainer.is_world_process_zero():
            total_results = compute_accuracy(test_preds, tokenizer)
            logger.info(f"Accuracy: {total_results['accuracy']}%")
            summarize_dataset = summarize_dataset.add_column("summarize", total_results["summarize"])
            summarize_dataset = summarize_dataset.add_column("prediction", total_results["prediction"])
            if "analyze" in total_results.keys():
                summarize_dataset = summarize_dataset.add_column("analyze", total_results["analyze"])
            summarize_dataset = summarize_dataset.remove_columns(['input_ids', 'attention_mask'])
            summarize_dataset_list.append(summarize_dataset)

        if dpo_type == 'recall':
            trainer.args.recall_generation_config.temperature -= config.temperature_increase
        elif dpo_type == 'analyze':
            trainer.args.analyze_generation_config.temperature -= config.temperature_increase

    return concatenate_datasets(summarize_dataset_list)


def generate_dpo_recall_data(trainer, tokenizer, recall_dataset, original_df):
    def _extract_question(text):
        match = re.search(r'(.*?)\nRecall:', text, re.DOTALL)
        return match.group(1)
    
    def _extract_recall(text):
        match = re.search(r'Recall: (.*?)\nAnalyze:', text, re.DOTALL)
        return match.group(1)
    
    def _extract_analyze(text):
        analyze = text.split("Analyze: ")[-1]
        split_analyze = re.findall(r'For option.*?(?=For option|$)', analyze, re.DOTALL)
        return tuple(r.split("\nSummarize")[0].strip() if "\nSummarize" in r else r.strip() for r in split_analyze)


    logger.info(f'=========== Generate DPO Recall Data ===========')
    summarize_dataset = generate_summarize_data(trainer, tokenizer, recall_dataset, "recall")
    generated_df = summarize_dataset.to_pandas()

    generated_df["question"] = generated_df["input"].apply(_extract_question)
    generated_df["recall"] = generated_df["input"].apply(_extract_recall)
    generated_df["analyze"] = generated_df["input"].apply(_extract_analyze)

    dpo_dataset_dict = {"prompt": [], "chosen": [], "rejected": [], "GT": [], "analyze": []}

    for question, group in generated_df.groupby('question'):
        chosen_data = group[group["GT"] == group["prediction"]].drop_duplicates(subset="recall")
        reject_data = group[group["GT"] != group['prediction']].drop_duplicates(subset="recall")

        if reject_data.empty:
            continue

        common_recall = set(chosen_data["recall"]).intersection(set(reject_data["recall"]))
        chosen_data = chosen_data[~chosen_data["recall"].isin(common_recall)]
        reject_data = reject_data[~reject_data["recall"].isin(common_recall)]

        if reject_data.empty:
            continue

        if len(chosen_data) > len(reject_data):
            reject_data = reject_data.sample(n=len(chosen_data), replace=True)
        elif len(reject_data) > len(chosen_data):
            sample_size = len(reject_data) - len(chosen_data)
            additional_chosen_data = original_df[original_df["input"].str.contains(question, regex=False)]

            sample_chosen_data = additional_chosen_data.sample(n=sample_size, replace=(sample_size > len(additional_chosen_data)))

            sample_chosen_data.loc[:, "recall"] = sample_chosen_data['input'].apply(_extract_recall)
            sample_chosen_data.loc[:, "analyze"] = sample_chosen_data['input'].apply(_extract_analyze)
            chosen_data = pd.concat([chosen_data, sample_chosen_data], ignore_index=True)
        
        for chosen, reject in zip(chosen_data.itertuples(), reject_data.itertuples()):
            dpo_dataset_dict["prompt"].append(f"{question}\nRecall:")
            dpo_dataset_dict["chosen"].append(chosen.recall)
            dpo_dataset_dict["rejected"].append(reject.recall)
            dpo_dataset_dict["GT"].append(chosen.GT)
            dpo_dataset_dict["analyze"].append(chosen.analyze)

    generated_recall_dpo = [
        {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        for prompt, chosen, rejected in zip(dpo_dataset_dict["prompt"], dpo_dataset_dict["chosen"], dpo_dataset_dict["rejected"])
    ]

    logger.info(f'>> Generate {len(generated_recall_dpo)} data for DPO recall training')
    with open(f"{config.generate_path}/recall_dpo.json", 'w') as f:
        json.dump(generated_recall_dpo, f, indent=4)

    # Prepare dataset for generating analyze dpo data
    analyze_data = []
    analyze_backup = {}
    options = sorted(re.findall(r'\(([A-Z])\)', dpo_dataset_dict['prompt'][0]))

    for prompt, recall, GT, analyze in zip(dpo_dataset_dict["prompt"], dpo_dataset_dict["chosen"], dpo_dataset_dict["GT"], dpo_dataset_dict["analyze"]):
        key = f"{prompt} {recall}"
        for option in options:
            analyze_data.append(
                {
                    "input": f"{key}\nAnalyze: For option {option},",
                    "GT": GT
                }
            )
        
        if key not in analyze_backup.keys():
            analyze_backup[key] = [analyze]
        else:
            analyze_backup[key].append(analyze)

    from datasets import Dataset
    dataset = Dataset.from_pandas(pd.DataFrame(analyze_data))
    def _preprocess_function(examples):
        return tokenizer(examples["input"], padding=True)
    analyze_dataset = dataset.map(_preprocess_function, batched=True)
    return analyze_dataset, analyze_backup


def generate_dpo_analyze_data(trainer, tokenizer, analyze_dataset, analyze_backup):
    ################################################
    ######### Generate DPO Reasoning Data ##########
    ################################################
    def _extract_recall(text):
        match = re.search(r'(.*?)\nAnalyze:', text, re.DOTALL)
        return match.group(1)

    def _extract_analyze(text):
        analyze = text.split("Analyze: ")[-1]
        split_analyze = re.findall(r'For option.*?(?=For option|$)', analyze, re.DOTALL)
        return tuple(r.split("\nSummarize")[0].strip() if "\nSummarize" in r else r.strip() for r in split_analyze)
    
    def _extract_prompt_analyze(analyze_list, prompt):
        for analyze in analyze_list:
            if prompt in analyze:
                return analyze.split(f"{prompt} ")[-1]
        return None

    logger.info(f'=========== Generate DPO Reasoning Data ===========')
    summarize_dataset = generate_summarize_data(trainer, tokenizer, analyze_dataset, "analyze")
    generated_df = summarize_dataset.to_pandas()

    generated_df["recall"] = generated_df["input"].apply(_extract_recall)
    generated_df["analyze"] = generated_df["input"].apply(_extract_analyze)

    dpo_dataset_dict = {"prompt": [], "chosen": [], "rejected": [], "GT": [], "analyze": []}

    for recall, group in generated_df.groupby('recall'):
        chosen_data = group[group["GT"] == group["prediction"]].drop_duplicates(subset="analyze")
        reject_data = group[group["GT"] != group['prediction']].drop_duplicates(subset="analyze")

        chosen_data = chosen_data.dropna(subset=['prediction'])
        reject_data = reject_data.dropna(subset=['prediction'])
        
        chosen_data = chosen_data[chosen_data['prediction'].str.match('[A-E]')]
        reject_data = reject_data[reject_data['prediction'].str.match('[A-E]')]

        if reject_data.empty:
            continue

        valid_chosen_options = set(re.findall(r'For option ([A-E]), ', ' '.join(chosen_data["analyze"].apply(lambda x: ' '.join(x)))))
        chosen_data = chosen_data[chosen_data['prediction'].isin(valid_chosen_options)]
        
        valid_reject_options = set(re.findall(r'For option ([A-E]), ', ' '.join(reject_data["analyze"].apply(lambda x: ' '.join(x)))))
        reject_data = reject_data[reject_data['prediction'].isin(valid_reject_options)]

        common_analyze = set(chosen_data["analyze"]).intersection(set(reject_data["analyze"]))
        chosen_data = chosen_data[~chosen_data["analyze"].isin(common_analyze)]
        reject_data = reject_data[~reject_data["analyze"].isin(common_analyze)]

        if reject_data.empty:
            continue

        if len(chosen_data) > len(reject_data):
            reject_data = reject_data.sample(n=len(chosen_data), replace=True)
        elif len(reject_data) > len(chosen_data):
            sample_size = len(reject_data) - len(chosen_data)
            if chosen_data.empty:
                additional_data = [{"GT": group["GT"].iloc[0], "analyze": analyze} for analyze in analyze_backup[recall]]
                chosen_data = pd.DataFrame(additional_data).sample(n=sample_size, replace=True)
            else:
                chosen_data = chosen_data.sample(n=sample_size, replace=True)

        # This part is for only consider wrong and GT analyze
        for chosen, reject in zip(chosen_data.itertuples(), reject_data.itertuples()):
            gt_prompt = f"For option {chosen.GT},"
            pd_prompt = f"For option {reject.prediction},"

            chosen_gt_analyze = _extract_prompt_analyze(chosen.analyze, gt_prompt)
            chosen_pd_analyze = _extract_prompt_analyze(chosen.analyze, pd_prompt)
            reject_gt_analyze = _extract_prompt_analyze(reject.analyze, gt_prompt)
            reject_pd_analyze = _extract_prompt_analyze(reject.analyze, pd_prompt)

            dpo_dataset_dict["prompt"].append(f"{recall}\nAnalyze: {gt_prompt}")
            dpo_dataset_dict["chosen"].append(chosen_gt_analyze)
            dpo_dataset_dict["rejected"].append(reject_gt_analyze)

            dpo_dataset_dict["prompt"].append(f"{recall}\nAnalyze: {pd_prompt}")
            dpo_dataset_dict["chosen"].append(chosen_pd_analyze)
            dpo_dataset_dict["rejected"].append(reject_pd_analyze)

            dpo_dataset_dict["GT"].append(chosen.GT)
    
    generated_analyze_dpo = [
        {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        for prompt, chosen, rejected in zip(dpo_dataset_dict["prompt"], dpo_dataset_dict["chosen"], dpo_dataset_dict["rejected"])
    ]
    logger.info(f'>> Generate {len(generated_analyze_dpo)} data for DPO reasoniong training')
    with open(f"{config.generate_path}/analyze_dpo.json", 'w') as f:
        json.dump(generated_analyze_dpo, f, indent=4)


def run_generation(training_args, model, tokenizer, tokenized_datasets):
    """
    Main training loop.
    """
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
        padding=True
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=None
    )

    if config.generation_file == "test":
        test_dataset = tokenized_datasets["test"]
        if config.dpo_iter > 0:
            trainer.args.stage_type = "recall_analyze_summarize"
        trainer.test_dataset = test_dataset
        summarize_dataset = generate_summarize_data(trainer, tokenizer, test_dataset, None)
        summarize_df = summarize_dataset.to_pandas()
        if config.generation_times > 1:
            self_consistency(summarize_df.to_dict(orient='records'))
        with open(f"{config.generate_path}", 'w') as f:
            json.dump(summarize_df.to_dict(orient='records'), f, indent=4)
        
    elif config.generation_file == "dpo":
        recall_dataset = tokenized_datasets["target_recall"]
        trainer.args.stage_type = "recall"
        trainer.test_dataset = recall_dataset
        original_df = tokenized_datasets["original_summarize"].to_pandas()
        analyze_dataset, analyze_backup = generate_dpo_recall_data(trainer, tokenizer, recall_dataset, original_df)

        if config.stage_type == "recall_analyze" or config.stage_type == "analyze":
            analyze_generation_configs = {
                "recall": GenerationConfig(
                    max_length=config.max_length,
                    do_sample=config.analyze_sample,
                    temperature=config.analyze_temperature,
                    top_k=config.analyze_top_k
                ),
                "analyze": GenerationConfig(
                    max_length=config.max_length,
                    do_sample=config.recall_sample,
                    temperature=config.recall_temperature,
                    top_k=config.recall_top_k
                ),
            }
            trainer.args.recall_generation_config = analyze_generation_configs["recall"]
            trainer.args.analyze_generation_config = analyze_generation_configs["analyze"]
            trainer.args.stage_type = "analyze"
            trainer.test_dataset = analyze_dataset
            generate_dpo_analyze_data(trainer, tokenizer, analyze_dataset, analyze_backup)


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Generation")

    parser.add_argument('--base_model', type=str, required=True, help='Base model name')
    parser.add_argument('--data_name', type=str, required=True, help='Data name')
    parser.add_argument('--training_type', type=str, required=True, help='Training type')
    parser.add_argument('--stage_type', choices=['recall', 'analyze', 'summarize', 'recall_summarize', 'analyze_summarize', 'recall_analyze', 'recall_analyze_summarize'], type=str, required=True, help='Stage type')
    parser.add_argument('--dpo_iter', type=int, default=0, help='DPO iteration')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--per_gpu_batch_size', type=int, required=True, help='Batch size per GPU')
    parser.add_argument('--generation_type', type=str, required=True, help='Generation type')
    parser.add_argument('--generation_file', type=str, required=True, help='Generation file')
    parser.add_argument('--temperature_increase', type=float, default=0.0, help='Temperature increase')
    parser.add_argument('--generation_times', type=int, default=1, help='Generation times')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = GenerationConfiguration(**vars(args))
    training_args = setup_logger(config)
    model, tokenizer = setup_model_and_tokenizer(config)
    data = load_and_prepare_data(config, tokenizer, training_args)
    run_generation(training_args, model, tokenizer, data)

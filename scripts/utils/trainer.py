import re
import pandas as pd
from copy import deepcopy
from functools import wraps
from collections import defaultdict
from accelerate import PartialState
from dataclasses import dataclass
from functools import partial
from transformers import (
    PreTrainedModel,
    DataCollatorForSeq2Seq
)
from transformers.trainer import *
from transformers.trainer_seq2seq import *
from transformers.training_args_seq2seq import *
from trl.trainer.dpo_trainer import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class EvalLoopContainer:
    """
    Container to store intermediate results of evaluation loop

    Args:
        do_nested_concat (`bool`, *optional*, defaults to `True`):
            If set to `True`, each iteration will recursively concatenate a new object containing tensors to
            the existing stored tensors, provided that the structure of the existing object and the new one
            are identical. If set to `False`, all newly added tensors will be stored in a list.
        padding_index (`int`, *optional*, defaults to -100):
            Value used to pad tensors of different shapes when `do_nested_concat=True`.
    """

    def __init__(self, do_nested_concat: bool = True, padding_index: int = -100):
        self.do_nested_concat = do_nested_concat
        self.padding_index = padding_index
        self.tensors = None
        self.arrays = None

    def add(self, tensors) -> None:
        """Add tensors to the stored objects. If `do_nested_concat=True`, the tensors will be concatenated recursively."""
        if self.tensors is None:
            self.tensors = tensors if self.do_nested_concat else [tensors]
        elif self.do_nested_concat:
            self.tensors = nested_concat(self.tensors, tensors, padding_index=self.padding_index)
        else:
            self.tensors.append(tensors)

    def to_cpu_and_numpy(self) -> None:
        """Move tensors in stored objects to CPU and convert them to numpy arrays."""

        # Check if we have something to add, if not just return
        if self.tensors is None:
            return

        new_arrays = nested_numpify(self.tensors)
        if self.arrays is None:
            self.arrays = new_arrays
        elif self.do_nested_concat:
            self.arrays = nested_concat(self.arrays, new_arrays, padding_index=self.padding_index)
        else:
            self.arrays.extend(new_arrays)

        # reset device tensors after adding to cpu
        self.tensors = None

    def get_arrays(self):
        """Returns the numpified and moved to CPU stored objects."""
        self.to_cpu_and_numpy()
        return self.arrays


class CoTTrainer(Trainer):

    def _get_train_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None or not has_length(train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(train_dataset)
    
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data1_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multipe eval datasets
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        _, output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {self.num_examples(dataloader) if has_length(dataloader) else 'Unknown'}")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        def _run_dataloader(batch_size, dataloader, all_preds, generation_config):
            # Will be useful when we have an iterable dataset so don't know its length.
            observed_num_examples = 0

            for _, inputs in enumerate(dataloader):
                # Update the observed num examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if batch_size is None:
                        batch_size = observed_batch_size

                # Prediction step
                _, logits, _ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys, **generation_config.to_dict())

                if is_torch_tpu_available():
                    xm.mark_step()

                # Update containers on host
                if logits is not None:
                    logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                    logits = self.accelerator.gather_for_metrics((logits))
                    all_preds.add(logits)

                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

                # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
                if args.eval_accumulation_steps is not None and self.accelerator.sync_gradients:
                    all_preds.to_cpu_and_numpy()
                    
                    del logits, inputs
                    torch.cuda.empty_cache()
            

            # Gather all remaining tensors and put them back on the CPU
            all_preds_arrays = all_preds.get_arrays()
            return all_preds_arrays

        def _preprocess_function(examples):
            if 't5' in self.model.config._name_or_path:
                return self.tokenizer(examples["input"], padding=True)
            '''elif 'gpt2' in self.model.config._name_or_path:
                return self.tokenizer(examples["input"], padding='max_length', truncation=True, max_length=256)'''
        
        def _prepare_analyze_dataset(dataset, recall_results):
            general_knowledge_data = []
            options = sorted(re.findall(r'\(([A-Z])\)', dataset[0]["input"]))

            for data, general_knowledge in zip(dataset, recall_results):
                if not general_knowledge.endswith(('.', '"', "'")):
                    general_knowledge = general_knowledge + '.'
                
                if 't5' in self.model.config._name_or_path:
                    general_knowledge_prompt = f"{data['input']} {general_knowledge}"
                '''elif 'gpt2' in self.model.config._name_or_path:
                    knowledge_prompt = knowledge
                    if "Analyze:" in knowledge_prompt:
                        knowledge_prompt = knowledge_prompt.split("Analyze:")[0].strip()
                '''
                if args.stage_type != "recall_summarize":
                    for option in options:
                        general_knowledge_data.append(
                            {
                                "input": f"{general_knowledge_prompt}\nAnalyze: For option {option},",
                                "GT": data["GT"]
                            }
                        )
                else:
                    general_knowledge_data.append(
                        {
                            "input": f"{general_knowledge_prompt}\nSummarize:",
                            "GT": data["GT"]
                        }
                    )
            from datasets import Dataset
            dataset = Dataset.from_pandas(pd.DataFrame(general_knowledge_data))
            from torch.utils.data import Dataset
            return dataset.map(_preprocess_function, batched=True)
    
        def _prepare_summarize_dataset(dataset, analyze_results):
            specific_knowledge_data = []
            options = sorted(re.findall(r'\(([A-Z])\)', dataset[0]["input"]))
            num_options = len(options)

            for i in range(0, len(dataset), num_options):
                batch_data = dataset[i:i+num_options]
                batch_specific_knowledge = analyze_results[i:i+num_options]

                prompt = batch_data['input'][0].split("Analyze:")[0]
                ground_truth = batch_data['GT'][0]

                clean_batch_specific_knowledge = []
                for specific_knowledge in batch_specific_knowledge:
                    clean_specific_knowledge = specific_knowledge

                    '''if 'gpt2' in self.model.config._name_or_path:
                        match = re.search(r"Analyze: For option [A-E], (.*)", clean_specific_knowledge)
                        if match:
                            clean_specific_knowledge = match.group(1)
                        else: clean_specific_knowledge = "None"'''
                        
                    if not clean_specific_knowledge.endswith(('.', '"', "'")):
                        clean_specific_knowledge += '.'
                    
                    clean_batch_specific_knowledge.append(clean_specific_knowledge)
                
                total_specific_knowledge = " ".join(
                    f"For option {options[index]}, {specific_knowledge}"
                    for index, specific_knowledge in enumerate(clean_batch_specific_knowledge)
                ).strip()
                    
                specific_knowledge_data.append(
                    {
                        "input": f"{prompt}Analyze: {total_specific_knowledge}\nSummarize:",
                        "GT": ground_truth
                    }
                )
            from datasets import Dataset
            dataset = Dataset.from_pandas(pd.DataFrame(specific_knowledge_data))
            from torch.utils.data import Dataset
            return dataset.map(_preprocess_function, batched=True)
        
        def _recall_stage(batch_size, dataloader):
            ##########################################
            ########## Recall Generation ##########
            ##########################################
            logger.info(f"** Generating General Knowledge **")
            self.callback_handler.eval_dataloader = dataloader
            all_preds = EvalLoopContainer(True, padding_index=-100)
            all_preds_arrays = _run_dataloader(batch_size, dataloader, all_preds, args.recall_generation_config)
            all_preds_arrays = np.where(all_preds_arrays != -100, all_preds_arrays, self.tokenizer.pad_token_id)
            recall_results = self.tokenizer.batch_decode(all_preds_arrays, skip_special_tokens=True)
            recall_results = [result.strip() for result in recall_results]
        
            if self.eval_dataset is not None:
                analyze_dataset = _prepare_analyze_dataset(self.eval_dataset, recall_results)
                analyze_dataloader = self.get_eval_dataloader(analyze_dataset)
            else:
                analyze_dataset = _prepare_analyze_dataset(self.test_dataset, recall_results)
                analyze_dataloader = self.get_test_dataloader(analyze_dataset)
            
            return analyze_dataset, analyze_dataloader
        
        def _analyze_stage(batch_size, analyze_dataset, analyze_dataloader):
            ##########################################
            ########## Analyze Generation ##########
            ##########################################
            logger.info(f"** Generating Specific Knowledge **")
            self.callback_handler.eval_dataloader = analyze_dataloader
            all_preds = EvalLoopContainer(True, padding_index=-100)
            all_preds_arrays = _run_dataloader(batch_size, analyze_dataloader, all_preds, args.analyze_generation_config)
            all_preds_arrays = np.where(all_preds_arrays != -100, all_preds_arrays, self.tokenizer.pad_token_id)
            analyze_results = self.tokenizer.batch_decode(all_preds_arrays, skip_special_tokens=True)
            analyze_results = [result.strip() for result in analyze_results]

            summarize_dataset = _prepare_summarize_dataset(analyze_dataset, analyze_results)

            if self.eval_dataset is not None:
                summarize_dataloader = self.get_eval_dataloader(summarize_dataset)
            else:
                summarize_dataloader = self.get_test_dataloader(summarize_dataset)
            
            return summarize_dataset, summarize_dataloader
        
        def _summarize_stage(batch_size, summarize_dataloader):
            ##########################################
            ########## Summarize Generation ##########
            ##########################################
            logger.info(f"** Generating Summarization **")
            self.callback_handler.eval_dataloader = summarize_dataloader
            all_preds = EvalLoopContainer(True, padding_index=-100)
            all_preds_arrays = _run_dataloader(batch_size, summarize_dataloader, all_preds, args.summarize_generation_config)
            return all_preds, all_preds_arrays
        
        if args.stage_type == "recall_analyze_summarize" or args.stage_type == "recall_analyze" or args.stage_type == "recall":
            analyze_dataset, analyze_dataloader = _recall_stage(batch_size, dataloader)
            summarize_dataset, summarize_dataloader = _analyze_stage(batch_size, analyze_dataset, analyze_dataloader)
            all_preds, all_preds_arrays = _summarize_stage(batch_size, summarize_dataloader)
        elif args.stage_type == "recall_summarize":
            analyze_dataset, analyze_dataloader = _recall_stage(batch_size, dataloader)
            all_preds, all_preds_arrays = _summarize_stage(batch_size, analyze_dataloader)
        elif args.stage_type == "analyze" or args.stage_type == "analyze_summarize":
            if self.eval_dataset is not None:
                analyze_dataset = self.eval_dataset
            else:
                analyze_dataset = self.test_dataset
            summarize_dataset, summarize_dataloader = _analyze_stage(batch_size, analyze_dataset, dataloader)
            all_preds, all_preds_arrays = _summarize_stage(batch_size, summarize_dataloader)
        elif args.stage_type == "summarize":
            all_preds, all_preds_arrays = _summarize_stage(batch_size, dataloader)


        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None:
            if args.stage_type != "analyze_summarize":
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds_arrays, label_ids=self.eval_dataset["GT"]))
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds_arrays, label_ids=summarize_dataset["GT"]))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        if self.eval_dataset is not None and args.stage_type != "analyze_summarize":
            return None, EvalLoopOutput(predictions=all_preds_arrays, label_ids=self.eval_dataset["GT"], metrics=metrics, num_samples=num_samples)
        elif args.stage_type != "summarize" and args.stage_type != "recall_summarize":
            return summarize_dataset, EvalLoopOutput(predictions=all_preds_arrays, label_ids=summarize_dataset["GT"], metrics=metrics, num_samples=num_samples)
        else:
            return self.test_dataset, EvalLoopOutput(predictions=all_preds_arrays, label_ids=self.test_dataset["GT"], metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


class SFTTrainer(CoTTrainer):
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        def _prepare_dataloader(dataset, data_collator):
            if is_datasets_available() and isinstance(dataset, datasets.Dataset):
                dataset = self._remove_unused_columns(dataset, description="training")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
            
            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = self._get_train_sampler(dataset)
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker
            
            return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, list):
            return [_prepare_dataloader(dataset, data_collator) for dataset in train_dataset]
        else:
            return _prepare_dataloader(train_dataset, data_collator)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if isinstance(train_dataloader, list) or has_length(train_dataloader):
            if isinstance(train_dataloader, list):
                len_dataloader = sum(len(dataloader) for dataloader in train_dataloader)
                num_examples = sum(self.num_examples(dataloader) for dataloader in train_dataloader)
            else:
                len_dataloader = len(train_dataloader)
                num_examples = self.num_examples(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
        
        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            if isinstance(epoch_iterator, list):
                steps_in_epoch = (
                    max(len(iterator) for iterator in epoch_iterator)
                    if len_dataloader is not None
                    else args.max_steps * args.gradient_accumulation_steps
                )
            else:
                steps_in_epoch = (
                    len(epoch_iterator)
                    if len_dataloader is not None
                    else args.max_steps * args.gradient_accumulation_steps
                )

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            '''if steps_trained_in_current_epoch > 0:
                if isinstance(epoch_iterator, list):
                    for i in range(len(epoch_iterator)):
                        epoch_iterator[i] = skip_first_batches(epoch_iterator[i], steps_trained_in_current_epoch)
                else:
                    epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True'''

            step = -1

            ###################################
            #######Iterator Switch Code########
            ###################################
            if isinstance(epoch_iterator, list):

                def _iterator_wrapper(iterator):
                    for step, data in enumerate(iterator):
                        yield step, data

                def _switch_dataloader(current_index, exhausted, iterators, switch_plan, plan_index):
                    next_index = (current_index + 1) % len(iterators)
                    if not exhausted[next_index]:  # if next iterator is not exhausted
                        if args.parallel_mode == ParallelMode.DISTRIBUTED:
                            dist.barrier()
                        logger.info(f"Switching to dataset: {switch_plan[next_index][1]}")
                        return next_index, iterators[next_index], (plan_index + 1) % len(switch_plan)
                    else:
                        for idx, ex in enumerate(exhausted):
                            if not ex:  # find the next iterator that not exhaused
                                if args.parallel_mode == ParallelMode.DISTRIBUTED:
                                    dist.barrier()
                                logger.info(f"Switching to dataset: {switch_plan[next_index][1]}")
                                return idx, iterators[idx], (plan_index + 1) % len(switch_plan)
                        return None, None, None  # both iterator are exhausted
    
                switch_plan = [
                    (interval, phase) 
                    for interval, phase in [
                        (args.recall_interval, "Recall Phase"),
                        (args.analyze_interval, "Analyze Phase"),
                        (args.summarize_interval, "Summarize Phase")
                    ] 
                    if interval != 0
                ]

                iterators = [_iterator_wrapper(iterator) for iterator in epoch_iterator]
                exhausted = [False] * len(epoch_iterator)
                current_iterator_index = 0 # 0 for recall, 1 for analyze, 2 for summarize
                current_iterator = iterators[current_iterator_index]
                plan_index = 0
                total_step = 0

                logger.info(f"Starting training with dataset: {switch_plan[current_iterator_index][1]}")
                while not all(exhausted):
                    try:
                        step, inputs = next(current_iterator)
                        total_step += 1
                        total_batched_samples += 1
                        if rng_to_sync:
                            self._load_rng_state(resume_from_checkpoint)
                            rng_to_sync = False

                        # Skip past any already trained steps if resuming training
                        if steps_trained_in_current_epoch > 0:
                            steps_trained_in_current_epoch -= 1
                            if steps_trained_progress_bar is not None:
                                steps_trained_progress_bar.update(1)
                            if steps_trained_in_current_epoch == 0:
                                self._load_rng_state(resume_from_checkpoint)
                            continue
                        elif steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None

                        if step % args.gradient_accumulation_steps == 0:
                            self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                        with self.accelerator.accumulate(model):
                            tr_loss_step = self.training_step(model, inputs)

                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_tpu_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                        else:
                            tr_loss += tr_loss_step

                        self.current_flos += float(self.floating_point_ops(inputs))

                        is_last_step_and_steps_less_than_grad_acc = (
                            steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                        )

                        if (
                            total_batched_samples % args.gradient_accumulation_steps == 0
                            or
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            is_last_step_and_steps_less_than_grad_acc
                        ):
                            # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                            # in accelerate. So, explicitly enable sync gradients to True in that case.
                            if is_last_step_and_steps_less_than_grad_acc:
                                self.accelerator.gradient_state._set_sync_gradients(True)

                            # Gradient clipping
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                                # deepspeed does its own clipping

                                if is_sagemaker_mp_enabled() and args.fp16:
                                    self.optimizer.clip_master_grads(args.max_grad_norm)
                                elif self.use_apex:
                                    # Revert to normal clipping otherwise, handling Apex or full precision
                                    nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer),
                                        args.max_grad_norm,
                                    )
                                else:
                                    self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )


                            # Optimizer step
                            self.optimizer.step()
                            optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                            if optimizer_was_run:
                                # Delay optimizer scheduling until metrics are generated
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                            model.zero_grad()
                            self.state.global_step += 1
                            self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                            self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                        else:
                            self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                        del inputs
                        torch.cuda.empty_cache()

                        if self.control.should_epoch_stop or self.control.should_training_stop:
                            break
                        
                        # Switch the strategy
                        if total_step == switch_plan[plan_index][0]:
                            total_step = 0
                            result = _switch_dataloader(current_iterator_index, exhausted, iterators, switch_plan, plan_index)
                            if result:
                                current_iterator_index, current_iterator, plan_index = result

                    except StopIteration:
                        print("im here!!!!!")
                        exhausted[current_iterator_index] = True
                        result = _switch_dataloader(current_iterator_index, exhausted, iterators, switch_plan, plan_index)
                        if result:
                            current_iterator_index, current_iterator, plan_index = result
            else:
                for step, inputs in enumerate(epoch_iterator):
                    total_batched_samples += 1
                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )

                    if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                    ):
                        # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                        # in accelerate. So, explicitly enable sync gradients to True in that case.
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def predict(
        self, 
        test_dataset: Dataset, 
        ignore_keys: Optional[List[str]] = None, 
        metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        summarize_dataset, output = eval_loop(
            test_dataloader, 
            description="Prediction", 
            ignore_keys=ignore_keys, 
            metric_key_prefix=metric_key_prefix
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return summarize_dataset, PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)


class RDPOTrainer(CoTTrainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`DPOConfig`):
            The DPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "dpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None
    ):

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the DPOTrainer/DPOConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs["torch_dtype"]
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_init_kwargs to the DPOTrainer/DPOConfig, but your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            torch_dtype = ref_model_init_kwargs["torch_dtype"]
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )

                ref_model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        if getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder to the DPOTrainer/DPOConfig."
            )
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        self.tokenizer = tokenizer

        self.reference_free = args.reference_free

        if ref_model:
            self.ref_model = ref_model
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")

        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_length = 512

        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_prompt_length = 128

        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_target_length = 128

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=self.tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else self.tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_target_length = args.max_target_length
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if (
            args.loss_type in ["hinge", "ipo", "bco_pair", "sppo_hard", "nca_pair", "apo_zero", "apo_down"]
            and args.label_smoothing > 0
        ):
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )
        if args.loss_type == "kto_pair":
            raise ValueError("Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.")

        self.beta = args.beta
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        
        self.dataset_num_proc = args.dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset, lower writer batch size to avoid OOM (frequent in vision models)
            tokenized_row_with_model = partial(self.tokenize_row, model=model)
            if isinstance(train_dataset, list):
                train_dataset = [dataset.map(tokenized_row_with_model, num_proc=self.dataset_num_proc, writer_batch_size=10) for dataset in train_dataset]
            else:
                train_dataset = train_dataset.map(tokenized_row_with_model, num_proc=self.dataset_num_proc, writer_batch_size=10)
            #if eval_dataset is not None:
                #eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc, writer_batch_size=10)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.is_deepspeed_enabled:
            self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with TR-DPO method. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))
        if self.loss_type == "bco_pair":
            self.running = RunningMoments(self.accelerator)

    # Called in _init_ function no change
    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """
        def _prepare_ref_log_probs(train_dataset):
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

                # Unnecessary cache clearing to avoid OOM
                torch.cuda.empty_cache()
                self.accelerator.free_memory()

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            train_dataset = train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            train_dataset = train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            return train_dataset

        def _prepare_dataloader(dataset, data_collator):
            if is_datasets_available() and isinstance(dataset, datasets.Dataset):
                dataset = self._remove_unused_columns(dataset, description="training")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
            
            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }

            if not isinstance(dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = self._get_train_sampler(dataset)
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker
            return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            if isinstance(self.train_dataset, list):
                self.train_dataset = [_prepare_ref_log_probs(dataset) for dataset in self.train_dataset]
            else:
                self.train_dataset = _prepare_ref_log_probs(self.train_dataset)
            
            self._precomputed_train_ref_log_probs = True

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, list):
            return [_prepare_dataloader(dataset, data_collator) for dataset in train_dataset]
        else:
            return _prepare_dataloader(train_dataset, data_collator)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model, 
            padding=True
        )

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            self.args.remove_unused_columns = True
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if isinstance(train_dataloader, list) or has_length(train_dataloader):
            if isinstance(train_dataloader, list):
                len_dataloader = sum(len(dataloader) for dataloader in train_dataloader)
                num_examples = sum(self.num_examples(dataloader) for dataloader in train_dataloader)
            else:
                len_dataloader = len(train_dataloader)
                num_examples = self.num_examples(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
        
        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            if isinstance(epoch_iterator, list):
                steps_in_epoch = (
                    max(len(iterator) for iterator in epoch_iterator)
                    if len_dataloader is not None
                    else args.max_steps * args.gradient_accumulation_steps
                )
            else:
                steps_in_epoch = (
                    len(epoch_iterator)
                    if len_dataloader is not None
                    else args.max_steps * args.gradient_accumulation_steps
                )

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            '''if steps_trained_in_current_epoch > 0:
                if isinstance(epoch_iterator, list):
                    for i in range(len(epoch_iterator)):
                        epoch_iterator[i] = skip_first_batches(epoch_iterator[i], steps_trained_in_current_epoch)
                else:
                    epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True'''

            step = -1

            ###################################
            #######Iterator Switch Code########
            ###################################
            if isinstance(epoch_iterator, list):

                def _iterator_wrapper(iterator):
                    for step, data in enumerate(iterator):
                        yield step, data

                def _switch_dataloader(current_index, exhausted, iterators, switch_plan, plan_index):
                    next_index = (current_index + 1) % len(iterators)
                    if not exhausted[next_index]:  # if next iterator is not exhausted
                        if args.parallel_mode == ParallelMode.DISTRIBUTED:
                            dist.barrier()
                        logger.info(f"Switching to dataset: {switch_plan[next_index][1]}")
                        return next_index, iterators[next_index], (plan_index + 1) % len(switch_plan)
                    else:
                        for idx, ex in enumerate(exhausted):
                            if not ex:  # find the next iterator that not exhaused
                                if args.parallel_mode == ParallelMode.DISTRIBUTED:
                                    dist.barrier()
                                logger.info(f"Switching to dataset: {switch_plan[next_index][1]}")
                                return idx, iterators[idx], (plan_index + 1) % len(switch_plan)
                        return None, None, None  # both iterator are exhausted

                switch_plan = [
                    (interval, phase) 
                    for interval, phase in [
                        (args.recall_interval, "Recall-DPO Phase"),
                        (args.analyze_interval, "Analyze-DPO Phase"),
                    ] 
                    if interval != 0
                ]

                iterators = [_iterator_wrapper(iterator) for iterator in epoch_iterator]
                exhausted = [False] * len(epoch_iterator)
                current_iterator_index = 0 # 0 for recall, 1 for analyze, 2 for summarize
                current_iterator = iterators[current_iterator_index]
                plan_index = 0
                total_step = 0

                logger.info(f"Starting training with dataset: {switch_plan[current_iterator_index][1]}")
                while not all(exhausted):
                    try:
                        step, inputs = next(current_iterator)
                        total_step += 1
                        total_batched_samples += 1
                        if rng_to_sync:
                            self._load_rng_state(resume_from_checkpoint)
                            rng_to_sync = False

                        # Skip past any already trained steps if resuming training
                        if steps_trained_in_current_epoch > 0:
                            steps_trained_in_current_epoch -= 1
                            if steps_trained_progress_bar is not None:
                                steps_trained_progress_bar.update(1)
                            if steps_trained_in_current_epoch == 0:
                                self._load_rng_state(resume_from_checkpoint)
                            continue
                        elif steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None

                        if step % args.gradient_accumulation_steps == 0:
                            self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                        with self.accelerator.accumulate(model):
                            tr_loss_step = self.training_step(model, inputs)
                            
                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_tpu_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                        else:
                            tr_loss += tr_loss_step

                        self.current_flos += float(self.floating_point_ops(inputs))

                        is_last_step_and_steps_less_than_grad_acc = (
                            steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                        )

                        if (
                            total_batched_samples % args.gradient_accumulation_steps == 0
                            or
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            is_last_step_and_steps_less_than_grad_acc
                        ):
                            # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                            # in accelerate. So, explicitly enable sync gradients to True in that case.
                            if is_last_step_and_steps_less_than_grad_acc:
                                self.accelerator.gradient_state._set_sync_gradients(True)

                            # Gradient clipping
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                                # deepspeed does its own clipping

                                if is_sagemaker_mp_enabled() and args.fp16:
                                    self.optimizer.clip_master_grads(args.max_grad_norm)
                                elif self.use_apex:
                                    # Revert to normal clipping otherwise, handling Apex or full precision
                                    nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer),
                                        args.max_grad_norm,
                                    )
                                else:
                                    self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )


                            # Optimizer step
                            self.optimizer.step()
                            optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                            if optimizer_was_run:
                                # Delay optimizer scheduling until metrics are generated
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                            model.zero_grad()
                            self.state.global_step += 1
                            self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                            self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                        else:
                            self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                        del inputs
                        torch.cuda.empty_cache()

                        if self.control.should_epoch_stop or self.control.should_training_stop:
                            break
                        
                        # Switch the strategy
                        if total_step == switch_plan[plan_index][0]:
                            total_step = 0
                            result = _switch_dataloader(current_iterator_index, exhausted, iterators, switch_plan, plan_index)
                            if result:
                                current_iterator_index, current_iterator, plan_index = result

                    except StopIteration:
                        print("im here!!!!!")
                        exhausted[current_iterator_index] = True
                        result = _switch_dataloader(current_iterator_index, exhausted, iterators, switch_plan, plan_index)
                        if result:
                            current_iterator_index, current_iterator, plan_index = result
            else:
                for step, inputs in enumerate(epoch_iterator):
                    total_batched_samples += 1
                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )

                    if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                    ):
                        # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                        # in accelerate. So, explicitly enable sync gradients to True in that case.
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    # Called in tokenize_row function
    def build_tokenized_answer(self, prompt, answer, images=None):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """
        
        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return_dict = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )
        if "pixel_values" in full_tokenized:
            return_dict["prompt_pixel_values"] = full_tokenized["pixel_values"]
        if "pixel_attention_mask" in full_tokenized:
            return_dict["prompt_pixel_attention_mask"] = full_tokenized["pixel_attention_mask"]

        return return_dict

    # Called in _init_ function
    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]
        images = feature.get("images")

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen, images)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected, images)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])
            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt. Avoid adding if it's already there
            bos_token_id = self.tokenizer.bos_token_id
            if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
                rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
                rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer. Avoid adding if it's already there
            eos_token_id = self.tokenizer.eos_token_id
            if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)
            if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                rejected_tokens["input_ids"].append(eos_token_id)
                rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])

            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch

    # Called in get_train_dataloader function
    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """
        Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset.
        """
        # compute reference logps
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    # Called in concatenated_forward function add new code in line 4702?
    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)
        
        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = (
                batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            )
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )
        return concatenated_batch

    # Called in get_batch_loss_metrics function
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps.to(self.accelerator.device)
        rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected_logps.to(self.accelerator.device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios = pi_logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = pi_logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://arxiv.org/pdf/2402.00856
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        elif self.loss_type == "aot_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)

            delta = chosen_logratios_sorted - rejected_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )
        elif self.loss_type == "aot":
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios_sorted, _ = torch.sort(pi_logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)

            delta = pi_logratios_sorted - ref_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'bco_pair', 'sppo_hard', 'nca_pair', 'robust', 'exo_pair']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    # Called in concatenated_forward function
    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Logits (batch and sequence length dim) {logits.shape[:-1]} and labels must have the same shape {labels.shape}."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    # Called in get_batch_loss_metrics and compute_reference_log_probs function
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits


        if all_logits.shape[:2] != concatenated_batch["concatenated_labels"].shape[:2]:
            # for llava, the model returns logits for the entire sequence, including the image tokens (placed before the text tokens)
            seq_len = concatenated_batch["concatenated_labels"].shape[1]
            all_logits = all_logits[:, -seq_len:]

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        if self.loss_type == "ipo":
            all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

    # Called in compute_loss function
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and (self.precompute_ref_log_probs or self.args.rpo_alpha is not None)
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + policy_nll_loss * self.args.rpo_alpha

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss, metrics

        return losses.mean(), metrics

    # Called in training_step function
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        return (loss, metrics) if return_outputs else loss

    # Called in compute_loss function
    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)


@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if k.endswith("input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # Set padding value based on the key
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0  # TODO: check if this is correct
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    # Set padding side based on the key
                    if k in ["prompt_input_ids", "prompt_attention_mask"]:
                        padding_side = "left"
                    else:
                        padding_side = "right"

                    # Set the dtype
                    if k.endswith("_pixel_values"):
                        dtype = torch.float32  # will be downcasted if necessary by the Trainer
                    else:
                        dtype = torch.int64

                    # Convert to tensor and pad
                    to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]
                    padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch
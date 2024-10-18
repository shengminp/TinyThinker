import os
import torch
from pathlib import Path
from trl.trainer.dpo_trainer import *
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import GenerationConfig

SCRIPT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, '../../../'))


SFT_OPTIONS = {
    "csqa": 5,
    "obqa": 4,
    "strategyqa": 2
}


@dataclass
class BaseConfiguration:
    base_model: str
    data_name: str
    training_type: str
    per_gpu_batch_size: int
    checkpoint_dir: str = None
    num_train_epochs: float = 0.0
    learning_rate: float = 0.0
    interval: int = 0
    seed: int = 42
    all_strategy: str = "steps"
    metric_name: str = "accuracy"
    log_level: str = "info"
    output_dir: str = "models/"
    dataset_dir: str = "datasets/"
    max_length: int = 200
    stage_type: str = "recall_analyze_summarize"
    resume_from_checkpoint: str = None

    def __post_init__(self):
        self.dataset_dir = os.path.abspath(os.path.join(ROOT_PATH, self.dataset_dir, self.data_name, "final"))
        self.output_dir = os.path.abspath(os.path.join(ROOT_PATH, self.output_dir, self.data_name, self.base_model))


@dataclass
class SFTConfiguration(BaseConfiguration):
    def __post_init__(self):
        super().__post_init__()

        self.dataset_dir = os.path.join(self.dataset_dir, self.training_type, self.stage_type)
        self.output_dir = os.path.join(self.output_dir, self.training_type, self.stage_type)

        file_mapping = {
            "recall_analyze_summarize": ['recall', 'analyze', 'summarize', 'valid'],
            "summarize": ['summarize', 'valid'],
            "recall_summarize": ['recall', 'summarize', 'valid'],
            "analyze_summarize": ['analyze', 'summarize', 'valid'],
        }
        self.dataset_path = {split: os.path.join(self.dataset_dir, f"{split}.json") for split in file_mapping[self.stage_type]}

        self.num_option = SFT_OPTIONS[self.data_name]

        strategy_mapping = {
            "recall_analyze_summarize": (self.interval, self.interval * self.num_option, self.interval),
            "summarize": (0, 0, self.interval),
            "recall_summarize": (self.interval, 0, self.interval),
            "analyze_summarize": (0, self.interval * self.num_option, self.interval),
        }
        self.recall_interval, self.analyze_interval, self.summarize_interval = strategy_mapping[self.stage_type]
        self.all_strategy_steps = sum(strategy_mapping[self.stage_type])


@dataclass
class SFTTrainingArguments(Seq2SeqTrainingArguments):
    stage_type: str = field(default="recall_analyze_summarize", metadata={"help": "Which stage to start generation."})
    recall_interval: int = field(default=0, metadata={"help": "The interval steps for generating general knowledge."})
    analyze_interval: int = field(default=0, metadata={"help": "The interval steps for generating specific knowledge for each option."})
    summarize_interval: int = field(default=0, metadata={"help": "The interval steps for summerizing."})
    recall_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
    analyze_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
    summarize_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )


@dataclass
class RDPOConfiguration(BaseConfiguration):
    ref_model: str = None
    dpo_iter: int = 0
    beta: float = 0.1
    rpo_alpha: float = 0.5
    max_prompt_length: int = 200
    max_target_length: int = 200

    def __post_init__(self):
        super().__post_init__()
        
        self.dataset_path = {'valid': os.path.join(self.dataset_dir, 'sft/recall_analyze_summarize/valid.json')}
        model_type = self.base_model.split('/')[-1]
        self.dataset_dir = os.path.join(self.dataset_dir, 'dpo', self.stage_type, model_type,)

        file_mapping = {
            "recall_analyze": ['recall_dpo', 'analyze_dpo'],
            "recall": ['recall_dpo'],
            "analyze": ['analyze_dpo']
        }
        self.dataset_path.update({
            split: os.path.join(self.dataset_dir, f"iter_{self.dpo_iter}/{split}.json")
            for split in file_mapping[self.stage_type]
        })

        if self.dpo_iter == 1:
            self.ref_model = os.path.join(self.output_dir, "sft", "recall_analyze_summarize", self.ref_model)
        elif self.dpo_iter > 1:
            self.ref_model = os.path.join(self.output_dir, self.training_type, self.stage_type, f'iter_{self.dpo_iter-1}', self.ref_model)
        
        self.output_dir = os.path.join(self.output_dir, self.training_type, self.stage_type, f'iter_{self.dpo_iter}')


@dataclass
class RDPOTrainingArguments(DPOConfig):
    stage_type: str = field(default="recall_analyze_summarize", metadata={"help": "Which stage to start generation."})
    recall_interval: int = field(default=0, metadata={"help": "The interval steps for generating general knowledge."})
    analyze_interval: int = field(default=0, metadata={"help": "The interval steps for generating specific knowledge for each option."})
    summarize_interval: int = field(default=0, metadata={"help": "The interval steps for summerizing."})
    recall_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
    analyze_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
    summarize_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `num_beams` value of the model configuration."
            )
        },
    )
    generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )


@dataclass
class GenerationConfiguration(BaseConfiguration):
    dpo_iter: int = 0
    generate_dir: str = "results/"
    generation_type: str = None
    generation_file: str = None
    temperature_increase: float = 0.5
    generation_times: int = 1

    def __post_init__(self):
        super().__post_init__()

        if self.generation_file == "test":
            if self.dpo_iter > 0:
                self.dataset_path = {'test': os.path.join(self.dataset_dir, "sft/recall_analyze_summarize/test.json")}
            else:
                self.dataset_path = {'test': os.path.join(self.dataset_dir, self.training_type, self.stage_type, "test.json")}
            self.generate_dir = os.path.abspath(os.path.join(ROOT_PATH, self.generate_dir, self.data_name, self.base_model))

            if self.dpo_iter:
                suffix = f'{self.training_type}_{self.generation_file}_{self.stage_type}_{self.dpo_iter}_{self.generation_type}'
            else:
                suffix = f'{self.training_type}_{self.generation_file}_{self.stage_type}_{self.generation_type}'
            self.generate_path = os.path.join(self.generate_dir, f'{suffix}.json')
            
            if self.checkpoint_dir:
                subdir = f"{self.stage_type}/iter_{self.dpo_iter}" if self.dpo_iter > 0 else f"{self.stage_type}/"
                self.checkpoint_dir = os.path.join(self.output_dir, self.training_type, subdir, self.checkpoint_dir)

            self.recall_sample = True
            self.recall_temperature = 0.7
            self.recall_top_k = 0
            self.analyze_sample = True
            self.analyze_temperature = 0.7
            self.analyze_top_k = 0 
        elif self.generation_file == "dpo":
            self.dataset_path = {
                'original_summarize': os.path.join(self.dataset_dir, "sft/recall_analyze_summarize/summarize.json"),
                'target_recall': os.path.join(self.dataset_dir, "sft/recall_analyze_summarize/recall_single.json")
            }

            model_type = self.base_model.split('/')[-1]
            self.generate_dir = os.path.join(self.dataset_dir, 'dpo', self.stage_type)
            self.generate_path = os.path.join(self.generate_dir, model_type, f'iter_{self.dpo_iter}/')

            if self.checkpoint_dir:
                self.output_dir = os.path.join(self.output_dir, self.training_type)
                if self.training_type == "sft":
                    self.checkpoint_dir = os.path.join(self.output_dir, "recall_analyze_summarize", self.checkpoint_dir)
                elif self.training_type == "dpo":
                    self.checkpoint_dir = os.path.join(self.output_dir, self.stage_type, f'iter_{self.dpo_iter-1}', self.checkpoint_dir)

            self.recall_sample = True
            self.recall_temperature = 0.7
            self.recall_top_k = 0
            self.analyze_sample = False
            self.analyze_temperature = 1.0
            self.analyze_top_k = 50
        
        self._create_generate_dir()
    
    def _create_generate_dir(self):
        dir_path = os.path.dirname(self.generate_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class GenerationArguments(Seq2SeqTrainingArguments):
    stage_type: str = field(default="recall_analyze_summarize", metadata={"help": "Which stage to start generation."})
    recall_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
    analyze_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
    summarize_generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
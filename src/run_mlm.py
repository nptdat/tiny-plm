#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
# This script are customized from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py


import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, List, Optional, Union

import evaluate
import transformers
from accelerate.utils import DistributedType
from datasets import DatasetDict, load_dataset, utils
from torch import nn
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertJapaneseTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

# import datasets


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)
logger.setLevel("INFO")  # temporary

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
SUPPORT_DATA_FILE = ["csv", "json", "txt", "parquet"]
HYPER_PARAM_FILE = Path("/opt/ml/input/config/hyperparameters.json")

# For TrainingArguments details, ref to https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )

    tokenizer_class: str = field(
        default="BertJapaneseTokenizer",
        metadata={
            "help": "Class of the tokenizer to initialize. Support [BertJapaneseTokenizer, AutoTokenizer]"
        },
    )

    mecab_dic_type: str = field(
        default="unidic_lite", metadata={"help": "Dictionary used in MeCab"}
    )

    tokenizer_vocab_file: str = field(
        default="vocab.txt",
        metadata={"help": "Path to the vocab.txt to init the tokenizer"},
    )

    tokenizer_path: str = field(
        default="sp_tokenizer",
        metadata={
            "help": "Path to the folder to init the tokenizer with AutoTokenizer.from_pretrained"
        },
    )

    model_config_file: str = field(
        default="model-config.json",
        metadata={"help": "Path to json config for model inialization"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_path: str = field(
        default="", metadata={"help": "The input training data file."}
    )
    validation_path: str = field(
        default="",
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={
            "help": "Ratio of tokens to mask for masked language modeling loss"
        },
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self) -> None:
        if (
            self.dataset_name is None
            and self.train_path is None
            and self.validation_path is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_path is not None:
                extension = self.train_path.split(".")[-1]
                if "/" not in extension:  # file
                    if extension not in SUPPORT_DATA_FILE:
                        raise ValueError(
                            f"`train_path` should has file extension in the list: {SUPPORT_DATA_FILE}"
                        )
            if self.validation_path is not None:
                extension = self.validation_path.split(".")[-1]
                if "/" not in extension:  # file
                    if extension not in SUPPORT_DATA_FILE:
                        raise ValueError(
                            f"`validation_path` should has file extension in the list: {SUPPORT_DATA_FILE}"
                        )


def build_datasets(
    data_args: DataTrainingArguments, model_args: ModelArguments
) -> DatasetDict:
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # Ensure that data_args.dataset_name is None
    data_files: dict[str, Union[str, List[str]]] = {}
    if data_args.train_path is not None:
        train_path = Path(data_args.train_path)
        if train_path.is_file():
            data_files["train"] = data_args.train_path
            extension = data_args.train_path.split(".")[-1]
        else:
            files_ = list(train_path.glob("*.*"))
            extension = files_[0].suffix[1:]
            files = [str(fpath) for fpath in files_]
            data_files["train"] = files
        if extension == "txt":
            extension = "text"

    # validation file must be the same type as that of train file
    if data_args.validation_path is not None:
        validation_path = Path(data_args.validation_path)
        if validation_path.is_file():
            data_files["validation"] = data_args.validation_path
        else:
            data_files["validation"] = [
                str(fpath) for fpath in validation_path.glob("*.*")
            ]

    # raw_datasets is a DatasetDict with 2 keys (2 Dataset): train & validation.
    # Each dataset contains only 1 'text' feature
    logger.info(f"--- {extension=} ---")
    logger.info(f"--- {data_files=} ---")

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    return raw_datasets


def init_tokenizer(
    tokenizer_class: str,
    tokenizer_path: str,
    vocab_file: str,
    mecab_dic_type: str,
) -> "PreTrainedTokenizer":
    if tokenizer_class == "BertJapaneseTokenizer":
        tokenizer = BertJapaneseTokenizer(
            vocab_file,
            do_lower_case=False,
            word_tokenizer_type="mecab",
            subword_tokenizer_type="wordpiece",
            tokenize_chinese_chars=False,
            mecab_kwargs={"mecab_dic": mecab_dic_type},
            # model_max_length = MAX_LENGTH
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def init_config(
    model_args: ModelArguments, pad_token_id: int = 0
) -> "PretrainedConfig":
    if model_args.model_name_or_path:
        model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    elif Path(model_args.model_config_file).exists():
        with open(model_args.model_config_file, "rt") as f:
            config_data = json.load(f)
        config_class = CONFIG_MAPPING[model_args.model_type]
        model_config = config_class(pad_token_id=pad_token_id, **config_data)
    return model_config


def init_model(
    model_config: PretrainedConfig, model_name_or_path: str
) -> "PreTrainedModel":
    if model_name_or_path:
        logger.info(f"Loading model from {model_name_or_path=}")
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    else:
        logger.info(f"Init model from {model_config=}")
        model = AutoModelForMaskedLM.from_config(model_config)
    return model


def count_model_params(model: nn.Module) -> int:
    total = 0
    for _, params in model.named_parameters():
        total += params.numel()
    return total


def get_model_max_input_len(model: "nn.Module") -> Any:
    """Return the max input_len of a model using the size of the position_embedding layer."""
    return list(model.base_model.embeddings.position_embeddings.parameters())[
        0
    ].shape[0]


def main() -> None:
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    logger.info("==========================", sys.argv)
    if sys.argv[-1] == "train":
        sys.argv = sys.argv[:-1]
        logger.info("argv after adjusting: ", sys.argv)

    # Dump partition info
    for line in subprocess.run(
        ["df", "-h"], encoding="utf-8", stdout=subprocess.PIPE
    ).stdout.split("\n"):
        logger.info(line)

    # Overwrite params from hypermeters.json
    args_json_file = os.path.abspath(sys.argv[-1])
    with open(args_json_file, "rt") as f:
        args = json.load(f)
    if HYPER_PARAM_FILE.exists():
        logger.info(
            f"File {str(HYPER_PARAM_FILE)} exists -> overwrite arguments"
        )
        with open(HYPER_PARAM_FILE, "rt") as f:
            hyper_params = json.load(f)
            for k in hyper_params.keys():
                if k in args:
                    hyper_params[k] = type(args[k])(hyper_params[k])
            logger.info(hyper_params)
        args.update(hyper_params)

    model_args, data_args, training_args = parser.parse_dict(args)

    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None

    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if (
            last_checkpoint is None
            and len(os.listdir(training_args.output_dir)) > 0
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None
            and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    logger.info("Building datasets...")
    raw_datasets = build_datasets(data_args, model_args)

    # Custom
    tokenizer = init_tokenizer(
        tokenizer_class=model_args.tokenizer_class,
        tokenizer_path=model_args.tokenizer_path,
        vocab_file=model_args.tokenizer_vocab_file,
        mecab_dic_type=model_args.mecab_dic_type,
    )
    model_config = init_config(model_args, tokenizer.pad_token_id)
    model = init_model(model_config, model_args.model_name_or_path)
    logger.info(f"-------- Model size: {count_model_params(model):,} params")
    logger.info(f"{model=}")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        model_max_seq_length = get_model_max_input_len(model)
        if max_seq_length > model_max_seq_length:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Picking {model_max_seq_length} instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = model_max_seq_length
        # TODO: adjust max_seq_length with respect to model's max_position_embeddings
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(
            data_args.max_seq_length, tokenizer.model_max_length
        )

    if data_args.line_by_line:
        logger.info("Running tokenizer on dataset line_by_line...")

        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples: dict) -> Any:
            # Remove empty lines
            examples[text_column_name] = [
                line
                for line in examples[text_column_name]
                if len(line) > 0 and not line.isspace()
            ]
            tokenized_data = tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

            # Compute lengths of samples here to utilize multi-threading of Datast.map
            # This processing help to avoid single-thread computation in LengthGroupedSampler,
            # which takes very long time on huge dataset
            if training_args.group_by_length:
                tokenized_data["length"] = [
                    len(input_ids) for input_ids in tokenized_data["input_ids"]
                ]
            return tokenized_data

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenizing line_by_line",
            )
            # the datasets in tokenized_datasets now has features=['input_ids', 'token_type_ids', 'special_tokens_mask', 'attention_mask']
    else:
        logger.info(
            "Running tokenizer on dataset with NOT line_by_line mode..."
        )

        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples: dict) -> Any:
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenizing NOT line_by_line",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples: dict) -> dict:
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples
            )
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples
            )
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits: Any, labels: Any) -> Any:
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds: Any) -> Any:
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = (
        data_args.line_by_line
        and training_args.fp16
        and not data_args.pad_to_max_length
    )
    # TODO: consider to use DataCollatorForWholeWordMask instead (same as cl-tohoku/bert-base-japanese-whole-word-masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )
    logger.info(f"----------- {data_collator=}")

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(
            compute_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None
        ),
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None
        ),
    )
    logger.info(f"----------- {trainer=}")

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        logger.info(f"--- {checkpoint=}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {
        # "finetuned_from": model_args.model_name_or_path,
        "finetuned_from": None,
        "tasks": "fill-mask",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
        else:
            kwargs["dataset"] = data_args.dataset_name

    trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()

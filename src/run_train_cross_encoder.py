"""
Train Cross-Encoder model for Japanese, using POSITIVE/NEGATIVE (Query, Doc) pairs dataset.
This script refer to the script `train_cross-encoder_scratch.py` in [1] from sentence-transformers.

INPUT
    - `config_file`: config file in YAML format.
        e.g., `src/config/cross-encoder-4-distil/cross_encoder_hr.yml`.
        The config file contains the following keys:
            - `input_file`: path to input file. (e.g., `data/cross-encoder-hr-train-18M_pairs.pkl.gz`)
            - `output_path`: path to output folder. (e.g., `model/bert-30M-hrvocab-cross-encoder`)
            - `model_path`: path to save the output model and other training result.
            - `max_train_size`: limit num of training samples. 0 for no limit.
            - `max_eval_size`: limit num of validation samples. 0 for no limit.
            - `max_token_len`: limit num of tokens for samples, including `query`, `doc` and several special tokens.
                - Note that this limit must be less than or equal to model `max_position_embeddings`
            - `num_epochs`: num of epoch to train
            - `train_batch_size`: training batch size
            - `evaluation_steps`: run evaluation for every training steps
            - `warmup_steps`: initial warmup steps for the learning rate

OUTPUT
    - Output the trained model & other training results to the folder defined in config `output_path` key.

REFERENCES:
[1] https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_scratch.py
"""

import gzip
import pickle
from collections import Counter
from datetime import datetime
from logging import basicConfig, getLogger
from pathlib import Path
from time import time

import numpy as np
import transformers
import typer
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.io.file import load_yaml

logger = getLogger(__name__)
basicConfig(
    level="INFO",
    format="%(asctime)s : %(levelname)s : %(name)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

np.random.seed(42)

LOSS_FUNCTION_MAP = {
    "MSELoss": nn.MSELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "L1Loss": nn.L1Loss,
}


def main(
    config_file: Path = Path(
        "src/config/cross-encoder-4-distil/cross_encoder_train.yml"
    ),
) -> None:
    logger.info(
        "=== STARTED cross-encoder training from run_train_cross_encoder.py"
    )

    start_time = time()
    transformers.logging.set_verbosity_error()

    cfg = load_yaml(config_file)
    logger.info(f"{cfg=}")

    input_file = cfg.get("input_file")
    eval_file = cfg.get("eval_file")
    model_path = cfg.get("model_path")
    output_path = cfg.get("output_path", "output")
    tensorboard_path = cfg.get("tensorboard_path")

    max_train_size = cfg.get("max_train_size", 0)
    max_eval_size = cfg.get("max_eval_size", 100)
    max_token_len = cfg.get("max_token_len", 512)
    char_per_token_ratio = cfg.get(
        "char_per_token_ratio", 1.0
    )  # separately confirmed the ratio

    num_epochs = cfg.get("num_epochs", 1)
    train_batch_size = cfg.get("train_batch_size", 32)

    evaluation_steps = cfg.get("evaluation_steps", 1000)
    logging_steps = cfg.get("logging_steps", 100)
    warmup_steps = cfg.get("warmup_steps", 5000)

    oversampling_ratio = cfg.get("oversampling_ratio", 1)
    oversampling_min_score = cfg.get("oversampling_min_score", 1)

    loss_function = cfg.get("loss_function")

    max_len = int(max_token_len * char_per_token_ratio)

    logger.info(f"{char_per_token_ratio=}")
    logger.info(f"{max_token_len=}")
    logger.info(f"{max_len=}")
    logger.info(f"{oversampling_ratio=}")
    logger.info(f"{oversampling_min_score=}")
    logger.info(f"{loss_function=}")

    # Load data
    logger.info("Loading training data...")
    with gzip.open(input_file, "rb") as f:
        (dev_samples, raw_train_samples) = pickle.load(f)

    if max_eval_size > 0:
        dev_samples = dev_samples[:max_eval_size]

    eval_scout_samples, eval_reply_samples = [], []
    if eval_file:
        with gzip.open(eval_file, "rb") as f:
            (eval_scout_samples, eval_reply_samples) = pickle.load(f)

    logger.info(f"dev_samples={len(dev_samples)}")
    logger.info(f"raw_train_samples={len(raw_train_samples)}")
    logger.info(f"eval_scout_samples={len(eval_scout_samples)}")
    logger.info(f"eval_reply_samples={len(eval_reply_samples)}")

    train_samples = []
    for query, passage, label in tqdm(raw_train_samples):
        data_len = len(query) + len(passage)
        if data_len > max_len:
            continue

        train_samples.append(InputExample(texts=[query, passage], label=label))
        # oversampling on POSITIVE samples
        if label >= oversampling_min_score:
            for i in range(oversampling_ratio):
                train_samples.append(
                    InputExample(texts=[query, passage], label=label)
                )

    logger.info(f"{len(train_samples)=}")

    if max_train_size > 0 and len(train_samples) >= max_train_size:
        np.random.shuffle(train_samples)
        logger.info(f"After shuffling: {len(train_samples)=}")
        train_samples = train_samples[:max_train_size]

    counter = Counter([sample.label for sample in train_samples])

    logger.info(f"Final: {len(train_samples)=}")
    logger.info(f"Ratios of labels: {counter}")

    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=train_batch_size
    )

    # breakpoint()
    evaluator = CERerankingEvaluator(dev_samples, name="train-validation")
    scout_evaluator = (
        CERerankingEvaluator(eval_scout_samples, name="scout-evaluation")
        if eval_scout_samples
        else None
    )
    reply_evaluator = (
        CERerankingEvaluator(eval_reply_samples, name="reply-evaluation")
        if eval_reply_samples
        else None
    )

    model = CrossEncoder(
        model_path,
        num_labels=1,
        max_length=max_token_len,
    )

    # Train the model
    loss_fct = LOSS_FUNCTION_MAP.get(loss_function)
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        loss_fct=loss_fct,
        evaluation_steps=evaluation_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        output_path=output_path,
        use_amp=True,
        tensorboard_path=tensorboard_path,
    )
    """
    NOTE: customize the sentence-bert's CrossEncoder class to enable `logging_steps` and `tensorboard_path` to the `fit` function.
    https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/CrossEncoder.py#L180
    """

    # Save latest model
    # output_path: store the model which gives highest validation score
    # output_path/final: store the model trained till the final step. Note that model in this folder may be worse than the model in `output_path`.
    model.save(output_path + "/final")

    if scout_evaluator:
        logger.info("Evaluating on scout data")
        scout_evaluator(model)
    if reply_evaluator:
        logger.info("Evaluating on reply data")
        reply_evaluator(model)

    logger.info(f"Finished cross-encoder training in {time() - start_time}")


if __name__ == "__main__":
    typer.run(main)

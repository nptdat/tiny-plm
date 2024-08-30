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
import logging
import pickle
from datetime import datetime
from pathlib import Path

import transformers
import typer
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.io.file import load_yaml

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main(
    config_file: Path = Path(
        "src/config/cross-encoder-4-distil/cross_encoder.yml"
    ),
) -> None:
    transformers.logging.set_verbosity_error()

    cfg = load_yaml(config_file)
    logging.info(f"{cfg=}")

    # Load data
    logging.info("Loading training data...")
    with gzip.open(cfg["input_file"], "rb") as f:
        (dev_samples, raw_train_samples) = pickle.load(f)

    if cfg["max_eval_size"] > 0:
        dev_samples = dev_samples[: cfg["max_eval_size"]]

    logging.info(f"dev_samples={len(dev_samples)}")
    logging.info(f"raw_train_samples={len(raw_train_samples)}")

    train_samples = []
    cnt = 0
    for query, passage, label in tqdm(raw_train_samples):
        data_len = len(query) + len(passage)
        if data_len > cfg["max_token_len"]:
            continue

        train_samples.append(InputExample(texts=[query, passage], label=label))
        cnt += 1
        if cfg["max_train_size"] > 0 and cnt >= cfg["max_train_size"]:
            break

    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=cfg["train_batch_size"]
    )
    evaluator = CERerankingEvaluator(dev_samples, name="train-eval")

    model = CrossEncoder(
        cfg["model_path"], num_labels=1, max_length=cfg["max_token_len"]
    )

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=cfg["num_epochs"],
        evaluation_steps=cfg["evaluation_steps"],
        warmup_steps=cfg["warmup_steps"],
        output_path=cfg["output_path"],
        use_amp=True,
    )

    # Save latest model
    model.save(cfg["model_path"])


if __name__ == "__main__":
    typer.run(main)

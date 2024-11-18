import gzip
import pickle
import warnings
from collections import defaultdict
from logging import basicConfig, getLogger
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
import typer
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm
from transformers import PreTrainedModel

from utils.io.file import load_yaml

warnings.filterwarnings("ignore")

logger = getLogger(__name__)
basicConfig(
    level="INFO",
    format="%(asctime)s : %(levelname)s : %(name)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def inverse_sigmoid(prob: float) -> float:
    """
    s = sigmoid(x) -> x = 1/(1 + e^(-x))
    -> x = ln(s) - ln(1-s)
    """
    if prob >= 1:
        return 10 ^ 6
    if prob <= 0:
        return -10 ^ 6
    return float(np.log(prob) - np.log(1 - prob))


def save_pkl_gzip(
    filepath: str, similarity_scores: dict[int, dict[int, float]]
) -> None:
    filepath_ = Path(filepath)
    if not filepath_.parent.exists():
        filepath_.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(filepath_, "wb") as f:
        pickle.dump(similarity_scores, f)


def get_all_query_ids(cfg: dict[str, Any]) -> list[int]:
    with gzip.open(cfg["hard_negative_init_score_file"], "rb") as f:
        init_scores = pickle.load(f)
    sorted_qids: list[int] = sorted(init_scores.keys())
    return sorted_qids


def load_master(train_file_path: str, valid_file_path: str) -> dict[int, str]:
    df1 = pd.read_csv(train_file_path, delimiter="\t", header=None)
    df2 = pd.read_csv(valid_file_path, delimiter="\t", header=None)
    df = pd.concat([df1, df2])
    df.columns = ["item_id", "text"]
    df.drop_duplicates(subset=["item_id"], inplace=True)

    id2item: dict[int, str] = dict()
    for item_id, query in tqdm(zip(df["item_id"], df["text"])):
        id2item[item_id] = query
    return id2item


def load_data(
    cfg: dict[str, Any]
) -> tuple[dict[int, dict[int, float]], dict[int, str], dict[int, str]]:
    logger.info("Loading data...")

    id2doc: dict[int, str] = load_master(
        cfg["train_doc_master"], cfg["valid_doc_master"]
    )
    id2query: dict[int, str] = load_master(
        cfg["train_query_master"], cfg["valid_query_master"]
    )
    with gzip.open(cfg["hard_negative_init_score_file"], "rb") as f:
        init_scores = pickle.load(f)

    num_pairs = sum([len(scores) for query_id, scores in init_scores.items()])
    logger.info(f"Num of docs in the master: {len(id2doc):,}")
    logger.info(f"Num of queries in the master: {len(id2query):,}")
    logger.info(f"Num of queries to compute scores={len(init_scores):,}")
    logger.info(f"Num of (query, doc) pairs to compute scores={num_pairs:,}")

    return init_scores, id2query, id2doc


def prepare_data(
    cfg: dict[str, Any], target_qids: set[int], max_len: int
) -> tuple[list[tuple[int, int, int]], list[tuple[str, str]]]:
    logger.info(f"Preparing data for prediction with {len(target_qids)=}...")

    init_scores, id2query, id2doc = load_data(cfg)
    sorted_qids = sorted(init_scores.keys())
    logger.info(
        f"{len(init_scores)=}, {len(sorted_qids)=}, {len(id2query)=}, {len(id2doc)=}"
    )

    len_triples: list[tuple[int, int, int]] = []
    # breakpoint()
    for qid in target_qids:
        query = id2query[qid]
        max_p_len = max(max_len - len(query), 0)
        doc_ids = list(init_scores[qid].keys())
        for doc_id in doc_ids:
            # NOTE: if the query_text is longer than max_len, then all of the doc_text will be empty.
            doc = id2doc[doc_id][:max_p_len]
            len_triples.append((qid, doc_id, len(query) + len(doc)))
    sorted_len_triples: list[tuple[int, int, int]] = sorted(
        len_triples, key=lambda x: -x[2]
    )
    logger.info(f"{len(sorted_len_triples)=}")

    # accumulate all pairs sorted by text len (desc order)
    logger.info("Building (query, doc) pairs ordered by text len...")
    text_pairs: list[tuple[str, str]] = []
    for qid, doc_id, text_len in tqdm(sorted_len_triples):
        query = id2query[qid]
        max_p_len = max(max_len - len(query), 0)
        doc = id2doc[doc_id][:max_p_len]
        text_pairs.append((query, doc))
    logger.info(f"Num of pairs: {len(text_pairs):,}")

    logger.info("Finished preparing data for prediction!")

    return sorted_len_triples, text_pairs


def predict(
    text_pairs: list[tuple[str, str]], model: PreTrainedModel, batch_size: int
) -> list[float]:
    # predict sim scores with cross-encoder
    logger.info("Predicting scores with cross-encoder...")
    pred_scores: list[float] = model.predict(
        text_pairs, batch_size=batch_size, show_progress_bar=True
    ).tolist()

    # because we predict regression on label 0..1 by passing num_labels=1,
    # the CrossEncoder apply sigmoid on the logits.
    # Here, we apply inverse_sigmoid to convert the prob score back to logits.
    pred_scores = [inverse_sigmoid(score) for score in pred_scores]

    logger.info(
        f"Finished predicting with cross-encoder! Num of scores: {len(pred_scores)}..."
    )
    return pred_scores


def build_similarity_scores(
    sorted_len_triples: list[tuple[int, int, int]], pred_scores: list[float]
) -> dict[int, dict[int, float]]:
    tuple_data: tuple = tuple(zip(*sorted_len_triples))
    (qids, doc_ids, lens_) = tuple_data

    # return similarity scores
    logger.info("Building similarity_scores...")
    similarity_scores: dict[int, dict[int, float]] = defaultdict(dict)
    for qid, doc_id, score in zip(qids, doc_ids, pred_scores):
        similarity_scores[qid][doc_id] = score
    return similarity_scores


def main(
    config_file: Path = Path(
        "src/config/cross-encoder-4-distil/cross_encoder_predict.yml"
    ),
) -> None:
    logger.info(
        "=== STARTED cross-encoder prediction from run_predict_cross_encoder.py"
    )

    start_time = time()

    # To avoid errors from transformers, like:
    # Be aware, overflowing tokens are not returned for the setting you have chosen,
    # i.e. sequence pairs with the 'longest_first' truncation strategy.
    # So the returned list will always be empty even if some tokens have been removed.
    transformers.logging.set_verbosity_error()

    cfg = load_yaml(config_file)
    logger.info(f"{cfg=}")

    file_index_from = cfg["file_index_from"]
    max_token_len = cfg["max_token_len"]
    char_per_token_ratio = cfg["char_per_token_ratio"]
    max_len = int(max_token_len * char_per_token_ratio)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Cross-Encoder model...")
    model = CrossEncoder(
        cfg["model_path"],
        num_labels=1,
        max_length=max_token_len,
        device=device,
    )

    sorted_qids = get_all_query_ids(cfg)
    logger.info(f"All sorted_qids: {len(sorted_qids):,}!")

    query_from = cfg["predict_query_from"]
    query_to = cfg["predict_query_to"]
    if query_to == 0:
        query_to = len(sorted_qids)
    logger.info(
        f"{max_token_len=}, {char_per_token_ratio=}, {max_len=}, {query_from=}, {query_to=}"
    )

    start = 0
    chunk_size = cfg["predict_query_chunk"]
    for idx, start in tqdm(enumerate(range(query_from, query_to, chunk_size))):
        file_index = idx + file_index_from
        end = min(start + chunk_size, query_to)
        if start >= end:
            break

        logger.info(
            f"--- Predicting for the chunk {file_index}(th), {start=}, {end=}..."
        )
        sorted_len_triples, text_pairs = prepare_data(
            cfg=cfg, target_qids=set(sorted_qids[start:end]), max_len=max_len
        )
        pred_scores = predict(text_pairs, model, cfg["predict_batch_size"])
        similarity_scores = build_similarity_scores(
            sorted_len_triples, pred_scores
        )

        output_file_tpl = cfg["hard_negative_cross_encoder_score_file"]
        output_file = output_file_tpl.format(file_index)
        logger.info(f"Writing similarity scores to {output_file}")
        save_pkl_gzip(output_file, similarity_scores)

    logger.info(f"Finished cross-encoder prediction in {time() - start_time}")


if __name__ == "__main__":
    typer.run(main)

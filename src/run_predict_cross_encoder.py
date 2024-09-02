import gzip
import pickle
import warnings
from collections import defaultdict
from logging import basicConfig, getLogger
from pathlib import Path
from time import time
from typing import Any, Dict, Tuple, Union

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


def save_pkl_gzip(
    filepath: str, similarity_scores: Dict[int, Dict[int, float]]
) -> None:
    filepath_ = Path(filepath)
    if not filepath_.parent.exists():
        filepath_.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(filepath_, "wb") as f:
        pickle.dump(similarity_scores, f)


def load_data(
    cfg: Dict[str, Any]
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, str], Dict[int, str]]:
    logger.info("Loading data...")
    df1 = pd.read_csv(cfg["train_doc_master"], delimiter="\t", header=None)
    df2 = pd.read_csv(cfg["valid_doc_master"], delimiter="\t", header=None)
    df_docs = pd.concat([df1, df2])
    df_docs.columns = ["doc_id", "text"]
    df_docs.drop_duplicates(subset=["doc_id"], inplace=True)

    df1 = pd.read_csv(cfg["train_query_master"], delimiter="\t", header=None)
    df2 = pd.read_csv(cfg["valid_query_master"], delimiter="\t", header=None)
    df_queries = pd.concat([df1, df2])
    df_queries.columns = ["query_id", "text"]
    df_queries.drop_duplicates(subset=["query_id"], inplace=True)

    with gzip.open(cfg["hard_negative_init_score_file"], "rb") as f:
        init_scores = pickle.load(f)

    id2doc = {}
    for pid, passage in tqdm(zip(df_docs["doc_id"], df_docs["text"])):
        id2doc[pid] = passage

    id2query = {}
    for qid, query in tqdm(zip(df_queries["query_id"], df_queries["text"])):
        id2query[qid] = query

    num_pairs = sum([len(scores) for query_id, scores in init_scores.items()])

    logger.info(f"Num of docs in the master: {len(id2doc):,}")
    logger.info(f"Num of queries in the master: {len(id2query):,}")
    logger.info(f"Num of queries to compute scores={len(init_scores):,}")
    logger.info(f"Num of (query, doc) pairs to compute scores={num_pairs:,}")

    return init_scores, id2query, id2doc


def predict(
    init_scores: Dict[int, Dict[int, float]],
    model: PreTrainedModel,
    id2query: Dict[int, str],
    id2doc: Dict[int, str],
    batch_size: int,
    max_len: int,
) -> dict[int, dict[int, float]]:
    # Sorted pairs of (query, doc) by text len
    logger.info("Accumulate lens of (query, doc) pairs...")
    len_list = []
    # cnt = 0
    for qid, data in tqdm(init_scores.items()):
        doc_ids = list(data.keys())
        query = id2query[qid]
        max_p_len = max_len - len(query)
        for doc_id in doc_ids:
            doc = id2doc[doc_id][:max_p_len]
            len_list.append((qid, doc_id, len(query) + len(doc)))
        # cnt += 1
        # if cnt >= 50:
        #     break
    sorted_len_list = sorted(len_list, key=lambda x: -x[2])

    # accumulate all pairs sorted by text len (desc order)
    logger.info("Building (query, doc) pairs ordered by text len...")
    all_pairs = []
    for qid, doc_id, text_len in tqdm(sorted_len_list):
        query = id2query[qid]
        max_p_len = max_len - len(query)
        doc = id2doc[doc_id][:max_p_len]
        all_pairs.append([query, doc])
    logger.info(f"Num of pairs: {len(all_pairs):,}")

    # predict sim scores with cross-encoder
    logger.info("Predicting scores with cross-encoder...")
    scores = model.predict(
        all_pairs, batch_size=batch_size, show_progress_bar=True
    ).tolist()
    tuple_data: tuple = tuple(zip(*sorted_len_list))
    (qids, doc_ids, lens_) = tuple_data
    logger.info(
        f"Finished predicting with cross-encoder! Num of scores: {len(scores)}..."
    )

    # return similarity scores
    logger.info("Building similarity_scores...")
    similarity_scores: dict[int, dict[int, float]] = defaultdict(dict)
    for qid, doc_id, score in zip(qids, doc_ids, scores):
        similarity_scores[qid][doc_id] = score

    logger.info("Finished similarity_scores!")
    return similarity_scores


# def predict_bak(
#     init_scores: Dict[int, Dict[int, float]],
#     model: PreTrainedModel,
#     id2query: Dict[int, str],
#     id2doc: Dict[int, str],
#     batch_size: int,
#     max_len: int,
# ) -> dict[int, dict[int, float]]:
#     similarity_scores = {}

#     for qid, data in tqdm(init_scores.items()):
#         doc_ids = list(data.keys())

#         query = id2query[qid]
#         max_p_len = max_len - len(query)
#         scores = []
#         for start in range(0, len(doc_ids), batch_size):
#             end = start + batch_size
#             doc_ids_ = doc_ids[start:end]
#             pairs = []
#             for doc_id in doc_ids_:
#                 pairs.append([query, id2doc[doc_id][:max_p_len]])
#             scores_ = model.predict(
#                 pairs,
#                 batch_size=1,
#                 show_progress_bar=False
#             ).tolist()
#             scores.extend(scores_)

#         similarity_scores[qid] = dict(zip(doc_ids_, scores))

#     return similarity_scores


def main(
    config_file: Path = Path(
        "src/config/cross-encoder-4-distil/cross_encoder_predict.yml"
    ),
) -> None:
    start_time = time()

    # To avoid errors from transformers, like:
    # Be aware, overflowing tokens are not returned for the setting you have chosen,
    # i.e. sequence pairs with the 'longest_first' truncation strategy.
    # So the returned list will always be empty even if some tokens have been removed.
    transformers.logging.set_verbosity_error()

    cfg = load_yaml(config_file)
    logger.info(f"{cfg=}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_scores, id2query, id2doc = load_data(cfg)

    logger.info("Initializing Cross-Encoder model...")
    model = CrossEncoder(
        cfg["model_path"],
        num_labels=1,
        max_length=cfg["max_token_len"],
        device=device,
    )

    logger.info("Predict similarity scores for (query/doc) pairs...")
    similarity_scores = predict(
        init_scores,
        model,
        id2query,
        id2doc,
        cfg["predict_batch_size"],
        cfg["max_token_len"],
    )
    logger.info(f"similarity_scores={len(similarity_scores):,}")

    output_file = cfg["hard_negative_cross_encoder_score_file"]
    logger.info(f"Writing similarity scores to {output_file}")
    save_pkl_gzip(output_file, similarity_scores)

    logger.info(f"Finished cross-encoder prediction in {time() - start_time}")


if __name__ == "__main__":
    typer.run(main)

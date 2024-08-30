import gzip
import pickle
import warnings
from logging import basicConfig, getLogger
from pathlib import Path
from time import time
from typing import Any, Dict, Tuple, Union

import pandas as pd
import torch
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
    similarity_scores = {}

    # TODO: flatten the whole data into list of (query, doc) and pass to model for more effective inference
    for qid, data in tqdm(init_scores.items()):
        doc_ids = list(data.keys())

        query = id2query[qid]
        max_p_len = max_len - len(query)
        scores = []
        for start in range(0, len(doc_ids), batch_size):
            end = start + batch_size
            doc_ids_ = doc_ids[start:end]
            pairs = []
            for doc_id in doc_ids_:
                pairs.append([query, id2doc[doc_id][:max_p_len]])
            scores_ = model.predict(pairs).tolist()
            scores.extend(scores_)

        similarity_scores[qid] = dict(zip(doc_ids_, scores))

    return similarity_scores


def main(
    config_file: Path = Path(
        "src/config/cross-encoder-4-distil/cross_encoder_predict.yml"
    ),
) -> None:
    start_time = time()

    cfg = load_yaml(config_file)
    logger.info(f"{cfg=}")

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
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
    save_pkl_gzip(
        cfg["hard_negative_cross_encoder_score_file"], similarity_scores
    )

    logger.info(f"Finished cross-encoder prediction in {time() - start_time}")


if __name__ == "__main__":
    typer.run(main)

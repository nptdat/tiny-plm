"""
Extract longest sentences from a dataset. These long sentences can be used to execute a dryrun for pre-training, then to confirm hyperparameters of a pretraining (esp batch_size which affect GPU memory).
"""

import logging
from pathlib import Path

import typer
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(asctime)s : %(levelname)s : %(name)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(
    data_dir: str = "data/cc100_ja/train",
    long_sentence_file: str = "data/cc100_ja/long_sentences_only.txt",
    min_sent_count: int = 64,
) -> None:
    data_path = Path(data_dir)
    if data_path.is_file():
        data_files = [data_path]
    else:
        data_files = list(data_path.glob("*.txt"))

    if min_sent_count <= 0:
        min_sent_count = 64

    logger.info(f"{data_files=}")

    max_sent_len = 0
    cnt = 0
    with open(long_sentence_file, "wt") as fout:
        for data_file in data_files:
            logger.info(f"--- Processing {data_file=}")
            with open(data_file, "rt") as fin:
                for sent in tqdm(fin):
                    if cnt < min_sent_count or len(sent) >= max_sent_len:
                        if len(sent) >= max_sent_len:
                            max_sent_len = len(sent)
                        fout.write(sent)
                        cnt += 1

    logger.info(
        f"Extracted {cnt} long sentences mixed with some other sentences to ensure {min_sent_count=} with {max_sent_len=}"
    )


if __name__ == "__main__":
    typer.run(main)

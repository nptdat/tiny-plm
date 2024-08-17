"""
Build sentence corpus from jawiki.
- INPUT:
    - jawiki-YYYYMMDD-cirrussearch-content.json.gz
- OUTPUT:
    - jawiki_corpus_{:02d}.txt (multiple files)

Usage:
$ python build_corpus_wiki.py ../data/jawiki-20230220-cirrussearch-content.json.gz ../data/jawiki_corpus --n-workers=16 --max-document-num 1000000
"""

import gzip
import json
import re
import sys
import unicodedata
from logging import basicConfig, getLogger
from multiprocessing import Process, Queue
from pathlib import Path
from time import sleep, time
from typing import Optional

import typer
from tqdm import tqdm

from utils import LruDict
from utils.io import SimpleS3
from utils.text import SentenceSplitter

logger = getLogger(__name__)
basicConfig(
    level="INFO", format="%(asctime)s : %(levelname)s : %(name)s : %(message)s"
)

MAX_QUEUE_SIZE = 10000
FILE_TEMPLATE = "jawiki_{:02d}.txt"


# Copied from: https://github.com/cl-tohoku/bert-japanese/blob/main/make_corpus_wiki.py#L54
def preprocess_text(text: str, title: Optional[str] = None) -> str:
    text = unicodedata.normalize("NFKC", text)

    # remove invisible characters
    text = "".join(c for c in text if c.isprintable())

    # remove templates
    text = re.sub(r"\[\d+?\]", "", text)
    text = re.sub(r"\[要.+?\]", "", text)
    text = re.sub(r"\{\{+[^{}]+?\}\}+", "", text)

    # remove navigation
    if title is not None:
        text = re.sub(r"^.+? \> " + re.escape(title), "", text)

    # remove footnotes
    text = re.sub(r" \^ .+", "", text)
    # remove annotations
    text = re.sub(r"\[(要出典|リンク切れ|.+?\?)\]", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def job_producer(
    input_file: str,
    job_queue: Queue,
    n_workers: int,
    max_document_num: int = 0,
) -> None:
    with gzip.open(input_file) as fin:
        for idx, line in tqdm(enumerate(fin)):
            job_queue.put(line)
            if max_document_num > 0 and idx >= max_document_num:
                break
            if job_queue.qsize() >= MAX_QUEUE_SIZE:
                sleep(1)
    # Put N * None to finish the N consumer processes
    for i in range(n_workers):
        job_queue.put(None)


def job_consumer(
    worker_id: int,
    output_file: str,
    job_queue: Queue,
    min_sentence_length: int = 10,
    max_sentence_length: int = 1000,
) -> None:
    logger.info(f"STARTED the {worker_id=}")

    splitter = SentenceSplitter()
    cache = LruDict(cache_len=10000)
    with open(output_file, "a") as fout:
        while True:
            if job_queue.qsize() > 0:
                data = job_queue.get()
                if data is None:  # consume only 1 None data
                    break

                dict_data = json.loads(data)
                text = dict_data.get("text")
                if not text:
                    continue
                title = dict_data.get("title")
                text = preprocess_text(text, title=title)

                for sent in splitter(text):
                    if min_sentence_length <= len(sent) <= max_sentence_length:
                        if sent in cache:
                            cache[sent] = 1
                            continue
                        fout.write(sent + "\n")
                        cache[sent] = 1

    logger.info(f"FINISHED the {worker_id=}")


def main(
    input_file: str,
    output_dir: str,
    s3_bucket: str = "",
    s3_raw_file_path: str = "",
    s3_output_path: str = "",
    min_sentence_length: int = 10,
    max_sentence_length: int = 1000,
    max_document_num: int = 0,
    n_workers: int = 4,
) -> None:
    start_time = time()

    logger.info(f"{input_file=}")
    logger.info(f"{output_dir=}")
    logger.info(f"{s3_bucket=}")
    logger.info(f"{s3_raw_file_path=}")
    logger.info(f"{s3_output_path=}")

    logger.info(
        f"{min_sentence_length=}, {max_sentence_length=}, {max_document_num=}, {n_workers=}"
    )

    if not Path(input_file).exists():
        if s3_bucket and s3_raw_file_path:
            simple_s3 = SimpleS3(s3_bucket)
            simple_s3.download(s3_raw_file_path, input_file)
        else:
            raise FileNotFoundError(
                f"{input_file} not found and no S3 bucket is set!"
            )

    # create the output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_dir = Path(output_dir) / "train"
    valid_dir = Path(output_dir) / "valid"
    train_dir.mkdir(exist_ok=True)
    valid_dir.mkdir(exist_ok=True)

    output_file_tpl = str(Path(output_dir) / FILE_TEMPLATE)

    processes = []
    job_queue: Queue = Queue()

    processes.append(
        Process(
            target=job_producer,
            args=(input_file, job_queue, n_workers, max_document_num),
        )
    )

    for worker_id in range(n_workers):
        processes.append(
            Process(
                target=job_consumer,
                args=(
                    worker_id,
                    output_file_tpl.format(worker_id),
                    job_queue,
                    min_sentence_length,
                    max_sentence_length,
                ),
            )
        )

    try:
        for p in processes:
            p.start()

        for p in processes:
            p.join()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)

    # Split train/valid
    # Move N-1 files to train folder
    num_files = n_workers
    for i in range(0, num_files - 1):
        file_path = Path(output_file_tpl.format(i))
        target = str(train_dir / FILE_TEMPLATE).format(i)
        file_path.rename(target)
    logger.info(f"{str(train_dir)} contains {num_files-1} files")

    # Move the Nth file to valid folder
    last_file_path = Path(output_file_tpl.format(num_files - 1))
    target = str(valid_dir / FILE_TEMPLATE).format(num_files - 1)
    last_file_path.rename(target)
    logger.info(f"{str(valid_dir)} contains 1 files")

    if s3_bucket and s3_output_path:
        simple_s3 = SimpleS3(s3_bucket)
        simple_s3.upload(output_dir, s3_output_path)

    logger.info(f"Finished building wiki corpus in {time() - start_time}")


if __name__ == "__main__":
    typer.run(main)

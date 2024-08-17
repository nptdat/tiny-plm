"""
Build sentence corpus from jawiki.
- INPUT:
    - ja.txt.xz
- OUTPUT:
    - cc100_ja_corpus_{:02d}.txt (multiple files)

Usage:
$ python build_corpus_cc100_ja.py ../data/ja.txt.xz ../data/cc100_ja_corpus --num-files 32 --max-sentence-num 1000000
"""

import gzip
import lzma
import random
import unicodedata
from logging import basicConfig, getLogger
from pathlib import Path
from time import time
from typing import List, TextIO, Union

import typer
from tqdm import tqdm

from utils.io import SimpleS3

logger = getLogger(__name__)
basicConfig(
    level="INFO", format="%(asctime)s : %(levelname)s : %(name)s : %(message)s"
)


FILE_TEMPLATE = "cc100_ja_{:02d}.txt"

random.seed(42)


def open_file_for_read(filepath: str) -> TextIO:
    if filepath.endswith(".xz"):
        return lzma.open(filepath, "rt")
    elif filepath.endswith(".gz"):
        return gzip.open(filepath, "rt")
    else:
        return open(filepath, "rt")


def open_files_for_write(output_file_tpl: str, num_files: int) -> List[TextIO]:
    fout_handles: List[TextIO] = []
    logger.info(f"{output_file_tpl=}")
    for i in range(num_files):
        file_name = output_file_tpl.format(i)
        fout = open(file_name, "wt")
        fout_handles.append(fout)
    return fout_handles


def close_files(file_handles: Union[TextIO, List[TextIO]]) -> None:
    if isinstance(file_handles, list):
        for file_handle in file_handles:
            file_handle.close()
    else:
        file_handles.close()


def get_input_files(input_path: str) -> List[str]:
    _input_path = Path(input_path)
    if _input_path.is_file():
        return [input_path]
    else:
        input_files = [str(path) for path in _input_path.glob("*.*")]
        return input_files


def normalize_text(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    return text


def limit_lines_on_file(
    file_path: str, max_lines: int, is_random: bool = True
) -> None:
    with open(file_path, "rt") as f:
        lines = f.read().split("\n")

    if len(lines) > max_lines:
        if is_random:
            indices = random.choices(range(len(lines)), k=max_lines)
        else:
            indices = list(range(max_lines))
        with open(file_path, "wt") as f:
            for index in indices:
                f.write(lines[index] + "\n")


def main(
    input_path: str,
    output_dir: str,
    s3_bucket: str = "",
    s3_raw_file_path: str = "",
    s3_output_path: str = "",
    num_files: int = 32,
    max_sentence_num: int = 0,
    max_validation_sentence_num: int = 0,
) -> None:
    logger.info(f"{input_path=}")
    logger.info(f"{output_dir=}")
    logger.info(f"{s3_bucket=}")
    logger.info(f"{s3_raw_file_path=}")
    logger.info(f"{s3_output_path=}")

    logger.info(
        f"{num_files=}, {max_sentence_num=}, {max_validation_sentence_num=}"
    )

    start_time = time()

    if not Path(input_path).exists():
        if s3_bucket and s3_raw_file_path:
            simple_s3 = SimpleS3(s3_bucket)
            simple_s3.download(s3_raw_file_path, input_path)
        else:
            raise FileNotFoundError(
                f"{input_path} not found and no S3 bucket is set!"
            )

    input_files = get_input_files(input_path)

    # create the output dir and open files for writing
    logger.info(f"{output_dir=}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_dir = Path(output_dir) / "train"
    valid_dir = Path(output_dir) / "valid"
    train_dir.mkdir(exist_ok=True)
    valid_dir.mkdir(exist_ok=True)

    output_file_tpl = str(Path(output_dir) / FILE_TEMPLATE)

    fouts = open_files_for_write(output_file_tpl, num_files)
    file_index = random.randint(0, num_files - 1)
    count_sents = 0
    for input_file in input_files:
        with open_file_for_read(input_file) as fin:
            for line in tqdm(fin):
                line = normalize_text(line)
                if line:
                    fouts[file_index].write(line + "\n")
                    count_sents += 1
                    if max_sentence_num > 0 and count_sents >= max_sentence_num:
                        break
                if line == "":
                    file_index = random.randint(0, num_files - 1)

            if max_sentence_num > 0 and count_sents >= max_sentence_num:
                logger.info(
                    f"Extracted {count_sents} sentences, hit the {max_sentence_num=} -> Early stopped!"
                )
                break

    close_files(fouts)

    # Split train/valid
    # Move N-1 files to train folder
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

    # Limit num of sentences for validation set
    if max_validation_sentence_num > 0:
        logger.info(
            f"Limiting the validation file {file_path} to {max_validation_sentence_num} sents..."
        )
        limit_lines_on_file(target, max_validation_sentence_num, is_random=True)

    if s3_bucket and s3_output_path:
        simple_s3 = SimpleS3(s3_bucket)
        simple_s3.upload(output_dir, s3_output_path)

    logger.info(f"Finished building cc100-ja corpus in {time() - start_time}")


if __name__ == "__main__":
    typer.run(main)

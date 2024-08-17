"""
Train a Tokenizer from scratch given a corpus.
"""

# Some code is copied from https://github.com/cl-tohoku/bert-japanese/blob/main/train_tokenizer.py

from logging import basicConfig, getLogger
from pathlib import Path
from time import time
from typing import Union

import typer

from japanese_tokenizers.implementations import JapaneseWordPieceTokenizer
from utils.io import SimpleS3

logger = getLogger(__name__)
basicConfig(
    level="INFO", format="%(asctime)s : %(levelname)s : %(name)s : %(message)s"
)


def main(
    input_dir: Path,
    output_dir: Path,
    s3_bucket: str = "",
    s3_output_path: str = "",
    mecab_dic_type: str = "unidic_lite",
    vocab_size: int = 32768,
    limit_alphabet: int = 6129,
    num_unused_tokens: int = 10,
) -> None:
    start_time = time()

    output_dir.mkdir(exist_ok=True)

    input_files: Union[str, list]
    if input_dir.is_file():
        input_files = str(input_dir)
    else:
        input_files = list(map(str, Path(input_dir).glob("*")))

    tokenizer = JapaneseWordPieceTokenizer(
        num_unused_tokens=num_unused_tokens,
        mecab_dic_type=mecab_dic_type,
    )

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    special_tokens += ["<unused{}>".format(i) for i in range(num_unused_tokens)]

    logger.info("Training the tokenizer")
    tokenizer.train(
        input_files,
        vocab_size=vocab_size,
        limit_alphabet=limit_alphabet,
        special_tokens=special_tokens,
    )

    logger.info("Saving the tokenizer to files")
    tokenizer.save_model(str(output_dir))

    if s3_bucket and s3_output_path:
        simple_s3 = SimpleS3(s3_bucket)
        simple_s3.upload(str(output_dir), s3_output_path)

    logger.info(f"Finished training the tokenizer in {time() - start_time}")


if __name__ == "__main__":
    typer.run(main)

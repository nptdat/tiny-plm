# Overview
- This repository provides an end-to-end pipeline to pre-train BERT models for Japanese from scratch.
- This code is heavily based on [HF Transformers' run_mlm](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py) (pytorch version) and [bert-japanese](https://github.com/cl-tohoku/bert-japanese)

# Requirements
- Hardware: GPU (consumer GPU is enough)
- Software:
    - Python 3.8 or above
    - nvidia-docker

# Data
- Current setting uses the following data to train a tiny bert model
    - [cc100-ja](https://data.statmt.org/cc-100/ja.txt.xz)
        - Use the first 128M sentences
        - Randomly choose 124M sentences for training, 1M sentences for validation
    - JaWiki ([2024/7/1 dump](https://dumps.wikimedia.org/other/cirrussearch/20240701/jawiki-20240701-cirrussearch-content.json.gz))
        - Use all data from the dump
        - Randomize 38.4M sentences for training, other 1.2M sentences for validation
    - From JaWiki training data, random 10M sentences to train the tokenizer
- To adjust the amount of data, please change environment variables in docker-compose.yml

# Model training
- Current setting trains a tiny BERT model with 12M parameters
- BERT model is pre-trained in 2-folds
    - (1) A model is pre-trained from scratch with the above cc100-ja data.
    - (2) The above model is then pre-trained with the above JaWiki data.
- To change model size
    - Identify the training config file in TRAIN_PARAM_JSONS env var, defined in docker-compose.yml
    - In the training config file, update the path fo model config in `model_config_file` key.
    - You can define the model structure in the model config file.

# Train BERT model with Docker
### Train the whole pipeline with cc100-ja & JaWiki
```shell
$ cd src
$ docker-compose build
$ docker-compose up all
```

### Train step-by-step
```shell
$ cd src
$ docker-compose build
$ docker-compose up download-cc100-ja
$ docker-compose up download-jawiki
$ docker-compose up build-dataset-cc100-ja
$ docker-compose up build-dataset-jawiki
$ docker-compose up train-tokenizer
$ docker-compose up train-model-cc100-ja
$ docker-compose up train-model-jawiki
```

- Replace `up` by `run` command to display progress bar.

# Overview
- This repository provides an end-to-end pipeline to pre-train BERT models for Japanese from scratch.
- This code is heavily based on [HF Transformers' run_mlm](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py) (PyTorch version) and [bert-japanese](https://github.com/cl-tohoku/bert-japanese)

# Requirements
- Hardware: GPU (a consumer GPU is sufficient)
- Software:
    - Python 3.9 or newer
    - nvidia-docker

# Setting information
tiny-plm supports three types of settings

## 1. Settings via environment variables
- The entry point `src/pretrain_pipeline.sh` accepts settings via environment variables.
- All available environment variables are defined in `src/config/docker_env_file.env`.
- Currently, keys in this .env file are exported to the environment automatically via docker-compose.
- NOTE: `docker_env_file.env` uses [Docker interpolation](https://docs.docker.com/compose/compose-file/12-interpolation/) to reduce boilerplate (e.g., `${DATA_PATH}`). If you load the .env file outside Docker, please remove the interpolation syntax.

## 2. Settings via Python script arguments
- The `pretrain_pipeline.sh` script receives settings and calls Python scripts, passing settings as arguments.

## 3. Settings from JSON files
- Some settings (for example, training parameters) are stored in `src/config/*.json` files.
- Python scripts can load those JSON setting files directly.

# Data
The current settings use the following data to train a tiny BERT model.
- [cc100-ja](https://data.statmt.org/cc-100/ja.txt.xz)
    - Use the first 128M sentences.
    - Randomly select 124M sentences for training and 1M sentences for validation.
    - To change the amount of data used for training, update `CC100_MAX_SENTENCE_NUM` and `CC100_MAX_VALIDATION_SENTENCE_NUM` in `src/config/docker_env_file.env`.
- JaWiki ([2024/7/1 dump](https://dumps.wikimedia.org/other/cirrussearch/20240701/jawiki-20240701-cirrussearch-content.json.gz))
    - Use all data from the dump.
    - Randomly select 38.4M sentences for training and 1.2M sentences for validation.
    - Note: the JaWiki dump may become unavailable. If so, do the following to update the dump:
      - Visit https://dumps.wikimedia.org/other/cirrussearch/ and find the latest available datetime.
      - Update the datetime in the JaWiki dump URL by changing the `JAWIKI_URL` variable in `docker_env_file.env`.
    - Change `JAWIKI_MAX_DOCUMENTS` in `docker_env_file.env` to adjust data size.
- From the JaWiki training data, randomly sample 10M sentences to train the tokenizer.

# Model training
- The current settings train a tiny BERT model with about 12M parameters.
- The BERT model is pre-trained in two stages:
    1. A model is trained from scratch on the cc100-ja data.
    2. That model is further pre-trained on the JaWiki data.
- To change the model size:
    - Specify the training config file in the `TRAIN_PARAM_JSONS` environment variable (defined in `docker-compose.yml`).
    - In that training config file, update the `model_config_file` path (default: `src/config/bert-tiny-custom.json`).
    - Define the model architecture in the referenced model config file.

# S3 storage
- You can use S3 to store intermediate data and final results throughout the pipeline.
- Enable S3 storage by setting the `*_S3_*` configuration keys.
- When S3 storage is enabled, modules behave as follows:
    - At the start, if an input file does not exist locally, it will be downloaded from S3.
    - At the end, outputs will be uploaded to S3.
- Authentication and permissions:
    - S3 usage assumes the pipeline runs on AWS instances (for example, SageMaker or EC2). Provide permissions for S3 access to those instances (for example, via an assumed role).
    - If you run S3 storage on a local machine or on-premise server, set AWS access keys in the environment (for example, using [aws configure](https://docs.aws.amazon.com/cli/v1/userguide/cli-authentication-user.html#cli-authentication-user-configure-wizard)).

# Train BERT model with Docker
- Update configuration in `src/config/docker_env_file.env`.

### Train the whole pipeline with cc100-ja & JaWiki
```shell
cd src
docker-compose build
docker-compose up all
```

### Train step-by-step
```shell
cd src
docker-compose build
docker-compose up download-cc100-ja
docker-compose up download-jawiki
docker-compose up build-dataset-cc100-ja
docker-compose up build-dataset-jawiki
docker-compose up train-tokenizer
docker-compose up train-model-cc100-ja
docker-compose up train-model-jawiki
```

- Replace `up` with `run` to show the progress bar.
```shell
cd src
docker-compose build
docker-compose run download-cc100-ja
```

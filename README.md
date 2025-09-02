# Overview
- This repository provides an end-to-end pipeline to pre-train BERT models for Japanese from scratch.
- This code is heavily based on [HF Transformers' run_mlm](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py) (pytorch version) and [bert-japanese](https://github.com/cl-tohoku/bert-japanese)

# Requirements
- Hardware: GPU (consumer GPU is enough)
- Software:
    - Python 3.9 or above
    - nvidia-docker

# Data
Current setting uses the following data to train a tiny bert model
- [cc100-ja](https://data.statmt.org/cc-100/ja.txt.xz)
    - Use the first 128M sentences
    - Randomly choose 124M sentences for training, 1M sentences for validation
    - To adjust the amount of data used for training, please change `CC100_MAX_SENTENCE_NUM` & `CC100_MAX_VALIDATION_SENTENCE_NUM` variables in `src/config/docker_env_file.env` file.
- JaWiki ([2024/7/1 dump](https://dumps.wikimedia.org/other/cirrussearch/20240701/jawiki-20240701-cirrussearch-content.json.gz))
    - Use all data from the dump
    - Randomize 38.4M sentences for training, other 1.2M sentences for validation
    - Note that the jawiki may become unavailable. If that is the case, do the followings to update the URL:
      - Access to https://dumps.wikimedia.org/other/cirrussearch/, look for the latest available datetime.
      - Then update the datetime to the URL
    - Change `JAWIKI_MAX_DOCUMENTS` variable in `docker_env_file.env` file to adjust data size.
- From JaWiki training data, random 10M sentences to train the tokenizer

# Model training
- Current setting trains a tiny BERT model with 12M parameters
- BERT model is pre-trained in 2-folds
    - (1) A model is pre-trained from scratch with the above cc100-ja data.
    - (2) The above model is then pre-trained with the above JaWiki data.
- To change model size
    - Identify the training config file in TRAIN_PARAM_JSONS env var, defined in docker-compose.yml
    - In the training config file, update the path fo model config in `model_config_file` key.
    - You can define the model structure in the model config file.

# Setting information
tiny-plm support 3 types of settings

## 1. Settings via environment variables
- The starting point `src/pretrain_pipeline.sh` receives settings via environment variables
- All of the available environment variables are defined in `src/config/docker_env_file.env`
- Currently, keys in this .env file are transferred to environment variables automatically via docker-compose.
- Please note that the `docker_env_file.env` is using [docker interpolation](https://docs.docker.com/compose/compose-file/12-interpolation/) to remove boilerplate (e.g., `${DATA_PATH}`). If you load the .env file from somewhere other than docker, please update the file to remove the interpolation.

## 2. Settings via python script arguments
- The pretrain_pipeline.sh script receives settings and call python scripts with settings are passed via arguments

## 3. Settings from json files
- A partial of settings (e.g., settings for training models) are stored in `src/config/*.json` setting files.
- Python scripts can load those json setting files directly.


# S3 storage
- You can use S3 to store intermediate data and final result through out the pipeline.
- S3 storage can be enabled be specifying `*_S3_*` setting keys
- If S3 storage is enabled, the modules will use S3 as the followings:
    - At the beginning, if the input does not exist in the local file system, the input will be downloaded from S3
    - At the end, upload the output to S3
- Authentication to S3 permissions:
    - S3 storage function is assumed to be used on AWS instances (e.g., SageMaker, EC2...), so please provide permissions for S3 access to the instances (e.g., via assume role...)
    - If you use S3 storage on local PCs or on-premise servers, please set AWS access keys to the environment (e.g, by using [aws configure](https://docs.aws.amazon.com/cli/v1/userguide/cli-authentication-user.html#cli-authentication-user-configure-wizard)...)

# Train BERT model with Docker
- Update configurations in `tiny-plm/src/config/docker_env_file.env`

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

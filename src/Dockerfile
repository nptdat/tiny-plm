FROM nvidia/cuda:12.5.0-devel-ubuntu22.04
# FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV PYTHONUNBUFFERED=1

# Install OS packages
RUN apt-get update \
    && apt-get install -y vim wget curl htop jq libopenmpi-dev python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /opt/ml/app

RUN mkdir data model
COPY . ./src

ENTRYPOINT src/pretrain_pipeline.sh

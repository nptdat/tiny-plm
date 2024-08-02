FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV PYTHONUNBUFFERED=1

# Install OS packages
RUN apt-get update && \
    apt-get -y upgrade
COPY install_packages.sh .
RUN sh install_packages.sh
RUN rm -rf /var/cache/yum

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python --version

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip uninstall -y apex

WORKDIR /opt/ml/app

RUN mkdir data model
COPY . ./src

ENTRYPOINT src/pretrain_pipeline.sh

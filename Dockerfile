# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster as base

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel as cuda

# Set bash as the default shell
ENV SHELL=/bin/bash

# Create a working directory
WORKDIR /home/jovyan

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    sudo \
    apt-utils \
    vim \
    git \
    wget
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN wget https://mirror.clarkson.edu/blender/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz \
	&& tar -xvf blender-3.2.2-linux-x64.tar.xz --strip-components=1 -C /bin \
	&& rm -rf blender-3.2.2-linux-x64.tar.xz \
	&& rm -rf blender-3.2.2-linux-x64

# RUN pip install --no-cache-dir --upgrade setuptools wheel pip
RUN pip install --no-cache pybind11 opencv-python matplotlib scikit-learn h5py
RUN pip install --no-cache jupyterlab jupyter -U

WORKDIR /home/jovyan

EXPOSE 8888

ENTRYPOINT jupyter lab \
    --notebook-dir=/home/jovyan \
    --ip=0.0.0.0 \
    --no-browser \
    --allow-root \
    --port=8888 \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --FileContentsManager.delete_to_trash=True


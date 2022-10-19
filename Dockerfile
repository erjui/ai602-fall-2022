FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

# To fix GPG error: https://sseongju1.tistory.com/61, https://linuxconfig.org/ubuntu-20-04-gpg-error-the-following-signatures-couldn-t-be-verified
RUN apt-key del 7fa2af80 && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Install apt dependencies as root
RUN apt update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    nano \
    sudo

# Install python dependencies
RUN pip install timm==0.3.2 tensorboard matplotlib pandas

# Add new user to avoid running as root
RUN useradd -ms /bin/bash docker
USER docker
WORKDIR /home/docker/ai602-fall-2022

ENV PATH="/home/docker/.local/bin:${PATH}"

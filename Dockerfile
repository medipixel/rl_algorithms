FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN apt-get install -y libsm6 libxext6 libxrender-dev libusb-1.0-0-dev && apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip
RUN apt-get update

# set workspace
RUN mkdir /workspace/
WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
COPY requirements-dev.txt /workspace/requirements-dev.txt

RUN pip install -U -r requirements.txt
RUN pip install -U -r requirements-dev.txt

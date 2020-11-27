FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN apt-get install -y libsm6 libxext6 libxrender-dev libusb-1.0-0-dev && apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip

# set workspace
RUN mkdir /workspace/
WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install -U Cython numpy
RUN pip install -U -r requirements.txt

# set cuda path
ENV CUDA_HOME /usr/local/cuda
ENV PATH "/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
ENV LIBRARY_PATH "$LIBRARY_PATH:/usr/local/cuda/lib64"

RUN apt-get update && apt-get install -y libgl1-mesa-glx
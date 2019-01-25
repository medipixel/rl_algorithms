FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

# copy requirements
RUN mkdir /data/
COPY requirements.txt /data/requirements.txt
COPY requirements-dev.txt /data/requirements-dev.txt

# install requirements
WORKDIR /data
RUN python3.6 -m pip install -r requirements.txt
RUN python3.6 -m pip install -r requirements-dev.txt

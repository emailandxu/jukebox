FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
COPY . /root/jukebox
WORKDIR /root/jukebox

ENV https_proxy=http://172.17.200.65:7890
ENV http_proxy=http://172.17.200.65:7890

RUN apt update
RUN apt install -y build-essential libopenmpi-dev
RUN pip install -r requirements.txt
RUN pip install -e .

RUN apt-get install -y libsndfile1-dev 
RUN apt-get install -y ssh
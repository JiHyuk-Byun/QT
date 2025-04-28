FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt install software-properties-common libopencv-dev git -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.9 python3-pip python3.9-dev -y
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.9 /usr/bin/python3
RUN apt install python3.9-distutils -y
RUN pip3 install setuptools==68.0.0
RUN pip3 install --upgrade pip
RUN pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

RUN mkdir /root/external
ADD requirements.txt /root/external/
RUN pip3 install -r /root/external/requirements.txt

WORKDIR /workspace
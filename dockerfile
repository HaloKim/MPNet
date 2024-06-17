FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace


RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install python3-pip -y
RUN apt-get install git vim -y
RUN pip install numpy==1.22.2 Cython datasets pytorch_transformers==1.0.0 transformers==4.28.0 scipy scikit-learn 

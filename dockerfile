FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace


RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install python3-pip -y
RUN apt-get install git

RUN pip install jupyterlab

RUN jupyter lab --generate-config

RUN echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py
RUN echo 'c.NotebookApp.terminado_settings = { "shell_command": ["/bin/bash"] }' \
          >> /root/.jupyter/jupyter_lab_config.py


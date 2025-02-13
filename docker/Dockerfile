FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime
WORKDIR /tmp/
COPY requirements.txt .
RUN apt-get update
RUN pip install -r requirements.txt
RUN apt-get install ffmpeg libsm6 libxext6 -y git
RUN pip install  ftfy regex tqdm

FROM python:3.11-buster

WORKDIR /app

RUN apt-get update
RUN pip install --upgrade pip setuptools

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ARG TEXT_MODEL_NAME
ARG CLIP_MODEL_NAME
ARG OPEN_CLIP_MODEL_NAME
ARG OPEN_CLIP_PRETRAINED


COPY download.py .
RUN ./download.py

COPY . .

RUN apt-get install git -y
RUN apt-get install pkg-config -y
RUN apt-get install libhdf5-serial-dev -y

RUN pip3 install -e CTCLIP/CT_CLIP
RUN pip3 install -e CTCLIP/transformer_maskgit

RUN apt-get install libgl1 -y

#RUN python3 test.py

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port 8080"]

# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get -y install gcc

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY how-to-guides/neptune-docker/scripts/training.py .
CMD [ "python3", "-W ignore" ,"training.py" ]
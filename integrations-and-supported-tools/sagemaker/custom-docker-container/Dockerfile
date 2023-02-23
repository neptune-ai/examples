FROM python:3-slim-buster

RUN pip --quiet --no-cache-dir install \
    numpy scipy scikit-learn pandas \
    neptune-sklearn

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY train /opt/program/train
WORKDIR /opt/program

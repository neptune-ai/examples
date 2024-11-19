FROM ubuntu:22.10

ARG PATH="/root/miniconda3/bin:${PATH}"
ENV PATH="/root/miniconda3/bin:${PATH}"

ARG BINANCE_TESTNET_API
ARG BINANCE_TESTNET_SECRET
ARG NEPTUNE_API_TOKEN
ARG NEPTUNE_PROJECT
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV BINANCE_TESTNET_API=${BINANCE_TESTNET_API}
ENV BINANCE_TESTNET_SECRET=${BINANCE_TESTNET_SECRET}
ENV NEPTUNE_API_TOKEN=${NEPTUNE_API_TOKEN}
ENV NEPTUNE_PROJECT=${NEPTUNE_PROJECT}
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

ENV VAR_NAME=$VAR_NAME


RUN apt update \
    && apt install -y python3-dev wget cron gcc vim build-essential

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-py37_4.9.2-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.9.2-Linux-x86_64.sh

RUN conda create -y -n env python=3.7

COPY . binance_trading/

RUN /bin/bash -c " source activate env \ 
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && rm ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd ~ \
    && rm -rf ta-lib/ \
    && pip install ta-lib"

RUN /bin/bash -c "source activate env \
    && pip install --upgrade pip \
    && pip install -r binance_trading/requirements.txt"


COPY cron-job /etc/cron.d/cron-job

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/cron-job

# Apply cron job
RUN /usr/bin/crontab /etc/cron.d/cron-job

# Run the command on container startup
CMD printenv > /etc/environment && cron -f
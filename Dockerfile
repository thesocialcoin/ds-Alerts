FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common=* \
    build-essential=* \
    gcc=* \
    curl=* \
    git=* \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends python3.10=* python3.10-dev=* python3-distutils=* \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir pip==22.2.2 \
    && pip install --no-cache-dir poetry==1.3.2

RUN useradd -U -m ds && mkdir /app 

WORKDIR /app

COPY  . .

RUN chown -R ds:ds /app

FROM builder as prod

RUN pip install .

EXPOSE 8888

USER ds

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root" ]


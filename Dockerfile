FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Systemowe zależności
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    unzip \
    && apt-get clean

# Symlink python -> python3 (jeśli nie istnieje)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Aktualizacja pip + instalacja bibliotek
RUN pip install --upgrade pip && \
    pip install \
        tensorflow==2.10.1 \
        torch==2.4.0 \
        transformers==4.37.2 \
        pandas==1.5.3 \
        # pandas-ta==0.3.14b0 \
        scikit-learn \
        matplotlib \
        numpy==1.23.5 \
        yfinance \
        datasets \
        tqdm \
        scipy \
        pymongo

WORKDIR /workspace

CMD ["/bin/bash"]

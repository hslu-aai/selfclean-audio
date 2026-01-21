ARG PYTHON_VERSION=3.11
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    build-essential \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    libsox-dev \
    libpq-dev \
    portaudio19-dev \
    libasound2-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to match volume mount
WORKDIR /workspace

RUN pip install --upgrade pip build

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Install pyright globally for pre-commit hooks
RUN pip install pyright

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME /data

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

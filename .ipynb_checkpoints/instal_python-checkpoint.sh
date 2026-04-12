#!/bin/bash

set -e  # para se der erro

PYTHON_VERSION=3.8.19
ENV_NAME=carla38

echo "=== Instalando dependências ==="
sudo apt update
sudo apt install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    wget \
    libbz2-dev

echo "=== Baixando Python $PYTHON_VERSION ==="
cd /tmp
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz

echo "=== Compilando Python (isso pode demorar) ==="
tar -xf Python-$PYTHON_VERSION.tgz
cd Python-$PYTHON_VERSION
./configure --enable-optimizations
make -j$(nproc)

echo "=== Instalando Python ==="
sudo make altinstall  # não sobrescreve python do sistema

echo "=== Criando ambiente virtual ==="
python3.8 -m venv ~/$ENV_NAME

echo "=== Ativando ambiente ==="
source ~/$ENV_NAME/bin/activate

echo "=== Atualizando pip ==="
pip install --upgrade pip

echo "======================================"
echo "✅ Python instalado e ambiente ativado!"
echo "👉 Para usar depois:"
echo "source ~/$ENV_NAME/bin/activate"
echo "======================================"

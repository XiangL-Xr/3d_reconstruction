# !/usr/bin/bash

### 步骤1: 安装依赖包
apt-get update
apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libatlas-base-dev \
    libsuitesparse-dev

### 步骤2：ceres-solver 运行环境配置
cd src/submodule/ceres-solver
sh build.sh
cd ../../..

### 步骤3：运行环境配置
cd src/submodule/colmap
sh build.sh
cd ../../..

### 步骤4: 3dreconstruction 运行环境配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --upgrade pip
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast/

imageio_download_bin freeimage
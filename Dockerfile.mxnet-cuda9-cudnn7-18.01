# Copyright 2017 Chi-Hung Weng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# This Dockerfile builds a Deep Learning & Python3 environment including:
# Keras (MXNET backend, beta version), MXNET and OpenCV.

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER Chi-Hung Weng <wengchihung@gmail.com>

ARG NUM_CPUS_FOR_BUILD=16
ARG MXNET_VER=1.0.0
ARG OPENCV_VER=3.4.0
# Remark: 786e376 is roughly a month and ten days after the release of MXNet 1.0.0.

RUN apt update && apt install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3-dev \
        python3-setuptools \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        qt4-default \
        apt-utils \
        cmake \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libdc1394-22-dev \
        libopenblas-dev \
        liblapack-dev \
        && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Get pip for Python3.
RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Install some useful and machine/deep-learning-related packages for Python3.
RUN pip3 --no-cache-dir install \
         h5py==2.7.0 \
         jupyter \
         matplotlib \
         seaborn \
         bokeh \
         numpy==1.13.3 \
         scipy \
         pandas \
         sklearn \
         scikit-image \
         autograd \
         mlxtend \
         graphviz

# Set up our notebook config.
RUN mkdir /root/.jupyter && \
    cd /root/.jupyter && \
    wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/docker/jupyter_notebook_config.py

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# It was resolved via adding a little wrapper script.
RUN cd / && \
    wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/docker/run_jupyter.sh && \
    chmod +x run_jupyter.sh

# Install OpenCV 3.
RUN git clone https://github.com/opencv/opencv.git /root/opencv && \
    cd /root/opencv && \
    git checkout ${OPENCV_VER} && \
    mkdir build && \
    cd build && \
    cmake -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF -D WITH_QT=ON -D WITH_OPENGL=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 .. && \
    make -j${NUM_CPUS_FOR_BUILD}  && \
    make install && \
    ldconfig && \
    rm -rf /root/opencv
# The source folder of OpenCV 3 is too big. We remove it after the installation.

# Add jemalloc to increase performance 
RUN apt update && apt install -y --no-install-recommends \
        libjemalloc-dev \
        && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
# Get MXNET from Github
ARG MXNET_VER=f30bb4a
RUN git clone --recursive https://github.com/dmlc/mxnet /opt/mxnet && \
    cd /opt/mxnet && \
    git checkout ${MXNET_VER}
# RUN git clone https://github.com/dmlc/mxnet.git /opt/mxnet --recursive --branch ${MXNET_VER} --depth 1
WORKDIR /opt/mxnet

# Build and Install MXNET.
RUN make -j${NUM_CPUS_FOR_BUILD} USE_OPENCV=1 \
                                 USE_BLAS=openblas \
                                 USE_CUDA=1 \
                                 USE_CUDA_PATH=/usr/local/cuda \
                                 USE_CUDNN=1
RUN cd python && python3 setup.py install

# # Install Keras for MXNET.
# RUN pip3 --no-cache-dir install keras-mxnet
RUN git clone https://github.com/dmlc/keras.git /opt/keras && \
    cd /opt/keras && \
    python3 setup.py install

RUN mkdir /notebooks && \
    wget -O /notebooks/KerasMNISTDemoMXNET.ipynb https://raw.githubusercontent.com/chi-hung/PythonTutorial/master/code_examples/KerasMNISTDemoMXNET.ipynb
WORKDIR /notebooks

# IPython
EXPOSE 8888

CMD ["/run_jupyter.sh", "--allow-root"]
#RUN ["/bin/bash"]

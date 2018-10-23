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
# Keras, Tensorflow and OpenCV.
#
# This file modifies a Dockerfile that is originally 
# maintained by the TensorFlow Authors (see the link below).
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.devel-gpu

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER Chi-Hung Weng <wengchihung@gmail.com>

# Specify number of CPUs can be used while building Tensorflow and OpenCV.
ARG NUM_CPUS_FOR_BUILD=20

# Specify the version of Bazel.
ARG BAZEL_VER=0.11.0
# Specify the version of Tensorflow.
ARG TF_VER=v1.6.0
# Specify the version of OpenCV.
ARG OPENCV_VER=3.4.1

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
        python \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
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
        graphviz \
        vim \
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
         pydot-ng \
         imgaug

# Set up our notebook config.
RUN mkdir /root/.jupyter && \
    cd /root/.jupyter && \
    wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/docker/jupyter_notebook_config.py

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
RUN cd / && \
    wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/docker/run_jupyter.sh && \
    chmod +x run_jupyter.sh

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install Bazel.
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VER}/bazel-${BAZEL_VER}-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-${BAZEL_VER}-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-${BAZEL_VER}-installer-linux-x86_64.sh

# Download the TensorFlow source folder from Github.
RUN cd /opt && git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout ${TF_VER}
WORKDIR /opt/tensorflow

# Configure the build (CUDA9, cuDNN7, Python3, etc).
ENV CI_BUILD_PYTHON=python3 \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    TF_NEED_CUDA=1 \
    TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1,7.0 \
    TF_CUDA_VERSION=9.0 \
    TF_CUDNN_VERSION=7 \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu \
    PYTHON_BIN_PATH=/usr/bin/python3 \
    PYTHON_LIB_PATH=/usr/local/lib/python3.5/dist-packages

# Build and install TensorFlow.
# Also, Keras will be installed when the installation of TensorFlow is complete.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt \
                --copt=-msse4.1 \
                --copt=-msse4.2 \
                --copt=-mavx \
                --copt=-mavx2 \
                --copt=-mfma \
                --copt=-O3 \
                --config=cuda \
                --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
                --jobs=${NUM_CPUS_FOR_BUILD} \
                tensorflow/tools/pip_package:build_pip_package && \
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip3 --no-cache-dir install --upgrade --upgrade-strategy only-if-needed /tmp/pip/tensorflow-*.whl && \
    pip3 --no-cache-dir install keras && \
    rm -rf /tmp/pip && \
    rm -rf /root/.cache
# Clean up pip wheel and Bazel cache when done.

# Install OpenCV
RUN git clone https://github.com/opencv/opencv.git /root/opencv && \
    cd /root/opencv && \
    git checkout ${OPENCV_VER} && \
    mkdir build && \
    cd build && \
    cmake -D WITH_TBB=ON \
          -D BUILD_NEW_PYTHON_SUPPORT=ON \
          -D WITH_V4L=ON \
          -D INSTALL_C_EXAMPLES=OFF \
          -D INSTALL_PYTHON_EXAMPLES=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D WITH_QT=ON \
          -D WITH_OPENGL=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=1 .. && \
    make -j${NUM_CPUS_FOR_BUILD}  && \
    make install && \
    ldconfig && \
    rm -rf /root/opencv
# Remark: the source folder of OpenCV is too big. We remove it after the installation.

RUN mkdir /notebooks && \
    wget -O /notebooks/MNISTDemoKeras.ipynb https://raw.githubusercontent.com/chi-hung/PythonTutorial/master/code_examples/KerasMNISTDemo.ipynb
WORKDIR /notebooks

# Add the "ipyrun" command. It runs the notebook & stores the obtained results into a HTML file.
RUN printf '#!/bin/bash\njupyter nbconvert --ExecutePreprocessor.timeout=None \
                                           --allow-errors \
                                           --to html \
                                           --execute $1' > /sbin/ipyrun && \
    chmod +x /sbin/ipyrun

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

CMD ["/run_jupyter.sh", "--allow-root"]

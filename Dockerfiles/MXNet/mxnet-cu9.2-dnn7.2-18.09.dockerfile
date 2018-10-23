#################################################################################
#
# Dockerfile for:
#   * MXNet + GluonCV (both are built nightly)
#   * keras-mxnet v2.2.2
#   * CUDA9.2 + cuDNN7.2 + NCCL2.2
#
# This image is based on "honghu/intelpython3:gpu-cu9.2-dnn7.2-18.09",
# where "Intel® Distribution for Python" is installed.
#
#################################################################################
#
# More Information
#   * GluonCV:
#       https://gluon-cv.mxnet.io
#   * Intel® Distribution for Python:
#       https://software.intel.com/en-us/distribution-for-python
#   * keras-mxnet: 
#       https://github.com/awslabs/keras-apache-mxnet/wiki
#
#################################################################################
#
# Software License Agreement
#   If you use the docker image built from this Dockerfile, it means 
#   you accept the following agreements:
#     * Intel® Distribution for Python:
#         https://software.intel.com/en-us/articles/end-user-license-agreement
#     * NVIDIA cuDNN:
#         https://docs.nvidia.com/deeplearning/sdk/cudnn-sla/index.html
#     * NVIDIA NCCL:
#         https://docs.nvidia.com/deeplearning/sdk/nccl-sla/index.html
#
#################################################################################
FROM honghu/intelpython3:gpu-cu9.2-dnn7.2-18.09
LABEL maintainer="Chi-Hung Weng <wengchihung@gmail.com>"

ARG NUM_CPUS_FOR_BUILD=16

# ARG MXNET_VER=1.2.1
# The MXNET_VER arg is disabled, as GluonCV requires the nightly build of MXNet.
# For the moment, we'll install the nightly build of MXNet and GluonCV.
ARG MXNET_COMMIT=445967e
# the version of commit "445967e" is in between v1.2.1 and the up-coming v1.3.0.
ARG GLUONCV_COMMIT=9669d72
# the version of commit "9669d72" is in between v0.2.0 and the up-coming v0.3.0.

ARG KERAS_VER=2.2.2

# Install dependent libs & compilers.
RUN apt install -y --no-install-recommends \
        libjemalloc-dev \
        libopenblas-dev \
        liblapack-dev \
        gcc-6 \
        g++-6 \
        && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Get MXNET from GitHub.
WORKDIR /opt/mxnet
RUN git clone --recursive https://github.com/dmlc/mxnet /opt/mxnet && \
    git checkout ${MXNET_COMMIT}
# RUN git clone --recursive https://github.com/dmlc/mxnet /opt/mxnet && \
#     git checkout ${MXNET_VER}

# Build MXNET.
RUN C_INCLUDE_PATH=/opt/intel/intelpython3/pkgs/opencv-3.1.0-np114py36_intel_8/include/ \
    PKG_CONFIG_PATH=/opt/intel/intelpython3/lib/pkgconfig \
    make -j${NUM_CPUS_FOR_BUILD} CC=gcc-6 \
                                 CXX=g++-6 \
                                 USE_OPENCV=1 \
                                 USE_BLAS=openblas \
                                 USE_LAPACK_PATH=/usr/lib/x86_64-linux-gnu/lapack \
                                 USE_CUDA=1 \
                                 USE_CUDA_PATH=/usr/local/cuda \
                                 USE_CUDNN=1 \
                                 USE_NCCL=1 \
                                 USE_NCCL_PATH=/usr/local/cuda/lib
# C_INCLUDE_PATH and PKG_CONFIG_PATH are set such that OpenCV can be found.
# Also, error may occur while building MXNet with gcc-7 & g++-7. We choose gcc-6 & g++-6 instead.
# More info: 
#   https://github.com/apache/incubator-mxnet/issues/9267

# Install MXNet & keras-mxnet.
WORKDIR /opt/mxnet/python
RUN python3 setup.py install && \
    pip install --no-cache-dir keras-mxnet==${KERAS_VER} && \
    rm -rf /tmp/pip* && \
    rm -rf /root/.cache

# Fetch and install GluonCV.
WORKDIR /opt/gluon-cv
RUN git clone https://github.com/dmlc/gluon-cv /opt/gluon-cv && \
    git checkout ${GLUONCV_COMMIT} && \
    python3 setup.py install

# Tell Keras to use MXNet as its backend.
WORKDIR /root/.keras
RUN wget -O /root/.keras/keras.json https://raw.githubusercontent.com/chi-hung/DockerbuildsKeras/master/keras-mxnet.json

# Add an example.
WORKDIR /workspace/
RUN wget https://raw.githubusercontent.com/chi-hung/PythonTutorial/master/code_examples/KerasMNISTDemoMXNET.ipynb

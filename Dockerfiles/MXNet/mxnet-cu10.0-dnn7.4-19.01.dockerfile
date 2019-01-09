########################################################################################
#
# Dockerfile for:
#   * MXNet v1.4.0rc0 + GluonCV v0.3.0
#   * keras-mxnet v2.2.4.1
#   * CUDA10.0 + cuDNN7.4
#
# This image is based on "honghu/intelpython3:gpu-cu10.0-dnn7.4-19.01",
# where "Intel® Distribution for Python" was pre-installed.
#
########################################################################################
#
# More Information
#   * Intel® Distribution for Python:
#       https://software.intel.com/en-us/distribution-for-python
#
########################################################################################
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
########################################################################################
FROM honghu/intelpython3:gpu-cu10.0-dnn7.4-19.01
LABEL maintainer="Chi-Hung Weng <wengchihung@gmail.com>"

ARG NUM_CPUS_FOR_BUILD=16

# MXNet v1.4.0.rc0
ARG MXNET_COMMIT=1.4.0.rc0
# GluonCV v.0.3.0
ARG GLUONCV_COMMIT=b55232d
# Keras-mxnet v2.2.4.1
ARG KERAS_VER=2.2.4.1

# Install dependent libs & compilers.
RUN apt update && apt install -y --allow-change-held-packages --no-install-recommends \
        libjemalloc-dev \
        libopenblas-dev \
        liblapack-dev \
        gcc-6 \
        g++-6 \
        libcudnn7 \
        && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Link NCCL2 libray and header where the build script expects them.
RUN mkdir /usr/local/cuda/lib && \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib/libnccl.so.2 && \
    ln -s /usr/include/nccl.h /usr/local/cuda/include/nccl.h

# Get MXNET from GitHub.
WORKDIR /opt/mxnet
RUN git clone --recursive https://github.com/dmlc/mxnet /opt/mxnet &&  \
git checkout ${MXNET_VER}

# Build MXNET.
RUN C_INCLUDE_PATH=/opt/opencv/include \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
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
# Compilation rror may occur while building MXNet with gcc-7 & g++-7. We choose gcc-6 & g++-6 instead.
# More info: 
#   https://github.com/apache/incubator-mxnet/issues/9267

ENV LD_LIBRARY_PATH=/opt/mxnet/lib/:${LD_LIBRARY_PATH}
# libmklml_intel.so is at /opt/mxnet/lib

# Remove OpenCV-dev folder as it is not necessary and occupies a large space.
RUN rm -rf /opt/opencv

# Install MXNet & keras-mxnet.
WORKDIR /opt/mxnet/python
RUN sed -i -e 's/numpy<=1.15.2/numpy<=1.15.4/g' /opt/mxnet/python/setup.py && \
    python3 setup.py install && \
    pip install --no-cache-dir keras-mxnet==${KERAS_VER} && \
    rm -rf /tmp/pip* && \
    rm -rf /root/.cache
# let MXNet not to downgrade/upgrade the intel-optimized NumPy

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

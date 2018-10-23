#################################################################################
#
# Dockerfile for:
#   * CNTK v2.5.1
#   * CUDA9.2 + cuDNN7.2 + NCCL2.2
#   * Keras v2.2.2
#
# This image is based on "honghu/intelpython3:gpu-cu9.2-dnn7.2-18.09",
# where "Intel® Distribution for Python" is installed.
#
#################################################################################
#
# More Information
#   * Intel® Distribution for Python:
#       https://software.intel.com/en-us/distribution-for-python
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

ARG CNTK_VER=2.5.1
ARG KERAS_VER=2.2.2
# Install "openmpi-bin", which is required by CNTK.
RUN apt update && apt install -y --no-install-recommends \
        openmpi-bin && \
        apt clean && \
        rm -rf /var/lib/apt/lists/*

# CNTK looks for "libmpi_cxx.so.1", not "libmpi_cxx.so.20".
RUN ln -s /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.20 /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.1

# Install CNTK & Keras.
RUN pip install https://cntk.ai/PythonWheel/GPU/cntk_gpu-${CNTK_VER}-cp36-cp36m-linux_x86_64.whl && \
    pip --no-cache-dir install keras==${KERAS_VER} && \
    rm -rf /tmp/pip* && \
    rm -rf /root/.cache

# Tell Keras to use CNTK as its backend.
WORKDIR /root/.keras
RUN wget -O /root/.keras/keras.json https://raw.githubusercontent.com/chi-hung/DockerKeras/master/keras-cntk.json

# Add a MNIST example.
WORKDIR /workspace
RUN wget -O /workspace/DemoKerasMNIST.ipynb https://raw.githubusercontent.com/chi-hung/PythonDataMining/master/code_examples/KerasMNISTDemo.ipynb

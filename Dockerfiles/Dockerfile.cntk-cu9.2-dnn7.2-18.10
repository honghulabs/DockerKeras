#################################################################################
#
# Dockerfile for:
#   * CNTK v2.6
#   * CUDA9.2 + cuDNN7.2 + NCCL2.2
#   * Keras v2.2.4
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

ARG CNTK_VER=2.6
ARG KERAS_VER=2.2.4
# Install "openmpi-bin", which is required by CNTK.
RUN apt update && apt install -y --no-install-recommends \
        openmpi-bin && \
        apt clean && \
        rm -rf /var/lib/apt/lists/*

# CNTK looks for "libmpi_cxx.so.1", not "libmpi_cxx.so.20".
RUN ln -s /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.20 /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.1

# Get keras_applications & keras_preprocessing
RUN pip --no-cache-dir install keras_applications keras_preprocessing --no-deps

# Install CNTK & Keras.
RUN pip install https://cntk.ai/PythonWheel/GPU/cntk_gpu-${CNTK_VER}-cp36-cp36m-linux_x86_64.whl && \
    pip --no-cache-dir install keras==${KERAS_VER} && \
    rm -rf /tmp/pip* && \
    rm -rf /root/.cache

# Fix a warning message: CNTK cannot find where MKL & OpenCV are.
RUN echo "/opt/intel/intelpython3/lib/" > /etc/ld.so.conf.d/intelpy3.conf && ldconfig

# Ubuntu 16.04 or higher should be compatible with CNTK. However, CNTK complains Ubuntu 18.04 is not okay.
# Let's fix it.
RUN sed -i "s/__my_distro__ != 'ubuntu' or __my_distro_ver__ != '16.04'/__my_distro__ != 'ubuntu' or (__my_distro_ver__ != '16.04' and  __my_distro_ver__ != '18.04')/g" /opt/intel/intelpython3/lib/python3.6/site-packages/cntk/cntk_py_init.py

# Tell Keras to use CNTK as its backend.
WORKDIR /root/.keras
RUN wget -O /root/.keras/keras.json https://raw.githubusercontent.com/chi-hung/DockerKeras/master/keras-cntk.json

# Add a MNIST example.
WORKDIR /workspace
RUN wget -O /workspace/DemoKerasMNIST.ipynb https://raw.githubusercontent.com/chi-hung/PythonDataMining/master/code_examples/KerasMNISTDemo.ipynb

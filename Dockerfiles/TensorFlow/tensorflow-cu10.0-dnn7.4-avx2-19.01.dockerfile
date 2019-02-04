#########################################################################################################
#
# Dockerfile for:
#   * TensorFlow v1.12
#   * CUDA10.0 + cuDNN7.4 + NCCL2
#   * Keras v2.2.4
#
# Several lines of this Dockerfile are from:
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.devel-gpu
#
# This image is based on "honghu/intelpython3:gpu-cu10.0-dnn7.4-19.01",
# where "Intel® Distribution for Python" was pre-installed.
#
#########################################################################################################
#
# More Information
#   * Intel® Distribution for Python:
#       https://software.intel.com/en-us/distribution-for-python
#
#########################################################################################################
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
#########################################################################################################
FROM honghu/intelpython3:gpu-cu10.0-dnn7.4-19.01
LABEL maintainer="Chi-Hung Weng <wengchihung@gmail.com>"

# Specify number of CPUs to be used while building TensorFlow.
ARG NUM_CPUS_FOR_BUILD=16

# Bazel version
ARG BAZEL_VER=0.15.0

# TensorFlow version
ARG TF_BR=r1.12
ARG TF_COMMIT=a6d8ffa
# The commit `a6d8ffa` refers to TensorFlow v1.12.0 (stable).

# Keras version
ARG KERAS_VER=2.2.4

# TensorRT version
ARG TENSORRT_VERSION=5.0.2

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
WORKDIR /bazel
RUN curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VER}/bazel-${BAZEL_VER}-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-${BAZEL_VER}-installer-linux-x86_64.sh && \
    rm -f /bazel/bazel-${BAZEL_VER}-installer-linux-x86_64.sh

# # Get & set up NCCL2
# WORKDIR /opt
# COPY ${NCCL2_FNAME}.* /opt/
# RUN tar xvf ${NCCL2_FNAME}.* && \
#     rm ${NCCL2_FNAME}.* && \
#     ln -s nccl* nccl2
# ENV LD_LIBRARY_PATH=/opt/nccl2/lib:${LD_LIBRARY_PATH}

# Link NCCL2 libray and header where the build script expects them.
RUN mkdir /usr/local/cuda/lib && \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib/libnccl.so.2 && \
    ln -s /usr/include/nccl.h /usr/local/cuda/include/nccl.h

# Get TensorRT runtime.
RUN apt update && \
    apt install -y nvinfer-runtime-trt-repo-ubuntu1804-$TENSORRT_VERSION-ga-cuda10.0
RUN apt update && apt install -y --no-install-recommends \
        libnvinfer5=$TENSORRT_VERSION-1+cuda10.0 \
        libnvinfer-dev=$TENSORRT_VERSION-1+cuda10.0 && \
        apt clean && \
        rm -rf /var/lib/apt/lists/*

# "keras_applications" & "keras_preprocessing" has to be pre-installed.
# issue: https://github.com/tensorflow/tensorflow/issues/21518
# Also, google-cloud has to be installed:
# issue: https://github.com/tensorflow/tensorflow/issues/6341
RUN pip --no-cache-dir install keras_applications keras_preprocessing --no-deps && \
    pip --no-cache-dir install protobuf

# Get TensorFlow
RUN git clone --branch=${TF_BR} --depth=1 https://github.com/tensorflow/tensorflow.git /opt/tensorflow && \
    cd /opt/tensorflow && \
    git branch ${TF_COMMIT} && \
    git checkout ${TF_COMMIT}

# Configs of TensorFlow
ENV CI_BUILD_PYTHON=python3 \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    TF_NEED_CUDA=1 \
    TF_NEED_TENSORRT=1 \
    TF_CUDA_COMPUTE_CAPABILITIES=5.2,6.0,6.1,7.0,7.5 \
    TF_CUDA_VERSION=10.0 \
    TF_CUDNN_VERSION=7 \
    TF_NCCL_VERSION=2
# ENV TF_NEED_MPI=1
# ENV NCCL_INSTALL_PATH=/opt/nccl2

# Build and install TensorFlow.
WORKDIR /opt/tensorflow
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
    pip --no-cache-dir install --upgrade --upgrade-strategy only-if-needed /tmp/pip/tensorflow-*.whl && \
    pip --no-cache-dir install --upgrade --upgrade-strategy only-if-needed keras==${KERAS_VER} && \
    rm -rf /tmp/pip* && \
    rm -rf /root/.cache
# Clean up pip wheel and Bazel cache when done.

# Add a MNIST example.
WORKDIR /workspace
RUN wget -O /workspace/DemoKerasMNIST.ipynb https://raw.githubusercontent.com/chi-hung/PythonDataMining/master/code_examples/KerasMNISTDemo.ipynb

# Expose the port for TensorBoard.
EXPOSE 6006

# Remove OpenCV-dev folder as it is not necessary and occupies a large space.
RUN rm -rf /opt/opencv

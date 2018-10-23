#########################################################################################################
#
# Dockerfile for:
#   * TensorFlow v1.10.1
#   * CUDA9.2 + cuDNN7.2 + NCCL2.2
#   * Keras v2.2.2
#
# Several lines of this Dockerfile are from:
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.devel-gpu
#
# This image is based on "honghu/intelpython3:gpu-cu9.2-dnn7.2-18.09",
# where "Intel® Distribution for Python" is installed.
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
FROM honghu/intelpython3:gpu-cu9.2-dnn7.2-18.09
LABEL maintainer="Chi-Hung Weng <wengchihung@gmail.com>"

# Specify number of CPUs to be used while building TensorFlow.
ARG NUM_CPUS_FOR_BUILD=16
# Specify the version of Bazel.
ARG BAZEL_VER=0.15.0
# Specify the branch of TensorFlow.
ARG TF_BR=r1.10
# Specify the version of Keras.
ARG KERAS_VER=2.2.2
# Specify the version of NCCL2.
#ARG NCCL2_FNAME=nccl_2.2.13-1+cuda9.2_x86_64

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

# Get TensorFlow.
RUN git clone --branch=${TF_BR} --depth=1 https://github.com/tensorflow/tensorflow.git /opt/tensorflow

# Settings of TensorFlow (CUDA9, cuDNN7, NCCL2, Python3, etc).
ENV CI_BUILD_PYTHON=python3 \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    TF_NEED_CUDA=1 \
    TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0 \
    TF_CUDA_VERSION=9.2 \
    TF_CUDNN_VERSION=7 \
    TF_NCCL_VERSION=2
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
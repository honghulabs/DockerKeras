#########################################################################################################
#
# Dockerfile for:
#   * TensorFlow v1.11 (built for CPU that has AVX512 instuctions)
#   * Keras v2.2.4
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
#
#########################################################################################################
FROM honghu/intelpython3:cpu-18.09
LABEL maintainer="Chi-Hung Weng <wengchihung@gmail.com>"

# Specify number of CPUs to be used while building TensorFlow.
ARG NUM_CPUS_FOR_BUILD=16
# Specify the version of Bazel.
ARG BAZEL_VER=0.15.0
# Specify the branch of TensorFlow.
ARG TF_BR=r1.11
# Specify the version of Keras.
ARG KERAS_VER=2.2.4

# Remove preinstalled TensorFlow, Keras
RUN pip --no-cache-dir uninstall -y tensorflow keras && \
    rm -rf /tmp/pip* && \
    rm -rf /root/.cache

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

# "keras_applications" & "keras_preprocessing" has to be pre-installed.
# Issue: https://github.com/tensorflow/tensorflow/issues/21518
RUN pip --no-cache-dir install keras_applications keras_preprocessing --no-deps && \
    rm -rf /tmp/pip* && \
    rm -rf /root/.cache

# Get TensorFlow.
RUN git clone --branch=${TF_BR} --depth=1 https://github.com/tensorflow/tensorflow.git /opt/tensorflow

# Settings of TensorFlow
ENV CI_BUILD_PYTHON=python3 \
    TF_NEED_CUDA=0

# Build and install TensorFlow for CPU.
WORKDIR /opt/tensorflow
RUN tensorflow/tools/ci_build/builds/configured CPU \
    bazel build -c opt \
                --copt=-msse4.1 \
                --copt=-msse4.2 \
                --copt=-mavx \
                --copt=-mavx2 \
                --copt=-mfma \
                --copt=-mavx512f \
                --copt=-mavx512pf \
                --copt=-mavx512cd \
                --copt=-mavx512er \
                --copt=-O3 \
                --config=mkl \
                --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
                --jobs=${NUM_CPUS_FOR_BUILD} \
                tensorflow/tools/pip_package:build_pip_package && \
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

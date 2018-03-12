FROM microsoft/cntk:2.4-gpu-python3.5-cuda9.0-cudnn7.0

MAINTAINER Chi-Hung Weng <wengchihung@gmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgtk2.0-0 \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV source_cntk "source /cntk/activate-cntk"

# Install OpenCV through Anaconda.
RUN /bin/bash -c "${source_cntk} && conda install -y -c menpo opencv3"
# Obtain/Upgrade some useful packages through Anaconda.
RUN /bin/bash -c "${source_cntk} && conda update -y scipy \
                                                    seaborn \
                                                    matplotlib \
                                                    pandas \
                                                    scikit-image \
                                                    scikit-learn \
                                                    jupyter \
                                                    numpy"
# Install some packages through pip.
RUN /bin/bash -c "${source_cntk} && pip --no-cache-dir install \
         autograd \
         mlxtend \
         pydot-ng \
         imgaug \
         bokeh"
# Install Keras.
RUN /bin/bash -c "${source_cntk} && pip --no-cache-dir install --upgrade --no-deps keras"
# Clean cache.
RUN rm -rf /tmp/pip && \
    rm -rf /root/.cache
# Tell Keras to use CNTK as its backend.
RUN mkdir /root/.keras && \
    wget -O /root/.keras/keras.json https://raw.githubusercontent.com/chi-hung/DockerbuildsKeras/master/keras-cntk.json

# Set up our notebook config.
RUN mkdir /root/.jupyter && \
    cd /root/.jupyter && \
    wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/docker/jupyter_notebook_config.py

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
RUN echo 'source /cntk/activate-cntk && jupyter notebook "$@"' > /root/run_jupyter.sh

RUN mkdir /notebooks && \
    wget -O /notebooks/MNISTDemoKeras.ipynb https://raw.githubusercontent.com/chi-hung/PythonTutorial/master/code_examples/KerasMNISTDemo.ipynb
WORKDIR /notebooks

# Add the "ipyrun" command. It runs the notebook & stores the obtained results into a HTML file.
RUN printf '#!/bin/bash\njupyter nbconvert --ExecutePreprocessor.timeout=None \
                                           --allow-errors \
                                           --to html \
                                           --execute $1' > /sbin/ipyrun && \
    chmod +x /sbin/ipyrun

# IPython
EXPOSE 8888

ENTRYPOINT ["/bin/bash"]
CMD ["/root/run_jupyter.sh", "--allow-root"]

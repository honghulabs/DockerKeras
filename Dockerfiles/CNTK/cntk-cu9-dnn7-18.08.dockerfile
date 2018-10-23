FROM microsoft/cntk:2.5.1-gpu-python3.5-cuda9.0-cudnn7.0
LABEL maintainer="Chi-Hung Weng <wengchihung@gmail.com>"
#
# This Dockerfile is based on the official CNTK docker image.
#
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
# Fix an issue that seaborn cannot imported successfully.
# More info:
#   https://github.com/ContinuumIO/docker-images/issues/49
RUN apt-get update && \
    apt-get install libgl1-mesa-glx -y
ENV QT_QPA_PLATFORM=offscreen

# Install some packages through pip.
RUN /bin/bash -c "${source_cntk} && pip --no-cache-dir install \
         autograd \
         mlxtend \
         pydot-ng \
         imgaug \
         bokeh"

# Install Keras.
RUN /bin/bash -c "${source_cntk} && pip --no-cache-dir install --upgrade --no-deps keras" && \
    /bin/bash -c "${source_cntk} && pip --no-cache-dir install keras-applications" && \
    rm -rf /tmp/pip && \
    rm -rf /root/.cache

# Tell Keras to use CNTK as its backend.
WORKDIR /root/.keras
RUN wget -O /root/.keras/keras.json https://raw.githubusercontent.com/chi-hung/DockerbuildsKeras/master/keras-cntk.json

# Set up our notebook config.
WORKDIR /root/.jupyter
RUN wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/docker/jupyter_notebook_config.py

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
RUN printf '#!/bin/bash\nsource /cntk/activate-cntk && jupyter notebook "$@"' > /root/run_jupyter.sh && \
    chmod +x /root/run_jupyter.sh

WORKDIR /workspace
RUN wget https://raw.githubusercontent.com/chi-hung/PythonTutorial/master/code_examples/KerasMNISTDemo.ipynb

# Add the "ipyrun" command. It executes the given notebook & transform it into a HTML file.
RUN printf '#!/bin/bash\njupyter nbconvert --ExecutePreprocessor.timeout=None \
                                           --allow-errors \
                                           --to html \
                                           --execute $1' > /sbin/ipyrun && \
    chmod +x /sbin/ipyrun

# Expose port 8888 for Jupyter.
EXPOSE 8888

# Shorten "nvidia-smi" as "smi" ; shorten "watch -n 1 nvidia-smi" as "wsmi".
RUN { printf 'alias smi="nvidia-smi"\nalias wsmi="watch -n 1 nvidia-smi"\n'; cat /etc/bash.bashrc; } >/etc/bash.bashrc.new && \
    mv /etc/bash.bashrc.new /etc/bash.bashrc

RUN printf '#!/bin/bash\nsource /cntk/activate-cntk && "$@"' > /root/activate_cntk_and_run.sh && \
    chmod +x /root/activate_cntk_and_run.sh

ENTRYPOINT ["/root/activate_cntk_and_run.sh"]
# If no specific command to execute, run jupyter notebook.
CMD ["/root/run_jupyter.sh", "--allow-root"]

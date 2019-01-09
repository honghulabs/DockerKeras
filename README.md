# DockerKeras
<img src="https://i.imgur.com/xsEfL7j.png" alt="drawing" width="25px"/> *Supported by HonghuTech, a Taiwanese Deep Learning Solutions Provider*
---
[![Docker Pulls](https://img.shields.io/docker/pulls/honghu/keras.svg)](https://hub.docker.com/r/honghu/keras/) [![GithubStars](https://img.shields.io/github/stars/chi-hung/DockerKeras.svg?style=social&label=Stars)](https://github.com/chi-hung/DockerKeras/)

Having trouble setting-up environments for **Deep Learning**? We do this for you! From now on, you shall say goodbye to annoying messages such as "**Build  failed...**" or "**An error occurred during installation...**".

Currently, we maintain the following docker images:

* *Keras using TensorFlow Backened* 
* *Keras using CNTK Backend*
* *Keras using MXNET Backend*
* *Keras using Theano Backend*

Apparently, all these environments support using **[Keras](https://keras.io)** as the frontend.

See below for more details about these environments.
## Table of Contents
* [Before Getting Started](#before-getting-started)
* [Summary of the Images](#summary-of-the-images)
* [Keras using TensorFlow Backend](#keras-using-tensorflow-backend)
* [Keras using MXNET Backend](#keras-using-mxnet-backend)
* [Keras using CNTK Backend](#keras-using-cntk-backend)
* [Keras using Theano Backend](#keras-using-theano-backend)
* [ndrun - Run a Docker Container for Your Deep-Learning Research](#ndrun---run-a-docker-container-for-your-deep-learning-research)
* [Getting Started with the Command Line](#getting-started-with-the-command-line)
    * [Example: Check a Framework’s Version](#example-check-a-frameworks-version)
    * [Example: Classify Handwritten-Digits With TensorFlow](#example-classify-handwritten-digits-with-tensorflow)
    * [Example: Train a Multi-GPU Model Using TensorFlow](#example-train-a-multi-gpu-model-using-tensorflow)
* [Advanced Usage of the Command Line](#advanced-usage-of-the-command-line)
    * [Avoid Giving Your Script Extra GPUs](#avoid-giving-your-script-extra-gpus)
    * [Set `NV_GPU` If You Know Which GPUs You’d Like to Use](#set-nv_gpu-if-you-know-which-gpus-youd-like-to-use)
    * [Run Commands on Behalf of Yourself](#run-commands-on-behalf-of-yourself)
* [Getting Started with Jupyter Notebook](#getting-started-with-jupyter-notebook)
    * [Learn Deep Learning with MXNET Gluon](#example-learn-deep-learning-with-mxnet-gluon)

---
## Before Getting Started
* *NVIDIA-Docker2* has to be installed. See [[here]](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)#prerequisites) for how to install and [[here]](https://devblogs.nvidia.com/gpu-containers-runtime/) for its introduction.
* Docker needs to be configured. For example, you may have to add your user to the ```docker``` group. see [[here]](https://docs.docker.com/install/linux/linux-postinstall/) for Docker setup.
* Beware: the recent images we've built contain CUDA```9.2```, which requires NVIDIA driver version ```>=396```. You can get the latest NVIDIA driver [[here]](http://www.nvidia.com/Download/index.aspx).

[[Back to Top]](#table-of-contents)
## Summary of the Images
The following tables list the docker images maintained by us. All these listed images are retrievable through [Docker Hub](https://hub.docker.com). 

* Images within the repository:  [**honghu/keras**](https://hub.docker.com/r/honghu/keras/)

    |Keras Backend |  Image's Tag  |  Description | Dockerfile |  Suggested NV Driver |
    |:---:|---|---|---|:---:|
    |TensorFlow| tf-cu10.0-dnn7.4-py3-avx2-19.01 /<Br> **tf-latest** | *TensorFlow* ```v1.12.0``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2019.0-047``` <br/> *Keras* ```v2.2.4```<br/> [*NCCL*](https://developer.nvidia.com/nccl) ```v2.3.7-1```| [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/TensorFlow/tensorflow-cu10.0-dnn7.4-avx2-19.01.dockerfile)| R410 |
    |TensorFlow| tf-cu9.2-dnn7.2-py3-avx2-18.10 | *TensorFlow* ```v1.11.0``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> *Keras* ```v2.2.4```<br/> [*NCCL*](https://developer.nvidia.com/nccl) ```v2.2.13```| [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/TensorFlow/tensorflow-cu9.2-dnn7.2-avx2-18.10.dockerfile)| R396 |
    |TensorFlow| tf-cu9.2-dnn7.2-py3-avx2-18.09 | *TensorFlow* ```v1.10.1``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> *Keras* ```v2.2.2```<br/> [*NCCL*](https://developer.nvidia.com/nccl) ```v2.2.13```| [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/TensorFlow/tensorflow-cu9.2-dnn7.2-avx2-18.09.dockerfile) | R396 |
    |TensorFlow| tf-cu9.2-dnn7.1-py3-avx2-18.08| *TensorFlow* ```v1.10.0``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> *Keras* ```v2.2.2```<br/> [*NCCL*](https://developer.nvidia.com/nccl) ```v2.2.13```| [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/TensorFlow/tensorflow-cu9.2-dnn7.1-avx2-18.08.dockerfile) | R396 |
    |TensorFlow| tf-cu9-dnn7-py3-avx2-18.03  | *TensorFlow* ```v1.6.0``` <br/> *Keras* ```v2.1.5``` | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/TensorFlow/tensorflow-cu9.0-dnn7.0-avx2-18.03.dockerfile)| R384 |
    |TensorFlow| tf-cu9-dnn7-py3-avx2-18.01  | *TensorFlow* ```v1.4.1``` <br/> *Keras* ```v2.1.2```   | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/TensorFlow/tensorflow-cu9.0-dnn7.0-avx2-18.01.dockerfile)| R384 |
     |MXNet| mx-cu10.0-dnn7.4-py3-19.01 / <br/> **mx-latest**  | *MXNet* ```v1.4.0.rc0``` <br/> [*GluonCV*](https://gluon-cv.mxnet.io) ```v0.3.0``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2019.0-047``` <br/> [*Keras-MXNet*](https://github.com/awslabs/keras-apache-mxnet/wiki) ```v2.2.4.1``` <br/> [*NCCL*](https://developer.nvidia.com/nccl) ```v2.3.7-1``` | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/MXNet/mxnet-cu10.0-dnn7.4-19.01.dockerfile) | R410 |
     |MXNet| mx-cu9.2-dnn7.2-py3-18.10 | *MXNet* ```v1.3.0-dev``` <br/> [*GluonCV*](https://gluon-cv.mxnet.io) ```v0.3.0-dev``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> [*Keras-MXNet*](https://github.com/awslabs/keras-apache-mxnet/wiki) ```v2.2.4.1``` <br/> [*NCCL*](https://developer.nvidia.com/nccl) ```v2.2.13``` | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/MXNet/mxnet-cu9.2-dnn7.2-18.10.dockerfile) | R396 |
    |MXNet| mx-cu9.2-dnn7.2-py3-18.09 | *MXNet* ```v1.3.0-dev``` <br/> [*GluonCV*](https://gluon-cv.mxnet.io) ```v0.3.0-dev``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> [*Keras-MXNet*](https://github.com/awslabs/keras-apache-mxnet/wiki) ```v2.2.2``` <br/> [*NCCL*](https://developer.nvidia.com/nccl) ```v2.2.13``` | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/MXNet/mxnet-cu9.2-dnn7.2-18.09.dockerfile) | R396 |
    |MXNet| mx-cu9.2-dnn7.1-py3-18.08 | *MXNet* ```v1.3.0-dev``` <br/> [*GluonCV*](https://gluon-cv.mxnet.io) ```v0.3.0-dev``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> [*Keras-MXNet*](https://github.com/awslabs/keras-apache-mxnet/wiki) ```v2.2.0``` <br/> [*NCCL*](https://developer.nvidia.com/nccl) ```v2.2.13``` | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/MXNet/mxnet-cu9.2-dnn7.1-18.08.dockerfile) | R396 |
    |MXNet| mx-cu9-dnn7-py3-18.03  | *MXNet* ```v1.2.0``` <br/> [*Keras-MXNet*](https://github.com/awslabs/keras-apache-mxnet/wiki) ```v2.1.3``` | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/MXNet/mxnet-cu9.0-dnn7.0-18.03.dockerfile) | R384 |
    |MXNet| mx-cu9-dnn7-py3-18.01  | *MXNet* ```v1.0.1``` <br/> [*Keras-MXNet*](https://github.com/awslabs/keras-apache-mxnet/wiki)  ```v1.2.2``` | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/MXNet/mxnet-cu9.0-dnn7.0-18.01.dockerfile) | R384 |
     |CNTK| cntk-cu10.0-dnn7.4-py3-19.01 / <br/> **cntk-latest**| *CNTK* ```v2.6``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2019.0-047``` <br/> *Keras* ```v2.2.4``` <br/> [*NCCL*](https://developer.nvidia.com/nccl) ```v2.3.7-1```  | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/CNTK/cntk-cu10.0-dnn7.4-19.01.dockerfile) | R410 |
    |CNTK| cntk-cu9.2-dnn7.2-py3-18.10 | *CNTK* ```v2.6``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> *Keras* ```v2.2.4```<br/> [*NCCL*](https://developer.nvidia.com/nccl) ```v2.2.13```  | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/CNTK/cntk-cu9.2-dnn7.2-18.10.dockerfile) | R396 |
    |CNTK| cntk-cu9.2-dnn7.2-py3-18.09| *CNTK* ```v2.5.1``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> *Keras* ```v2.2.2```  | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/CNTK/cntk-cu9.2-dnn7.2-18.09.dockerfile) | R396 |
    |CNTK| cntk-cu9-dnn7-py3-18.08 | *CNTK* ```v2.5.1``` <br/> *Keras* ```v2.2.2```  | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/CNTK/cntk-cu9-dnn7-18.08.dockerfile) | R384 |
    |CNTK| cntk-cu9-dnn7-py3-18.03  | *CNTK* ```v2.4``` <br/> *Keras* ```v2.1.5```  | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/CNTK/cntk-cu8-dnn6-18.03.dockerfile) | R384 |
    |CNTK| cntk-cu8-dnn6-py3-18.01  | *CNTK* ```v2.2``` <br/> *Keras* ```v2.1.2```  | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/CNTK/cntk-cu8-dnn6-18.01.dockerfile) |
    |Theano| theano-cu9.0-dnn7.0-py3-18.09 <br/> *theano-latest*  | Theano ```v1.0.2``` <br/> [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> *Keras* ```v2.2.2``` | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/Theano/theano-cu9.0-dnn7.2-18.09.dockerfile)|
    |Theano| theano-cu9-dnn7-py3-18.03 | Theano ```v1.0.1``` <br/> *Keras* ```v2.1.5``` | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/Theano/theano-cu9.0-dnn7.0-18.03.dockerfile)|
    |Theano| theano-cu9-dnn7-py3-18.01  | Theano ```v1.0.1``` <br/> *Keras* ```v2.1.2``` | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/Theano/theano-cu9.0-dnn7.0-18.01.dockerfile)|
    
* Images within the repository:  [**honghu/intelpython3**](https://hub.docker.com/r/honghu/intelpython3/)

    |  Tag  |  Description | Dockerfile |
    |---|---|:---:|
    | gpu-cu10.0-dnn7.4-19.01  | [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2019.0-054``` <br/> *Ubuntu* ```18.04``` |[[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/IntelPython3/intelpy3-gpu-cu10.0-dnn7.4-19.01.dockerfile) |
    | gpu-cu9.2-dnn7.2-18.09  | [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> *Ubuntu* ```18.04``` |[[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/IntelPython3/intelpy3-gpu-cu9.2-dnn7.2-18.09.dockerfile) |
    | gpu-cu9.0-dnn7.2-18.09  | [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> *Ubuntu* ```16.04``` |[[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/IntelPython3/intelpy3-gpu-cu9.0-dnn7.2-18.09.dockerfile) |
    | gpu-cu9.2-dnn7.1-18.08  | [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> *Ubuntu* ```18.04``` |[[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/IntelPython3/intelpy3-gpu-cu9.2-dnn7.1-18.08.dockerfile) |
    | cpu-18.09  |  [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> *Ubuntu* ```18.04```<br/> | [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/IntelPython3/intelpy3-cpu-18.09.dockerfile)|
    | cpu-18.08  |  [*Intel® Distribution for Python*](https://software.intel.com/en-us/distribution-for-python) ```v2018.3-039``` <br/> *Ubuntu* ```18.04```| [[Click]](https://github.com/chi-hung/DockerKeras/blob/master/Dockerfiles/IntelPython3/intelpy3-cpu-18.08.dockerfile)|

[[Back to Top]](#table-of-contents)
## Keras using TensorFlow Backend
This environment can be obtained via:
```bash
docker pull honghu/keras:tf-cu10.0-dnn7.4-py3-avx2-19.01
```
which includes 
* *Keras* ```v2.2.4```
* *TensorFlow* ```v1.12.0```
* *[Intel® Distribution for Python](https://software.intel.com/en-us/distribution-for-python)* ```v2019.0-047```, including accelerated NumPy and scikit-learn.
* *NVIDIA CUDA*```10.0```, *cuDNN*```7.4``` and *NCCL*```2.3.7-1```.
* Must-to-have packages such as *XGBOOST*, *Pandas*, *OpenCV*, *imgaug*, *Matplotlib*, *Seaborn* and *Bokeh*.

[[Back to Top]](#table-of-contents)

## Keras using MXNET Backend
This environment can be obtained via:
```bash
docker pull honghu/keras:mx-cu10.0-dnn7.4-py3-19.01
```
which includes
* *[Keras-MXNet](https://github.com/awslabs/keras-apache-mxnet/wiki)* ```v2.2.4.1```
* *MXNet* ```v1.4.0.rc0```
* *[GluonCV](https://gluon-cv.mxnet.io)* ```v0.3.0```
* *[Intel® Distribution for Python](https://software.intel.com/en-us/distribution-for-python)* ```v2019.0-047```, including accelerated NumPy and scikit-learn.
* *NVIDIA CUDA*```10.0```, *cuDNN*```7.4``` and *NCCL*```2.3.7-1```.
* Must-to-have packages such as *XGBOOST*, *Pandas*, *OpenCV*, *imgaug*, *Matplotlib*, *Seaborn* and *Bokeh*.

[[Back to Top]](#table-of-contents)
## Keras using CNTK Backend
This environment can be obtained via:
```bash
docker pull honghu/keras:cntk-cu10.0-dnn7.4-py3-19.01
```
which includes
* *Keras* ```v2.2.4```
* *CNTK* ```v2.6```
* *NVIDIA CUDA*```10.0```, *cuDNN*```7.4``` and *NCCL*```2.3.7-1```.
* Must-to-have packages such as *Pandas*, *OpenCV*, *imgaug*, *Matplotlib*, *Seaborn* and *Bokeh*.

Remark
* According to Microsoft, CNTK backend of Keras is [still in beta](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras). But, never mind! For the task such as text generation, switching the backend from TensorFlow to CNTK could possibly increase the speed of training significantly. See a [[benchmark]](http://minimaxir.com/2017/06/keras-cntk/) made by Max Woolf.

[[Back to Top]](#table-of-contents)
## Keras using Theano Backend
This environment can be obtained via:
```bash
docker pull honghu/keras:theano-cu9.0-dnn7.0-py3-18.09
```
which includes
* *Keras* ```v2.2.2```
* *Theano* ```v1.0.2```
* *NVIDIA CUDA*```9.0``` and *cuDNN*```7.0```.
* Must-to-have packages such as *Pandas*, *OpenCV*, *imgaug*, *Matplotlib*, *Seaborn* and *Bokeh*.

Remark
* As Theano has [stopped developing](https://groups.google.com/forum/#!msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ), we will not update this image regularly.

[[Back to Top]](#table-of-contents)
## ndrun - Run a Docker Container for Your Deep-Learning Research
Before you proceed to the next section, please get ```ndrun``` first:
```bash
# Create the "bin" directory if you don't have one inside your home folder.
if [ ! -d ~/bin ] ;then
  mkdir ~/bin
fi
# Get the wrapper file and save it to "~/bin/ndrun".
wget -O ~/bin/ndrun https://raw.githubusercontent.com/chi-hung/DockerbuildsKeras/master/ndrun.sh
# Make the wrapper file executable.
chmod +x ~/bin/ndrun
```
```ndrun``` is a tool that helps you to run a deep-learning environment. Before using it, please be sure to re-open your terminal in order to let the system know where this newly-added script ```ndrun``` is. In other words, make sure ```$HOME/bin``` is within your system's ```$PATH``` and then reload ```bash```.

Remark: 
* ```ndrun``` has to be used along with the recent images (images made starting Sep. 2018). There's no garantee that it will work fine with the older images.

[[Back to Top]](#table-of-contents)
## Getting Started with the Command Line

### Example: Check a Framework's Version
Let's prepare a script that will import *TensorFlow* and print its version out:
```bash
# Create a script that prints TensorFlow's version. 
printf "import tensorflow as tf \
        \nprint('TensorFlow version=',tf.__version__)" \
        > check_tf_version.py
```
Now, using ```ndrun```, the script ```check_tf_version.py```  can be executed easily using our TensorFlow image. All you have to do is add ```ndrun``` before ```python3 check_tf_version.py```:
```bash
ndrun python3 check_tf_version.py
```
And you should get the following output:
```
TensorFlow version= 1.12.0
```
which indicates that the current version of *TensorFlow* is ```1.12.0```. Now, the question then arises: where is this *TensorFlow* installed? Indeed, the *TensorFlow*'s version you've seen is from the *TensorFlow* installed inside our latest *TensorFlow* image.

To activate another image, we can use the option ```-t [IMG_TYPE]```. For example, let's now prepare a script that will import CNTK and print its version out:
```bash
# Create a script that checks CNTK's version. 
printf "import cntk \
        \nprint('CNTK version=',cntk.__version__)" \
        > check_cntk_version.py
```
To run this script using the *CNTK* image, simply add the option ```-t cntk```:
```bash
# Print CNTK's version out.
ndrun -t cntk python3 check_cntk_version.py
```

Its output:
```
CNTK version= 2.6
```

Currently, the possible choices of ```[IMG_TYPE]``` are:

* ```tensorflow``` 
* ```cntk``` 
* ```mxnet```
* ```theano```

Remark
* If you select an image via its type, i.e. via ```[IMG_TYPE]```, then, the latest image of that type will be selected.
* The latest *TensorFlow* image will be selected, if you do not inform ```ndrun``` which image it should select.
* If you don't have the selected image locally, docker will pull it from [Docker Hub](https://hub.docker.com) and that might take some time. 
* You can also select an image via its tag. Type ```ndrun --help``` for more details.

[[Back to Top]](#table-of-contents)
### Example: Classify Handwritten-Digits With TensorFlow
Now, let's retrieve an example from Google's GitHub repository aimed at handwritten-digits classification. This simple model (has only one hidden layer) is written in *TensorFlow* and [*MNIST*](http://yann.lecun.com/exdb/mnist/) is the dataset it's using.
```bash
# Get "mnist_with_summaries.py" from Google's GitHub repository.
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
```
Then, the retrieved script ```mnist_with_summaries.py``` can now be easily executed, via:
```bash
ndrun python3 mnist_with_summaries.py
```
The output should be similar to the following:
```
...
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
WARNING:tensorflow:From /opt/intel/intelpython3/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
2019-01-09 16:41:24.505114: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: TITAN V major: 7 minor: 0 memoryClockRate(GHz): 1.455
pciBusID: 0000:04:00.0
totalMemory: 11.75GiB freeMemory: 11.34GiB
2019-01-09 16:41:24.505174: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-01-09 16:41:25.018713: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-09 16:41:25.018764: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2019-01-09 16:41:25.018771: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2019-01-09 16:41:25.019071: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10957 MB memory) -> physical GPU (device: 0, name: TITAN V, pci bus id: 0000:04:00.0, compute capability: 7.0)
Accuracy at step 0: 0.0867
Accuracy at step 10: 0.7258
Accuracy at step 20: 0.8332
Accuracy at step 30: 0.8655
Accuracy at step 40: 0.87
Accuracy at step 50: 0.8883
...
```

[[Back to Top]](#table-of-contents)
### Example: Train a Multi-GPU Model Using TensorFlow
The previous example utilizes only one GPU. In this example, we suppose you have multiple GPUs at hand and you would like to train a model that utilizes multi-GPUs.

To be more specific:
* Our goal is to demostrate how you can run a script that classifies images of the [*CIFAR10*](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. 
* The model we are going to train is a small Convolutional Neural Network. For more details of this model, see the *TensorFlow*'s [official tutorial](https://www.tensorflow.org/tutorials/deep_cnn).

First, let's pull some models from the Google's GitHub repository. We also need to get the *CIFAR10* dataset, which is roughly 162MB:
```bash
# Clone tensorflow/models to your local folder.
# Say, to your home directory.
git clone https://github.com/tensorflow/models.git $HOME/models

# Checkout to a screenshot that I'm positive that the benchmark can be executed succesfully
cd $HOME/models && \
git checkout c803290 

# Let's also retrieve the CIFAR10 Dataset and put it into
# our home directory.
wget -O $HOME/cifar-10-binary.tar.gz \
https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
```
Now, you should have:
* ```cifar-10-binary.tar.gz``` (the *CIFAR10* dataset)
* ```models``` (a folder that contains many deep-learning models)

in your home directory. 

The script we are going to run is```cifar10_multi_gpu_train.py```, which is located at ```$HOME/models/tutorials/image/cifar10/```. Before we train the model, we need to set up configurations for training. Let's use ```--help``` to find out the acceptable configurations of ```cifar10_multi_gpu_train.py```:
```bash
ndrun python3 models/tutorials/image/cifar10/cifar10_multi_gpu_train.py --help
```
which returns the following output:

```
...
flags:

/workspace/models/tutorials/image/cifar10/cifar10_multi_gpu_train.py:
  --[no]log_device_placement: Whether to log device placement.
    (default: 'false')
  --max_steps: Number of batches to run.
    (default: '1000000')
    (an integer)
  --num_gpus: How many GPUs to use.
    (default: '1')
    (an integer)
  --train_dir: Directory where to write event logs and checkpoint.
    (default: '/tmp/cifar10_train')

cifar10:
  --batch_size: Number of images to process in a batch.
    (default: '128')
    (an integer)
  --data_dir: Path to the CIFAR-10 data directory.
    (default: '/tmp/cifar10_data')
  --[no]use_fp16: Train the model using fp16.
    (default: 'false')
...
```
As you can see, you can choose number of GPUs to be used via ```--num_gpus NUM_GPUS```  and you can set ```--data_dir TRAIN_DIR```, which tells the script where the downloaded *CIFAR10* dataset is.

Now, we are ready to train the model:
```bash
# Switch to your home directory.
cd $HOME
# Train the model.
ndrun -n 2 python3 models/tutorials/image/cifar10/cifar10_multi_gpu_train.py \
                       --num_gpus=2 \
                       --data_dir=/workspace \
                       --batch_size=128 \
                       --max_steps=100 \
                       --fp16
```
Your output should be similar to what I've got (2x NVIDIA Titan V):
```
...
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.FixedLengthRecordDataset`.
Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
...
2019-01-09 16:54:33.376230: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: TITAN V major: 7 minor: 0 memoryClockRate(GHz): 1.455
pciBusID: 0000:04:00.0
totalMemory: 11.75GiB freeMemory: 11.34GiB
2019-01-09 16:54:34.278037: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 1 with properties:
name: TITAN V major: 7 minor: 0 memoryClockRate(GHz): 1.455
pciBusID: 0000:06:00.0
totalMemory: 11.75GiB freeMemory: 11.34GiB
2019-01-09 16:54:34.280095: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0, 1
2019-01-09 16:54:35.130083: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-09 16:54:35.130130: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 1
2019-01-09 16:54:35.130137: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N Y
2019-01-09 16:54:35.130141: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 1:   Y N
2019-01-09 16:54:35.130998: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10957 MB memory) -> physical GPU (device: 0, name: TITAN V, pci bus id: 0000:04:00.0, compute capability: 7.0)
2019-01-09 16:54:35.131761: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10957 MB memory) -> physical GPU (device: 1, name: TITAN V, pci bus id: 0000:06:00.0, compute capability: 7.0)
WARNING:tensorflow:From /workspace/models/tutorials/image/cifar10/cifar10_multi_gpu_train.py:237: start_queue_runners (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
2019-01-09 16:54:41.808379: step 0, loss = 4.68 (40.7 examples/sec; 3.149 sec/batch)
2019-01-09 16:54:42.546906: step 10, loss = 4.61 (17071.7 examples/sec; 0.007 sec/batch)
2019-01-09 16:54:42.659138: step 20, loss = 4.46 (25444.7 examples/sec; 0.005 sec/batch)
2019-01-09 16:54:42.755444: step 30, loss = 4.40 (29393.4 examples/sec; 0.004 sec/batch)
2019-01-09 16:54:42.861769: step 40, loss = 4.27 (27947.5 examples/sec; 0.005 sec/batch)
2019-01-09 16:54:42.951161: step 50, loss = 4.18 (29182.5 examples/sec; 0.004 sec/batch)
2019-01-09 16:54:43.035104: step 60, loss = 4.19 (30707.3 examples/sec; 0.004 sec/batch)
2019-01-09 16:54:43.123850: step 70, loss = 4.21 (16449.5 examples/sec; 0.008 sec/batch)
2019-01-09 16:54:43.229280: step 80, loss = 4.17 (32346.5 examples/sec; 0.004 sec/batch)
2019-01-09 16:54:43.307614: step 90, loss = 4.27 (34882.1 examples/sec; 0.004 sec/batch)
```
Remark
*  As you activate a docker image using ```ndrun```, your current working directory on the host machine, e.g. ```$HOME```,  will automatically be mounted to ```/workspace```, a default working directory inside the docker container. 

    Since the script runs inside the docker container, it can only find the *CIFAR10* dataset at ```/workspace``` (caution! not at ```$HOME```!). Therefore, you should set ```--data_dir=/workspace```.
* Use ```-n [NUM_GPUS]```  to specify number of GPUs visible to the running image. If you don't pass this option to ```ndrun```, then, by default, ```ndrun``` will use only 1 GPU to run your script.

[[Back to Top]](#table-of-contents)

## Advanced Usage of the Command Line 

### Avoid Giving Your Script Extra GPUs
Here's a mistake: we create a docker container that sees two available GPUs and we however use only one of the available GPUs for training (specifying ```--num_gpu=1```):

```bash
# Switch to your home directory.
cd $HOME

# Train the model.
ndrun -n 2 python3 models/tutorials/image/cifar10/cifar10_multi_gpu_train.py \
                       --num_gpus=1 \
                       --data_dir=/workspace \
                       --batch_size=128 \
                       --max_steps=100 \
                       --fp16
```
During the run-time of this script, we can check the status of GPUs via ```nvidia-smi```:
```
chweng@server1:~$ nvidia-smi
Thu Jan 10 00:58:22 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.79       Driver Version: 410.79       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  TITAN V             On   | 00000000:04:00.0 Off |                  N/A |
| 31%   49C    P2    92W / 250W |  11780MiB / 12036MiB |     67%      Default |
+-------------------------------+----------------------+----------------------+
|   1  TITAN V             On   | 00000000:06:00.0 Off |                  N/A |
| 32%   46C    P8    30W / 250W |    502MiB / 12036MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      4783      C   python3                                    11769MiB |
|    1      4783      C   python3                                      491MiB |
+-----------------------------------------------------------------------------+
```
The above output indicates that *GPU0* was being utilized (*GPU-Util*=67%). Interestingly, although *GPU1*'s RAM was somewhat occupied, *GPU1* was not computing at all (*GPU-Util*=0%).

Avoid this mistake, otherwise you'll waste your GPU resources. (*GPU1* was occupied but was not computing at all.)

[[Back to Top]](#table-of-contents)
### Set ```NV_GPU``` If You Know Which GPUs You'd Like to Use
If you'd like to run your script using *GPU6* and *GPU7*, you can pass ```NV_GPU=6,7``` to *ndrun*. Let's look an example:
```bash
# Switch to your home directory.
cd $HOME

# Train the model using GPU6 and GPU7.
NV_GPU=6,7 ndrun -n 2 python3 models/tutorials/image/cifar10/cifar10_multi_gpu_train.py \
                --num_gpus=2 \
                --data_dir=/workspace \
                --batch_size=128 \
                --fp16
```
or, if you want to utlize 4 GPUs, say, *GPU0, GPU1, GPU2 and GPU3*:
```bash
# Switch to your home directory.
cd $HOME

# Train the model using GPU0, GPU1, GPU2 and GPU3.
NV_GPU=0,1,2,3 ndrun -n 4 python3 models/tutorials/image/cifar10/cifar10_multi_gpu_train.py \
                  --num_gpus=4 \
                  --data_dir=/workspace \
                  --batch_size=128 \
                  --fp16
```
However, I would suggest you avoid passing *NV_GPU* to ```ndrun```, unless you are pretty sure that's what you want. That's because ```ndrun``` will **automatically find available GPU devices for you**. Here, an *available* GPU device means it has [GPU-Utilization](http://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries) *< 30%* and has free memory *> 2048MB.*  If you wish, you can rewrite these criteria inside ```ndrun```.

[[Back to Top]](#table-of-contents)

### Run Commands on Behalf of Yourself
Normally, any command you type inside the terminal is excuted by yourself. However, this is not the default behaviour if you are using Docker. For example, if you touch a file called ```newfile``` inside a docker container:
```bash
# create an empty file and see who owns it
ndrun touch newfile && ls -hl newfile
```
You shall see the following output:
```
NV_GPU=0
-rw-r--r-- 1 root root 0 Oct 23 09:27 newfile
```
i.e. the new file you have created is owned by ```root``` because ```touch``` is executed by ```root``` inside the activated docker container. To avoid this behavior, you can feed the option  ```-u [USERNAME]``` while calling ```ndrun```:

```bash
# create an empty file and see who owns it. 
# the command "touch newfile2" will be executed
# on behalf of the user: chweng
ndrun -u chweng touch newfile2 && ls -hl newfile2
```
Its output is as follows:
```
You have provided a username. We will now create a docker image for that user.
Sending build context to Docker daemon  2.048kB
Step 1/7 : FROM honghu/keras:tf-latest
 ---> 1423013fa0c0
Step 2/7 : RUN groupadd -g 1001 chweng
 ---> Running in 64dd90f26709
Removing intermediate container 64dd90f26709
 ---> 101e44ef92f6
Step 3/7 : RUN useradd -rm -s /bin/bash -u 1001 -g 1001 chweng
 ---> Running in d3960f3010a4
Removing intermediate container d3960f3010a4
 ---> ebb5a45244c3
Step 4/7 : USER chweng
 ---> Running in fbd7557eb97c
Removing intermediate container fbd7557eb97c
 ---> 686ed15296bd
Step 5/7 : ENV HOME /home/chweng
 ---> Running in 52719f65b75d
Removing intermediate container 52719f65b75d
 ---> acdfc026fea0
Step 6/7 : RUN mkdir -p /home/chweng/.jupyter && wget -O /home/chweng/.jupyter/jupyter_notebook_config.py https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/docker/jupyter_notebook_config.py
 ---> Running in 35de7ae55b28
wget: /opt/intel/intelpython3/lib/libuuid.so.1: no version information available (required by wget)
--2018-10-23 01:34:24--  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/docker/jupyter_notebook_config.py
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1220 (1.2K) [text/plain]
Saving to: ‘/home/chweng/.jupyter/jupyter_notebook_config.py’

     0K .                                                     100%  142M=0s

2018-10-23 01:34:25 (142 MB/s) - ‘/home/chweng/.jupyter/jupyter_notebook_config.py’ saved [1220/1220]

Removing intermediate container 35de7ae55b28
 ---> a90ee4846193
Step 7/7 : WORKDIR /workspace
Removing intermediate container a283db0b088c
 ---> 006bd114ac71
Successfully built 006bd114ac71
Successfully tagged honghu/keras:tf-latest-chweng
NV_GPU=0
-rw-r--r-- 1 chweng chweng 0 Oct 23 09:34 newfile2
```

which shows that, the newly-created file ```newfile2``` is not owned by ```root```! Instead, it is owned by ```chweng```.

Hence, with the help of ```-u [USERNAME]```, you are now able to run any command on behalf of a specific user.

Remark: 

* In order to run commands on behalf of a specific user, a new image for the specified user will be created and named as ```honghu/keras:[SELECTED_IMG_TAG]-[USERNAME]```.
* In terminal, type ```cut -d: -f1 /etc/passwd```  for a list of users of your system.

[[Back to Top]](#table-of-contents)
## Getting Started with Jupyter Notebook

If you don't pass any script to ```ndrun```, then, ```ndrun``` will activate a docker container that runs [Jupyter Notebook](http://jupyter-notebook.readthedocs.io/en/stable/notebook.html#) for you. See the example below for more details.

### Example: Learn Deep Learning with MXNET Gluon
The easiest way to dive into Deep Learning with MXNet's new interface, gluon, is to follow the tutorials of [[Deep Learning - The Straight Dope]](http://gluon.mxnet.io/). These tutorials are written in the nice form of Jupyter Notebook, which allows executable scripts, equations, explanations and figures to be contained at one place - a notebook.

Let's clone these tutorials into, say, our home directory:
```bash
cd $HOME && git clone https://github.com/zackchase/mxnet-the-straight-dope.git
```
Now, you'll see a folder called ```mxnet-the-straight-dope``` within your home directory. Let's switch to this directory and initialize a daemon of *Jupyter Notebook* from there:
```bash
cd $HOME/mxnet-the-straight-dope
ndrun -n 1 -t mxnet -p 8889
```
The above command activates the latest *MXNet* image. It utilize 1 GPU and is now served as a daemon that listens to ```Port 8889``` on the side of your host machine.

Its output:
```
An intepreter such as python3 or bash, is not given.
You did not provide me the script you would like to execute.
NV_GPU=0
Starting Jupyter Notebook...

 * To use Jupyter Notebook, open a browser and connect to the following address:
   http://localhost:8889/?token=c5676caa643ecf9ebbfd8781381d0c0dbfbfcc1e67028e7a
 * To stop and remove this container, type:
   docker stop 5fb5489f198b && docker rm 5fb5489f198b
 * To enter into this container, type:
   docker exec -it 5fb5489f198b bash
```
Now, by opening a web browser and connecting to the URL given above, we are able to start learning *MXNet gluon*:

![IMG_MXNET_GLUON_TUTORIALS](https://i.imgur.com/wODQhGG.png)

Bravo!

[[Back to Top]](#table-of-contents)


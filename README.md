# Dockerbuilds-Keras
[![GithubStars](https://img.shields.io/github/stars/chi-hung/DockerbuildsKeras.svg?style=social&label=Stars)](https://github.com/chi-hung/DockerbuildsKeras/) [![Docker Pulls](https://img.shields.io/docker/pulls/honghu/keras.svg)](https://hub.docker.com/r/honghu/keras/) [![SponsoredByHonghuTech](https://img.shields.io/badge/sponsored%20by-Honghu%20Tech-red.svg)](http://www.honghutech.com/)

Wouldn't it be amazing, if you can run your scripts easily, without the need to install any deep-learning frameworks, dependent packages and libraries on your own? 

Now, with [Docker](https://www.docker.com/), [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker) and docker images provided by us, ***installation failed...*** or ***an error occurred during installation...*** is no more a pain in the ass, since installations are no longer required and you no more have to be overwhelmed by tons of installation guides.

We prepare, configure and maintain docker images of different deep-learning environments for you. From now on, all you have to do is type:
```
# run this if your script uses TensorFlow or TensorFlow+Keras.
ndrun -t tensorflow python3 my_script_using_tensorflow_and_keras.py

# or run this if your script uses CNTK or CNTK+Keras.
ndrun -t cntk python3 my_script_using_cntk_and_keras.py
```
which activates a suitable docker image that can run your script successfully.

Currently, I maintain the following docker images:

1. *Keras using Tensorflow Backened* 
2. *Keras using CNTK Backend*
3. *Keras using MXNET Backend*
4. *Keras using Theano Backend*

Apparantly, all of which support my beloved Keras. You can retrieve these images easily through [Dockerhub](https://hub.docker.com/r/honghu/keras/).

Give me a star if you find this project helpful.
亲，如觉不错便给颗星。
Geben Sie mir bitte einen Stern, wenn du findest, dass dieses Projekt sich lohnt!

See below for further information about what packages are included in each of the images we provide and how to use. Also, if you find any important package which is not included, do not hesitate to contact me!

## Table of Contents

* [Keras using TensorFlow Backend](#keras-using-tensorflow-backend)
* [Keras using CNTK Backend](#keras-using-cntk-backend)
* [Keras using MXNET Backend](#keras-using-mxnet-backend)
* [Keras using Theano Backend](#keras-using-theano-backend)
* [Before Getting Started](#before-getting-started)
* [Getting Started with the Command Line](#getting-started-with-the-command-line)
    * [Example: check the current framework’s version](#example-check-the-current-framework’s-version)
    * [Example: classify handwritten-digits with TensorFlow](#example-classify-handwritten-digits-with-tensorflow)
    * [Example: train a multi-GPU model with TensorFlow](#example-train-a-multi-gpu-model-with-tensorflow)
* [Subtle Issues When Using the Command Line](#subtle-issues-when-using-the-command-line)
    * [Avoid giving your script extra GPUs](#avoid-giving-your-script-extra-gpus)
    * [Use NV_GPU if you know which GPUs you’d like to use](#use-nv-gpu-if-you-know-which-gpus-you’d-like-to-use)
* [Getting Started with Jupyter Notebook](#getting-started-with-jupyter-notebook)
    * [Example: Learn MXNET and its gluon interface](#example-learn-mxnet-and-its-gluon-interface)

---
## Keras using TensorFlow Backend
This environment is retrievable by issuing the following command
```bash
docker pull honghu/keras:tf-cu9-dnn7-py3-avx2
```
which includes 
1. Keras```v2.0.8```
2. Tensorflow```v1.3.0``` &nbsp;&nbsp;![](https://img.shields.io/badge/build-from%20source-brightgreen.svg)
3. OpenCV```v3.3.0-dev``` ![](https://img.shields.io/badge/build-from%20source-brightgreen.svg)
4. common packages for Data Mining: Pandas, Scikit-Learn, Matplotlib and Seaborn.

Remark
* All the above-mentioned packages are built for Python```3```.
* Tensorflow and OpenCV are built from source. Both of them take advantage of NVIDIA's CUDA```9``` and cuDNN```7```.
* This image supports CPU instructions such as SSE4.2, AVX2 and FMA. If your CPU doesn't support these instructions, please pull ```honghu/keras:tf-cu9-dnn7-py3``` instead.

[go back](#table-of-contents)
## Keras using CNTK Backend
This environment is retrievable by issuing the following command
```bash
docker pull honghu/keras:cntk-cu8-dnn6-py3
```
which includes
1. Keras```v2.0.6```
2. CNTK```v2.2```
3. OpenCV```v3.2.0```
4. common packages for Data Mining: Pandas, Scikit-Learn, Matplotlib and Seaborn.

Remark
* All the above-mentioned packages are built for Python```3```.
* This image is based on the [official CNTK image](https://hub.docker.com/r/microsoft/cntk/), which takes advantage of NVIDIA's CUDA```8``` and cuDNN```6```. 
* According to Microsoft, CNTK backend is still [in beta](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras). Still, it was [tested](http://minimaxir.com/2017/06/keras-cntk/) that using CNTK backend is faster than using Tensorflow backend for some tasks such as text generation+LSTM.

[go back](#table-of-contents)
## Keras using MXNET Backend
This environment is retrievable by issuing the following command
```bash
docker pull honghu/keras:mx-cu9-dnn7-py3
```
which includes
1. [keras-mxnet](https://pypi.python.org/pypi/keras-mxnet/1.2.2)```v1.2.2```
2. MXNET```v0.11.1```&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](https://img.shields.io/badge/build-from%20source-brightgreen.svg)
3. OpenCV```v3.3.0-dev``` ![](https://img.shields.io/badge/build-from%20source-brightgreen.svg)
4. common packages for Data Mining: Pandas, Scikit-Learn, Matplotlib and Seaborn.

Remark
* All the above-mentioned packages are built for Python```3```.
* MXNET and OpenCV are built from source. Both of them take advantage of NVIDIA's CUDA```9``` and cuDNN```7```.
* The MXNET backend is under development. Go to [DMLC's Github](https://github.com/dmlc/keras) for some more details.

[go back](#table-of-contents)
## Keras using Theano Backend
This environment is retrievable by issuing the following command
```bash
docker pull honghu/keras:theano-cu9-dnn7-py3
```
which includes
1. Keras```v2.0.8```
2. Theano```0.10.0beta3``` ![](https://img.shields.io/badge/build-from%20source-brightgreen.svg)
3. OpenCV```v3.3.0```
4. common packages for Data Mining: Pandas, Scikit-Learn, Matplotlib and Seaborn.

Remark
* All the above-mentioned packages are built for Python```3```.
* Theano is built from source. It takes advantage of NVIDIA's CUDA```9``` and cuDNN```7```.

[go back](#table-of-contents)
## Before Getting Started
Before we start to learn how to use the docker images you just pulled, let us first get a wrapper file that helps us to get things done easily:
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
Now, armed with this wrapper file, running scripts (or interacting with Jupyter Notebooks) using different kind of environments is easier than ever.

Before we start, please be sure to re-open your terminal in order to let the system know where this newly added script *ndrun* is. In other words, make sure ```$HOME/bin``` is within your system's ```$PATH``` and then reload ```bash```.

[go back](#table-of-contents)
## Getting Started with the Command Line

### Example: check the framework's version of the running image
As a starting example, let us prepare and run a script that imports and checks TensorFlow's version:
```bash
# Create a script that checks TensorFlow's version. 
printf "import tensorflow as tf \
        \nprint('Tensorflow version=',tf.__version__)" \
        > check_tf_version.py
# Check TensorFlow's version.
ndrun python3 check_tf_version.py
```
You should get the following output:
```
Tensorflow version= 1.3.0
```
In the above example, the default docker image ```honghu/keras:tf-cu9-dnn7-py3-avx2``` was activated, which has TensorFlow and some other useful packages installed. The script then runs within it, printing out the version of TensorFlow it detects.

Furthermore, the option ```-t``` allows us to specify the type of the image to be activated. For example, you'll be able to check CNTK's version using the option ```-t cntk```, which shows the version of CNTK inside the activated CNTK's image:
```bash
# Create a script that checks CNTK's version. 
printf "import cntk \
        \nprint('CNTK version=',cntk.__version__)" \
        > check_cntk_version.py

# Check CNtk's version.
ndrun -t cntk python3 check_cntk_version.py
```
Its output is the following:
```
************************************************************
CNTK is activated.

Please checkout tutorials and examples here:
  /cntk/Tutorials
  /cntk/Examples

To deactivate the environment run

  source /root/anaconda3/bin/deactivate

************************************************************
CNTK version= 2.2
```

Currently, the available types are:

1. ```-t tensorflow```  ***(for TensorFlow)***
2. ```-t cntk``` ***(for CNTK)***
3. ```-t mxnet``` ***(for MXNET)***
4. ```-t theano``` ***(for Theano)***

Remark: ```-t tensorflow``` will be used by default, if you did not pass the option ```-t``` to *ndrun*.

The following table lists the defined types and their corresponding docker images.

|Framework |  Type  |  Docker Image (publisher/name:tag)|
|---|---|---|
|TensorFlow + Keras| tensorflow  |  honghu/keras:tf-cu9-dnn7-py3-avx2 |
|CNTK + Keras| cntk  |  honghu/keras:cntk-cu8-dnn6-py3 |
|MXNET + Keras| mxnet  |  honghu/keras:mx-cu9-dnn7-py3 |
|Theano + Keras| theano  | honghu/keras:theano-cu9-dnn7-py3  |

[go back](#table-of-contents)
### Example: classify handwritten-digits with TensorFlow
Now, let's retreive an example from Google's repository, which uses a small neural network (it has only one hidden layer) designed for handwritten-digits classification. This model is written in TensorFlow and [MNIST](http://yann.lecun.com/exdb/mnist/) is the dataset it's using.
```bash
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
```
This script can be executed simply through:
```bash
ndrun python3 mnist_with_summaries.py
```
and you should see some outputs similar to the following:
```bash
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
2017-10-16 17:33:59.597331: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: Tesla V100-SXM2-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.53
pciBusID: 0000:06:00.0
totalMemory: 15.77GiB freeMemory: 15.36GiB
2017-10-16 17:33:59.597368: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:06:00.0, compute capability: 7.0)
Accuracy at step 0: 0.1426
Accuracy at step 10: 0.6942
Accuracy at step 20: 0.8195
Accuracy at step 30: 0.8626
...
```

[go back](#table-of-contents)
### Example: train a multi-GPU model with TensorFlow
The previous example utilizes single GPU only. In this example, we suppose you have multiple GPUs at hand and you would like to train a model that utilizes multi-GPUs.

To be more specific:
1. our goal is to demostrate how you can run a script that classifies images from the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). 
2. the model we are going to train can be found in [a TensorFlow's official tutorial](https://www.tensorflow.org/tutorials/deep_cnn).

First, let's pull some models from the Google's repository. We also need to get the *CIFAR10* dataset, which is roughly 162MB:
```bash
# Clone tensorflow/models to your local folder.
# Say, to your home directory.
git clone https://github.com/tensorflow/models.git $HOME/models

# There's a bug in the latest CIFAR10 example.
# We temporarily switch to an older version of this repository.
cd $HOME/models && \
git checkout c96ef83 

# Let's also retrieve the CIFAR10 Dataset and put it into
# our home directory.
wget -O $HOME/cifar-10-binary.tar.gz \
https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
```
Now, you should have 
1. ```cifar-10-binary.tar.gz``` (*CIFAR10* dataset)
2. ```models``` (models pulled from TensorFlow's repository)

within your home directory. 

Next, we will run ```cifar10_multi_gpu_train.py```, which is located at ```models/tutorials/image/cifar10/```. Let's first ask this script for help in order to find out what the acceptable input arguments are:

```bash
ndrun python3 models/tutorials/image/cifar10/cifar10_multi_gpu_train.py --help
```
which returns the following output:

```bash
usage: cifar10_multi_gpu_train.py [-h] [--batch_size BATCH_SIZE]
                                  [--data_dir DATA_DIR]
                                  [--use_fp16 [USE_FP16]] [--nouse_fp16]
                                  [--train_dir TRAIN_DIR]
                                  [--max_steps MAX_STEPS]
                                  [--num_gpus NUM_GPUS]
                                  [--log_device_placement [LOG_DEVICE_PLACEMENT]]
                                  [--nolog_device_placement]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Number of images to process in a batch.
  --data_dir DATA_DIR   Path to the CIFAR-10 data directory.
  --use_fp16 [USE_FP16]
                        Train the model using fp16.
  --nouse_fp16
  --train_dir TRAIN_DIR
                        Directory where to write event logs and checkpoint.
  --max_steps MAX_STEPS
                        Number of batches to run.
  --num_gpus NUM_GPUS   How many GPUs to use.
  --log_device_placement [LOG_DEVICE_PLACEMENT]
                        Whether to log device placement.
  --nolog_device_placement
```
As you can see, you can select number of GPUs to be used via the option ```--num_gpus```  and you can use the option ```--data_dir``` to tell the script the location of the downloaded *CIFAR10* dataset. Here's a working example:
```bash
# Switch to your home directory.
cd $HOME

# Train the Resnet model.
ndrun -n 2 python3 models/tutorials/image/cifar10/cifar10_multi_gpu_train.py \
                       --num_gpus=2 \
                       --data_dir=/notebooks \
                       --batch_size=128 \
                       --max_steps=100 \
                       --fp16
```
Remark
*  As you activate a docker image using *ndrun*, your current working directory, i.e.```$HOME```,  will be  mounted to ```/notebooks```, a default working directory inside the docker container. Consequently, you should set ```--data_dir=/notebooks```, since *CIFAR10* dataset is only visible at ```/notebooks``` on the docker container's side.
* *ndrun* accepts the option ```-n``` (number of GPUs). By default it is ```-n 1```. If you'd like to use 2 GPUs, set ```-n 2``` so that there will be two GPUs visible to the activated docker image.
* You should get an output similar to the following (2x NVIDIA Tesla V100):
    ```bash
    Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
    2017-10-17 04:36:51.811596: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
    name: Tesla V100-SXM2-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.53
    pciBusID: 0000:06:00.0
    totalMemory: 15.77GiB freeMemory: 15.36GiB
    2017-10-17 04:36:52.434640: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 1 with properties: 
    name: Tesla V100-SXM2-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.53
    pciBusID: 0000:07:00.0
    totalMemory: 15.77GiB freeMemory: 15.36GiB
    2017-10-17 04:36:52.434689: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Device peer to peer matrix
    2017-10-17 04:36:52.434702: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1051] DMA: 0 1 
    2017-10-17 04:36:52.434726: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 0:   Y Y 
    2017-10-17 04:36:52.434748: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 1:   Y Y 
    2017-10-17 04:36:52.434758: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:06:00.0, compute capability: 7.0)
    2017-10-17 04:36:52.434765: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:1) -> (device: 1, name: Tesla V100-SXM2-16GB, pci bus id: 0000:07:00.0, compute capability: 7.0)
    2017-10-17 04:36:59.790431: step 0, loss = 4.68 (38.3 examples/sec; 3.346 sec/batch)
    2017-10-17 04:37:00.205024: step 10, loss = 4.59 (24464.4 examples/sec; 0.005 sec/batch)
    2017-10-17 04:37:00.323271: step 20, loss = 4.58 (20327.2 examples/sec; 0.006 sec/batch)
    2017-10-17 04:37:00.439341: step 30, loss = 4.50 (23105.6 examples/sec; 0.006 sec/batch)
    2017-10-17 04:37:00.558475: step 40, loss = 4.35 (22412.1 examples/sec; 0.006 sec/batch)
    2017-10-17 04:37:00.675634: step 50, loss = 4.48 (23193.5 examples/sec; 0.006 sec/batch)
    2017-10-17 04:37:00.791710: step 60, loss = 4.21 (23634.6 examples/sec; 0.005 sec/batch)
    2017-10-17 04:37:00.911417: step 70, loss = 4.26 (21293.4 examples/sec; 0.006 sec/batch)
    2017-10-17 04:37:01.028642: step 80, loss = 4.22 (22391.1 examples/sec; 0.006 sec/batch)
    2017-10-17 04:37:01.149847: step 90, loss = 3.98 (20516.7 examples/sec; 0.006 sec/batch)
    ```
    
[go back](#table-of-contents)
## Subtle Issues When Using the Command Line

### Avoid giving your script extra GPUs.
Here's the command that assigns 2 GPUs to run the script and we however let the script use only single GPU:

```bash
# Switch to your home directory.
cd $HOME

# Train the Resnet model.
ndrun -n 2 python3 models/tutorials/image/cifar10/cifar10_multi_gpu_train.py \
                       --num_gpus=1 \
                       --data_dir=/notebooks \
                       --batch_size=128 \
                       --max_steps=100 \
                       --fp16
```
We can check the status of GPUs while this script is running, via ```nvidia-smi```:
```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.81                 Driver Version: 384.81                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:01:00.0  On |                  N/A |
| 31%   58C    P2   143W / 250W |  10844MiB / 11171MiB |     72%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  Off  | 00000000:02:00.0 Off |                  N/A |
| 27%   51C    P2    58W / 250W |  10622MiB / 11172MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      3046      C   python3                                    10451MiB |
|    1      3046      C   python3                                    10611MiB |
+-----------------------------------------------------------------------------+
```
which indicates that only *GPU0* is in use. (Although *GPU1*'s memory is almost fully occupied, it's *GPU-Util* is *0\%*, meaning that it's not used at all.) This is a mistake that should be avoided otherwise you'll waste resources of your GPU.

[go back](#table-of-contents)
### Use *NV_GPU* if you know which GPUs you'd like to use.
If you'd like to use, say, *GPU 6* and *GPU7* to run your script, you can pass ```NV_GPU=6,7``` to *ndrun*, as the following example:
```bash
# Switch to your home directory.
cd $HOME

# Train the Resnet model using GPU6 and GPU7.
NV_GPU=6,7 ndrun -n 2 python3 models/tutorials/image/cifar10/cifar10_multi_gpu_train.py \
                --num_gpus=2 \
                --data_dir=/notebooks \
                --batch_size=128 \
                --fp16
```
or if you want to use 4 GPUs, say, *GPU0, GPU1, GPU2 and GPU3*:
```bash
# Switch to your home directory.
cd $HOME

# Train the Resnet model using GPU0,GPU1,GPU2 and GPU3.
NV_GPU=0,1,2,3 ndrun -n 4 python3 models/tutorials/image/cifar10/cifar10_multi_gpu_train.py \
                  --num_gpus=4 \
                  --data_dir=/notebooks \
                  --batch_size=128 \
                  --fp16
```
However, I would suggest you avoid passing *NV_GPU* to *ndrun*, unless you are pretty sure that's what you want, since *ndrun* will automatically find *available GPUs* for you. Here, an *available* GPU means it has [GPU-Utilization](http://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries) *< 30%* and has free memory *> 2048MB.*  If you wish, you can modify these criterions inside *ndrun*.

[go back](#table-of-contents)
## Getting Started with Jupyter Notebook

### Example: Learn MXNET and its gluon interface.
[MXNET + its gluon interface](http://gluon.mxnet.io/) has nice tutorials written in the format of *Jupyter Notebook*. Let's fetch them into, say, our home directory:
```
cd $HOME
git clone https://github.com/zackchase/mxnet-the-straight-dope.git
```
Now, you'll see a folder called ```mxnet-the-straight-dope``` within your home directory. Let's switch to this directory and initializes a *Jupyter Notebook server* from there:
```
cd $HOME/mxnet-the-straight-dope
ndrun -n 1 -t mxnet -p 8889
```
The above lines activate an environment that has MXNET installed. It also use one GPU via ```-n 1``` and open a port at *8889* via ```-p 8889```.

Its output is like this:
```
 An intepreter such as python/python3, is not given.
 You did not provide me the script you wish to execute.
Starting Jupyter Notebook...
NV_GPU=0
 * To use Jupyter Notebook, open a browser and connect to the following address:
      http://localhost:8889/?token=c0820fb56079312ca967a1355c298d21e35872090753eaa1
   Replace "localhost" to the IP address that is visible to other computers, if you are not coming from localhost.
 * To stop and remove this docker daemon, type:
      docker stop d982d8d571f8
```
Now, by opening a web-browser and connecting to the IP-address given, we are able to start learning *gluon*:

![IMG_MXNET_GLUON_TUTORIALS](https://i.imgur.com/wODQhGG.png)

Bravo!

[go back](#table-of-contents)
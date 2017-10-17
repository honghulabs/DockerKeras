# DockerbuildsKeras
[![Docker Pulls](https://img.shields.io/docker/pulls/honghu/keras.svg)](https://hub.docker.com/r/honghu/keras/) [![DockerStars](https://img.shields.io/docker/stars/honghu/keras.svg)](https://hub.docker.com/r/honghu/keras/) [![SponsoredByHonghuTech](https://img.shields.io/badge/sponsored%20by-Honghu%20Tech-red.svg)](http://www.honghutech.com/)

Wouldn't it be amazing, if you can run any scripts easily, without having to worry about whether the necessary deep-learning frameworks, dependent packages and libraries are all installed and configured properly? 

Now, with [Docker](https://www.docker.com/) and [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker), ***installation failed...*** or ***an error occurred during installation...*** is no more the pain in the ass, since installations are no longer required and you no more have to be overwhelmed by tons of installation guides.

We prepare, configure and maintain docker images of different deep-learning environments for you. From now on, all you have to do is type:
```
# run this if your script uses TensorFlow or TensorFlow+Keras.
ndrun -t tensorflow python3 my_script_using_tensorflow_and_keras.py

# or run this if your script uses CNTK or CNTK+Keras.
ndrun -t mxnet python3 my_script_using_cntk_and_keras.py
```
which activates a suitable docker image that can run your script successfully.

Currently, I maintain the following docker images:

1. *Keras using Tensorflow Backened* 
2. *Keras using CNTK Backend*
3. *Keras using MXNET Backend*
4. *Keras using Theano Backend*

Apparantly, all of which support my beloved Keras. You can retrieve these images easily through [Dockerhub](https://hub.docker.com/r/honghu/keras/).

Give me a star if you find any of these environments helpful. See below for further information about what packages are included in each of the images we provide and how to use. Also, if you find any important package which is not included, do not hesitate to contact me!

## Keras using Tensorflow Backend
This environment is retrievable by issuing the following command
```bash
docker pull honghu/keras:tf-cu9-dnn7-py3
```
which includes 
1. Keras```v2.0.8```
2. Tensorflow```v1.3.0``` &nbsp;&nbsp;![](https://img.shields.io/badge/build-from%20source-brightgreen.svg)
3. OpenCV```v3.3.0-dev``` ![](https://img.shields.io/badge/build-from%20source-brightgreen.svg)
4. common packages for Data Mining: Pandas, Scikit-Learn, Matplotlib and Seaborn.

Remark
* All the above-mentioned packages are built for Python```3```.
* Tensorflow and OpenCV are built from source. Both of them take advantage of NVIDIA's CUDA```9``` and cuDNN```7```.

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

## Getting Started with the Command Line
To begin with, let us get a wrapper file that helps us to execute Python scripts:
```bash
# get the wrapper file and save it as "ndrun".
wget -O ndrun https://raw.githubusercontent.com/chi-hung/DockerbuildsKeras/master/ndrun.sh
# make the wrapper file executable.
chmod +x ndrun
```
Now, armed with this wrapper file, running scripts written for different frameworks is easier than ever.
### check supported frameworks' version
As a starting example, let us prepare and run a script that imports and checks Tensorflow's version. 
```bash
# Create a script that checks Tensorflow's version. 
printf "import tensorflow as tf \
        \nprint('Tensorflow version=',tf.__version__)" \
        > check_tf_version.py
# Check Tensorflow's version.
./ndrun python3 check_tf_version.py
```
You should get the following output:
```
Tensorflow version= 1.3.0
```
In the above example, the image *honghu/keras:tf-cu9-dnn7-py3* was activated by default, which has TensorFlow and some other useful packages installed. On the other hand, if you'd like to activate an image of another framework, use ```-t``` to select an image based on its coressponding tag. Currently, the available tags are:
1. *tf-cu9-dnn7-py3* ***(for TensorFlow)***
2. *cntk-cu8-dnn6-py3* ***(for CNTK)***
3. *mx-cu9-dnn7-py3* ***(for MXNET)***
4. *theano-cu9-dnn7-py3* ***(for Theano)***

For example, you'll be able to check CNTK's version when using the option ```-t cntk-cu8-dnn6-py3``` (or even```-t cntk```):
```bash
# Create a script that checks Tensorflow's version. 
printf "import cntk \
        \nprint('CNTK version=',cntk.__version__)" \
        > check_cntk_version.py
# Check Tensorflow's version.
./ndrun -t cntk-cu8-dnn6-py3 python3 check_cntk_version.py
# or even simpler:
./ndrun -t cntk python3 check_cntk_version.py
```
which leads to the following output:
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
### handwritten-digits classification
Now, let's retreive an example from Google's repository, which uses TensorFlow to classify handwritten-digits.
```bash
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
```
this script can be executed via:
```bash
./ndrun python3 mnist_with_summaries.py
```
You should see some outputs similar to this
```bash
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
2017-10-16 17:33:58.981769: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
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
### multi-GPU example: *Resnet* + *CIFAR10* Dataset
The previous example utilizes single GPU only. In this example, we suppose you have multiple GPUs at hand and you would like to train a model that utilizes multi-GPUs.

To be more specific:
1. we are going to train a model (*Resnet*) for image classification.
2. our goal is to demostrate how you can run a script that classifies images of [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). 

First, let's retreive some models and *CIFAR10* dataset:
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
1. *cifar-10-binary.tar.gz* (*CIFAR10* dataset)
2. *models* (a folder that contain's models of TensorFlow)

within your home directory. 

Next, we will run *cifar10_multi_gpu_train.py*, which is located at *models/tutorials/image/cifar10/*. Let's first ask for help in order to find out what the acceptable input arguments are:

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
As you can see, you can select number of GPUs to be used via the option ```-num_gpus```  and you can use the option ```-data_dir``` to tell the script the location of the downloaded *CIFAR10* dataset. Here's a working example:
```bash
# Switch to your home directory.
cd $HOME

# Train the Resnet model.
./ndrun -n 2 python3 models/tutorials/image/cifar10/cifar10_multi_gpu_train.py \
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
2017-10-17 04:36:51.182380: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
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

# DockerbuildsKeras
[![Docker Pulls](https://img.shields.io/docker/pulls/honghu/keras.svg)](https://hub.docker.com/r/honghu/keras/) [![DockerStars](https://img.shields.io/docker/stars/honghu/keras.svg)](https://hub.docker.com/r/honghu/keras/) [![SponsoredByHonghuTech](https://img.shields.io/badge/sponsored%20by-Honghu%20Tech-red.svg)](http://www.honghutech.com/)

You've just found docker images built for Keras.

My goal is to provide various deep learning environments for Keras.



Currently, I maintain the following four types of docker images

1. *Keras using Tensorflow Backened* 
2. *Keras using CNTK Backend*
3. *Keras using MXNET Backend*
4. *Keras using Theano Backend*

You can retrieve them easily through [Dockerhub](https://hub.docker.com/r/honghu/keras/).

Give me a star if you find any of these environments helpful.
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
1. [Keras-MXNET](https://pypi.python.org/pypi/keras-mxnet/1.2.2)```v1.2.2```
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

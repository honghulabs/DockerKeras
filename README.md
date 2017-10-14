# DockerbuildsKeras
[![Docker Pulls](https://img.shields.io/docker/pulls/honghu/keras.svg)](https://hub.docker.com/r/honghu/keras/) [![DockerStars](https://img.shields.io/docker/stars/honghu/keras.svg)](https://hub.docker.com/r/honghu/keras/) [![SponsoredByHonghuTech](https://img.shields.io/badge/sponsored%20by-Honghu%20Tech-red.svg)](http://www.honghutech.com/)

You've just found docker images built for Keras.

My goal is to provide a variety of deep learning environments for Keras.

Give me a star if you find any of these environments helpful.

Currently, I maintain the following three types of docker images

1. *Keras using Tensorflow Backened* 
2. *Keras using CNTK Backend*
3. *Keras using MXNET Backend*

You can retrieve them easily through [Dockerhub](https://hub.docker.com/r/honghu/keras/).

## Keras using Tensorflow Backend
This environment is retrievable by issuing the following command
```bash
docker pull honghu/keras:tf-cu9-dnn7-py3
```
which covers Keras```v2.0.8```, Tensorflow```v1.3.0```, OpenCV```v3.3.0-dev``` and some packages that are common to Data Mining such as Pandas, Scikit-Learn, Matplotlib and Seaborn.

Remark
* All the above-mentioned packages are built for Python```3```.
* This image takes advantage of NVIDIA's CUDA```9``` and cuDNN```7``` while building Tensorflow and OpenCV.

## Keras using CNTK Backend
This environment is retrievable by issuing the following command
```bash
docker pull honghu/keras:cntk-cu8-dnn6-py3
```
which covers Keras```v2.0.6```, CNTK```v2.2```, OpenCV```v3.2.0``` and some packages that are common to Data Mining such as Pandas, Scikit-Learn, Matplotlib and Seaborn.

Remark
* All the above-mentioned packages are built for Python```3```.
* This image is based on the official CNTK image, which takes advantage of NVIDIA's CUDA```8``` and cuDNN```6``` while building CNTK. 
* According to Microsoft, the CNTK backend is still [in beta](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras). Still, it was [tested](http://minimaxir.com/2017/06/keras-cntk/) that using CNTK backend is faster than using Tensorflow backend for some tasks such as text generation+LSTM.

## Keras using MXNET Backend
This environment is retrievable by issuing the following command
```bash
docker pull honghu/keras:mx-cu9-dnn7-py3
```
which covers Keras```v1.2.2```, MXNET```v0.11.1```, OpenCV```v3.3.0-dev``` and some packages that are common to Data Mining such as Pandas, Scikit-Learn, Matplotlib and Seaborn.

Remark
* All the above-mentioned packages are built for Python```3```.
* This image takes advantage of NVIDIA's CUDA```9``` and cuDNN```7``` while building MXNET and OpenCV.

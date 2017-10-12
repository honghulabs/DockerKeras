# Dockerbuilds-Keras

We like Keras.

My goal is to provide a variety of deep learning environments primarily for Keras.

Give me a star if you find any of these environments helpful.

Currently, I maintain the following two types of docker images

1. *Keras using Tensorflow Backened* 
2. *Keras using CNTK Backend*

## Keras using Tensorflow Backend
this environment is retrievable by issuing the following command
```bash
docker pull honghu/keras:tf-cu9-dnn7-py3
```
which covers Keras, Tensorflow, OpenCV and some packages that are common to Data Mining such as Pandas, Scikit-Learn, Matplotlib and Seaborn.

All the above-mentioned packages are built for Python3.

This image also take advantage of NVIDIA's CUDA 9 and cuDNN7.

## Keras using CNTK Backend
this environment is retrievable by issuing the following command
```bash
docker pull honghu/keras:cntk-cu8-dnn6-py3
```
which covers Keras, CNTK, OpenCV and some packages that are common to Data Mining such as Pandas, Scikit-Learn, Matplotlib and Seaborn.

All the above-mentioned packages are built for Python3.

This image is based on the official CNTK image, which takes advantage of NVIDIA's CUDA 8 and cuDNN6. 

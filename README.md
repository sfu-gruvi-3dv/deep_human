# deep_human
Code for iccv2019 paper "A Neural Network for Detailed Human Depth Estimation from a Single Image" (Under construction)

Requirements
CUDA 9.0
OpenCV 3.2
Python 3.5
numpy


Preparation
Download model from https://drive.google.com/file/d/1xE_-KUPBI4S2FUbLyOQeqj1L4uS-hAg4/view?usp=sharing
extract the files into folder models.

Test:
run python demo.py 
model and test dir can be set in file params/params_iccv.py
results will be saved in output/

References:
@InProceedings{Tang_2019_ICCV,
author = {Tang, Sicong and Tan, Feitong and Cheng, Kelvin and Li, Zhaoyang and Zhu, Siyu and Tan, Ping},
title = {A Neural Network for Detailed Human Depth Estimation From a Single Image},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}




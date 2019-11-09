# deep_human
Code for iccv2019 paper "A Neural Network for Detailed Human Depth Estimation from a Single Image" (Under construction)

Requirements<br/>
CUDA 9.0<br/>
OpenCV 3.2<br/>
Python 3.5<br/>
tensorflow >= 1.6.0<br/>
numpy<br/>


Preparation<br/>
Download model from https://drive.google.com/file/d/1xE_-KUPBI4S2FUbLyOQeqj1L4uS-hAg4/view?usp=sharing<br/>
mkdir models<br/>
tar -xf models.tar -C models<br/>

Demo:<br/>
run python demo.py <br/>
model and test dir can be set in file params/params_iccv.py<br/>
results will be saved in output/<br/>

Acutally the network predict the depth for all pixels, and the computed depth image needs to be cropped by silhouette, you can use some off-the-shelf tools(e.g. MaskRCNN) to get the foreground region, or use the segmentation result returned by segmentation-net/<br/>

Training data can be downloaded here:<br/>
https://drive.google.com/file/d/1fWxF6dpdzJH_Hknmr3RKTIyiHegwiirF/view?usp=sharing<br/>

References:<br/>
@InProceedings{Tang_2019_ICCV,
author = {Tang, Sicong and Tan, Feitong and Cheng, Kelvin and Li, Zhaoyang and Zhu, Siyu and Tan, Ping},
title = {A Neural Network for Detailed Human Depth Estimation From a Single Image},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}




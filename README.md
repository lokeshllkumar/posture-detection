# fall-detector
A fall detection program written using OpenCV in Python.

## Description
The simple fall detection program written using Python, with the help of the OpenCV and Mediapipe libraries. 

The main.py program performs skeletonization on moving objects in each frame of a given video file and draws a box around the moving objects. 
The posture of the moving body is estimated after computing the aspect ratio of the drawn box. No pre-trained models have been used to derermine the posture of objects.

## Getting started
Install the following on your system
1. python 3.10
2. pip - the Python package manager to install the necessary libraries
3. opencv2 - the OpenCV library for Python
4. matplotlib - the plotting library to plot an accuracy graph
5. mediapipe - the library based on the framework to perform skeletonization on objects in frame

# Tab2Img: from tabular data to images
A tool to convert tabular data into images for CNN. Inspired by the [DeepInsight](https://www.nature.com/articles/s41598-019-47765-6) paper.

## Background

In the [paper](https://www.nature.com/articles/s41598-019-47765-6) "*DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture*" the autors propose  a method to convert tabular data into images, in order to utilize the power of convolutional neural network (CNN) for non-image structured data.

<p align="center">
  <img src="https://github.com/nicomignoni/tab2img/blob/master/docs/feature_mapping.png"/>
</p>

The Figure illustrates the main idea: given a training dataset ![equation](https://latex.codecogs.com/gif.latex?X%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bm%5Ctimes%20n%7D) with ![equation](https://latex.codecogs.com/gif.latex?m) samples and ![equation](https://latex.codecogs.com/gif.latex?n) features, we are required to find a function ![equation](https://latex.codecogs.com/gif.latex?M%20%3A%20%5Cmathbb%7BR%7D%5E%7Bm%5Ctimes%20n%7D%20%5Crightarrow%20%5Cmathbb%7BR%7D%5E%7Bm%5Ctimes%20d%20%5Ctimes%20d%7D), where ![equation](https://latex.codecogs.com/gif.latex?d%20%3D%20%5Clceil%20%5Csqrt%7Bn%7D%5Crceil). 
There are numerous ways to choose ![equation](https://latex.codecogs.com/gif.latex?M). In this implementation, the features are organized with respect to the correlation vector ![equation](https://latex.codecogs.com/gif.latex?%5Crho%28X%2C%20Y%29), where ![equation](https://latex.codecogs.com/gif.latex?Y%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B1%20%5Ctimes%20m%7D) is the target vector.
Given ![equation](https://latex.codecogs.com/gif.latex?X) and ![equation](https://latex.codecogs.com/gif.latex?Y) as

![equation](https://latex.codecogs.com/gif.latex?X%20%3D%20%5Cbegin%7Bpmatrix%7D%20x%5E%7B%281%29%7D_1%20%26%20%5Ccdots%20%26%20x%5E%7B%281%29%7D_n%20%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20x%5E%7B%28m%29%7D_1%20%26%20%5Ccdots%20%26%20x%5E%7B%28m%29%7D_n%20%5Cend%7Bpmatrix%7D%2C%20%5Cquad%20Y%20%3D%20%5Cbegin%7Bpmatrix%7D%20y%5E%7B%281%29%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20y%5E%7B%28m%29%7D%29%20%5Cend%7Bpmatrix%7D)


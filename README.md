# Tab2Img: from tabular data to images
A tool to convert tabular data into images for CNN. Inspired by the [DeepInsight](https://www.nature.com/articles/s41598-019-47765-6) paper.

## Background

In the [paper](https://www.nature.com/articles/s41598-019-47765-6) "*DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture*" the autors propose  a method to convert tabular data into images, in order to utilize the power of convolutional neural network (CNN) for non-image structured data.

<p align="center">
  <img src="https://github.com/nicomignoni/tab2img/blob/master/docs/feature_mapping.png"/>
</p>

The Figure illustrates the main idea: given a training dataset ![equation](https://latex.codecogs.com/gif.latex?X%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bm%5Ctimes%20n%7D)

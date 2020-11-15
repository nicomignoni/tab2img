# Tab2Img: from tabular data to images
A tool to convert tabular data into images for CNN. Inspired by the [DeepInsight](https://www.nature.com/articles/s41598-019-47765-6) paper.

## Installation 
```
pip install tab2img
```

## Background

In the [paper](https://www.nature.com/articles/s41598-019-47765-6) "*DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture*" the autors propose  a method to convert tabular data into images, in order to utilize the power of convolutional neural network (CNN) for non-image structured data.

<p align="center">
  <img src="https://github.com/nicomignoni/tab2img/blob/master/docs/feature_mapping.png"/>
</p>

The Figure illustrates the main idea: given a training dataset ![equation](https://latex.codecogs.com/gif.latex?X%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bm%5Ctimes%20n%7D) with ![equation](https://latex.codecogs.com/gif.latex?m) samples and ![equation](https://latex.codecogs.com/gif.latex?n) features, we are required to find a function ![equation](https://latex.codecogs.com/gif.latex?M%20%3A%20%5Cmathbb%7BR%7D%5E%7Bm%5Ctimes%20n%7D%20%5Crightarrow%20%5Cmathbb%7BR%7D%5E%7Bm%5Ctimes%20d%20%5Ctimes%20d%7D), where ![equation](https://latex.codecogs.com/gif.latex?d%20%3D%20%5Clceil%20%5Csqrt%7Bn%7D%5Crceil). 

There are numerous ways to choose ![equation](https://latex.codecogs.com/gif.latex?M). In this implementation, the features are organized with respect to the correlation vector ![equation](https://latex.codecogs.com/gif.latex?%5Crho%28X%2C%20Y%29), where ![equation](https://latex.codecogs.com/gif.latex?Y%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B1%20%5Ctimes%20m%7D) is the target vector.
Given ![equation](https://latex.codecogs.com/gif.latex?X) and ![equation](https://latex.codecogs.com/gif.latex?Y) as

<div align="center"><img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20X%20%3D%20%5Cbegin%7Bpmatrix%7D%20x%5E%7B%281%29%7D_1%20%26%20%5Ccdots%20%26%20x%5E%7B%281%29%7D_n%20%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20x%5E%7B%28m%29%7D_1%20%26%20%5Ccdots%20%26%20x%5E%7B%28m%29%7D_n%20%5Cend%7Bpmatrix%7D%2C%20%5Cquad%20Y%20%3D%5Cbegin%7Bpmatrix%7D%20y_1%20%5C%5C%20%5Cvdots%20%5C%5C%20y_m%20%5Cend%7Bpmatrix%7D%20%5Cend%7Balign*%7D"></div>

 the vector ![equation](https://latex.codecogs.com/gif.latex?%5Crho%28X%2C%20Y%29%20%3D%20%28%5Crho_1%2C%20...%2C%20%5Crho_n%29) express the Pearson correlation coefficient*
 
<div align="center"><img src="https://latex.codecogs.com/gif.latex?%5Crho%20%3D%20%5Cfrac%7B%5Ctext%7Bcov%7D%28x%2C%20y%29%7D%7B%5Csigma%28x%29%5Csigma%28y%29%7D"></div>

where 

<div align="center"><img src="https://latex.codecogs.com/gif.latex?%5Crho_i%20%3D%20%5Crho%28X_i%2C%20Y%29%2C%20%5Cquad%20X_i%20%3D%20%5Cbegin%7Bpmatrix%7D%20x%5E%7B%281%29%7D_i%20%5C%5C%20%5Cvdots%20%5C%5C%20x%5E%7B%28m%29%7D_i%20%5Cend%7Bpmatrix%7D"></div>

At this point ![equation](https://latex.codecogs.com/gif.latex?%5Crho%28X%2C%20Y%29) is sorted from the greatest to the smallest, generating the vector of indices 

<div align="center"><img src="https://latex.codecogs.com/gif.latex?%5Cbold%7BJ%7D%20%3D%20%28J_k%20%5C%20%3A%20%5C%20%5Crho%28X_%7BJ_k%7D%29%20%5Cgeq%20%5Crho%28X_%7BJ_%7Bk-1%7D%7D%29%2C%20k%20%5Cin%20%5C%5B1%2C%20...%2C%20n%5D%29"></div>

Eventually, the final tensor ![equation](https://latex.codecogs.com/gif.latex?M) is

The function that maps ![equation](https://latex.codecogs.com/gif.latex?J_k) to the right row and column ![equation](https://latex.codecogs.com/gif.latex?%28r%2C%20c%29_k) of ![equation](https://latex.codecogs.com/gif.latex?M) is 

<div align="center"><img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%28r%2Cc%29_k%20%3D%20%5Cbegin%7Bcases%7D%20%28%5Csqrt%7Bk%7D%2C%20%5Csqrt%7Bk%7D%29%20%26%20%5Ctext%7Bif%7D%20%5C%20%5Csqrt%7Bk%7D%20%5Cin%20%5Cmathbb%7BN%7D%20%5C%5C%20%28%5Clceil%5Csqrt%7Bk%7D%5Crceil%2C%20%5Clceil%5Csqrt%7Bk%7D%5Crceil%20-%20%5Cfrac%7B1%7D%7B2%7D%28%5Clceil%5Csqrt%7Bk%7D%5Crceil%5E2%20-%20k%29%29%20%26%20%5Ctext%7Bif%7D%20%5C%20%5Csqrt%7Bk%7D%20%5Cnotin%20%5Cmathbb%7BN%7D%20%5C%20%5Ctext%7Band%7D%20%5C%20%5Clceil%5Csqrt%7Bk%7D%5Crceil%5E2%20-%20k%20%3D%200%20%5Cmod%7B2%7D%20%5C%5C%20%28%5Clceil%5Csqrt%7Bk%7D%5Crceil%20-%20%5Clceil%5Cfrac%7B1%7D%7B2%7D%28%5Clceil%5Csqrt%7Bk%7D%5Crceil%5E2%20-%20k%29%5Crceil%2C%20%5Clceil%5Csqrt%7Bk%7D%5Crceil%29%20%26%20%5Ctext%7Bif%7D%20%5C%20%5Csqrt%7Bk%7D%20%5Cnotin%20%5Cmathbb%7BN%7D%20%5C%20%5Ctext%7Band%7D%20%5C%20%5Clceil%5Csqrt%7Bk%7D%5Crceil%5E2%20-%20k%20%5Cneq%200%20%5Cmod%7B2%7D%20%5Cend%7Bcases%7D%20%5Cend%7Balign*%7D"></div>

---

In this case, being ![equation](https://latex.codecogs.com/gif.latex?X) a sample, the coefficient is implemented as 

<div align="center"><img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Crho%28x%2Cy%29%20%3D%20%5Cfrac%20%7B%5Csum%20_%7Bi%3D1%7D%5E%7Bn%7D%28x_%7Bi%7D-%7B%5Cbar%20%7Bx%7D%7D%29%28y_%7Bi%7D-%7B%5Cbar%20%7By%7D%7D%29%7D%7B%7B%5Csqrt%20%7B%5Csum%20_%7Bi%3D1%7D%5E%7Bn%7D%28x_%7Bi%7D-%7B%5Cbar%20%7Bx%7D%7D%29%5E%7B2%7D%7D%7D%7B%5Csqrt%20%7B%5Csum%20_%7Bi%3D1%7D%5E%7Bn%7D%28y_%7Bi%7D-%7B%5Cbar%20%7By%7D%7D%29%5E%7B2%7D%7D%7D%7D%20%5Cend%7Balign*%7D"></div>

## Example

```python
from sklearn.datasets import fetch_covtype
from tab2img.converter import Tab2Img

dataset = fetch_covtype()

train = dataset.data
target = dataset.target

model = Tab2Img()
images = model.fit_transform(train, target)
```


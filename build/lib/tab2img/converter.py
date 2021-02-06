import numpy as np
from math import ceil, sqrt

class Tab2Img:
    def __init__(self, save_path=None, allow_pickle=False):
        self.save_path = save_path
        self.allow_pickle = allow_pickle

    def fit(self, X, Y):
        n_sample, n_attrs = X.shape
        size = ceil(sqrt(n_attrs))

        # Delete constant features (std == 0) from X
        std_X             = np.std(X, axis=0, ddof=0)
        constant_features = np.where(std_X == 0)
        X                 = np.delete(X, constant_features, axis=1)
        
        # Pearson's Corrleation Coefficient
        # en.wikipedia.org/wiki/Pearson_correlation_coefficient
        std_X  = np.std(X, axis=0, ddof=0)
        std_Y  = np.std(Y, ddof=0)
        mean_X = np.mean(X, axis=0)
        mean_Y = np.mean(Y)
        cov  = (X - mean_X) * (Y - mean_Y).reshape(n_sample, 1)
        corr = np.divide(cov.sum(axis=0), std_X*std_Y*n_sample)

        # The correlation vector ('corr') gets sorted in ascending mode
        # and for each element in the sorted vector, we map the index to
        # its place (i.e. find the row and col) in the matrix representing
        # the sample as image. 
        indices = np.argsort(corr)[::-1]

        # The map from tabular data to image data:
        # [table column index, tensor row, tensor column]
        self.mapping = np.zeros((len(indices), 3), np.int8)
        
        for i, index in enumerate(indices):
            closest_sqrt = ceil(sqrt(i+1))
            distance = closest_sqrt ** 2 - (i+1)
            if distance == 0:
                row, col = closest_sqrt, closest_sqrt
            else:
                if distance % 2 == 0:
                    row, col = closest_sqrt - distance/2, closest_sqrt
                else:
                    row, col = closest_sqrt, closest_sqrt - ceil(distance/2)
            self.mapping[i, :] = [index, int(row-1), int(col-1)]

    def transform(self, X):
        n_sample, n_attrs = X.shape
        size = ceil(sqrt(n_attrs))
        
        # The final tensor
        images  = np.zeros((n_sample, size, size), np.float32)
        for index, row, col in self.mapping:
            images[:, row, col] = X[:, index]

        if self.save_path: np.save(save_path, allow_pickle=allow_pickle)
            
        return images

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        
        return self.transform(X)

        
        
        

from tab2img import Tab2Img

import os
import numpy as np
import pandas as pd

'''Loading datasets'''

train  = pd.read_csv('train.csv')
target = train['SalePrice'].values

TRAIN_SIZE, NUM_ATTRIBUTES = train.shape

'''Data Manipulation'''
        
# Dividing features into categorical,convertible, numerical and deletable ones
categorical_attrs = {'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
                     'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
                     'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Foundation',
                     'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'CentralAir', 
                     'Electrical', 'GarageType', 'PavedDrive', 'MiscFeature', 'MoSold',
                     'SaleType', 'SaleCondition'}

numerical_attrs   = {'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                     'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                     'BsmtUnfSF', '1stFlrSF','2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 
                     'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
                     'TotRmsAbvGrd', 'Fireplaces','GarageYrBlt', 'GarageCars', 'GarageArea', 
                     'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                     'PoolArea', 'MiscVal', 'YrSold'}

convertible_attrs = {'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                     'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 
                     'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence'}

deletable_attrs   = {'TotalBsmtSF'}

# Revoming the target column ('SalePrice')
train.drop(columns='SalePrice', inplace=True)

# Concatenating the train and test datasetes 
dataset = pd.concat([train], ignore_index=True)

'''Converting data'''

from sklearn.preprocessing import OneHotEncoder

# Dropping the 'deletable attributes'
dataset.drop(columns=deletable_attrs, inplace=True)

# Converting the 'convertible attributes' to integers 
for attr in convertible_attrs:
    dataset[attr] = dataset[attr].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 
                                       'Av': 3, 'Mn': 2, 'No': 1,
                                       'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1,
                                       'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj': 3, 'Sev': 2, 'Sal': 1,
                                       'Fin': 3, 'RFn': 2, 'Unf': 1,
                                       'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1})
    dataset[attr].fillna(0, inplace=True)

# Filling NaNs for 'numerical attributes' w/ 0s
for attr in numerical_attrs:
    dataset[attr].fillna(0, inplace=True)

data = dataset[set().union(*[numerical_attrs, convertible_attrs])].values

# Converting 'categorical attributes' into OneHot representation
ohe = OneHotEncoder(sparse=False, handle_unknown='error')
for attr in categorical_attrs:
    dataset[attr].fillna('NN', inplace=True)
    one_shot_feature = ohe.fit_transform(dataset[attr].values.reshape(-1, 1))
    data = np.concatenate((data, one_shot_feature), axis=1)

# Deleting data having column totally equal to zeros
data = np.delete(data, [data[:, np.sum(data, axis=0) == 0]], axis=1)
    
training_data = data[:TRAIN_SIZE, :]
testing_data  = data[TRAIN_SIZE:, :]

'''Model testing'''

model = Tab2Img()
img = model.fit_transform(training_data, target)

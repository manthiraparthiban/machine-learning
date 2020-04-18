import pandas as pd
import numpy as np

train=pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\train.csv')
test=pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\test.csv')

print(train)
print(test)
print(train.describe())

train.isnull()
test.isnull()

train.isnull().sum()
test.isnull().sum()

from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

imputer=imputer.fit(test.loc[:, ['Fare']])
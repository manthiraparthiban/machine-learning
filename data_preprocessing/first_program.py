import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data_preprocessing.csv")
dataset.isnull().sum()

x = dataset.iloc[:,0:3].values
y = dataset.iloc[:,3:4].values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_x = LabelEncoder()
x[:,0] = label_x.fit_transform(x[:,0])
onehotencoder_x = OneHotEncoder(categorical_features=[0])
x = onehotencoder_x.fit_transform(x).toarray()
x = x[:,1:]
label_y = LabelEncoder()
y = label_y.fit_transform(y)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.20,random_state = 0)

from sklearn.preprocessing import StandardScaler
Sc_x = StandardScaler()
xtrain = Sc_x.fit_transform(xtrain)
xtest = Sc_x.transform(xtest)
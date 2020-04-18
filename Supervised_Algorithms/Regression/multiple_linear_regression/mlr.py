import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("data_startups.csv")
dataset.isnull().sum()

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4:5].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
x[:,3] = le_x.fit_transform(x[:,3])
one_x = OneHotEncoder(categorical_features=[3])
x = one_x.fit_transform(x).toarray() 
x = x[:,1:]

from sklearn.model_selection import train_test_split
xtrain,xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.30, random_state = 0)

from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(xtrain,ytrain)

y_pred = regress.predict(xtest)

regress.score(xtrain,ytrain)
regress.score(xtest,ytest)

import statsmodels.regression.linear_model as sm
x = np.append(arr = np.ones(shape = (50,1), dtype=int), values = x, axis=1)

x_ov = x[:,[0,1,2,3,4,5]]
regress_ols = sm.OLS(endog = y, exog = x_ov).fit()
regress_ols.summary()


#check video code
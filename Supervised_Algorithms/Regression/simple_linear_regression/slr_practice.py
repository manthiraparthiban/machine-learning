import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("height_weight.csv")
dataset.isnull().sum()

x = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1:2].values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:,0:1])
x[:,0:1] = imputer.transform(x[:,0:1])

from sklearn.model_selection import train_test_split
xtrain,xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.30, random_state = 0)

from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(xtrain,ytrain)

y_pred = regress.predict(xtest)

plt.title("simple linear regression")
plt.xlabel("height")
plt.ylabel("weight")
plt.scatter(xtrain,ytrain,color = "red")
plt.plot(xtrain,regress.predict(xtrain), color = "blue")
plt.show()

plt.title("simple linear regression with test data")
plt.xlabel("height")
plt.ylabel("weight")
plt.scatter(xtest,ytest,color = "red")
plt.plot(xtest,regress.predict(xtest), color = "blue")
plt.show()

plt.title("simple linear regression with test data and train_slope")
plt.xlabel("height")
plt.ylabel("weight")
plt.scatter(xtest,ytest,color = "red")
plt.plot(xtrain,regress.predict(xtrain), color = "blue")
plt.show()

my_height = [[185]]
my_weight = regress.predict(my_height)
print(my_weight)

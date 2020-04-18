import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("salary_data.csv")
x = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
xtrain,xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.30, random_state = 0)

from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(xtrain,ytrain)

y_pred = regress.predict(xtest)

plt.title("simple linear regression")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.scatter(xtrain,ytrain,color = "red")
plt.plot(xtrain,regress.predict(xtrain), color = "blue")
plt.show()

plt.title("simple linear regression with test data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.scatter(xtest,ytest,color = "red")
plt.plot(xtest,regress.predict(xtest), color = "blue")
plt.show()

plt.title("simple linear regression with test data and train_slope")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.scatter(xtest,ytest,color = "red")
plt.plot(xtrain,regress.predict(xtrain), color = "blue")
plt.show()
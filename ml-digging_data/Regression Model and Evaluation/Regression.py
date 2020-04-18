# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:08:22 2020

@author: Admin
"""


#linear regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\Regression Model and Evaluation\data\LinearRegression\data.csv')

data.isnull().sum()

x = data.iloc[:,:-1].values
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

a = regressor.predict([[2]])

#mean absolute error

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

#mean square error

from sklearn.metrics import mean_squared_error

train_MSE=mean_squared_error(y_train, regressor.predict(x_train))
test_MSE=mean_squared_error(y_test, y_pred)


#r2 score

from sklearn.metrics import r2_score

rscore = r2_score(y_test, y_pred)

# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Hight vs Weights (Training set)')
plt.xlabel('Hight')
plt.ylabel('Weight')
plt.show()

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Hight vs weights (Test set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()


#multi regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\Regression Model and Evaluation\data\MultipleLinearRegression\insurance.csv')

data.info()

data.isnull().sum()

x = data.iloc[:,:-1].values
y = data.iloc[:,6].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder = LabelEncoder()

x[:,1] = labelencoder.fit_transform(x[:,1])
x[:,4] = labelencoder.fit_transform(x[:,4])
x[:,5] = labelencoder.fit_transform(x[:,5])

onehotencoder = OneHotEncoder(categorical_features = [5])
x = onehotencoder.fit_transform(x).toarray()

x = x[:, 1:]

import statsmodels.formula.api as sm
x = np.append(arr = np.ones((1338, 1)).astype(int), values = x, axis = 1)

x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
regressor_OLS = sm.ols(endog = y, exog = x_opt).fit()
regressor_OLS.summary()


x_opt = x[:, [0, 1, 2, 3, 4, 6, 7, 8]]
# we removed 5th column. lets execute this
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 2, 3, 4, 6, 7, 8]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 2, 4, 6, 7, 8]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 4, 6, 7, 8]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#Splitting x-opt into training and Split test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


#ploynomial regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\Regression Model and Evaluation\data\PolynomialRegression\HousingData.csv')

data.info()

x = data[["Purchase time passed(1990)"]]
y = data[["Pricing"]]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import PolynomialFeatures

model = PolynomialFeatures(degree = 3)
x_train = model.fit_transform(x_train)
x_test = model.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
pred = regressor.predict(x_test) 

from sklearn.metrics import r2_score
r2_score(y_test, pred)



#regression spline

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 
data = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\Regression Model and Evaluation\data\regression_spline\Wage.csv')

data_x = data['age']
data_y = data['wage']

from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.33, random_state = 1)

# Dividing the data into 4 bins
df_cut, bins = pd.cut(train_x, 4, retbins=True, right=True)
df_cut.value_counts(sort=False)
df_steps = pd.concat([train_x, df_cut, train_y], keys=['age','age_cuts','wage'], axis=1)

# Create dummy variables for the age groups
df_steps_dummies = pd.get_dummies(df_cut)
df_steps_dummies.head()

df_steps_dummies.columns = ['17.938-33.5','33.5-49','49-64.5','64.5-80']

# Fitting Generalised linear models
fit3 = sm.GLM(df_steps.wage, df_steps_dummies).fit()
# Binning validation set into same 4 bins
bin_mapping = np.digitize(valid_x, bins)

bin_mapping = pd.DataFrame(bin_mapping)
bin_mapping.columns = ['bins']
bin_mapping.bins = bin_mapping.bins.astype('O')

X_valid = pd.get_dummies(bin_mapping)
X_valid.head()

# Removing any outliers
X_valid = pd.get_dummies(bin_mapping).drop(['bins_5'], axis=1)

# Prediction
pred2 = fit3.predict(X_valid)
# We will plot the graph for 70 observations only
xp = np.linspace(valid_x.min(),valid_x.max()-1,70) 
bin_mapping = np.digitize(xp, bins) 
X_valid_2 = pd.get_dummies(bin_mapping) 
pred2 = fit3.predict(X_valid_2)
# Visualisation
fig, (ax1) = plt.subplots(1,1, figsize=(12,5))
fig.suptitle('Piecewise Constant', fontsize=14)

# Scatter plot with polynomial regression line
ax1.scatter(train_x, train_y, facecolor='None', edgecolor='k', alpha=0.3)
ax1.plot(xp, pred2, c='b')

ax1.set_xlabel('age')
ax1.set_ylabel('wage')
plt.show()



#support vector regression

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\Regression Model and Evaluation\data\svr\SVR_Data.csv')

X = data.iloc[:,0:1].values
y = data.iloc[:,1:].values

plt.scatter(X, y, color='darkviolet', label='data')
plt.show()

# Feature Scaling SVR model don't have inbuilt feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler(with_mean=True, with_std=True)
X= sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=0)

from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf')
svr_lin = SVR(kernel='linear')
svr_poly = SVR(kernel='poly', degree=4)
svr_rbf.fit(X_train, y_train.ravel())
svr_lin.fit(X_train, y_train.ravel())
svr_poly.fit(X_train, y_train.ravel())


y_pred_rbf = svr_rbf.predict(X_test)
y_pred_lin = svr_lin.predict(X_test)
y_pred_poly = svr_poly.predict(X_test)

from sklearn.metrics import r2_score

r2_score(y_test, y_pred_rbf)
r2_score(y_test, y_pred_lin)
r2_score(y_test, y_pred_poly)

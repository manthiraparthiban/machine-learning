import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\final_machine_learning\machine_learning_programs\Supervised_Algorithms\Regression\random_forest_regression\dataset.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

dataset.isnull().sum()

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
regressor.fit(x, y) 
regressor.score(x, y)

y_pred = regressor.predict([[6.5]])
print(y_pred)


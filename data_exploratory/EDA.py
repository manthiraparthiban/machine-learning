# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:18:12 2020

@author: Admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("50_Startups_EDA.csv")

dataset.shape
dataset.describe()


pd.set_option('display.max_rows',500)

dataset.info()

dataset.dtypes

dataset.nunique()

dataset.iloc[:,4].value_counts()

sns.countplot(x = 'State', data = dataset)


dataset[dataset.isnull()].sum()
sns.heatmap(dataset.isnull())
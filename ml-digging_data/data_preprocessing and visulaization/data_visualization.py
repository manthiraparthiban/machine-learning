# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 10:15:17 2020

@author: Admin
"""

#line chart
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\Iris.csv', sep=",")

dataset['Species'].unique()

#Create 3 DataFrame for each Species
setosa=dataset[dataset['Species']=='Iris-setosa']
versicolor =dataset[dataset['Species']=='Iris-versicolor']
virginica =dataset[dataset['Species']=='Iris-virginica']

plt.figure()
fig,ax=plt.subplots(1,2,figsize=(17, 9))
dataset.plot(x="SepalLengthCm",y="SepalWidthCm",kind="line",ax=ax[0],sharex=False,sharey=False,label="sepal")
dataset.plot(x="PetalLengthCm",y="PetalWidthCm",kind="line",ax=ax[1],sharex=False,sharey=False,label="petal")
ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()


#bar chart
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

#Reading IRIS Dataset in Pandas Dataframe
dataset = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\Iris.csv', sep=",")
dataset
#findout unique classification/type of iris flower.
dataset['Species'].unique()

plt.figure()
fig,ax=plt.subplots(1,2,figsize=(17, 9))
dataset.plot(x="SepalLengthCm",y="SepalWidthCm",kind="bar",ax=ax[0],sharex=False,sharey=False,label="sepal")
dataset.plot(x="PetalLengthCm",y="PetalWidthCm",kind="bar",ax=ax[1],sharex=False,sharey=False,label="petal")
ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()

import seaborn as sns
sns.barplot(dataset["Species"],dataset["SepalLengthCm"])
plt.title("Barplot for Sepal Length")
sns.barplot(dataset["Species"],dataset["SepalWidthCm"])
plt.title("Barplot for Sepal Width")
sns.barplot(dataset["Species"],dataset["PetalLengthCm"])
plt.title("Barplot for Petal Length")
sns.barplot(dataset["Species"],dataset["PetalWidthCm"])
plt.title("Barplot for Petal Width")


#histogram 

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

#Reading IRIS Dataset in Pandas Dataframe
dataset = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\Iris.csv', sep=",")

#findout unique classification/type of iris flower.
dataset['Species'].unique()

plt.figure()
fig,ax=plt.subplots(1,2,figsize=(17, 9))
dataset.plot(x="SepalLengthCm",y="SepalWidthCm",kind="hist",ax=ax[0],sharex=False,sharey=False,label="sepal")
dataset.plot(x="PetalLengthCm",y="PetalWidthCm",kind="hist",ax=ax[1],sharex=False,sharey=False,label="petal")
ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()

#Create 3 DataFrame for each Species
setosa=dataset[dataset['Species']=='Iris-setosa']
versicolor =dataset[dataset['Species']=='Iris-versicolor']
virginica =dataset[dataset['Species']=='Iris-virginica']

plt.figure()
fig,ax=plt.subplots(1,2,figsize=(21, 10))

setosa.plot(x="SepalLengthCm", y="SepalWidthCm", kind="hist",ax=ax[0],label='setosa',color='r')
versicolor.plot(x="SepalLengthCm",y="SepalWidthCm",kind="hist",ax=ax[0],label='versicolor',color='b')
virginica.plot(x="SepalLengthCm", y="SepalWidthCm", kind="hist", ax=ax[0], label='virginica', color='g')

setosa.plot(x="PetalLengthCm", y="PetalWidthCm", kind="hist",ax=ax[1],label='setosa',color='r')
versicolor.plot(x="PetalLengthCm",y="PetalWidthCm",kind="hist",ax=ax[1],label='versicolor',color='b')
virginica.plot(x="PetalLengthCm", y="PetalWidthCm", kind="hist", ax=ax[1], label='virginica', color='g')

ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()


#Box plot
# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns

#Reading IRIS Dataset in Pandas Dataframe
dataset = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\Iris.csv', sep=",")

#findout unique classification/type of iris flower.
dataset['Species'].unique()

sns.boxplot(x="Species", y="PetalLengthCm", data=dataset )
plt.show()


#scatter plot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

dataset = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\Iris.csv', sep=",")

dataset['Species'].unique()

plt.figure()
fig,ax=plt.subplots(1,2,figsize=(17, 9))
dataset.plot(x="SepalLengthCm",y="SepalWidthCm",kind="scatter",ax=ax[0],sharex=False,sharey=False,label="sepal")
dataset.plot(x="PetalLengthCm",y="PetalWidthCm",kind="scatter",ax=ax[1],sharex=False,sharey=False,label="petal")
ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()

setosa=dataset[dataset['Species']=='Iris-setosa']
versicolor =dataset[dataset['Species']=='Iris-versicolor']
virginica =dataset[dataset['Species']=='Iris-virginica']

plt.figure()
fig,ax=plt.subplots(1,2,figsize=(21, 10))

setosa.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter",ax=ax[0],label='setosa',color='r')
versicolor.plot(x="SepalLengthCm",y="SepalWidthCm",kind="scatter",ax=ax[0],label='versicolor',color='b')
virginica.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter", ax=ax[0], label='virginica', color='g')

setosa.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter",ax=ax[1],label='setosa',color='r')
versicolor.plot(x="PetalLengthCm",y="PetalWidthCm",kind="scatter",ax=ax[1],label='versicolor',color='b')
virginica.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter", ax=ax[1], label='virginica', color='g')

ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()


# Violin plot

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns

#Reading IRIS Dataset in Pandas Dataframe
dataset = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\Iris.csv', sep=",")

#findout unique classification/type of iris flower.
dataset['Species'].unique()

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=dataset)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=dataset)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=dataset)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=dataset)

#heat map
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns

dataset = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\Iris.csv', sep=",")

dataset['Species'].unique()
fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.heatmap(dataset.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
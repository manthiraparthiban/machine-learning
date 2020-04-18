import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv(r'D:\final_machine_learning\machine_learning_programs\Unsupervised_Algorithms\PCA\data.csv', encoding= 'unicode_escape')
X = df[list(df.columns)[:-1]]
y = df['Flower']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)  

pca = PCA(n_components=2)
x = StandardScaler().fit_transform(X_train)
new_x = pd.DataFrame(data = pca.fit_transform(x), columns = ['x1', 'x2'])

df2 = pd.concat([new_x, df[['Flower']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('x1', fontsize = 15)
ax.set_ylabel('x2', fontsize = 15)
ax.set_title('2 Components', fontsize = 20)
for i, j in zip(['Rose', 'Jasmin', 'Lotus'],['g', 'b', 'r']):
    print(i,j)
    ax.scatter(df2.loc[df2['Flower'] == i, 'x1'], df2.loc[df2['Flower'] == i, 'x2'], c = j)
ax.legend(['Rose', 'Jasmin', 'Lotus'])
ax.grid()
plt.show()

print (pca.explained_variance_ratio_)

print (df.columns)
print (df2.columns)
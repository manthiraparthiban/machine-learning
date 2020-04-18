import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],
             [3,5],
             [2,4],
             [2,5],
             [7,9],
             [8,4]])

plt.scatter(X[:,0],X[:,1], s=100)  

clf = KMeans(n_clusters = 2)
clf.fit(X)

centeroids = clf.cluster_centers_
print(centeroids)

labels = clf.labels_
print(labels)

colors = 10*["g.","r.","b.","c,","k."]

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1], colors[labels[i]], markersize = 25)
   
plt.scatter(centeroids[:,0], centeroids[:,1], marker = 'x', s = 150, linewidth=5)
plt.show()

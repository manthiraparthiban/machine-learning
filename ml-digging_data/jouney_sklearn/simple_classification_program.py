from sklearn import datasets
iris = datasets.load_iris()

from sklearn import svm
clf = svm.SVC()

X, y = iris.data, iris.target

clf.fit(X, y) 

clf.predict(X[0:2])
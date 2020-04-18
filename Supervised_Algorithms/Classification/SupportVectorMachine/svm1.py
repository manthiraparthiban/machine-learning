
#non linear dataaset
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.linear_model.logistic import LogisticRegression

df = pd.read_csv(r'D:\final_machine_learning\machine_learning_programs\Supervised_Algorithms\Classification\SupportVectorMachine\svm_dataset.csv', encoding= 'unicode_escape')
X = df[list(df.columns)[:-1]]
y = df['Flower']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

logistic = LogisticRegression()
logistic.fit(X_train, y_train)  
y_pred = logistic.predict(X_test) 
print ('Accuracy-logistic:', accuracy_score(y_test, y_pred))

gaussian = SVC(kernel='rbf') 
gaussian.fit(X_train, y_train)  
y_pred = gaussian.predict(X_test) 
print ('Accuracy-svm:', accuracy_score(y_test, y_pred))


#linear dataset
  
def classifier():
    xx = np.linspace(1,10)
    yy = -regressor.coef_[0][0] / regressor.coef_[0][1] * xx - regressor.intercept_[0] / regressor.coef_[0][1]
    plt.plot(xx, yy)
    plt.scatter(x1,x2)
    plt.show()

x1 = [2,6,3,9,4,10]
x2 = [3,9,3,10,2,13]

X = np.array([[2,3],[6,9],[3,3],[9,10],[4,2],[10,13]])
y = [0,1,0,1,0,1]

regressor = LogisticRegression()
regressor.fit(X,y)
classifier()

regressor = svm.SVC(kernel='linear',C = 1.0)
regressor.fit(X,y)
classifier()

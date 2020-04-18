from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
  
df = pd.read_csv(r'D:\final_machine_learning\machine_learning_programs\Supervised_Algorithms\Classification\LogisticRegression\multi_class_classification\dataset.csv')
X = df[list(df.columns)[:-1]]
y = df['Flower']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)  

tree = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
tree_predictions = tree.predict(X_test)
data = [[1,2,3,4]]  #checking
tree.predict(data)   
print (tree.score(X_test, y_test))
print (confusion_matrix(y_test, tree_predictions))
print (precision_recall_fscore_support(y_test, tree_predictions))

svc = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svc_predictions = svc.predict(X_test)  
print (svc.score(X_test, y_test))
print (confusion_matrix(y_test, svc_predictions))
print (precision_recall_fscore_support(y_test, svc_predictions))

knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)  
knn_predictions = knn.predict(X_test)  
knn.score(X_test,knn_predictions)
print (knn.score(X_test, y_test)) 
print (confusion_matrix(y_test, knn_predictions))
print (precision_recall_fscore_support(y_test, knn_predictions))

gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
print (gnb.score(X_test, y_test))
print (confusion_matrix(y_test, gnb_predictions))
print (precision_recall_fscore_support(y_test, gnb_predictions))
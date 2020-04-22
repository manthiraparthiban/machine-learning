#Logistic Regression

import pandas as pd
pima = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\Classification Model\data\Logistic-Regression\pima-indians-diabetes-database\diabetes.csv')

pima.isnull().sum()

X = pima[['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']]
y = pima[["Outcome"]]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

logreg.score(X_train,y_train)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

#### Visualizing Confusion Matrix using Heatmap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')  #y_test
plt.xlabel('Predicted label')  #y_pred


#### Confusion Matrix Evaluation Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,'%')
print("Precision:",metrics.precision_score(y_test, y_pred)*100,'%')
print("Recall:",metrics.recall_score(y_test, y_pred)*100,'%')


#roc curve 
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------

#SVM

import pandas as pd
bankdata = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\Classification Model\data\SVM\bill_authentication.csv')

bankdata.isnull().sum()

X = bankdata.drop('Class', axis = 1)
y = bankdata[["Class"]]

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 


from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)  
y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix 
confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))  

#-----------------------------------------------------------------------------------------------------------------------------------------------------

#kernal svm

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)

X = irisdata.drop('Class', axis=1)  
y = irisdata['Class']


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

##### 1. Polynomial Kernel

from sklearn.svm import SVC  
svclassifier = SVC(kernel='poly', degree=8)  
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))   

print(classification_report(y_test, y_pred))

##### 2. Gaussian Kernel

from sklearn.svm import SVC  
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred))  

##### 3. Sigmoid Kernel

from sklearn.svm import SVC  
svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Random Forest Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\Classification Model\data\Random-Forest\breast-cancer-wisconsin.csv')

data.isnull().sum()
data.BareNuclei.unique()

data = data[data.BareNuclei != '?']

X = data.drop('CancerType', axis=1)  
y = data['CancerType']

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred))  


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# SGD Classifier


from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame

X, y = make_blobs(n_samples=100, centers=3, n_features=2)

# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

features=df.drop('label',axis=1)
target=df.label

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.20)

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train) 

y_pred=clf.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred)) 

print(classification_report(y_test, y_pred))


#parameter tuning
clf = SGDClassifier(penalty="l2", max_iter=5)
clf.fit(X_train, y_train) 
y_pred=clf.predict(X_test) 
print(classification_report(y_test, y_pred)) 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Naive Bayes

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
sns.set(style="white", color_codes=True)

train_df = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\Classification Model\data\Naive-bayes\train.csv', dtype={"Age": np.float64})
test_df = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\Classification Model\data\Naive-bayes\test.csv', dtype={"Age": np.float64})

train_df.info()

test_df.info()

train_df = train_df.drop(['Ticket', 'Cabin','Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin','Name', 'PassengerId'], axis=1)

from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(test_df.loc[:, ['Fare']])
test_df.loc[:, ['Fare']]=imputer.transform(test_df.loc[:, ['Fare']])
train_df = train_df[pd.notnull(train_df['Embarked'])]

full_data=[train_df,test_df]

for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train_df['CategoricalAge'] = pd.cut(train_df['Age'], 5)

print (train_df[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


for dataset in full_data:
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']  

train_df.isnull().sum()

test_df.isnull().sum()

# Encoding the catagorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_X=LabelEncoder()
train_df.loc[:, ['Sex']]=labelEncoder_X.fit_transform(train_df.loc[:, ['Sex']])
test_df.loc[:, ['Sex']]=labelEncoder_X.fit_transform(test_df.loc[:, ['Sex']])

train_df.loc[:, ['Embarked']]=labelEncoder_X.fit_transform(train_df.loc[:, ['Embarked']])
test_df.loc[:, ['Embarked']]=labelEncoder_X.fit_transform(test_df.loc[:, ['Embarked']])


# Get one hot encoding of column embarked for training,testing data
trainEmbarkedencoder = pd.get_dummies(train_df['Embarked'])
testEmbarkedencoder= pd.get_dummies(test_df['Embarked'])
# Drop columns embarked as they are now encoded
train_df = train_df.drop('Embarked',axis = 1)
test_df = test_df.drop('Embarked',axis = 1)
# Join the encoded dataframe
train_df = train_df.join(trainEmbarkedencoder)
test_df = test_df.join(testEmbarkedencoder)
train_df 


from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
train_OneHotEncoded = onehot_encoder.fit_transform(train_df.loc[:, ['Sex']])
test_OneHotEncoded = onehot_encoder.fit_transform(test_df.loc[:, ['Sex']])
train_df["Male"] = train_OneHotEncoded[:,0]
train_df["Felame"] = train_OneHotEncoded[:,1]
test_df["Male"] = test_OneHotEncoded[:,0]
test_df["Felame"] = test_OneHotEncoded[:,1]

X = train_df.drop(['Survived','CategoricalAge'], axis=1)  
y = train_df['Survived']


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred))  


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#KNN Algorithm

import pandas as pd
df =pd.read_csv(r"D:\final_machine_learning\machine-learning\ml-digging_data\Classification Model\data\KNN\pima-indians-diabetes-database\diabetes.csv")
X = df.drop(columns=['Outcome'])
y = df['Outcome'].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1, stratify=y)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train,y_train)

knn.predict(X_test)[0:5]

knn.score(X_test, y_test)

### k-Fold Cross-Validation


from sklearn.model_selection import cross_val_score
import numpy as np

#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)

#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=5)

#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

### Hypertuning model parameters using GridSearchCV

from sklearn.model_selection import GridSearchCV

#create new a knn model
knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}

#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

#fit model to data
knn_gscv.fit(X, y)

#check top performing n_neighbors value
knn_gscv.best_params_

#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
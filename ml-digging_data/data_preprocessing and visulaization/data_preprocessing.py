import pandas as pd
import numpy as np

train=pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\train.csv')
test=pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\test.csv')

print(train)
print(test)
print(train.describe())

train.isnull()
test.isnull()

train.isnull().sum()
test.isnull().sum()


from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

imputer=imputer.fit(test.loc[:, ['Fare']])
# fit() is used to fit the data 
test.loc[:, ['Fare']]=imputer.transform(test.loc[:, ['Fare']])

test.isnull().sum()

# Removing the data row wise
train = train[pd.notnull(train['Embarked'])]

train.isnull().sum()

#combine train and test data
full_data=[train,test]

for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)

print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

for dataset in full_data:
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    
train = train.drop(['Cabin','Name','Ticket'], axis = 1)
test = test.drop(['Cabin','Name','Ticket'], axis = 1)

print(train['Sex'].unique())
print(test['Sex'].unique())
print(train['Embarked'].unique())
print(test['Embarked'].unique())

# Encoding the catagorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_X=LabelEncoder()
train.loc[:, ['Sex']]=labelEncoder_X.fit_transform(train.loc[:, ['Sex']])
test.loc[:, ['Sex']]=labelEncoder_X.fit_transform(test.loc[:, ['Sex']])


train.loc[:, ['Embarked']]=labelEncoder_X.fit_transform(train.loc[:, ['Embarked']])
test.loc[:, ['Embarked']]=labelEncoder_X.fit_transform(test.loc[:, ['Embarked']])

train

# One hot Encoding using Pandas
# Loading the data
train_one=pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\train.csv')
test_one=pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\test.csv')


# Get one hot encoding of column sex , embarked for training,testing data
trainsexencoder = pd.get_dummies(train_one['Sex'])
testsexencoder= pd.get_dummies(test_one['Sex'])
trainEmbarkedencoder = pd.get_dummies(train_one['Embarked'])
testEmbarkedencoder= pd.get_dummies(test_one['Embarked'])
# Drop columns sex , embarked as they are now encoded
train_one = train_one.drop('Sex',axis = 1)
train_one = train_one.drop('Embarked',axis = 1)
test_one = test_one.drop('Sex',axis = 1)
test_one = test_one.drop('Embarked',axis = 1)
# Join the encoded dataframe
train_one = train_one.join(trainsexencoder)
test_one = test_one.join(testsexencoder)
train_one = train_one.join(trainEmbarkedencoder)
test_one = test_one.join(testEmbarkedencoder)
train_one


from sklearn.model_selection import train_test_split
# dividing train data into dependent and independent variable
y=train['Survived']
train=train.drop(['Survived'], axis = 1)
X=train
X_train,X_test,y_Train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)
X_train


import matplotlib.pyplot as plt
X_train.hist(column='Fare')

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
std_scaleTrain=sc_X.fit(X[['Age','Fare']])
X[['Age','Fare']]=std_scaleTrain.transform(X[['Age','Fare']])
X


#extract data from json
import json
data = json.loads(open(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\example.json').read())
df = pd.DataFrame(data)
df.head(10)
# Loading a text file
data = pd.read_csv(r'D:\final_machine_learning\machine-learning\ml-digging_data\data_preprocessing and visulaization\data\text.txt', sep=",")
data
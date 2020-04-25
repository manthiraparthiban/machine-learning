import pandas as pd
import numpy as np
data=pd.read_csv(r"D:\final_machine_learning\machine-learning\ml-digging_data\Natural Language Processing\spam.csv",encoding='cp1252')
data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

data = data.rename(columns={"v1":"label", "v2":"text"})

import nltk
nltk.download()

data

# Importing LabelEncoder()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data.label = le.fit_transform(data.label)

import re # Importing Regular Expression
X = []
for i in range(0, 5572):

    # Applying Regular Expression
    
    '''
    Replace email addresses with 'emailaddr'
    Replace URLs with 'httpaddr'
    Replace money symbols with 'moneysymb'
    Replace phone numbers with 'phonenumbr'
    Replace numbers with 'numbr'
    Remove all 'punctuations'
    '''
    msg = data['text'][i]
    msg = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', data['text'][i])
    msg = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', data['text'][i])
    msg = re.sub('£|\$', 'moneysymb', data['text'][i])
    msg=re.sub(r'\..\w*', '.', data['text'][i])
    msg = re.sub('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', data['text'][i])
    msg = re.sub(r'\d+', '', data['text'][i])

    ''' Remove all punctuations '''
    msg = re.sub('[^\w\d\s]', ' ', data['text'][i])
    
    if i<2:
        print("\t\t\t\t MESSAGE ", i)
    
    if i<2:
        print("\n After Regular Expression - Message ", i, " : ", msg)
    
    # preparing Messages with Remaining Tokens
    msg = ''.join(msg)
    if i<2:
        print("\n Final Prepared - Message ", i, " : ", msg, "\n\n")
    
    # Preparing WordVector Corpus
    X.append(msg)
    
data['text']=pd.DataFrame(X)

data['text']= data['text'].astype(str).apply(lambda x: " ".join(x.lower() for x in x.split()))

# Import stop words
from nltk.corpus import stopwords

stop = stopwords.words('english')

data['text'] = data['text'].astype(str).apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#common word remove
freq = pd.Series(' '.join(data['text']).split()).value_counts()[:10]
freq

req = list(freq.index)
data['text']= data['text'].astype(str).apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#rare words removal

freq = pd.Series(' '.join(data['text']).split()).value_counts()[-10:]
freq

freq = list(freq.index)
data['text'] = data['text'].astype(str).apply(lambda x: " ".join(x for x in x.split() if x not in freq))


from textblob import TextBlob
data['text'] = data['text'].apply(lambda x: "".join(str(TextBlob(x).correct())))

#tokenization
#word
import time
start = time.time()
data["unigrams"] = data["text"].apply(nltk.word_tokenize)
print ("series.apply", (time.time() - start))

#sentence
import time
start = time.time()
data["unigramSent"] = data["text"].apply(nltk.sent_tokenize)
print ("series.apply", (time.time() - start))


from nltk.stem import PorterStemmer
st = PorterStemmer()
data['text'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


nltk.download('wordnet')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

data['text'] = data.text.apply(lemmatize_text)


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(lowercase=False)

VX = vectorizer.fit_transform(data['text'].astype(str))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(VX, data['label'], random_state=0)



# Training the model
# MultinomialNB or Multinomial Naive Bayes Classifier

prediction = dict()
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
prediction["MultinomialNB"] = model.predict(X_test)
print('Accuracy: %.2f%%' % (accuracy_score(y_test, prediction["MultinomialNB"]) * 100))

# Logistic Regression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
prediction["Logistic"] = model.predict(X_test)
accuracy_score(y_test,prediction["Logistic"])*100


#### k-NN classifier

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
prediction["knn"] = model.predict(X_test)
accuracy_score(y_test,prediction["knn"])


### Ensemble classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
prediction["random_forest"] = model.predict(X_test)
accuracy_score(y_test,prediction["random_forest"])


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train, y_train)
prediction["adaboost"] = model.predict(X_test)
accuracy_score(y_test,prediction["adaboost"])

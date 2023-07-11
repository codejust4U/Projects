import numpy as np
import pandas as pd
import os
import re
import nltk #natural language tool kit
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv(r'D:\copy of htdocs\practice\Python\200days\Day121 simple project #2\archive\IMDB Dataset.csv')


dataset['review'][0]

dataset = dataset.sample(10000)

dataset.shape

dataset.info()

dataset['sentiment'].replace({'positive':1,'negative':0},inplace=True)

clean = re.compile('<.*?>')
re.sub(clean, '',dataset.iloc[2].review)

def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '',text)

dataset['review']=dataset['review'].apply(clean_html)

def con_lower(text):
    return text.lower()

dataset['review']=dataset['review'].apply(con_lower)

def rem_special(text):
    x=''
    for i in text:
        if i.isalnum():
            x+=i
        else:
            x=x+' '
    return x

rem_special('Do not @wa#tch this movie, go see something else ... I was very disappointed, I cannot rate this movie any better than 3.')

dataset['review'] = dataset['review'].apply(rem_special)

stopwords.words('english')

def rem_stopwords(text):
    x=[]
    for i in text.split():
        if i not in  stopwords.words('english'):
            x.append(i)
    y=x[:]
    x.clear()
    return y

dataset['review']=dataset['review'].apply(rem_stopwords)


ps = PorterStemmer()

y = []
def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z

stem_words(['I','loved','loving','it'])

dataset['review'] = dataset['review'].apply(stem_words)

def join_back(list_input):
    return " ".join(list_input)


dataset['review']=dataset['review'].apply(join_back)

dataset['review']

x=dataset.iloc[:,0:1].values
x.shape


cv = CountVectorizer(max_features=1000)         #for used 1000 words

x=cv.fit_transform(dataset['review']).toarray()
x.shape

x[0].max()
x[0].mean()

y=dataset.iloc[:,-1].values
y.shape

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

X_train.shape

y_test.shape

clf1 = GaussianNB()
clf2 = MultinomialNB()
clf3 = BernoulliNB()

print(clf1.fit(X_train,y_train))
print(clf2.fit(X_train,y_train))
print(clf3.fit(X_train,y_train))

y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)
y_pred3 = clf3.predict(X_test)

y_test.shape

y_pred1.shape


print("Guassian : ",accuracy_score(y_test,y_pred1))
print("Multinomial : ",accuracy_score(y_test,y_pred2))
print("Bernouli : ",accuracy_score(y_test,y_pred3))
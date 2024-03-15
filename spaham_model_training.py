#!/usr/bin/env python
# coding: utf-8

# # Import libreries

# In[8]:
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import random
import pickle

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# # Collecting Data
# In[9]:

ds=pd.read_csv("ds_train.csv",index_col=None)
ds.fillna('', inplace=True)

# # Train & Test 

# In[10]:
X=ds.iloc[:,:-1]                            #saperating dependent and independent coulmns
y=ds.iloc[:,-1]

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
# Transform the text into vectors
X = vectorizer.fit_transform(ds["text"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)

# # Models Declaration

# In[11]:


rnf_clf=RandomForestClassifier(n_estimators=200,min_samples_split=10,random_state=1,n_jobs=4)
knn_clf=knn_classifier = KNeighborsClassifier(n_neighbors=3)
desT=DecisionTreeClassifier()
log_res=LogisticRegression(n_jobs=4)
svm_clf = SVC(kernel='linear')


# # Random forest Training

# In[12]:

rnf_clf.fit(X_train,y_train)
knn_clf.fit(X_train,y_train)
desT.fit(X_train,y_train)
log_res.fit(X_train,y_train)
svm_clf.fit(X_train,y_train)

# # Saving pickles

# In[17]:


with open("rf.pkl", "wb") as f:                                                           # random forest model
    pickle.dump(rnf_clf, f)

with open("knn.pkl", "wb") as f:                                                           # KNN model
    pickle.dump(knn_clf, f)
    
with open("destree.pkl", "wb") as f:                                                       # Decision Tree model
    pickle.dump(desT, f)
    
with open("log_res.pkl", "wb") as f:                                                       # Logistric model
    pickle.dump(log_res, f)
    
with open("svm.pkl", "wb") as f:                                                           # SVM model
    pickle.dump(svm_clf, f)
    
    
with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:                                # vertorizer
    pickle.dump(vectorizer, vectorizer_file)                                             

with open('df_X_test.pkl', 'wb') as file:                                                  # X test dataframe
    pickle.dump(X_test, file)
    
with open('df_y_test.pkl', 'wb') as file:                                                  # y test dataframe
    pickle.dump(y_test, file)
    
with open('ds.pkl', 'wb') as file:                                                         # dataframe
    pickle.dump(ds, file)

# # Random forest prediction

# In[15]:


# accuracy = sklearn.metrics.accuracy_score(y_test,predictions)
# precision = sklearn.metrics.precision_score(y_test,predictions)

# print('Accuracy:', accuracy)
# print('Precision:', precision)


# # KNN classifier
# # Decision tree classifier

# # Logistic Regression

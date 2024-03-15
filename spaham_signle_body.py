#!/usr/bin/env python
# coding: utf-8

# # Import libreries

# In[ ]:


import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# # Create a file object for the saved model

# In[ ]:


with open("rf.pkl", "rb") as f:
    rf_cls = pickle.load(f)
    
with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    tfidf_vectorizer = pickle.load(vec_file)


# # Email Body Input 

# In[ ]:


msg="We are writing to inform you that your Amazon account has been suspended due to suspected fraudulent activity.To reactivate your account, please click on the following link If you did not make any fraudulent purchases, please contact us,Thank you for your cooperation."
msg2="Subject: soul mateone of your buddies hooked you up on a date with another buddy .your invitation : a free dating web site created by women no more invitation :"
msg3="LinkedIn  updates-noreply@linkedin.com Akshat Shrivastava and others share their thoughts on LinkedIn"
user_input_transformed = tfidf_vectorizer.transform([msg3])


# # Use the model to make predictions
# 

# In[ ]:


print(type(X))
print(X)
pred = rf_cls.predict(user_input_transformed)


# In[ ]:


pred[0]


# In[ ]:





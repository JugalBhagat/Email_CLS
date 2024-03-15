def predict2(uploaded_dataframe,model):
    import sys
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split 
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    import pickle
    import pandas as pd
    import sklearn.metrics
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt

    # if len(sys.argv) != 4:
    #     print("Usage: python spaham_uploaded_classify.py selected_model uploaded_file dataframe")
    #     sys.exit(1)

    # Retrieve the command-line arguments
    # model = sys.argv[1]
    # uploaded_file = sys.argv[2]

    #ds= pd.read_csv(uploaded_dataframe)
    ds=uploaded_dataframe
    ds.fillna('', inplace=True)

    # ds=pd.read_csv("Spam_ham.csv")
    # ds.fillna('', inplace=True)
    # ds=ds.iloc[:,:-1]

    #model=selected_model
    print(model)
    if(model=="knn"):
        with open("knn.pkl", "rb") as f:                                                     # KNN model
            knn_cls = pickle.load(f) 
        model=knn_cls
    elif(model=="random_forest"):
        with open("rf.pkl", "rb") as f:                                                     # Random forest model
            rf = pickle.load(f) 
            print("rf")
        model=rf
    elif(model=="svm"):
        with open("svm.pkl", "rb") as f:                                                     # Random forest model
            svm = pickle.load(f) 
            print("svm")
        model=svm
    elif(model=="log_res"):
        with open("log_res.pkl", "rb") as f:                                                 # Logistical Regression model
            log_res= pickle.load(f) 
            print("log_res")
        model=log_res
    elif(model=="dest"):
        with open("destree.pkl", "rb") as f:                                                 # Decision tree model
            dest = pickle.load(f) 
            print("dest")
        model=dest
    elif(model=="dest"):
        print("Not available")
        return 0,0
        # with open("destree.pkl", "rb") as f:                                               # Artifical neural netwrok
        #     dest = pickle.load(f) 
        #     print("dest")
        # model=dest
        
    # with open("rf.pkl", "rb") as f:                                                    
    #     rf_cls = pickle.load(f) 
    # model=rf_cls

    with open("tfidf_vectorizer.pkl", "rb") as vec_file:
        tfidf_vectorizer = pickle.load(vec_file)

    X = tfidf_vectorizer.transform(ds["text"])


    # # Make prediction

    predictions = model.predict(X)
    ds['label_num'] = predictions
    print(predictions)

    # # Diffrentiate Datasets

    # Create a dataset where 'new_column' is equal to 1
    ds_spam = ds[ds['label_num'] == '1']

    # Create a dataset where 'new_column' is equal to 0
    ds_ham = ds[ds['label_num'] == '0']

    # Display the two datasets

    ds_ham.to_csv("ham_email.csv")
    ds_spam.to_csv("spam_email.csv")
    return ds_ham,ds_spam
    #return 0,0

def temp():
    print("helo")
    return "helo"
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer# importing TF-IDF library
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_score #importing cross validation library
wikipedia_300 = pd.read_csv("C:/Program Files/Weka-3-8/data/wikipedia_300.csv")
wikipedia_300_x=wikipedia_300["Text"]
wikipedia_300_y=wikipedia_300["Category"]
vectorizer = TfidfVectorizer()#Term Frequency-Inverse Document Frequency
TF_IDF = vectorizer.fit_transform(wikipedia_300_x)
MNB = MultinomialNB()
accuracy = cross_val_score(MNB,TF_IDF,wikipedia_300_y,scoring='accuracy', cv=10)#applying 10 Cross Validation
print('Accuracy with 10 CV (TF-IDF)=',accuracy.mean() * 100)
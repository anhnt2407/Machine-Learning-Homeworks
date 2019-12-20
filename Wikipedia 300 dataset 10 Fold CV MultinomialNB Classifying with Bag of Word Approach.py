import numpy as np
import pandas as pd # library for loading dataset
wikipedia_300 = pd.read_csv("C:/Program Files/Weka-3-8/data/wikipedia_300.csv")#loading dataset
wikipedia_300_x=wikipedia_300["Text"]
wikipedia_300_y=wikipedia_300["Category"]
from sklearn.feature_extraction.text import CountVectorizer#importing bag-of-words library
cv = CountVectorizer()
bag_of_words = cv.fit_transform(wikipedia_300_x)
from sklearn.model_selection import cross_val_score#importing Cross Validation library
from sklearn.naive_bayes import MultinomialNB#importing model
MNB = MultinomialNB()#multinominal naive bayes model
accuracy = cross_val_score(MNB,bag_of_words,wikipedia_300_y,scoring='accuracy', cv=10)
print('Accuracy with 10 C.V.=',accuracy.mean() * 100) #printing Accuracy
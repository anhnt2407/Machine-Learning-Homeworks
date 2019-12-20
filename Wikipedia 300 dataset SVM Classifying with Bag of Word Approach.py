import numpy as np
import pandas as pd # library for loading dataset
wikipedia_300 = pd.read_csv("C:/Program Files/Weka-3-8/data/wikipedia_300.csv")#loading dataset
wikipedia_300_x=wikipedia_300["Text"]
wikipedia_300_y=wikipedia_300["Category"]
from sklearn.feature_extraction.text import CountVectorizer#importing bag-of-wordslibrary
cv = CountVectorizer()
bag_of_words = cv.fit_transform(wikipedia_300_x)
from sklearn.model_selection import train_test_split#Train-Test split
x_train, x_test, y_train, y_test = train_test_split(bag_of_words, wikipedia_300_y, test_size=0.2, random_state=4)
from sklearn.svm import LinearSVC#importing model
SVC = LinearSVC()#multinominal naivbe bayes model
SVC.fit(x_train,y_train)
y_pred = SVC.predict(x_test)
from sklearn import metrics
print('Accuracy= ' ,metrics.accuracy_score(y_test,y_pred))#Finding Accuracy
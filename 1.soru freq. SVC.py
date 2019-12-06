import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
wikipedia_300 = pd.read_csv("C:/Program Files/Weka-3-8/data/wikipedia_300.csv")
wikipedia_300_x=wikipedia_300["Text"]
wikipedia_300_y=wikipedia_300["Category"]
vectorizer = TfidfVectorizer()
TF_IDF = vectorizer.fit_transform(wikipedia_300_x)
x_train, x_test, y_train, y_test = train_test_split(TF_IDF, wikipedia_300_y, test_size=0.2, random_state=4)
SVC = LinearSVC()
SVC.fit(x_train,y_train)
y_pred = SVC.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))
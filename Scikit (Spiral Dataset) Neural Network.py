import numpy as np
import pandas as pd # library for loading dataset
import seaborn as sns #for plotting graphs
import matplotlib.pyplot as plt
spiral_dataset= pd.read_csv("C:/Program Files/Weka-3-8/data/spiral.csv").values
# Assign data to X variable
X = spiral_dataset[:,0:2]
# Assign data to y variable
y = spiral_dataset[:,2]
from sklearn.model_selection import train_test_split # Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=60)
from sklearn.preprocessing import StandardScaler #Before making actual predictions, 
#it is always a good practice to scale the features so that all of them can be uniformly evaluated.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier # Model Training and Predictions
clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=1000) 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix #evaluating the the Algorithm
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
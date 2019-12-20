import numpy as np
import pandas as pd # library for loading dataset
diabetes_dataset= pd.read_csv("C:/Program Files/Weka-3-8/data/diabetes.csv") #loading dataset to pandas dataframe
diabetes_dataset.head() # information about data set
# Assign data from first four columns to X variable
X = diabetes_dataset.iloc[:, 0:4]
# Assign data from first fifth columns to y variable
y = diabetes_dataset.select_dtypes(include=[object])
from sklearn.model_selection import train_test_split # Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
from sklearn.preprocessing import StandardScaler # Before making actual predictions 
#it is always a good practice to scale the features so that all of them can be uniformly evaluated
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier # Applying Model
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix #evaluating the performance of the model.
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
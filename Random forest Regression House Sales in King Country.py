#Import pandas library and read the CSV file.
import pandas as pd
import numpy as np
dataset = pd.read_csv("C:/Program Files/Weka-3-8/data/projeturkey.csv") # loading data
y = dataset['AverageTemperature']
X = dataset[['dt']]
from sklearn.model_selection import train_test_split #train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestRegressor # import random forest regressor 
regressor = RandomForestRegressor(n_estimators=1000, random_state=0)#The n_estimators parameter defines the number of trees in the random forest.
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn import metrics #Evaluating the random forest regressor model algorithm using the error metrics.
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
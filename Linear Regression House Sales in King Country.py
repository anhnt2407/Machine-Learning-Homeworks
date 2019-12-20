#importing necessary library for loading data,plotting
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Program Files/Weka-3-8/data/projegreenland.csv")
#importing library for linear regression models
from sklearn.model_selection import train_test_split #training the data, and the rest for testing the data.
from sklearn import linear_model #linear models
#In KNeighborsRegressor the target is predicted by local interpolation of the targets associated of the nearest neighbors in the training set.
from sklearn.neighbors import KNeighborsRegressor
#PolynomialFeatures generates a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree.
from sklearn.preprocessing import PolynomialFeatures 
from sklearn import metrics #The metrics is imported as the metric module implements functions assessing prediction error for specific purposes.
from mpl_toolkits.mplot3d import Axes3D
train_data,test_data=train_test_split(df,train_size=0.8,random_state=3) #Training and Testing data %80-%20
reg=linear_model.LinearRegression() # Applying simple linear regression model for prince vs living space
x_train=np.array(train_data['dt']).reshape(-1,1)
y_train=np.array(train_data['AverageTemperature']).reshape(-1,1)
reg.fit(x_train,y_train)
# Code for finding Sqaured mean error,R squared training,R sqaured testing
x_test=np.array(test_data['dt']).reshape(-1,1)
y_test=np.array(test_data['AverageTemperature']).reshape(-1,1)
pred=reg.predict(x_test)
print('linear model')
mean_squared_error=metrics.mean_squared_error(y_test,pred)
print('Sqaured mean error', round(np.sqrt(mean_squared_error),2))
print('R squared training',round(reg.score(x_train,y_train),3))
print('R sqaured testing',round(reg.score(x_test,y_test),3) )
print('intercept',reg.intercept_)
print('coefficient',reg.coef_)
# Code for linear regression model graph 
ax = plt.subplots(figsize= (12, 10))
plt.scatter(x_test, y_test, color= 'darkgreen', label = 'data')
plt.plot(x_test, reg.predict(x_test), color='red', label= ' Predicted Regression line')
plt.xlabel('Time')
plt.ylabel('AverageTemperature')
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

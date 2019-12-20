# import numpy package for arrays and stuff 
import numpy as np  
  
# import matplotlib.pyplot for plotting our result 
import matplotlib.pyplot as plt 
  
# import pandas for importing csv files  
import pandas as pd  

dataset = pd.read_csv("C:/Program Files/Weka-3-8/data/kc_house_data.csv")
# select all rows by : and column 1 
# by 1:2 representing features 
X = dataset[:, 1:2].astype(int)  
  
# print X 
print(X) 

# select all rows by : and column 2 
# by 2 to Y representing labels 
y = dataset[:, 2].astype(int)  
  
# print y 
print(y) 
# import the regressor 
from sklearn.tree import DecisionTreeRegressor  
  
# create a regressor object 
regressor = DecisionTreeRegressor(random_state = 0)  
  
# fit the regressor with X and Y data 
regressor.fit(X, y) 

# test the output by changing values, like 3750 
y_pred = regressor.predict(3750) 
  
# print the predicted price 
print("Predicted price: % d\n"% y_pred)  

# arange for creating a range of values  
# from min value of X to max value of X  
# with a difference of 0.01 between two 
# consecutive values 
X_grid = np.arange(min(X), max(X), 0.01) 
  
# reshape for reshaping the data into  
# a len(X_grid)*1 array, i.e. to make 
# a column out of the X_grid values 
X_grid = X_grid.reshape((len(X_grid), 1))  
  
# scatter plot for original data 
plt.scatter(X, y, color = 'red') 
  
# plot predicted data 
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')  
  
# specify title 
plt.title('Profit to Production Cost (Decision Tree Regression)')  
  
# specify X axis label 
plt.xlabel('Production Cost') 
  
# specify Y axis label 
plt.ylabel('Profit') 
  
# show the plot 
plt.show() 

# import export_graphviz 
from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='tree.dot', 
               feature_names =['Production Cost'])  
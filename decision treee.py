#Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
#loading data
data = pd.read_csv("C:/Program Files/Weka-3-8/data/kc_house_data.csv")
#target variables
y = data['price']
X = data[['bedrooms','bathrooms','sqft_living','sqft_lot','sqft_lot15']]
#train test split
x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,y,test_size=0.3,random_state=42,
shuffle=True)
#Training/model fitting
model =  DecisionTreeRegressor(max_depth=5,random_state=0)
model.fit(x_training_set, y_training_set)
#Model parameters study
from sklearn.metrics import mean_squared_error, r2_score
model_score = model.score(x_training_set,y_training_set)
# Have a look at R sq to give an idea of the fit 
# Explained variance score: 1 is perfect prediction
print('Coefficient of determination R^2 of the prediction.: ',model_score)
y_predicted = model.predict(x_test_set)
# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(y_test_set, y_predicted))
# Explained variance score: 1 is perfect prediction
print('Test Variance score: %.2f' % r2_score(y_test_set, y_predicted))
# So let's run the model against the test data
#Visualization
from sklearn.model_selection import cross_val_predict
fig, ax = plt.subplots()
ax.scatter(y_test_set, y_predicted, edgecolors=(0, 0, 0))
ax.plot([y_test_set.min(), y_test_set.max()], [y_test_set.min(), y_test_set.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()



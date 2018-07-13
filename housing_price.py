import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("C:/Users/Admin/Desktop/Linearreg/kc_house_data.csv")

# Assign columns
area = dataset['sqft_living']
price = dataset['price']

X = np.array(area).reshape(-1,1)
y = np.array(price)

# Split data to training and testing sets
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=0)

# Perform Linear Regression algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
pred = regressor.predict(xtest)


# Plot the training data
plt.scatter(xtrain, ytrain, color='green')
plt.plot(xtrain, regressor.predict(xtrain), color='red')
plt.title("Training Data")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()


# Plot the test or predicted data
plt.scatter(xtest, ytest, color='green')
plt.plot(xtrain, regressor.predict(xtrain), color='red')
plt.title("Test Data")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()


from sklearn import metrics

# Perform additional calculations
MAE = metrics.mean_absolute_error(ytest, regressor.predict(xtest))
MSE = metrics.mean_squared_error(ytest, regressor.predict(xtest))
RMSE = metrics.mean_squared_error(ytest, regressor.predict(xtest))

print("Mean Absolute Error:", MAE)
print("Mean Squared Error:", MSE)
print("Root Mean Squared Error:", np.sqrt(RMSE))


# Mean Absolute Error: 173012.801995901
# Mean Squared Error: 71356849350.38338
# Root Mean Squared Error: 267127.0284909099


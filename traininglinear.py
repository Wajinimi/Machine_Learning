import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load and clean data
read_file = "C:\\Users\\WAJI\\OneDrive\\projects\\PYTHON\\house_price_dataset1.csv"
open_file = pd.read_csv(read_file)
cleaned_data = open_file.dropna(subset=["House_Price"])

# Select features (X) and target (y)
X = cleaned_data[["Number_of_Bedrooms", "Square_Feet"]].values
y = cleaned_data["House_Price"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare sklearn LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\nSklearn LinearRegression Results:")
print("Sklearn Coefficients:", model.coef_)
print("Sklearn Intercept:", model.intercept_)


"""
Gradient descent is just one way to  solve the optimization problem for multipurpose models(you could use it in L2, L1, Linear, Logisitic Regression,
Nueral Networks ect). It is not the model itself, but a computational approach to find the model's parameter. It is usually used for larger datasets to avoid
computational complexity due to matrices inversion. In those cases, GD is prefered because it goes about everything 
step-by-step and avoids inverting large matrices
Gradient Descent is one method (out of several) to find those parameters by minimizing the loss function. My linear regression 
model(Sci-kitlearn) already provided me with intercepts and coefficients. But here we are going to use GD to generate 
intercepts and coefficients(parameters), and we will see if it is going to be the same as Linear Regression
"""
# Gradient Descent Parameters
eta = 0.01  # Lowered learning rate
n_epochs = 1000  # Number of iterations
m = len(X_train_scaled)  # Number of training examples

# Initialize weights (coefficients) and bias (intercept)
n_features = X_train_scaled.shape[1]
theta = np.random.randn(n_features + 1) * 0.01  # Random small values

# Add a column of ones to X_train for the bias term
X_train_with_bias = np.c_[np.ones((m, 1)), X_train_scaled]

# Perform Gradient Descent
for epoch in range(n_epochs):
    predictions = X_train_with_bias @ theta  # Predicted values
    errors = predictions - y_train  # Errors
    gradients = 2 / m * X_train_with_bias.T @ errors  # Gradients
    theta -= eta * gradients  # Update theta

# Extract final coefficients and intercept
final_intercept = theta[0]
final_coefficients = theta[1:]

print("Gradient Descent Results:")
print("Final Coefficients:", final_coefficients)
print("Final Intercept:", final_intercept)

"""My Gradient Descent generated the same parameters as the Linear Regression.This is just for learning even though the
dataset is not that large. But when there are millions of datapoints, it is better to go for Gradient descent to avoid
computational complerxity
"""





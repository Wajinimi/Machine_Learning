from sklearn.preprocessing import add_dummy_feature
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
read_file = "C:\\Users\\WAJI\\OneDrive\\projects\\PYTHON\\house_price_dataset1.csv"
open_file = pd.read_csv(read_file)
cleaned_data = open_file.dropna(subset=["House_Price"])

# Select features (X) and target (y)
X = cleaned_data[["Number_of_Bedrooms", "Square_Feet"]]
y = cleaned_data["House_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add dummy feature (bias term)
X_train_with_bias = add_dummy_feature(X_train)

# Calculate theta (weights) using the Normal Equation
theta_best = np.linalg.inv(X_train_with_bias.T @ X_train_with_bias) @ X_train_with_bias.T @ y_train

print("Theta (weights including bias):", theta_best)


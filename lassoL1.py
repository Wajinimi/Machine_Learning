import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# Load and clean the dataset
load_file = "C:\\Users\\WAJI\\OneDrive\\projects\\PYTHON\\house_price_dataset1.csv"
open_file = pd.read_csv(load_file)
cleaned_data = open_file.dropna(subset=["House_Price"])  # Drop rows with missing target values

# Split features and target
y = cleaned_data["House_Price"]
#I am using all the features expcept the target, L1 will sort the most important and kick the less important out
X = cleaned_data.drop(columns=["House_Price"])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_trainedScale = scaler.fit_transform(X_train)
X_testScale = scaler.transform(X_test)

# Lasso regression with GridSearchCV
param_grid = {'alpha': [30.0, 60.0, 70.0, 100.0]}
lasso = Lasso()
searching = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
searching.fit(X_trainedScale, y_train)  # Pass scaled data here

# Output best parameters and RMSE
print("Best parameters found (Lasso):", searching.best_params_)
print("Best RMSE score (Lasso):", np.sqrt(-searching.best_score_))

"""Using Lasso L1, notice how I included all the features from the dataset except the target?. I did it because i dont have to worry about what
features is important or not. That is the duty of L1 to do for me. Features that are not helpful have been kicked out by shrinking some coefficiants
to 0
L1 found the best Apha to be 30.0 and RMSE score to be 10525.385975805975
"""
rmse = 10525.385975805975
target_from_dataset_mean = y.mean()
percentage_calculation = (rmse / target_from_dataset_mean * 100)
print("The mean price from the dataset is:", target_from_dataset_mean)
print(f"The model's prediction is off by: {percentage_calculation:.2f}%")

"""The mean price from the dataset is: 439046.5460937873
The model's prediction is off by: 2.40%
This shows that my model is working just fine. Considering the fact that the Mean House_Price is $439,046 and then the model
error is just 2.40% which is relatively small.
"""

percentage_error = (439046 * 2.40/ 100)
print(f"My model is ${percentage_error} away from the actual values in the dataset")

#The model is $10537.104 away from the actual values. This is a very reasonable amount consdering the fact that the mean taget value is 439,046



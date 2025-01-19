from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# Load the dataset
read_file = "C:\\Users\\WAJI\\OneDrive\\projects\\PYTHON\\house_price_dataset1.csv"
open_file = pd.read_csv(read_file)
cleaned_data = open_file.dropna(subset=["House_Price"])

# Select features (X) and target (y)
X = cleaned_data[["Number_of_Bedrooms", "Location_Score", "Square_Feet"]]
y = cleaned_data["House_Price"]

#Allocating 80 percent of the data for training and 20 percent for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)


model = LinearRegression()
model.fit(X_train, y_train)
print("The coefficient is:", model.coef_)
print("The inntercept is:", model.intercept_)

"""prediction = model.predict(X_test)
print("This is the prediction:", prediction)"""


#CROSS-VALIDATION
kfold = KFold(n_splits = 5, shuffle = True, random_state= 42)
scores = cross_val_score(model, X_test, y_test, cv = kfold, scoring='neg_mean_squared_error')
rmse = np.sqrt(-scores)
print ("RMSE SCORE", rmse)
print("RMSE MEAN", rmse.mean())
print("STANDARD DEVIATION", rmse.std())

#Now let us check the mean for our House_Price from the dataset to see if our model is performing well
House_price_mean = cleaned_data["House_Price"].mean()
print("This is the House price mean:", House_price_mean)


"""NOTE, THE mean for our Cross Validation test is 153498.4056977248, now to decide whether this mean is too large or small.
We can use this formula = RMSE/House Price Mean from the dataset * %100
We calculated the mean for the House_Price column and we got 439046, so:

153498/439046 * 100 = %34.96
Well, we might say a percentage of 34.96 is too high as this shows overfitting and we could decide to say a percentage of 10 is our threshold

With this, we could try to do  hyperparameter tuning"""


#HYPERPARAMETER LASSO
param_grid = {'alpha': [30.0, 60.0, 70.0, 100.0]}

lasso = Lasso()

grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search_lasso.fit(X_train, y_train)
print("Best parameters found (Lasso):", grid_search_lasso.best_params_)
print("Best RMSE score (Lasso):", np.sqrt(-grid_search_lasso.best_score_))

"""Now let us see if our Hyperparameter is making our model perform.
Lasso_Mean/House_Price mean * 100
20648/439046 *100 = %4.7
An overfitting of 4.7 is a very fair percentage considering that our House_Prices are in hundred of thousands
"""



# Plot Actual vs Predicted
plt.scatter(y_test, model.predict(X_test))
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
min_value = min(min(y_test), min(model.predict(X_test)))
max_value = max(max(y_test), max(model.predict(X_test)))
plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--')  # Perfect prediction line
plt.show()



















"""NOTE, we can also use MSE, PLOTTING to check our model's performance. But we choose to use CROSS-VALIDATION
1. MSE 
2. PLOTTING
3. CROSS-VALIDATION


#MSE
mse = mean_squared_error(y_test, prediction)
print("This is the mean square error of the actual price and predicted price:", mse)

#PLOT. If the dots are aroud the diagonla line, we will say our model is performing well
# Predicted vs Actual Plot
plt.scatter(y_test, prediction, label="Predicted vs Actual")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")

# Plotting the diagonal line for reference
min_value = min(min(y_test), min(prediction))
max_value = max(max(y_test), max(prediction))
plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', label="Perfect Prediction Line")
plt.legend()
plt.show()
"""


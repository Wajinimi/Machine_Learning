from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load the dataset
read_file = "C:\\Users\\WAJI\\OneDrive\\projects\\PYTHON\\house_price_dataset1.csv"
open_file = pd.read_csv(read_file)
cleaned_data = open_file.dropna(subset=["House_Price"])

# Select features (X) and target (y)
X = cleaned_data[["Number_of_Bedrooms", "Location_Score", "Square_Feet"]]
y = cleaned_data["House_Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


"""The output is 
Mean Squared Error: 731956497.5797203
R-squared: 0.9760839423437137
The MSE appears to be reasonable which suggests the model is  performing well, the squared difference between the predicted house ...
..prices and the actual house prices is not large.

R^2 value ranges from 0 to 1, with higher values indicating better model performance. An R^2 value close to 1 means the model...
...explains a large portion of the variance.

From our output, R^2 is positive 0.9760839423437137  which is close to 1 and suggests that our model is performing well...
this means the model is  capturing the variance in the dataset. """

#However to be sure what percentage difference is from the actual price and the predicted price. 
mean_house_price = y.mean()
mean_predictions = [mean_house_price] * len(y) 
mse = mean_squared_error(y, mean_predictions) 
print(f"Mean Squared Error: {mse}")

"""731956497.5797203/27751801008.83463 * 100 = %26.38
Well, a percentage of 26.38 is reasonable considering the fact that the Actual_House Prices are in Hundreds of Thousands... 
...So a difference of 26000 from the predicted value is not actually bad"""
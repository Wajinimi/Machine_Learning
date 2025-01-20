from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

read_file = "C:\\Users\\WAJI\\OneDrive\\projects\\PYTHON\\house_price_dataset1.csv"
open_file = pd.read_csv(read_file)
cleaned_data = open_file.dropna(subset=["House_Price"])

X = cleaned_data[["Number_of_Bedrooms", "Location_Score", "Square_Feet"]]
y = cleaned_data["House_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = SVR(kernel = 'rbf')
model.fit(X_train, y_train)

predict = model.predict(X_test)

mse = mean_squared_error(y_test, predict)
r2 = r2_score(y_test, predict) 


print(f"Mean Squared Error: {mse}") 
print(f"R-squared: {r2}")


"""The output is 
Mean Squared Error: 31663459326.286755
R-squared: -0.03457667409049092
The MSE appears too large which suggests the model is not performing well, the squared difference between the predicted house ...
..prices and the actual house prices is large.

R^2 value ranges from 0 to 1, with higher values indicating better model performance. An R^2 value close to 1 means the model...
...explains a large portion of the variance.

From our output, R^2 is negative -0.34567... which suggests that the model is performing worse than a simple mean...
...prediction; which means the model is not capturing that variance inteh dataset. """


#Lets say we want to calculate the error percentage from the actual price and the predicted price
#RMSE/House Price Mean from the dataset * %100
mean_house_price = y.mean()
mean_predictions = [mean_house_price] * len(y) 
mse = mean_squared_error(y, mean_predictions) 
print(f"Mean Squared Error: {mse}")

"""31663459326.286755/27751801008.83463 * 100 = %114
Well, a percentage of %114 is too high as this shows overfitting and we could decide to say a percentage of 10 is our threshold
Lets try and use another model = Random Forest
NOTE: THIS DID NOT WORK BECAUSE SVM IS USUALY MEANT FOR CLASSIFICATION AND NOT CONTINOUS PREDICTION (REGRESSION)"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
load_file = "C:\\Users\\WAJI\\OneDrive\\projects\\PYTHON\\Email_dataset.csv"
read_file = pd.read_csv(load_file, encoding='ISO-8859-1')

# Clean the data
cleaning = read_file.where(pd.notnull(read_file), '')
cleaning.loc[cleaning['v1'] == "spam", 'v1'] = 0
cleaning.loc[cleaning['v1'] == "ham", 'v1'] = 1

# Feature and target variables
X = cleaning['v2']
y = cleaning['v1']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train = feature_extraction.fit_transform(X_train)
X_test = feature_extraction.transform(X_test)

"""fit()Learns the vocabulary and computes the TF-IDF (Term Frequency-Inverse Document Frequency) weights based on the 
training data X_train

transform() Applies the learned vocabulary and TF-IDF weights to transform the training data into a sparse matrix format
where each row is a document and each column is a feature (term).

Why not fit() again on X_test, You should never fit() on the test set to avoid data leakage. Instead, transform the test 
set using the vocabulary and weights learned from the training data
"""
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Logistic Regression model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Training accuracy
prediction_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, prediction_train)
print("Accuracy on training data:", accuracy_train)

# Testing accuracy
prediction_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, prediction_test)
print("Accuracy on test data:", accuracy_test)

input_mail = ["Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."]
input_mail_feature = feature_extraction.transform(input_mail)
predict = model.predict(input_mail_feature)
print(predict)

if(predict[0]==1):
    print("It is Ham email")
else:
    print("It is Spam email")

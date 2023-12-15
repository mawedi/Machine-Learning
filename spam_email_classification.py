import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Setting the path of spam.csv file
file_path = "spam.csv"

# Open the file
df = pd.read_csv(file_path)

# Separating data for X as the input and Y as output
X = df.iloc[:, 1]
Y = df.iloc[:, 0]

# Transforming the X to a list and Y to the appropriate form
Y = list(Y)
X = np.array(X).reshape(-1)

# Splitting data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Transforming data from text to numeric
vector = TfidfVectorizer()
X_transformed_trained_data = vector.fit_transform(X_train)

# Traning of the model
model = LogisticRegression()
model.fit(X_transformed_trained_data, Y_train)

# Transforming testing data from text to numeric
X_test_transofrmed_data = vector.transform(X_test)

# Printing the accuracy of the model
output_tested_data = model.predict(X_test_transofrmed_data)
score = accuracy_score(Y_test, output_tested_data)
print(f"The accuracy of the model is {score}")

# Testing a sentence that can be in the email spam
message = ["Congratulations! You've won a prize. Click here to claim."]

# Transformation of the data
transformed_data = vector.transform(message)

# Predict the message's type
output = model.predict(transformed_data)
print(f"message: {message[0]}\nPrediction: {output[0]}")


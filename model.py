import pandas as pd
data=pd.read_csv("data/crop_data.csv")
print("First 5 rows of dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

# Separate input and output
X = data.drop("label", axis=1)
y = data["label"]

print("\nInput Features:")
print(X.head())

print("\nTarget Output:")
print(y.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Check model accuracy
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# Taking user input
N = float(input("Enter Nitrogen value: "))
P = float(input("Enter Phosphorus value: "))
K = float(input("Enter Potassium value: "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter pH value: "))
rainfall = float(input("Enter Rainfall: "))

# Making prediction
prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])

print("Recommended Crop:", prediction[0])

import pickle
pickle.dump(model, open("models/crop_model.pkl","wb"))

print("Model saved successfully!")
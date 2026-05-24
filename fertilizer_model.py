import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data/fertilizer_data.csv")

print("Dataset loaded")

# Convert all column names to lowercase
data.columns = data.columns.str.strip().str.lower()

print(data.columns)  # check

# 👇 MATCH YOUR DATASET EXACTLY
X = data[['nitrogen', 'phosphorous', 'potassium']]
y = data['fertilizer_name']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained")

# Save
pickle.dump(model, open("models/fertilizer_model.pkl", "wb"))

print("fertilizer_model.pkl saved!")
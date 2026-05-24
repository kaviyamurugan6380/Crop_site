import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv("data/crop_data.csv")

print("Dataset loaded!")

# Input → only 4 values
X = data[['temperature', 'humidity', 'ph', 'rainfall']]

# Output → NPK
y = data[['N', 'P', 'K']]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("models/npk_model.pkl", "wb"))

print("NPK model saved successfully!")
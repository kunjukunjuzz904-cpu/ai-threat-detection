import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("dataset/data.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Remove infinity values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove missing values
data.dropna(inplace=True)

# Features & Labels
X = data.drop("Label", axis=1)
y = data["Label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "model.pkl")
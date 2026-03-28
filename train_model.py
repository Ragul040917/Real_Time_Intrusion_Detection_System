print("🚀 Script started...")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("traffic_prp_200.csv")

# Handle missing values
data = data.fillna(method="ffill")

# Encode categorical columns
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == "object":
        data[column] = label_encoder.fit_transform(data[column])

# Define features and label
X = data.drop("label", axis=1)
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "rf_model.pkl")

print("✅ Model trained and saved as rf_model.pkl")

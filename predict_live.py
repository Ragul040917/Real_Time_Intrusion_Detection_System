import joblib
import pandas as pd
import numpy as np

print("🚀 Loading trained model...")

# Load saved model
model = joblib.load("rf_model.pkl")

print("✅ Model loaded successfully!")

# IMPORTANT:
# Replace this with exact feature count from your dataset
# (Exclude the 'label' column)

# Check how many features your model expects
n_features = model.n_features_in_
print(f"Model expects {n_features} features")

# Create dummy live traffic sample (random values for now)
sample_data = np.random.rand(1, n_features)

# Convert to DataFrame
sample_df = pd.DataFrame(sample_data)

# Predict
prediction = model.predict(sample_df)

print("🔍 Prediction Result:", prediction)

if prediction[0] == 0:
    print("🟢 Normal Traffic")
else:
    print("🔴 Intrusion Detected!")

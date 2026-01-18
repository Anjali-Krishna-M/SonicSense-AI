import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Create models folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

print("📊 Loading dataset...")
data = pd.read_csv('data.csv')

# Drop the filename column (not needed for training)
data = data.drop(['filename'], axis=1)

# Separate Features (X) and Labels (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert genre names (Rock, Pop) into numbers (0, 1, 2...)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (Normalize logic)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("🧠 Training the Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"🔥 Model Accuracy: {accuracy * 100:.2f}%")

# Save everything for the website to use
joblib.dump(model, 'models/music_genre_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/encoder.pkl')

print("💾 Model, Scaler, and Encoder saved to /models/")
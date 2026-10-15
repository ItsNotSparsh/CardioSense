import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# 1. Load your actual dataset
csv_file = 'heart_synthetic_50000.csv'

if not os.path.exists(csv_file):
    print(f"❌ Error: '{csv_file}' not found. Please upload it.")
    exit()

print(f"Loading {csv_file}...")
df = pd.read_csv(csv_file)

# 2. Prepare Features (X) and Target (y)
# The dataset has 'target' as the label, so we drop it from features
X = df.drop(columns=['target'])
y = df['target']

print(f"Dataset Shape: {df.shape}")
print("Features:", list(X.columns))

# 3. Split into Training and Testing sets (80% train, 20% test)
# This allows us to test how good the model is on data it hasn't seen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Accuracy
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("------------------------------------------------")
print(f"✅ Model Accuracy: {accuracy:.2%}")
print("------------------------------------------------")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 6. Save the trained model
filename = 'new_model.pkl'
with open(filename, 'wb') as f:
    pickle.dump(model, f)

print(f"💾 Model saved as '{filename}'. You can now run app.py!")
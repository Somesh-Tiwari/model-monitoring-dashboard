# app/train.py
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os

print("Loading data...")
X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=10, random_state=42)

print("Training model...")
clf.fit(X, y)

# Save the model
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")
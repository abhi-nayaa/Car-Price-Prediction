"""Train and save a car-price model compatible with app.py."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "CarPrice.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load data and derive brand consistently for training and prediction.
df = pd.read_csv(DATA_PATH)
df["brand"] = df["CarName"].astype(str).str.strip().str.split().str[0].str.lower()
df = df.drop(columns=["car_ID", "CarName"])
df = df.dropna(subset=["price"])

X = df.drop(columns=["price"])
y = df["price"]
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = [c for c in X.columns if c not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", "passthrough", numerical_features),
    ]
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )),
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Model: RandomForestRegressor pipeline")
print(f"R2:  {r2_score(y_test, predictions):.4f}")
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
print(f"MSE: {mean_squared_error(y_test, predictions):.2f}")

# Refit on all available data for the deployed model.
model.fit(X, y)
model_path = MODEL_DIR / "car_price_model.pkl"
joblib.dump(model, model_path)

categories = {
    col: sorted(X[col].astype(str).unique().tolist())
    for col in categorical_features
}
numeric_defaults = {
    col: float(X[col].median()) for col in numerical_features
}
metadata = {
    "features": X.columns.tolist(),
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "categories": categories,
    "numeric_defaults": numeric_defaults,
    "currency": "USD",
    "selected_model": "Random Forest Regressor",
}
with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved model: {model_path}")
print(f"Saved metadata: {MODEL_DIR / 'model_metadata.json'}")

# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from joblib import dump

# --- Config ---
CSV_PATH = "data.csv"          # your real dataset (if exists)
MODEL_OUT = "artifacts/model.joblib"
TARGET_COL = "math_score"      # chosen target name (change if your dataset uses different)

# Columns we expect (matches your earlier code)
categorical_features = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course"
]
numerical_features = ["reading_score", "writing_score"]
all_features = categorical_features + numerical_features

# --- Load or synthesize data ---
if os.path.exists(CSV_PATH):
    print(f"Found {CSV_PATH} — loading it.")
    df = pd.read_csv(CSV_PATH)
    # If the dataset doesn't have the expected columns, stop with a readable message
    missing = [c for c in all_features + [TARGET_COL] if c not in df.columns]
    if missing:
        raise SystemExit(f"Your {CSV_PATH} is missing columns: {missing}\n"
                         "Either provide these columns or remove/change TARGET_COL in the script.")
else:
    print(f"{CSV_PATH} not found. Creating a small synthetic dataset for quick training.")
    n = 500
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "gender": rng.choice(["male", "female"], size=n),
        "race_ethnicity": rng.choice(["group A","group B","group C","group D","group E"], size=n),
        "parental_level_of_education": rng.choice([
            "some high school","high school",
            "some college","associate's degree","bachelor's degree","master's degree"
        ], size=n),
        "lunch": rng.choice(["standard","free/reduced"], size=n),
        "test_preparation_course": rng.choice(["none","completed"], size=n),
        "reading_score": rng.integers(30, 100, size=n),
        "writing_score": rng.integers(30, 100, size=n),
    })
    # Synthetic target: math_score correlated with reading+writing
    df[TARGET_COL] = (0.45 * df["reading_score"] + 0.45 * df["writing_score"] +
                      rng.normal(0, 8, size=n)).round().clip(0, 100).astype(int)

# --- Prepare X, y ---
X = df[all_features]
y = df[TARGET_COL]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocessing ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
    ],
    remainder="drop",
)

# --- Pipeline (preprocessing + model) ---
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# --- Train ---
print("Training model...")
pipeline.fit(X_train, y_train)

# --- Evaluate ---
y_pred = pipeline.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"R2 score on test set: {score:.4f}")

# --- Save model ---
os.makedirs(os.path.dirname(MODEL_OUT) or ".", exist_ok=True)
dump(pipeline, MODEL_OUT)
print(f"Saved trained pipeline to: {MODEL_OUT}")

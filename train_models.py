"""Rebuild the two models from Cindy's notebook (Assignment1_Iris_HeartDisease_ipynb.ipynb).

Run once from the project root:  python train_models.py
It writes models/iris_model.joblib and models/heart_model.joblib.
The code follows the notebook exactly (same features, split, models and seed).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent
OUT = BASE / "models"
OUT.mkdir(exist_ok=True)

# ---------- Part A: Iris ----------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
iris_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
iris_model.fit(X_train, y_train)
print("Iris test accuracy:", round(accuracy_score(y_test, iris_model.predict(X_test)), 4))
joblib.dump({"model": iris_model, "features": list(iris.feature_names),
             "classes": list(iris.target_names)}, OUT / "iris_model.joblib")

# ---------- Part B: Heart disease ----------
df = pd.read_csv(BASE / "heart_disease.csv")
TARGET = "heart_disease"
num_cols = ["age", "blood_pressure", "cholesterol", "bmi", "physical_activity", "stress_level"]
cat_cols = ["sex", "smoker", "diabetes", "family_history"]
X = df.drop(columns=TARGET)
y = df[TARGET]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}
aucs = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)]).fit(X_train, y_train)
    aucs[name] = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"{name}: test ROC-AUC = {aucs[name]:.3f}")

# Same rule as the notebook: keep the model with the best test ROC-AUC, refit on all data.
best_name = max(aucs, key=aucs.get)
final = Pipeline([("prep", preprocessor), ("clf", models[best_name])]).fit(X, y)
joblib.dump({"model": final, "name": best_name, "num_cols": num_cols, "cat_cols": cat_cols},
            OUT / "heart_model.joblib")
print("Saved models/heart_model.joblib using:", best_name)

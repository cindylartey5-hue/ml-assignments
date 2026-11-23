"""Assignment 1 web app - Lartey Naadu Cindy (PUIT/23210097).

Part A: Iris species classifier.  Part B: heart disease classifier.
Models are built by train_models.py from the notebook code and loaded from models/.
"""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder="public", static_url_path="")

IRIS = joblib.load(BASE_DIR / "models" / "iris_model.joblib")
HEART = joblib.load(BASE_DIR / "models" / "heart_model.joblib")

STUDENT = {
    "name": "Lartey Naadu Cindy",
    "id": "PUIT/23210097",
    "course": "AI & Machine Learning",
    "repo": "https://github.com/cindylartey5-hue/ml-assignments",
}

# ---------- results taken from the notebook / report ----------
IRIS_RESULTS = {
    "test_size": 30, "correct": 27, "accuracy": 0.90,
    "macro": {"precision": 0.9024, "recall": 0.90, "f1": 0.8997},
    "per_class": [
        {"name": "Setosa", "precision": 1.00, "recall": 1.00, "f1": 1.00},
        {"name": "Versicolor", "precision": 0.82, "recall": 0.90, "f1": 0.86},
        {"name": "Virginica", "precision": 0.89, "recall": 0.80, "f1": 0.84},
    ],
    "cv_mean": 0.96, "cv_std": 0.0389,
}
HEART_RESULTS = {
    "records": 500, "features": 10, "positive": 266, "negative": 234,
    "models": [
        {"name": "Logistic Regression", "accuracy": 0.560, "precision": 0.562,
         "recall": 0.774, "f1": 0.651, "auc": 0.505, "cv_auc": 0.508, "shuffled": 0.484},
        {"name": "Random Forest", "accuracy": 0.500, "precision": 0.529,
         "recall": 0.509, "f1": 0.519, "auc": 0.512, "cv_auc": 0.507, "shuffled": 0.533},
    ],
}

IRIS_FIELDS = [
    ("sepal_length", "Sepal length", 4.3, 7.9),
    ("sepal_width", "Sepal width", 2.0, 4.4),
    ("petal_length", "Petal length", 1.0, 6.9),
    ("petal_width", "Petal width", 0.1, 2.5),
]
HEART_NUMERIC = [
    ("age", "Age", "years", 30, 79, 1),
    ("blood_pressure", "Blood pressure", "mmHg", 90, 179, 1),
    ("cholesterol", "Cholesterol", "mg/dL", 150, 299, 1),
    ("bmi", "BMI", "kg/m²", 18.1, 40.0, 0.1),
    ("physical_activity", "Physical activity", "0–4", 0, 4, 1),
    ("stress_level", "Stress level", "1–9", 1, 9, 1),
]
HEART_CATEGORIES = [
    ("sex", "Sex", ["Male", "Female"]),
    ("smoker", "Smoker", ["Yes", "No"]),
    ("diabetes", "Diabetes", ["Yes", "No"]),
    ("family_history", "Family history", ["Yes", "No"]),
]

NAV = [
    ("home", "Overview", "home", "Home"),
    ("iris", "Iris predictor", "flower", "Iris"),
    ("iris_results", "Iris results", "chart", "Iris stats"),
    ("heart", "Heart predictor", "heart", "Heart"),
    ("heart_results", "Heart results", "pulse", "Heart stats"),
]


@app.context_processor
def inject_globals():
    return {"student": STUDENT, "nav": NAV}


def read_number(form, key, low, high):
    """Return (value, error) for one numeric field, checked against the training range."""
    raw = (form.get(key) or "").strip()
    try:
        value = float(raw)
    except ValueError:
        return None, "Enter a number."
    if not (low <= value <= high):
        return value, f"Use a value between {low:g} and {high:g}."
    return value, None


@app.route("/")
def home():
    return render_template("home.html", iris=IRIS_RESULTS, heart=HEART_RESULTS, active="home")


@app.route("/iris", methods=["GET", "POST"])
def iris():
    values, errors, result = {}, {}, None
    if request.method == "POST":
        for key, _label, low, high in IRIS_FIELDS:
            values[key], err = read_number(request.form, key, low, high)
            if err:
                errors[key] = err
        if not errors:
            row = pd.DataFrame([[values[k] for k, *_ in IRIS_FIELDS]], columns=IRIS["features"])
            proba = IRIS["model"].predict_proba(row)[0]
            ranked = sorted(zip(IRIS["classes"], proba), key=lambda p: p[1], reverse=True)
            result = {
                "species": ranked[0][0].capitalize(),
                "confidence": round(ranked[0][1] * 100),
                "all": [(name.capitalize(), round(p * 100)) for name, p in ranked],
            }
    return render_template("iris.html", fields=IRIS_FIELDS, values=values, errors=errors,
                           result=result, active="iris")


@app.route("/iris/results")
def iris_results():
    return render_template("iris_results.html", r=IRIS_RESULTS, active="iris_results")


@app.route("/heart", methods=["GET", "POST"])
def heart():
    values, errors, result = {}, {}, None
    if request.method == "POST":
        for key, _label, _unit, low, high, _step in HEART_NUMERIC:
            values[key], err = read_number(request.form, key, low, high)
            if err:
                errors[key] = err
        for key, _label, options in HEART_CATEGORIES:
            values[key] = request.form.get(key, "")
            if values[key] not in options:
                errors[key] = "Choose one option."
        if not errors:
            columns = ["age", "sex", "blood_pressure", "cholesterol", "bmi", "smoker",
                       "diabetes", "physical_activity", "family_history", "stress_level"]
            row = pd.DataFrame([[values[c] for c in columns]], columns=columns)
            p_yes = float(HEART["model"].predict_proba(row)[0][1])
            result = {
                "label": "Heart disease" if p_yes >= 0.5 else "No heart disease",
                "positive": p_yes >= 0.5,
                "p_yes": round(p_yes * 100),
                "p_no": 100 - round(p_yes * 100),
                "model": HEART["name"],
            }
    return render_template("heart.html", numeric=HEART_NUMERIC, categories=HEART_CATEGORIES,
                           values=values, errors=errors, result=result, active="heart")


@app.route("/heart/results")
def heart_results():
    return render_template("heart_results.html", r=HEART_RESULTS, active="heart_results")


@app.errorhandler(404)
def not_found(_e):
    return render_template("404.html", active=None), 404


if __name__ == "__main__":
    app.run(debug=True)

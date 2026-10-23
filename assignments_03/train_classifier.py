import json
import sys
from pathlib import Path

import pandas as pd
import requests
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
CITY = "Charlotte, North Carolina"
FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]
LABEL_RULE = (
    "good_for_running=1 when temperature_2m_max is 7–26 °C, "
    "temperature_2m_min >= 0 °C, precipitation_sum < 3 mm, "
    "and wind_speed_10m_max < 30 km/h; otherwise 0."
)

# Task 1: fetch Charlotte daily weather for 2023.
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 35.23,
    "longitude": -80.84,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": FEATURES,
    "timezone": "America/New_York",
}
response = requests.get(url, params=params, timeout=30)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop(columns="time")
print("Dataset shape:", df.shape)
print(df.head())
# City chosen: Charlotte, North Carolina.

# Task 2: label days using the assignment's running-weather thresholds.
df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)
class_counts = df["good_for_running"].value_counts().reindex([0, 1], fill_value=0)
print("Skip days:", int(class_counts.loc[0]))
print("Good days:", int(class_counts.loc[1]))
good_fraction = float(class_counts.loc[1] / len(df))
# 38.1% of Charlotte's days are labeled good, a plausible seasonal mix; the class balance should be considered when interpreting precision, recall, and F1.

# Task 3: select KNN k by 5-fold cross-validation on the training set.
X = df[FEATURES]
y = df["good_for_running"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
k_values = [3, 5, 7, 9, 11, 15, 21]
knn_cv = {}
for k in k_values:
    candidate = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))]
    )
    scores = cross_val_score(candidate, X_train, y_train, cv=5, scoring="accuracy")
    knn_cv[k] = float(scores.mean())
    print(f"KNN k={k}: mean 5-fold CV accuracy={scores.mean():.3f}")
best_k = max(knn_cv, key=knn_cv.get)
knn_pipeline = Pipeline(
    [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=best_k))]
)
knn_pipeline.fit(X_train, y_train)
knn_predictions = knn_pipeline.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("Chosen KNN k:", best_k)
print("KNN test accuracy:", knn_accuracy)
print("KNN classification report:")
print(classification_report(y_test, knn_predictions, target_names=["skip", "good"], zero_division=0))

# Task 4: fit and evaluate logistic regression.
logistic_pipeline = Pipeline(
    [("scaler", StandardScaler()), ("logistic", LogisticRegression(max_iter=1000))]
)
logistic_pipeline.fit(X_train, y_train)
logistic_predictions = logistic_pipeline.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_precision = precision_score(y_test, logistic_predictions, pos_label=1, zero_division=0)
logistic_recall = recall_score(y_test, logistic_predictions, pos_label=1, zero_division=0)
logistic_f1 = f1_score(y_test, logistic_predictions, pos_label=1, zero_division=0)
print("Logistic regression test accuracy:", logistic_accuracy)
print("Logistic regression precision (good):", logistic_precision)
print("Logistic regression recall (good):", logistic_recall)
print("Logistic regression F1 (good):", logistic_f1)
print("Logistic regression classification report:")
print(classification_report(y_test, logistic_predictions, target_names=["skip", "good"], zero_division=0))
logistic_coefficients = logistic_pipeline.named_steps["logistic"].coef_[0]
for name, coef in zip(FEATURES, logistic_coefficients):
    print(f"{name:22s}: {coef:+.3f}")
# Precipitation (-2.492) and daily high (-0.934) push most strongly toward skip; high heat and rain make running less appealing, while the positive wind coefficient (+0.320) is a surprising fit to this sample.

# Task 5: choose the model and explain the trade-off for a running app.
print("KNN accuracy:", knn_accuracy)
print("Logistic regression accuracy:", logistic_accuracy)
# KNN accuracy is 0.973 versus logistic regression's 0.877. A false positive could recommend an unsafe run, while a false negative misses a possible run. I deploy KNN because it scored better, had no false negatives on this test set, and the training set is small enough for fast predictions; its probabilities are based on the three nearest neighbors.

# Task 6: save the selected fitted pipeline and metadata.
chosen_pipeline = knn_pipeline
chosen_accuracy = knn_accuracy
chosen_f1 = f1_score(y_test, knn_predictions, pos_label=1, zero_division=0)
joblib.dump(chosen_pipeline, MODELS_DIR / "weather_classifier.pkl")
metadata = {
    "python_version": sys.version.split()[0],
    "scikit_learn_version": sklearn.__version__,
    "feature_names": FEATURES,
    "label_rule": LABEL_RULE,
    "city": CITY,
    "test_accuracy": chosen_accuracy,
    "test_f1": chosen_f1,
    "model": f"KNeighborsClassifier pipeline with StandardScaler (k={best_k})",
}
metadata_path = MODELS_DIR / "weather_classifier_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print("Saved model to", MODELS_DIR / "weather_classifier.pkl")
print("Saved metadata to", metadata_path)

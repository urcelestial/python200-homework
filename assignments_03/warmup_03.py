import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
plt.switch_backend("Agg")

# Task Preprocessing Q1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaled training column means:", X_train_scaled.mean(axis=0))
# Fit the scaler on training data only to avoid letting test-set information leak into training.

# Task Preprocessing Q2
encoder = OneHotEncoder(sparse_output=False)
season_train = encoder.fit_transform(df.loc[X_train.index, ["season"]])
season_test = encoder.transform(df.loc[X_test.index, ["season"]])
print("Encoded season training shape:", season_train.shape)
print("Encoded season test shape:", season_test.shape)
print("Season feature names:", encoder.get_feature_names_out(["season"]))
# One season column becomes four indicator columns because the data contains four distinct seasons.

# Task Preprocessing Q3
features_with_season = df[numeric + ["season"]]
X_season_train, X_season_test, y_season_train, y_season_test = train_test_split(
    features_with_season, y, test_size=0.2, random_state=42, stratify=y
)
preprocessor = ColumnTransformer(
    [
        ("numeric", StandardScaler(), numeric),
        ("season", OneHotEncoder(handle_unknown="ignore"), ["season"]),
    ]
)
season_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("logistic", LogisticRegression(max_iter=1000))]
)
season_pipeline.fit(X_season_train, y_season_train)
print("Season pipeline test accuracy:", season_pipeline.score(X_season_test, y_season_test))
# The pipeline applies the same scaling and one-hot encoding steps automatically at fit and prediction time.

# Task KNN Q1
knn_pipeline = Pipeline(
    [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))]
)
knn_pipeline.fit(X_train, y_train)
knn_predictions = knn_pipeline.predict(X_test)
print("Scaled KNN test accuracy:", accuracy_score(y_test, knn_predictions))
print("Scaled KNN classification report:")
print(classification_report(y_test, knn_predictions, target_names=["skip", "good"], zero_division=0))

# Task KNN Q2
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
knn_cv_scores = {}
for k in k_values:
    candidate = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))]
    )
    scores = cross_val_score(candidate, X_train, y_train, cv=5, scoring="accuracy")
    knn_cv_scores[k] = scores.mean()
    print(f"k={k}: mean 5-fold CV accuracy={scores.mean():.3f}")
chosen_k = max(knn_cv_scores, key=knn_cv_scores.get)
# I would choose k=11, since it has the highest mean 5-fold training cross-validation accuracy (0.942).

# Task KNN Q3
raw_knn = KNeighborsClassifier(n_neighbors=5)
raw_knn.fit(X_train, y_train)
raw_knn_accuracy = accuracy_score(y_test, raw_knn.predict(X_test))
scaled_knn_accuracy = accuracy_score(y_test, knn_pipeline.predict(X_test))
print("Unscaled KNN test accuracy:", raw_knn_accuracy)
print("Scaled KNN test accuracy:", scaled_knn_accuracy)
# Scaling raised test accuracy from 0.925 to 0.958; it matters because distance-based KNN can otherwise be dominated by features with larger numeric ranges.

# Task Logistic Regression Q1
pipeline = Pipeline(
    [("scaler", StandardScaler()), ("logistic", LogisticRegression(max_iter=1000))]
)
pipeline.fit(X_train, y_train)
logistic_predictions = pipeline.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_precision = precision_score(y_test, logistic_predictions, pos_label=1, zero_division=0)
logistic_recall = recall_score(y_test, logistic_predictions, pos_label=1, zero_division=0)
logistic_f1 = f1_score(y_test, logistic_predictions, pos_label=1, zero_division=0)
print("Logistic regression accuracy:", logistic_accuracy)
print("Logistic regression precision (good):", logistic_precision)
print("Logistic regression recall (good):", logistic_recall)
print("Logistic regression F1 (good):", logistic_f1)

# Task Logistic Regression Q2
coefficients = pipeline.named_steps["logistic"].coef_[0]
for name, coef in zip(numeric, coefficients):
    print(f"{name:22s}: {coef:+.3f}")
# Wind speed (-2.444) and precipitation (-1.889) push most strongly toward skip, matching the intuition that windy or rainy days are less suitable for a run.

# Task Logistic Regression Q3
good_class_index = list(pipeline.named_steps["logistic"].classes_).index(1)
good_probabilities = pipeline.predict_proba(X_test.iloc[:5])[:, good_class_index]
print("Good-day probabilities for the first five test days:", np.round(good_probabilities, 2))
# A probability near 0.5 means the model is uncertain; the second test day's 0.47 probability is a borderline prediction.

# Task joblib Q1
joblib.dump(pipeline, "models/warmup_model.pkl")
loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == pipeline.predict(X_test)).all()
print("Predictions match.")
# Saving only LogisticRegression would omit the scaler, so raw values would be interpreted on the wrong scale at prediction time.

# Task Evaluation Q1
matrix = confusion_matrix(y_test, logistic_predictions, labels=[0, 1])
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["skip", "good"])
display.plot(cmap="Blues", colorbar=False)
plt.tight_layout()
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.close()
false_positives = int(matrix[0, 1])
false_negatives = int(matrix[1, 0])
print("False positives:", false_positives)
print("False negatives:", false_negatives)
# The model made 9 false positives and 7 false negatives; a false positive may matter more because it could encourage a run in unsafe weather.

"""Week 2 mini-project: predict daily high temperatures from weather data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Task 1: Charlotte, North Carolina, for the complete 2023 calendar year.


url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 35.23,
    "longitude": -80.84,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params, timeout=30)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
print("Dataset shape:", df.shape)
print("First five rows:")
print(df.head())


# Task 2: inspect distributions, missing values, and feature relationships.
print("Descriptive statistics:")
print(df.describe())
missing_counts = df.isna().sum()
print("Missing values by column:")
print(missing_counts)


if missing_counts.any():
    df = df.dropna(subset=[
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]).copy()
    print("Shape after dropping rows with missing measurements:", df.shape)

target = "temperature_2m_max"
correlations = df.select_dtypes(include="number").corr()[target].sort_values(ascending=False)
print("Numeric feature correlations with daily high (sorted):")
print(correlations)


plt.figure(figsize=(8, 5))
plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"], alpha=0.65)
plt.title("Charlotte Daily High vs Low (2023)")
plt.xlabel("Daily Low Temperature (°C)")
plt.ylabel("Daily High Temperature (°C)")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "high_vs_low_project.png")
plt.close()

# Task 3: flag summer months (June, July, and August).
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)
summer_counts = df["is_summer"].value_counts().sort_index()
print("Non-summer days:", summer_counts.get(0, 0))
print("Summer days:", summer_counts.get(1, 0))


# Task 4: baseline model using only the daily low.
X_baseline = df[["temperature_2m_min"]].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42
)
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = baseline_model.score(X_test, y_test)
print("Baseline slope:", baseline_model.coef_[0])
print("Baseline RMSE (°C):", baseline_rmse)
print("Baseline test R²:", baseline_r2)


# Task 5: add precipitation, wind, and the seasonal indicator.
feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer",
]
X = df[feature_cols].values
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train_full, y_train_full)
full_pred = model.predict(X_test_full)
train_r2 = model.score(X_train_full, y_train_full)
test_r2 = model.score(X_test_full, y_test_full)
full_rmse = np.sqrt(mean_squared_error(y_test_full, full_pred))
print("Full model train R²:", train_r2)
print("Full model test R²:", test_r2)
print("Full model test RMSE (°C):", full_rmse)
print("Baseline test R²:", baseline_r2)
print("Full model test R²:", test_r2)
print("Change in test R² (full minus baseline):", test_r2 - baseline_r2)
print("Feature coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")
# Each coefficient is the predicted high change for a one-unit feature change,
# holding the others fixed. Low and summer are expected to raise the high;
# precipitation and wind may lower it. Compare train and test R²: a small gap
# suggests similar performance on unseen days, while a large gap may signal
# overfitting. The printed R² difference shows how much the added features help.


# Task 6: evaluate the full model on held-out days.
plt.figure(figsize=(6, 6))
plt.scatter(full_pred, y_test_full, alpha=0.75)
line_min = min(full_pred.min(), y_test_full.min())
line_max = max(full_pred.max(), y_test_full.max())
plt.plot([line_min, line_max], [line_min, line_max], "k--", label="Perfect prediction")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted High Temperature (°C)")
plt.ylabel("Actual High Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "predicted_vs_actual_high.png")
plt.close()
# Errors look roughly even across the range, though a few mid-range days are
# noticeably under-predicted (their points sit above the diagonal).


# Charlotte's 2023 dataset has 365 days; the 20% test set has 73 days. The
# full model's RMSE is 2.54°C and its test R² is 0.909, so predictions are
# typically about 2.5°C from the recorded high. The largest absolute
# coefficient is is_summer (+1.295), which raises the predicted high by about
# 1.3°C holding the other features fixed. Low and summer raise predictions;
# precipitation and wind lower them. Train R² (0.882) and test R² (0.909) are
# close, so there is no large train/test gap. One surprising result is that
# test R² is slightly higher than train R²; the full model also improves test
# R² by 0.042 over the baseline, with RMSE falling from 3.07°C to 2.54°C.
largest_effect_idx = int(np.argmax(np.abs(model.coef_)))
largest_effect_feature = feature_cols[largest_effect_idx]
direction = "raises" if model.coef_[largest_effect_idx] > 0 else "lowers"
print("Summary:")
print(f"The dataset has {len(df)} days, with {len(y_test_full)} days in the test set.")
print(
    f"The full model has test RMSE {full_rmse:.2f}°C and R² {test_r2:.3f}; "
    f"a typical prediction is off by about {full_rmse:.2f}°C."
)
print(
    f"The largest absolute coefficient is {largest_effect_feature} "
    f"({model.coef_[largest_effect_idx]:+.3f}), which {direction} the predicted high."
)
print(
    "For your summary comment, note whether the full model beat the baseline, "
    "whether coefficient signs matched expectations, and one surprising result."
)

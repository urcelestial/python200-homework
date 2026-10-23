"""Week 2 warmups: scikit-learn and linear regression."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# --- scikit-learn API ---
# Q1
temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max)
predictions = model.predict([[6], [18]])
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for a low of 6 degrees:", predictions[0])
print("Predicted high for a low of 18 degrees:", predictions[1])

# Q2
x = np.array([10, 20, 30, 40, 50])
print("Original x shape:", x.shape)
x_2d = x.reshape(-1, 1)
print("Reshaped x shape:", x_2d.shape)



rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Q1
plt.figure(figsize=(8, 5))
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature (°C)")
plt.ylabel("Daily High Temperature (°C)")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "high_vs_low.png")
plt.close()

# Q2
X = temp_min.reshape(-1, 1)
y = temp_max
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_baseline = model.score(X_test, y_test)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", rmse)
print("Test R²:", r2_baseline)

# Q4
X_full = np.column_stack([temp_min, is_rainy])
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)
model_full = LinearRegression()
model_full.fit(X_full_train, y_full_train)
y_full_pred = model_full.predict(X_full_test)
r2_full = model_full.score(X_full_test, y_full_test)
print("One-feature test R²:", r2_baseline)
print("Two-feature test R²:", r2_full)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# Q5
plt.figure(figsize=(6, 6))
plt.scatter(y_full_pred, y_full_test)
line_min = min(y_full_pred.min(), y_full_test.min())
line_max = max(y_full_pred.max(), y_full_test.max())
plt.plot([line_min, line_max], [line_min, line_max], "k--", label="Perfect prediction")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted High Temperature (°C)")
plt.ylabel("Actual High Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "predicted_vs_actual_warmup.png")
plt.close()
# A point above the diagonal has an actual high above the prediction (an
# under-prediction); a point below it has an actual high below the prediction
# (an over-prediction).

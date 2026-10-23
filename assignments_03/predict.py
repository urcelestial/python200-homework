import json
from pathlib import Path

import pandas as pd
import joblib

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "weather_classifier.pkl"
METADATA_PATH = BASE_DIR / "models" / "weather_classifier_metadata.json"

# Task 7: load the saved model and predict hypothetical days.
if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise FileNotFoundError(
        "Saved model files are missing. Run train_classifier.py first to create "
        "models/weather_classifier.pkl and models/weather_classifier_metadata.json."
    )

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
feature_names = metadata["feature_names"]
print("City:", metadata["city"])
print("Features:", feature_names)
print("Test accuracy:", metadata["test_accuracy"])

hypothetical_days = pd.DataFrame(
    [
        [18.0, 10.0, 0.0, 8.0],
        [35.0, 24.0, 7.0, 42.0],
        [26.5, 16.0, 0.0, 10.0],
        [25.0, 13.0, 2.0, 15.0],
    ],
    columns=feature_names,
    index=["clearly_good", "clearly_bad", "borderline_hot", "borderline_mild"],
)
predicted = model.predict(hypothetical_days)
good_class_index = list(model[-1].classes_).index(1)
good_probabilities = model.predict_proba(hypothetical_days)[:, good_class_index]
for day_name, row, label, probability in zip(
    hypothetical_days.index, hypothetical_days.to_dict(orient="records"), predicted, good_probabilities
):
    label_text = "good" if label == 1 else "skip"
    print(
        f"{day_name}: inputs={row}, prediction={label_text}, "
        f"probability_good={probability:.2f}"
    )

# Task 8: Reflection
# The borderline_hot day has probability 0.33 for good, so the model leans skip; probabilities near 0.5, including 0.52, are uncertain and should lead to a cautious recommendation or a request for more context.
# If this script runs before training, the model and metadata files do not exist; the FileNotFoundError above explains how to create them. A cloud pipeline would accept daily data as an input rather than use these hard-coded examples.

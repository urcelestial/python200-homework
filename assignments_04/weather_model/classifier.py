"""Reusable interface for the saved running-weather classifier."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


FEATURES = (
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
)


@dataclass
class Prediction:
    """A running-weather label and its probability of being a good day."""

    label: str
    probability: float


class WeatherClassifier:
    """Load a saved pipeline and predict whether weather is good for running."""

    def __init__(self, model_path: str | Path) -> None:
        """Load a fitted model pipeline from a pickle file.

        Args:
            model_path: Path to the saved pipeline.

        Raises:
            FileNotFoundError: If no model file exists at ``model_path``.
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)

    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """Predict running suitability for each weather record.

        Args:
            days: Daily weather dictionaries containing all four model features.

        Returns:
            One prediction per input day, in input order.

        Raises:
            ValueError: If any day is missing one or more required features.
        """
        if not days:
            return []

        rows = []
        for index, day in enumerate(days):
            missing = [feature for feature in FEATURES if feature not in day]
            if missing:
                missing_names = ", ".join(missing)
                raise ValueError(f"Day {index} is missing required features: {missing_names}")
            rows.append({feature: day[feature] for feature in FEATURES})

        feature_frame = pd.DataFrame(rows, columns=FEATURES)
        labels = self.model.predict(feature_frame)
        probabilities = self.model.predict_proba(feature_frame)[:, 1]
        return [
            Prediction(label="good" if label == 1 else "skip", probability=float(probability))
            for label, probability in zip(labels, probabilities)
        ]

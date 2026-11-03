"""Behavioral tests for the reusable weather classifier."""

import joblib
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import Prediction, WeatherClassifier


FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


@pytest.fixture
def classifier(tmp_path):
    """Build and save a tiny deterministic model for testing the interface."""
    training_days = [
        [18.0, 10.0, 0.0, 8.0],
        [22.0, 12.0, 1.0, 12.0],
        [16.0, 8.0, 0.5, 5.0],
        [28.0, 18.0, 0.0, 10.0],
        [-2.0, -8.0, 12.0, 45.0],
        [3.0, -4.0, 8.0, 38.0],
        [32.0, 23.0, 7.0, 42.0],
        [5.0, -1.0, 4.0, 35.0],
    ]
    labels = [1, 1, 1, 0, 0, 0, 0, 0]
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=1)),
        ]
    )
    model.fit(pd.DataFrame(training_days, columns=FEATURES), labels)
    model_path = tmp_path / "tiny_weather_model.pkl"
    joblib.dump(model, model_path)
    return WeatherClassifier(model_path)


GOOD_DAY = {
    "temperature_2m_max": 19.0,
    "temperature_2m_min": 11.0,
    "precipitation_sum": 0.0,
    "wind_speed_10m_max": 7.0,
}
BAD_DAY = {
    "temperature_2m_max": 2.0,
    "temperature_2m_min": -6.0,
    "precipitation_sum": 10.0,
    "wind_speed_10m_max": 40.0,
}


@pytest.mark.parametrize(
    ("day", "expected_label"),
    [(GOOD_DAY, "good"), (BAD_DAY, "skip")],
)
def test_clearly_good_and_bad_days_get_expected_labels(classifier, day, expected_label):
    result = classifier.predict([day])
    assert result[0].label == expected_label


def test_probability_is_between_zero_and_one(classifier):
    prediction = classifier.predict([GOOD_DAY])[0]
    assert 0.0 <= prediction.probability <= 1.0


def test_three_days_return_ordered_prediction_objects(classifier):
    days = [GOOD_DAY, BAD_DAY, GOOD_DAY]
    predictions = classifier.predict(days)
    assert len(predictions) == 3
    assert all(isinstance(prediction, Prediction) for prediction in predictions)
    assert [prediction.label for prediction in predictions] == ["good", "skip", "good"]


def test_empty_days_return_empty_list(classifier):
    assert classifier.predict([]) == []


def test_missing_feature_raises_value_error(classifier):
    incomplete_day = {key: value for key, value in GOOD_DAY.items() if key != "precipitation_sum"}
    with pytest.raises(ValueError, match="precipitation_sum"):
        classifier.predict([incomplete_day])


def test_missing_model_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        WeatherClassifier(tmp_path / "missing.pkl")


# Deliberately returning "skip" for every prediction made
# test_clearly_good_and_bad_days_get_expected_labels[day0-good] fail:
# it expected "good" but received "skip".

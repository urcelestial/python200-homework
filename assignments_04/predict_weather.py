"""Run the saved weather model against a few example forecasts."""

from pathlib import Path

from weather_model import WeatherClassifier


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "weather_classifier.pkl"


def main() -> None:
    """Print predictions for representative weather conditions."""
    classifier = WeatherClassifier(MODEL_PATH)
    days = [
        {
            "name": "mild and dry",
            "temperature_2m_max": 19.0,
            "temperature_2m_min": 10.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 8.0,
        },
        {
            "name": "cold, rainy, and windy",
            "temperature_2m_max": 5.0,
            "temperature_2m_min": -8.0,
            "precipitation_sum": 10.0,
            "wind_speed_10m_max": 60.0,
        },
        {
            "name": "hot and dry",
            "temperature_2m_max": 33.0,
            "temperature_2m_min": 22.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 10.0,
        },
        {
            "name": "mild with light rain",
            "temperature_2m_max": 24.0,
            "temperature_2m_min": 15.0,
            "precipitation_sum": 2.0,
            "wind_speed_10m_max": 17.0,
        },
    ]
    predictions = classifier.predict(days)
    for day, prediction in zip(days, predictions):
        inputs = {key: value for key, value in day.items() if key != "name"}
        print(
            f"{day['name']}: inputs={inputs}, label={prediction.label}, "
            f"probability_good={prediction.probability:.2f}"
        )


if __name__ == "__main__":
    main()


# If this guard were omitted, importing predict_weather would immediately load
# the model and print predictions, creating side effects for another script.

# Reflection:
# Fixed feature order prevents a caller's dictionary insertion order from
# silently swapping values between columns, which could produce wrong results.
# The tests train a tiny model in tmp_path, so they are isolated, repeatable,
# and do not depend on the real model file being present or up to date.
# This component is ready for batches because predict accepts a list and
# preserves its order. Before production, I would add input range/type checks
# and monitoring for missing or drifting weather data.

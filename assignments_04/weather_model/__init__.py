"""Public interface for the weather model component."""

from .classifier import Prediction, WeatherClassifier

__all__ = ["WeatherClassifier", "Prediction"]

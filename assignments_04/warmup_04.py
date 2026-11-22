"""Week 4 warmups: metadata, dataclasses, and pytest."""

import json
from dataclasses import dataclass
from pathlib import Path
import tempfile

import pytest


# Question 1
class ModelInfo:
    """Load and retain model metadata from a JSON file."""

    def __init__(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        """Return the number of model features."""
        return len(self.data["features"])


# Loading once in __init__ avoids repeated file I/O and JSON parsing when this
# method is called many times; feature_count can use the already-loaded data.


# Question 2
@dataclass
class DayForecast:
    """A concise forecast for one day."""

    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        """Return a readable one-line forecast."""
        rain_text = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c:.1f}C, {rain_text}"


# @dataclass wrote the __init__ method for us, assigning the three fields.


# Question 3
def parse_temperature(text: str) -> float:
    """Convert text to a temperature, raising a clear error for invalid input."""
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"Invalid temperature: {text!r}") from error


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Without match="temperature", pytest.raises(ValueError) would not catch an
# incorrect or unhelpful error message that still had the right exception type.


def main() -> None:
    """Print examples for the warmup questions."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        metadata_path = Path(temporary_directory) / "metadata.json"
        metadata_path.write_text(
            json.dumps({"features": ["temperature", "rain", "wind"]}),
            encoding="utf-8",
        )
        model_info = ModelInfo(metadata_path)
        print("Feature count:", model_info.feature_count())

    forecasts = [
        DayForecast("2023-06-15", 24.1, True),
        DayForecast("2023-06-16", 26.0, False),
    ]
    for forecast in forecasts:
        print(forecast.summary())


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from .schemas import WeatherResponse

# HourlyReading is a lightweight dataclass for parsed rows; WeatherResponse uses
# Pydantic because external response data needs validation before conversion.
@dataclass
class HourlyReading:
    """Represents a single hour's weather reading, including timestamp, temperature, and precipitation."""
    timestamp: str
    temperature_c: float
    precipitation_mm: float

def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert columnar hourly data into one HourlyReading per hour.

    Args:
        response: Validated weather response containing hourly columns.

    Returns:
        A list of hourly readings in timestamp order.
    """
    readings = []
    total_hours = len(response.hourly.time)

    for i in range(total_hours):
        reading = HourlyReading(
        timestamp=response.hourly.time[i],
        temperature_c=response.hourly.temperature_2m[i],
        precipitation_mm=response.hourly.precipitation[i],
    )
        readings.append(reading)

    return readings

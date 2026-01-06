"""
Mock Weather API for Hello World Weather Tutorial.

Provides a simple, self-contained weather data simulation service for the
Hello World Weather tutorial application. This mock API eliminates external
dependencies while demonstrating realistic weather data patterns and API
interaction workflows within the Osprey Agent Framework.

The mock service generates randomized but realistic weather data for any
location string provided, enabling complete tutorial functionality without
requiring external weather service API keys, network connectivity, or rate
limiting considerations.

Architecture Design:
    The mock API follows standard service patterns that can be easily replaced
    with real weather service integrations:

    1. **Data Model**: Structured weather reading with type safety
    2. **Service Interface**: Clean API methods matching real weather services
    3. **Randomization**: Realistic weather variation with random temperatures and conditions
    4. **Flexibility**: Accepts any location string from LLM extraction
    5. **Extensibility**: Easy to adjust temperature ranges or weather parameters

Location Handling:
    The mock API accepts **any location string** and returns simulated weather data
    for it. This allows the LLM-based location extraction to pass through arbitrary
    location names (cities, landmarks, "local", etc.) without restriction.

Weather Generation:
    - **Temperature Range**: Random value between 0°C and 35°C
    - **Conditions**: Randomly selected from a diverse set of weather conditions
    - **Timestamp**: Current time when the weather reading is generated

.. note::
   This is a tutorial-focused mock service designed for learning and development.
   Production applications should integrate with real weather APIs like
   OpenWeatherMap, WeatherAPI, or similar services.

.. warning::
   The mock service generates random data and should not be used for any
   real weather-dependent decisions. All temperature values are in Celsius.
"""

import random
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CurrentWeatherReading:
    """Structured data model for current weather conditions.

    Type-safe container for weather data with essential fields: location, temperature
    (Celsius), conditions, and timestamp. Designed to match common weather API patterns
    for easy replacement with real services in production.

    Args:
        location: Human-readable location name
        temperature: Temperature in degrees Celsius
        conditions: Weather conditions (e.g., "Sunny", "Rainy")
        timestamp: When the weather data was generated
    """

    location: str
    temperature: float  # Celsius
    conditions: str
    timestamp: datetime


class SimpleWeatherAPI:
    """Mock weather service for Hello World Weather tutorial.

    Generates randomized weather data for any location string without external APIs.
    Accepts any location (cities, "local", etc.) and returns structured weather data
    with random temperature (0-35°C) and conditions.

    This enables complete tutorial functionality with no API keys, network calls,
    or rate limiting. The simple design supports LLM-based location extraction that
    can parse arbitrary location names from natural language.

    Note:
        For tutorial/development only. Weather data is random - not for real decisions.
        All temperatures in Celsius.
    """

    # Weather condition options for random selection
    ALL_CONDITIONS = [
        "Sunny",
        "Partly Cloudy",
        "Cloudy",
        "Overcast",
        "Foggy",
        "Rainy",
        "Drizzle",
        "Thunderstorms",
        "Snow",
        "Windy",
        "Clear",
    ]
    """Diverse weather conditions for random selection."""

    def get_current_weather(self, location: str) -> CurrentWeatherReading:
        """Get current weather for any location string.

        Accepts any location (city names, "local", etc.) and generates random weather:
        - Temperature: 0-35°C
        - Conditions: Random from ALL_CONDITIONS list
        - Timestamp: Current time
        - Location: Preserved exactly as provided

        Args:
            location: Any location string (no validation/restriction)

        Returns:
            CurrentWeatherReading with randomized weather data
        """

        # Generate completely random weather data for any location
        temperature = random.randint(0, 35)  # Temperature range: 0-35°C
        conditions = random.choice(self.ALL_CONDITIONS)

        return CurrentWeatherReading(
            location=location,  # Preserve exact location string provided
            temperature=float(temperature),
            conditions=conditions,
            timestamp=datetime.now(),
        )


# Global API instance for application-wide weather data access
weather_api = SimpleWeatherAPI()
"""Global mock weather API instance for use throughout the application."""

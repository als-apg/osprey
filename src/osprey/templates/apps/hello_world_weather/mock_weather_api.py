"""
Mock Weather API for Hello World Weather Tutorial.

Provides a simple, self-contained weather data simulation service for the
Hello World Weather tutorial application. This mock API eliminates external
dependencies while demonstrating realistic weather data patterns and API
interaction workflows within the Osprey Agent Framework.

The mock service generates randomized but realistic weather data for three
supported cities, enabling complete tutorial functionality without requiring
external weather service API keys, network connectivity, or rate limiting
considerations.

Architecture Design:
    The mock API follows standard service patterns that can be easily replaced
    with real weather service integrations:

    1. **Data Model**: Structured weather reading with type safety
    2. **Service Interface**: Clean API methods matching real weather services
    3. **Randomization**: Realistic weather variation based on city characteristics
    4. **Error Handling**: Graceful fallback for unsupported locations
    5. **Extensibility**: Easy to add new cities or weather parameters

Supported Locations:
    - **San Francisco**: Base temperature 18°C, fog and coastal conditions
    - **New York**: Base temperature 15°C, varied seasonal conditions
    - **Prague**: Base temperature 12°C, central European weather patterns

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

    Provides a type-safe container for weather data retrieved from the mock weather
    service. This dataclass implements the standard weather data structure used
    throughout the Hello World Weather tutorial, ensuring consistency between
    API responses and framework context objects.

    The model includes essential weather information fields with appropriate
    Python types for validation and serialization. The structure matches
    common weather API response formats, making it easy to replace the mock
    service with real weather APIs in production applications.

    Data Fields:
        - **location**: Human-readable location name for display and identification
        - **temperature**: Current temperature in Celsius as floating-point value
        - **conditions**: Descriptive weather conditions string (e.g., "Sunny", "Rainy")
        - **timestamp**: Data generation timestamp for freshness tracking

    :param location: Human-readable name of the weather location
    :type location: str
    :param temperature: Current temperature in degrees Celsius
    :type temperature: float
    :param conditions: Descriptive weather conditions string
    :type conditions: str
    :param timestamp: Timestamp when weather data was generated
    :type timestamp: datetime

    .. note::
       This dataclass is designed for tutorial purposes and includes only
       essential weather fields. Production weather models might include
       additional fields like humidity, pressure, wind speed, and forecasts.

    .. warning::
       Temperature values are always in Celsius. Applications requiring
       different units should perform conversion when processing the data,
       not in the data model itself.

    Examples:
        Creating weather reading from mock data::

            >>> from datetime import datetime
            >>> reading = CurrentWeatherReading(
            ...     location="San Francisco",
            ...     temperature=18.5,
            ...     conditions="Foggy",
            ...     timestamp=datetime.now()
            ... )
            >>> print(f"{reading.location}: {reading.temperature}°C, {reading.conditions}")
            San Francisco: 18.5°C, Foggy

        Integration with framework context::

            >>> weather_context = CurrentWeatherContext(
            ...     location=reading.location,
            ...     temperature=reading.temperature,
            ...     conditions=reading.conditions,
            ...     timestamp=reading.timestamp
            ... )

    .. seealso::
       :class:`CurrentWeatherContext` : Framework context class using this data model
       :class:`SimpleWeatherAPI` : Mock service that generates instances of this class
       :doc:`/developer-guides/data-models` : Data modeling best practices guide
    """

    location: str
    temperature: float  # Celsius
    conditions: str
    timestamp: datetime


class SimpleWeatherAPI:
    """Mock weather service for Hello World Weather tutorial application.

    Provides a simple, self-contained weather data simulation service that generates
    realistic weather conditions for supported cities without requiring external
    API dependencies. The service implements standard weather API patterns that
    can be easily replaced with real weather service integrations.

    The mock service generates randomized weather data based on realistic temperature
    ranges and condition patterns for each supported city. This approach provides
    varied, believable weather data for tutorial demonstrations while maintaining
    deterministic behavior suitable for testing and development.

    Service Characteristics:
        - **No External Dependencies**: Completely self-contained with no network calls
        - **Realistic Data**: Weather patterns based on actual city characteristics
        - **Randomized Variation**: Dynamic weather changes for demonstration purposes
        - **Type Safety**: Returns structured CurrentWeatherReading objects
        - **Error Handling**: Graceful fallback for unsupported locations
        - **Extensible Design**: Easy to add new cities or weather parameters

    Supported Cities:
        The service provides weather data for three cities with distinct characteristics:

        - **San Francisco**: Maritime climate with fog and mild temperatures (base 18°C)
        - **New York**: Continental climate with varied conditions (base 15°C)
        - **Prague**: Central European climate with frequent precipitation (base 12°C)

    Weather Generation:
        Each city has:

        - **Base Temperature**: Realistic average temperature for the location
        - **Temperature Variation**: Random adjustment of -5°C to +8°C from base
        - **Condition Set**: Location-appropriate weather conditions list
        - **Random Selection**: Conditions chosen randomly from the appropriate set

    .. note::
       This mock service is designed for tutorial and development purposes.
       The weather data is generated randomly and should not be used for any
       real-world weather-dependent decisions or applications.

    .. warning::
       All temperature values are in Celsius. The service defaults to San Francisco
       for unrecognized location names to ensure consistent behavior during tutorials.

    Examples:
        Basic weather retrieval::

            >>> api = SimpleWeatherAPI()
            >>> weather = api.get_current_weather("San Francisco")
            >>> print(f"{weather.location}: {weather.temperature}°C, {weather.conditions}")
            San Francisco: 16.0°C, Foggy

        Handling unsupported locations::

            >>> weather = api.get_current_weather("Unknown City")
            >>> print(weather.location)  # Falls back to San Francisco
            San Francisco

        Integration with weather capability::

            >>> from hello_world_weather.mock_weather_api import weather_api
            >>> weather_data = weather_api.get_current_weather("New York")
            >>> # Use weather_data to create CurrentWeatherContext

    .. seealso::
       :class:`CurrentWeatherReading` : Data model returned by this service
       :class:`CurrentWeatherCapability` : Framework capability using this service
       :doc:`/developer-guides/mock-services` : Mock service development patterns
    """

    # City-specific weather pattern configuration
    CITY_DATA = {
        "San Francisco": {"base_temp": 18, "conditions": ["Sunny", "Foggy", "Partly Cloudy"]},
        "New York": {"base_temp": 15, "conditions": ["Sunny", "Rainy", "Cloudy", "Snow"]},
        "Prague": {"base_temp": 12, "conditions": ["Rainy", "Cloudy", "Partly Cloudy"]},
    }
    """Weather pattern configuration for supported cities.

    Defines base temperature and typical weather conditions for each supported
    city in the mock weather service. This configuration provides realistic
    weather variation while maintaining predictable patterns for tutorial
    and development purposes.

    Configuration Structure:
        Each city entry contains:

        - **base_temp**: Average temperature in Celsius used as baseline for variation
        - **conditions**: List of typical weather conditions for random selection
    """

    def get_current_weather(self, location: str) -> CurrentWeatherReading:
        """Retrieve current weather conditions for the specified location.

        Generates realistic weather data for the requested location using randomized
        temperature variations and condition selection based on city-specific weather
        patterns. The method provides consistent API behavior that matches real weather
        services while eliminating external dependencies for tutorial purposes.

        Location Processing:
            The method normalizes location names using title case and provides fallback
            behavior for unsupported locations:

            1. **Normalization**: Converts input to title case for consistent matching
            2. **Lookup**: Searches supported cities for weather pattern data
            3. **Fallback**: Defaults to San Francisco for unrecognized locations
            4. **Generation**: Creates randomized weather based on city patterns

        Weather Generation Process:
            For each request, the service:

            1. **Temperature**: Calculates random temperature within city-specific range
            2. **Conditions**: Selects random weather condition from city-appropriate list
            3. **Timestamp**: Records current time as data generation timestamp
            4. **Packaging**: Returns structured CurrentWeatherReading object

        :param location: Name of the location for weather data retrieval.
            Supported locations include "San Francisco", "New York", and "Prague".
            Location names are case-insensitive and normalized to title case.
        :type location: str
        :return: Current weather reading containing location, temperature, conditions,
            and timestamp data. Temperature is in Celsius, conditions are descriptive
            strings, and timestamp reflects data generation time.
        :rtype: CurrentWeatherReading

        .. note::
           Unsupported locations automatically default to San Francisco weather patterns
           to ensure consistent tutorial behavior. This fallback prevents errors while
           demonstrating the expected data structure.

        .. warning::
           Generated weather data is randomized and not based on actual weather conditions.
           Temperature variations range from -5°C to +8°C from the base temperature,
           which may produce unrealistic values in some cases.

        Examples:
            Retrieving weather for supported cities::

                >>> api = SimpleWeatherAPI()
                >>> sf_weather = api.get_current_weather("san francisco")
                >>> print(f"{sf_weather.location}: {sf_weather.temperature}°C")
                San Francisco: 20.0°C

                >>> ny_weather = api.get_current_weather("NEW YORK")
                >>> print(f"{ny_weather.conditions}")
                Sunny

            Fallback behavior for unsupported locations::

                >>> unknown_weather = api.get_current_weather("London")
                >>> print(unknown_weather.location)  # Falls back to San Francisco
                San Francisco

            Integration with capability workflows::

                >>> def retrieve_weather_data(city_name):
                ...     api = SimpleWeatherAPI()
                ...     weather = api.get_current_weather(city_name)
                ...     return {
                ...         "location": weather.location,
                ...         "temp": weather.temperature,
                ...         "conditions": weather.conditions,
                ...         "retrieved_at": weather.timestamp.isoformat()
                ...     }

        .. seealso::
           :class:`CurrentWeatherReading` : Return type structure and field descriptions
           :attr:`CITY_DATA` : City-specific weather pattern configuration
           :class:`CurrentWeatherCapability` : Framework capability using this method
        """

        # Normalize location name for consistent matching
        location = location.title()
        if location not in self.CITY_DATA:
            # Default to San Francisco if city not found to ensure consistent tutorial behavior
            location = "San Francisco"

        city_info = self.CITY_DATA[location]

        # Generate randomized weather data based on city patterns
        temperature = city_info["base_temp"] + random.randint(-5, 8)
        conditions = random.choice(city_info["conditions"])

        return CurrentWeatherReading(
            location=location,
            temperature=float(temperature),
            conditions=conditions,
            timestamp=datetime.now(),
        )


# Global API instance for application-wide weather data access
weather_api = SimpleWeatherAPI()
"""Global instance of the mock weather API service.

Provides a singleton weather service instance for use throughout the Hello World
Weather application. This global instance eliminates the need for dependency injection
while maintaining consistent weather data patterns across all application components.

The global instance is initialized once at module import time and can be safely
used by multiple capabilities and components concurrently. Since the mock service
is stateless and thread-safe, the singleton pattern provides optimal performance
for tutorial and development scenarios.

.. note::
   In production applications, consider using dependency injection or service
   locator patterns instead of global instances for better testability and
   configuration management.

.. warning::
   This global instance uses the default mock weather service configuration.
   Applications requiring different weather patterns or cities should create
   custom SimpleWeatherAPI instances with modified CITY_DATA.

Examples:
    Using the global weather API instance::

        >>> from hello_world_weather.mock_weather_api import weather_api
        >>> current_weather = weather_api.get_current_weather("Prague")
        >>> print(f"Temperature: {current_weather.temperature}°C")
        Temperature: 14.0°C

    Integration in capabilities::

        >>> # In CurrentWeatherCapability.execute()
        >>> weather = weather_api.get_current_weather(location)
        >>> context = CurrentWeatherContext(
        ...     location=weather.location,
        ...     temperature=weather.temperature,
        ...     conditions=weather.conditions,
        ...     timestamp=weather.timestamp
        ... )

.. seealso::
   :class:`SimpleWeatherAPI` : The weather service class instantiated here
   :class:`CurrentWeatherCapability` : Primary consumer of this global instance
"""

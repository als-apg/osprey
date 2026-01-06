"""
Tests for Hello World Weather template mock API.

Tests the mock weather API that is directly importable (not a template file).
This verifies the API works correctly with any location string, generating
random weather data regardless of the location provided.
"""

from datetime import datetime

from osprey.templates.apps.hello_world_weather.mock_weather_api import (
    SimpleWeatherAPI,
    weather_api,
)

# Common test locations used across multiple tests
COMMON_TEST_LOCATIONS = ["San Francisco", "New York", "Prague", "Tokyo", "London", "local"]


class TestSimpleWeatherAPI:
    """Test suite for SimpleWeatherAPI."""

    def test_mock_api_handles_local_location(self):
        """Test that mock API handles 'local' location string."""
        api = SimpleWeatherAPI()
        weather = api.get_current_weather("local")

        assert weather.location == "local"
        assert isinstance(weather.temperature, float)
        assert isinstance(weather.conditions, str)
        assert isinstance(weather.timestamp, datetime)

    def test_mock_api_handles_any_location_string(self):
        """Test that mock API accepts and preserves any location string."""
        api = SimpleWeatherAPI()

        # Test various location strings (common + edge cases)
        test_locations = COMMON_TEST_LOCATIONS + [
            "UnknownCity",
            "Some Random Place",
            "123 Main St",
        ]

        for location in test_locations:
            weather = api.get_current_weather(location)
            # Location should be preserved exactly as provided
            assert weather.location == location
            assert isinstance(weather.temperature, float)
            assert isinstance(weather.conditions, str)
            assert isinstance(weather.timestamp, datetime)

    def test_mock_api_preserves_case(self):
        """Test that mock API preserves the exact case of location strings."""
        api = SimpleWeatherAPI()

        # Test that case is preserved exactly
        test_cases = [
            "new york",
            "NEW YORK",
            "New York",
            "NeW yOrK",
            "LOCAL",
            "Local",
            "local",
        ]

        for location in test_cases:
            weather = api.get_current_weather(location)
            assert weather.location == location  # Exact preservation

    def test_weather_reading_structure(self):
        """Test that CurrentWeatherReading has expected structure."""
        api = SimpleWeatherAPI()
        weather = api.get_current_weather("San Francisco")

        # Verify all required fields exist
        assert hasattr(weather, "location")
        assert hasattr(weather, "temperature")
        assert hasattr(weather, "conditions")
        assert hasattr(weather, "timestamp")

        # Verify types
        assert isinstance(weather.location, str)
        assert isinstance(weather.temperature, float)
        assert isinstance(weather.conditions, str)
        assert isinstance(weather.timestamp, datetime)

    def test_temperature_range_universal(self):
        """Test that generated temperatures are within the universal range (0-35°C)."""
        api = SimpleWeatherAPI()

        # Test multiple times with various locations
        for _ in range(20):
            for location in COMMON_TEST_LOCATIONS:
                weather = api.get_current_weather(location)
                # Universal temperature range: 0-35°C
                assert 0 <= weather.temperature <= 35

    def test_conditions_from_valid_set(self):
        """Test that weather conditions come from the ALL_CONDITIONS list."""
        api = SimpleWeatherAPI()

        # Get valid conditions from the API
        valid_conditions = api.ALL_CONDITIONS

        # Test multiple times with various locations
        for _ in range(20):
            for location in COMMON_TEST_LOCATIONS:
                weather = api.get_current_weather(location)
                assert weather.conditions in valid_conditions

    def test_global_weather_api_instance(self):
        """Test that global weather_api instance works correctly."""
        # Test that global instance works with any location
        weather1 = weather_api.get_current_weather("San Francisco")
        assert weather1.location == "San Francisco"

        weather2 = weather_api.get_current_weather("Tokyo")
        assert weather2.location == "Tokyo"

        weather3 = weather_api.get_current_weather("local")
        assert weather3.location == "local"

    def test_timestamp_generation(self):
        """Test that timestamps are generated correctly."""
        api = SimpleWeatherAPI()

        before = datetime.now()
        weather = api.get_current_weather("Anywhere")
        after = datetime.now()

        # Timestamp should be between before and after
        assert before <= weather.timestamp <= after

    def test_randomization_works(self):
        """Test that weather data is randomized between calls."""
        api = SimpleWeatherAPI()

        # Generate multiple readings for the same location
        readings = [api.get_current_weather("San Francisco") for _ in range(20)]

        # Check that we get variation in temperature
        temperatures = [r.temperature for r in readings]
        assert len(set(temperatures)) > 1, "Should have temperature variation"

        # Check that we get variation in conditions
        conditions = [r.conditions for r in readings]
        assert len(set(conditions)) > 1, "Should have conditions variation"

    def test_empty_string_location(self):
        """Test that API handles empty string location."""
        api = SimpleWeatherAPI()
        weather = api.get_current_weather("")

        # Should still work and preserve empty string
        assert weather.location == ""
        assert isinstance(weather.temperature, float)
        assert isinstance(weather.conditions, str)

    def test_special_characters_in_location(self):
        """Test that API handles special characters in location strings."""
        api = SimpleWeatherAPI()

        special_locations = [
            "São Paulo",
            "Zürich",
            "Москва",
            "東京",
            "Location-With-Dashes",
            "Location_With_Underscores",
        ]

        for location in special_locations:
            weather = api.get_current_weather(location)
            assert weather.location == location

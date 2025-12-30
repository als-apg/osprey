"""
Base Test Class for Routing Components

Provides shared fixtures, utilities, and test patterns for routing-related tests.
This consolidates common patterns from test_routing_analytics.py, test_routing_cache.py,
and test_routing_feedback.py.
"""

import json
import time
from pathlib import Path
from typing import Any

import pytest

# ============================================================================
# Base Fixtures
# ============================================================================


class RoutingTestBase:
    """Base class for routing component tests with shared utilities."""

    # Subclasses should override these
    SYSTEM_CLASS = None
    DEFAULT_MAX_HISTORY = 100
    DEFAULT_PERSISTENCE_ENABLED = True

    @pytest.fixture
    def temp_storage_path(self, tmp_path):
        """Create temporary storage file path."""
        return tmp_path / "test_storage.json"

    @pytest.fixture
    def system_with_persistence(self, temp_storage_path):
        """Create system with persistence enabled."""
        if self.SYSTEM_CLASS is None:
            pytest.skip("SYSTEM_CLASS not defined")
        return self._create_system(
            persistence_path=temp_storage_path,
            enable_persistence=True,
        )

    @pytest.fixture
    def system_no_persistence(self):
        """Create system without persistence."""
        if self.SYSTEM_CLASS is None:
            pytest.skip("SYSTEM_CLASS not defined")
        return self._create_system(enable_persistence=False)

    def _create_system(self, **kwargs):
        """Create system instance with given parameters.

        Subclasses should override to provide system-specific defaults.
        """
        defaults = {
            "max_history": self.DEFAULT_MAX_HISTORY,
            "enable_persistence": self.DEFAULT_PERSISTENCE_ENABLED,
        }
        defaults.update(kwargs)
        return self.SYSTEM_CLASS(**defaults)

    # ========================================================================
    # Common Test Patterns
    # ========================================================================

    def verify_persistence_disabled(self, system, temp_path: Path):
        """Verify that persistence is disabled and no file is created."""
        assert not temp_path.exists()

    def verify_export_success(self, export_path: Path, expected_keys: list[str]):
        """Verify export file was created and contains expected keys."""
        assert export_path.exists()

        with open(export_path) as f:
            data = json.load(f)

        for key in expected_keys:
            assert key in data, f"Expected key '{key}' not found in export"

        assert "exported_at" in data
        return data

    def verify_export_creates_directory(self, system, tmp_path: Path, export_method: str):
        """Verify export creates parent directories."""
        export_path = tmp_path / "subdir" / "export.json"
        export_func = getattr(system, export_method)
        success = export_func(export_path)

        assert success
        assert export_path.exists()

    def verify_max_history_enforcement(
        self,
        system,
        max_history: int,
        add_item_func,
        get_items_func,
        item_count: int = None,
    ):
        """Verify max history limit is enforced.

        Args:
            system: The system instance
            max_history: Expected max history size
            add_item_func: Function to add items (called with index)
            get_items_func: Function to get items list
            item_count: Number of items to add (default: max_history * 2)
        """
        if item_count is None:
            item_count = max_history * 2

        # Add more than max_history items
        for i in range(item_count):
            add_item_func(i)

        items = get_items_func()
        assert len(items) == max_history, f"Expected {max_history} items, got {len(items)}"

        # Verify most recent items are kept
        return items

    def verify_clear_operation(self, system, clear_method: str, check_funcs: list[callable]):
        """Verify clear operation empties all data structures.

        Args:
            system: The system instance
            clear_method: Name of the clear method
            check_funcs: List of functions that should return 0 after clear
        """
        clear_func = getattr(system, clear_method)
        clear_func()

        for check_func in check_funcs:
            result = check_func()
            if isinstance(result, (list, dict)):
                assert len(result) == 0, f"Expected empty collection, got {len(result)} items"
            else:
                assert result == 0, f"Expected 0, got {result}"

    def verify_persistence_across_sessions(
        self,
        temp_path: Path,
        create_system_func,
        add_data_func,
        verify_data_func,
        data_count: int = 5,
    ):
        """Verify data persists across sessions.

        Args:
            temp_path: Path to persistence file
            create_system_func: Function to create system instance
            add_data_func: Function to add data (called with system, index)
            verify_data_func: Function to verify data (called with system)
            data_count: Number of data items to add
        """
        # Session 1: Create and populate
        system1 = create_system_func(temp_path)
        for i in range(data_count):
            add_data_func(system1, i)

        # Session 2: Load and verify
        system2 = create_system_func(temp_path)
        verify_data_func(system2, data_count)

    # ========================================================================
    # Common Assertion Helpers
    # ========================================================================

    def assert_dataclass_fields(self, obj, expected_fields: dict[str, Any]):
        """Assert dataclass has expected field values."""
        for field, expected_value in expected_fields.items():
            actual_value = getattr(obj, field)
            assert actual_value == expected_value, (
                f"Field '{field}': expected {expected_value}, got {actual_value}"
            )

    def assert_statistics_valid(self, stats, expected_totals: dict[str, int] | None = None):
        """Assert statistics object has valid values."""
        # All counts should be non-negative
        for attr in dir(stats):
            if not attr.startswith("_"):
                value = getattr(stats, attr)
                if isinstance(value, (int, float)) and "count" in attr.lower():
                    assert value >= 0, f"Statistic '{attr}' should be non-negative, got {value}"

        # Check expected totals if provided
        if expected_totals:
            for attr, expected in expected_totals.items():
                actual = getattr(stats, attr)
                assert actual == expected, f"Expected {attr}={expected}, got {actual}"

    def assert_time_series_valid(self, series: list[tuple], min_length: int = 0):
        """Assert time series data is valid."""
        assert len(series) >= min_length, (
            f"Expected at least {min_length} entries, got {len(series)}"
        )

        for entry in series:
            assert len(entry) == 2, f"Expected (datetime, value) tuple, got {entry}"
            dt, value = entry
            # First element should be datetime-like (has year attribute)
            assert hasattr(dt, "year"), f"Expected datetime object, got {type(dt)}"
            # Second element should be numeric
            assert isinstance(value, (int, float)), f"Expected numeric value, got {type(value)}"

    def assert_rate_calculation(self, rate: float, numerator: int, denominator: int):
        """Assert rate calculation is correct."""
        if denominator == 0:
            assert rate == 0.0, f"Expected rate=0.0 for zero denominator, got {rate}"
        else:
            expected = numerator / denominator
            assert abs(rate - expected) < 0.01, f"Expected rate={expected}, got {rate}"

    # ========================================================================
    # Pattern Extraction Helpers
    # ========================================================================

    def verify_pattern_extraction(
        self,
        system,
        test_cases: list[tuple],
        extract_method: str = "_extract_pattern",
    ):
        """Verify pattern extraction for multiple test cases.

        Args:
            system: The system instance
            test_cases: List of (input, expected_output) tuples
            extract_method: Name of the pattern extraction method
        """
        extract_func = getattr(system, extract_method)

        for input_text, expected_pattern in test_cases:
            actual_pattern = extract_func(input_text)
            assert actual_pattern == expected_pattern, (
                f"Input '{input_text}': expected '{expected_pattern}', got '{actual_pattern}'"
            )


# ============================================================================
# Shared Test Data Generators
# ============================================================================


class TestDataGenerator:
    """Generate test data for routing components."""

    @staticmethod
    def generate_queries(count: int, prefix: str = "query") -> list[str]:
        """Generate test queries."""
        return [f"{prefix} {i}" for i in range(count)]

    @staticmethod
    def generate_projects(count: int, prefix: str = "project") -> list[str]:
        """Generate test project names."""
        return [f"{prefix}{i}" for i in range(count)]

    @staticmethod
    def generate_confidences(count: int, base: float = 0.8, variance: float = 0.1) -> list[float]:
        """Generate test confidence values."""
        import random

        random.seed(42)  # Reproducible
        return [
            max(0.0, min(1.0, base + random.uniform(-variance, variance))) for _ in range(count)
        ]

    @staticmethod
    def generate_timestamps(count: int, start_time: float | None = None) -> list[float]:
        """Generate test timestamps."""
        if start_time is None:
            start_time = time.time()
        return [start_time + i for i in range(count)]


# ============================================================================
# Common Mock Helpers
# ============================================================================


class MockHelpers:
    """Helper functions for creating mocks in routing tests."""

    @staticmethod
    def create_mock_decision(
        project: str = "test_project",
        confidence: float = 0.9,
        **kwargs,
    ):
        """Create a mock routing decision."""
        from unittest.mock import Mock

        decision = Mock()
        decision.project_name = project
        decision.confidence = confidence
        decision.reasoning = kwargs.get("reasoning", "Test reasoning")
        decision.alternative_projects = kwargs.get("alternative_projects", [])
        decision.timestamp = kwargs.get("timestamp", time.time())
        decision.hit_count = kwargs.get("hit_count", 0)
        return decision

    @staticmethod
    def create_mock_metric(
        query: str = "test query",
        project: str = "test_project",
        confidence: float = 0.9,
        **kwargs,
    ):
        """Create a mock routing metric."""
        from unittest.mock import Mock

        metric = Mock()
        metric.query = query
        metric.project_selected = project
        metric.confidence = confidence
        metric.routing_time_ms = kwargs.get("routing_time_ms", 50.0)
        metric.cache_hit = kwargs.get("cache_hit", False)
        metric.mode = kwargs.get("mode", "automatic")
        metric.success = kwargs.get("success", True)
        metric.error = kwargs.get("error", None)
        metric.timestamp = kwargs.get("timestamp", time.time())
        return metric

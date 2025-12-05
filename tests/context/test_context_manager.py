"""
Comprehensive tests for ContextManager extract_from_step fix.

This test suite validates the critical bug fix for handling multiple contexts
of the same type in extract_from_step().
"""

from typing import ClassVar

import pytest
from pydantic import Field

from osprey.context.base import CapabilityContext
from osprey.context.context_manager import ContextManager

# ===================================================================
# Test Context Classes
# ===================================================================


class PVAddressesContext(CapabilityContext):
    """Test context for PV addresses."""

    CONTEXT_TYPE: ClassVar[str] = "PV_ADDRESSES"
    pvs: list[str] = Field(description="List of PV addresses")

    def get_summary(self) -> dict:
        return {"type": "PV Addresses", "count": len(self.pvs), "pvs": self.pvs}

    def get_access_details(self, key: str) -> dict:
        return {"pvs": self.pvs, "count": len(self.pvs)}


class TimeRangeContext(CapabilityContext):
    """Test context for time range."""

    CONTEXT_TYPE: ClassVar[str] = "TIME_RANGE"
    start: str = Field(description="Start time")
    end: str | None = Field(default=None, description="End time")

    def get_summary(self) -> dict:
        return {"type": "Time Range", "start": self.start, "end": self.end}

    def get_access_details(self, key: str) -> dict:
        return {"start": self.start, "end": self.end}


class CurrentWeatherContext(CapabilityContext):
    """Test context for current weather."""

    CONTEXT_TYPE: ClassVar[str] = "CURRENT_WEATHER"
    location: str = Field(description="Location name")
    temp: float = Field(description="Temperature")

    def get_summary(self) -> dict:
        return {"type": "Current Weather", "location": self.location, "temperature": self.temp}

    def get_access_details(self, key: str) -> dict:
        return {"location": self.location, "temp": self.temp}


class ArchiverDataContext(CapabilityContext):
    """Test context for archiver data."""

    CONTEXT_TYPE: ClassVar[str] = "ARCHIVER_DATA"
    data: list = Field(description="Archived data")

    def get_summary(self) -> dict:
        return {"type": "Archiver Data", "count": len(self.data)}

    def get_access_details(self, key: str) -> dict:
        return {"data_count": len(self.data)}


class PVValuesContext(CapabilityContext):
    """Test context for PV values."""

    CONTEXT_TYPE: ClassVar[str] = "PV_VALUES"
    values: dict = Field(description="PV values")

    def get_summary(self) -> dict:
        return {"type": "PV Values", "count": len(self.values)}

    def get_access_details(self, key: str) -> dict:
        return {"values_count": len(self.values)}


class AnalysisParamsContext(CapabilityContext):
    """Test context for analysis parameters."""

    CONTEXT_TYPE: ClassVar[str] = "ANALYSIS_PARAMS"
    method: str = Field(description="Analysis method")

    def get_summary(self) -> dict:
        return {"type": "Analysis Parameters", "method": self.method}

    def get_access_details(self, key: str) -> dict:
        return {"method": self.method}


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def mock_state():
    """Create a minimal AgentState for testing."""
    return {"capability_context_data": {}}


@pytest.fixture
def context_manager(mock_state):
    """Create a ContextManager instance for testing."""
    return ContextManager(mock_state)


# ===================================================================
# Test Section 4.1: Core Bug Fix - Multiple Same-Type Contexts
# ===================================================================


def test_extract_multiple_same_type_contexts(context_manager, mock_state):
    """Test the PRIMARY bug: multiple contexts of same type returned as list."""
    # Store two weather contexts with different keys
    sf_weather = CurrentWeatherContext(location="San Francisco", temp=15.0)
    ny_weather = CurrentWeatherContext(location="New York", temp=20.0)
    context_manager.set_context("CURRENT_WEATHER", "sf_weather", sf_weather, skip_validation=True)
    context_manager.set_context("CURRENT_WEATHER", "ny_weather", ny_weather, skip_validation=True)

    # Create step with both contexts as inputs
    step = {"inputs": [{"CURRENT_WEATHER": "sf_weather"}, {"CURRENT_WEATHER": "ny_weather"}]}

    # Execute
    contexts = context_manager.extract_from_step(step, mock_state)

    # Assert: Single key with list value
    assert len(contexts) == 1
    assert "CURRENT_WEATHER" in contexts
    assert isinstance(contexts["CURRENT_WEATHER"], list)
    assert len(contexts["CURRENT_WEATHER"]) == 2

    # Both contexts present (order may vary)
    locations = {ctx.location for ctx in contexts["CURRENT_WEATHER"]}
    assert locations == {"San Francisco", "New York"}


# ===================================================================
# Test Section 4.2: Regression Tests - Single Context Still Works
# ===================================================================


def test_extract_single_context_backwards_compatibility(context_manager, mock_state):
    """Ensure single context extraction returns object (not list)."""
    # Store single PV_ADDRESSES context
    pv_context = PVAddressesContext(pvs=["SR:C01:BI:CURRENT"])
    context_manager.set_context("PV_ADDRESSES", "main", pv_context, skip_validation=True)

    step = {"inputs": [{"PV_ADDRESSES": "main"}]}

    # Execute
    contexts = context_manager.extract_from_step(step, mock_state)

    # Assert: Returns single object (NOT list)
    assert len(contexts) == 1
    assert "PV_ADDRESSES" in contexts
    assert not isinstance(contexts["PV_ADDRESSES"], list)
    assert contexts["PV_ADDRESSES"].pvs == ["SR:C01:BI:CURRENT"]


# ===================================================================
# Test Section 4.3: List Detection Tests
# ===================================================================


def test_isinstance_check_for_single_context(context_manager, mock_state):
    """Verify isinstance check works for single context."""
    pv_context = PVAddressesContext(pvs=["PV1"])
    context_manager.set_context("PV_ADDRESSES", "main", pv_context, skip_validation=True)

    step = {"inputs": [{"PV_ADDRESSES": "main"}]}
    contexts = context_manager.extract_from_step(step, mock_state)

    # isinstance(single_context, list) should be False
    assert not isinstance(contexts["PV_ADDRESSES"], list)


def test_isinstance_check_for_multiple_contexts(context_manager, mock_state):
    """Verify isinstance check works for multiple contexts."""
    pv1 = PVAddressesContext(pvs=["PV1"])
    pv2 = PVAddressesContext(pvs=["PV2"])
    context_manager.set_context("PV_ADDRESSES", "set1", pv1, skip_validation=True)
    context_manager.set_context("PV_ADDRESSES", "set2", pv2, skip_validation=True)

    step = {"inputs": [{"PV_ADDRESSES": "set1"}, {"PV_ADDRESSES": "set2"}]}
    contexts = context_manager.extract_from_step(step, mock_state)

    # isinstance(multiple_contexts, list) should be True
    assert isinstance(contexts["PV_ADDRESSES"], list)
    assert len(contexts["PV_ADDRESSES"]) == 2


# ===================================================================
# Test Section 4.4: Constraint Validation Tests
# ===================================================================


def test_extract_with_hard_constraints_multiple_same_type(context_manager, mock_state):
    """Hard constraints work correctly with multiple same-type contexts."""
    # Store contexts
    context_manager.set_context(
        "PV_ADDRESSES", "pvs1", PVAddressesContext(pvs=["PV1"]), skip_validation=True
    )
    context_manager.set_context(
        "PV_ADDRESSES", "pvs2", PVAddressesContext(pvs=["PV2"]), skip_validation=True
    )
    context_manager.set_context(
        "TIME_RANGE", "time", TimeRangeContext(start="2025-01-01"), skip_validation=True
    )

    step = {"inputs": [{"PV_ADDRESSES": "pvs1"}, {"PV_ADDRESSES": "pvs2"}, {"TIME_RANGE": "time"}]}

    # Execute with constraints requiring both types
    contexts = context_manager.extract_from_step(
        step, mock_state, constraints=["PV_ADDRESSES", "TIME_RANGE"], constraint_mode="hard"
    )

    # Assert: Two keys (both types present), PV_ADDRESSES is list
    assert len(contexts) == 2
    assert "PV_ADDRESSES" in contexts
    assert "TIME_RANGE" in contexts
    assert isinstance(contexts["PV_ADDRESSES"], list)
    assert len(contexts["PV_ADDRESSES"]) == 2
    assert not isinstance(contexts["TIME_RANGE"], list)


def test_extract_with_soft_constraints_multiple_same_type(context_manager, mock_state):
    """Soft constraints work correctly with multiple same-type contexts."""
    # Store only ARCHIVER_DATA contexts (no PV_VALUES)
    context_manager.set_context(
        "ARCHIVER_DATA", "data1", ArchiverDataContext(data=[1, 2, 3]), skip_validation=True
    )
    context_manager.set_context(
        "ARCHIVER_DATA", "data2", ArchiverDataContext(data=[4, 5, 6]), skip_validation=True
    )

    step = {"inputs": [{"ARCHIVER_DATA": "data1"}, {"ARCHIVER_DATA": "data2"}]}

    # Execute with soft constraints (at least one required)
    contexts = context_manager.extract_from_step(
        step,
        mock_state,
        constraints=["PV_VALUES", "ARCHIVER_DATA"],  # Only ARCHIVER_DATA exists
        constraint_mode="soft",
    )

    # Assert: Should succeed because at least one constraint satisfied
    assert len(contexts) == 1
    assert "ARCHIVER_DATA" in contexts
    assert isinstance(contexts["ARCHIVER_DATA"], list)
    assert len(contexts["ARCHIVER_DATA"]) == 2


# ===================================================================
# Test Section 4.5: Edge Cases
# ===================================================================


def test_extract_from_step_with_empty_inputs(context_manager, mock_state):
    """Empty inputs returns empty dict (no error)."""
    step = {"inputs": []}

    contexts = context_manager.extract_from_step(step, mock_state)

    assert contexts == {}


def test_extract_from_step_missing_context_raises_error(context_manager, mock_state):
    """Clear error when requested context doesn't exist."""
    step = {"inputs": [{"NONEXISTENT_TYPE": "key"}]}

    with pytest.raises(ValueError) as exc_info:
        context_manager.extract_from_step(step, mock_state)

    assert "not found" in str(exc_info.value).lower()


def test_extract_from_step_with_mixed_context_types(context_manager, mock_state):
    """Multiple different context types extracted correctly (all single)."""
    # Store various context types (one of each)
    context_manager.set_context(
        "PV_ADDRESSES", "pvs", PVAddressesContext(pvs=["PV1"]), skip_validation=True
    )
    context_manager.set_context(
        "TIME_RANGE", "time", TimeRangeContext(start="2025-01-01"), skip_validation=True
    )
    context_manager.set_context(
        "ANALYSIS_PARAMS", "params", AnalysisParamsContext(method="fft"), skip_validation=True
    )

    step = {
        "inputs": [{"PV_ADDRESSES": "pvs"}, {"TIME_RANGE": "time"}, {"ANALYSIS_PARAMS": "params"}]
    }

    contexts = context_manager.extract_from_step(step, mock_state)

    # Assert: Three keys, all single objects (not lists)
    assert len(contexts) == 3
    assert "PV_ADDRESSES" in contexts
    assert "TIME_RANGE" in contexts
    assert "ANALYSIS_PARAMS" in contexts
    assert not isinstance(contexts["PV_ADDRESSES"], list)
    assert not isinstance(contexts["TIME_RANGE"], list)
    assert not isinstance(contexts["ANALYSIS_PARAMS"], list)


def test_extract_from_step_mixed_single_and_multiple(context_manager, mock_state):
    """Mix of single and multiple contexts handled correctly."""
    # One TIME_RANGE, two PV_ADDRESSES
    context_manager.set_context(
        "TIME_RANGE", "time", TimeRangeContext(start="2025-01-01"), skip_validation=True
    )
    context_manager.set_context(
        "PV_ADDRESSES", "set1", PVAddressesContext(pvs=["PV1"]), skip_validation=True
    )
    context_manager.set_context(
        "PV_ADDRESSES", "set2", PVAddressesContext(pvs=["PV2"]), skip_validation=True
    )

    step = {"inputs": [{"TIME_RANGE": "time"}, {"PV_ADDRESSES": "set1"}, {"PV_ADDRESSES": "set2"}]}

    contexts = context_manager.extract_from_step(step, mock_state)

    # Assert: TIME_RANGE is single, PV_ADDRESSES is list
    assert len(contexts) == 2
    assert not isinstance(contexts["TIME_RANGE"], list)
    assert isinstance(contexts["PV_ADDRESSES"], list)
    assert len(contexts["PV_ADDRESSES"]) == 2


# ===================================================================
# Test Section 4.6: Integration with get_summaries() - List Return Type
# ===================================================================


def test_get_summaries_returns_list(context_manager, mock_state):
    """get_summaries() returns list of summary dicts, not dict."""
    # Store single PV_ADDRESSES context
    pv_context = PVAddressesContext(pvs=["PV1", "PV2"])
    context_manager.set_context("PV_ADDRESSES", "main", pv_context, skip_validation=True)

    step = {"inputs": [{"PV_ADDRESSES": "main"}]}

    # Execute get_summaries with step
    summaries = context_manager.get_summaries(step=step)

    # Assert: Returns list, not dict
    assert isinstance(summaries, list)
    assert len(summaries) == 1

    # Assert: Summary contains type field and data
    summary = summaries[0]
    assert "type" in summary
    assert summary["type"] == "PV Addresses"
    assert "PV1" in str(summary)


def test_get_summaries_with_multiple_same_type_contexts(context_manager, mock_state):
    """get_summaries() returns all contexts in list, including multiples of same type."""
    # Store two CURRENT_WEATHER contexts
    sf_weather = CurrentWeatherContext(location="San Francisco", temp=15.0)
    ny_weather = CurrentWeatherContext(location="New York", temp=20.0)
    context_manager.set_context("CURRENT_WEATHER", "sf_weather", sf_weather, skip_validation=True)
    context_manager.set_context("CURRENT_WEATHER", "ny_weather", ny_weather, skip_validation=True)

    step = {"inputs": [{"CURRENT_WEATHER": "sf_weather"}, {"CURRENT_WEATHER": "ny_weather"}]}

    # Execute get_summaries with step
    summaries = context_manager.get_summaries(step=step)

    # Assert: Returns list with 2 items (both weather contexts)
    assert isinstance(summaries, list)
    assert len(summaries) == 2

    # Both should have type "Current Weather" (or similar)
    assert all("type" in s for s in summaries)

    # Both locations present in summaries
    all_summary_text = str(summaries)
    assert "San Francisco" in all_summary_text
    assert "New York" in all_summary_text


def test_get_summaries_with_mixed_single_and_multiple(context_manager, mock_state):
    """get_summaries() handles mix of single and multiple contexts in flat list."""
    # Single TIME_RANGE, multiple PV_ADDRESSES
    context_manager.set_context(
        "TIME_RANGE", "time", TimeRangeContext(start="2025-01-01"), skip_validation=True
    )
    context_manager.set_context(
        "PV_ADDRESSES", "set1", PVAddressesContext(pvs=["PV1"]), skip_validation=True
    )
    context_manager.set_context(
        "PV_ADDRESSES", "set2", PVAddressesContext(pvs=["PV2"]), skip_validation=True
    )

    step = {"inputs": [{"TIME_RANGE": "time"}, {"PV_ADDRESSES": "set1"}, {"PV_ADDRESSES": "set2"}]}

    summaries = context_manager.get_summaries(step=step)

    # Assert: Returns list with 3 items (1 TIME_RANGE + 2 PV_ADDRESSES)
    assert isinstance(summaries, list)
    assert len(summaries) == 3

    # Check types are present
    types = [s.get("type") for s in summaries]
    assert "Time Range" in types
    assert types.count("PV Addresses") == 2  # Two PV_ADDRESSES contexts


def test_get_summaries_empty_step(context_manager, mock_state):
    """get_summaries() returns empty list when no contexts available."""
    step = {"inputs": []}

    summaries = context_manager.get_summaries(step=step)

    # Assert: Returns empty list
    assert isinstance(summaries, list)
    assert len(summaries) == 0


# ===================================================================
# Test Section 4.7: Real-World Usage Pattern Tests
# ===================================================================


def test_extract_from_step_no_constraints_returns_dict(context_manager, mock_state):
    """Validate Python capabilities pattern - no constraints, just check emptiness.

    Real-world usage (data_analysis.py, data_visualization.py, machine_operations.py):
        contexts = context_manager.extract_from_step(step, state)
        if not contexts:
            raise DataValidationError("No contexts could be extracted")
    """
    # Store multiple contexts of different types
    context_manager.set_context(
        "ARCHIVER_DATA", "data1", ArchiverDataContext(data=[1, 2, 3]), skip_validation=True
    )
    context_manager.set_context(
        "PV_VALUES", "vals", PVValuesContext(values={}), skip_validation=True
    )

    step = {"inputs": [{"ARCHIVER_DATA": "data1"}, {"PV_VALUES": "vals"}]}

    # NO constraints (like data_analysis.py does)
    contexts = context_manager.extract_from_step(step, mock_state)

    # Should return dict with context types as keys
    assert len(contexts) == 2
    assert "ARCHIVER_DATA" in contexts
    assert "PV_VALUES" in contexts

    # CRITICAL: Emptiness check must still work (Pattern 3 does this)
    assert contexts  # Truthy check
    assert bool(contexts) is True


def test_extract_from_step_emptiness_check_still_works(context_manager, mock_state):
    """Verify that 'if not contexts:' pattern still works after migration.

    Pattern 3 capabilities rely on truthiness checks, not direct key access.
    This ensures backward compatibility for that usage pattern.
    """
    # Empty inputs case
    step = {"inputs": []}
    contexts = context_manager.extract_from_step(step, mock_state)

    # Python capabilities check: if not contexts
    assert not contexts  # Should be falsy
    assert bool(contexts) is False
    assert len(contexts) == 0

    # Non-empty case
    context_manager.set_context(
        "ARCHIVER_DATA", "test", ArchiverDataContext(data=[]), skip_validation=True
    )
    step = {"inputs": [{"ARCHIVER_DATA": "test"}]}
    contexts = context_manager.extract_from_step(step, mock_state)

    assert contexts  # Should be truthy
    assert bool(contexts) is True
    assert len(contexts) == 1


def test_pattern_3_capabilities_iterate_dict_values(context_manager, mock_state):
    """Pattern 3 capabilities iterate over dict values - ensure list handling works.

    These capabilities don't access contexts by key, they just iterate over
    whatever contexts are provided. List values should work fine with this pattern.
    """
    # Store two contexts of same type (will become list)
    context_manager.set_context(
        "ARCHIVER_DATA", "data1", ArchiverDataContext(data=[1, 2]), skip_validation=True
    )
    context_manager.set_context(
        "ARCHIVER_DATA", "data2", ArchiverDataContext(data=[3, 4]), skip_validation=True
    )

    step = {"inputs": [{"ARCHIVER_DATA": "data1"}, {"ARCHIVER_DATA": "data2"}]}

    contexts = context_manager.extract_from_step(step, mock_state)

    # Pattern 3 does: for context in contexts.values()
    # With list-based approach, need to flatten lists
    all_contexts = []
    for value in contexts.values():
        if isinstance(value, list):
            all_contexts.extend(value)
        else:
            all_contexts.append(value)

    # Should get both contexts
    assert len(all_contexts) == 2
    assert all(hasattr(ctx, "data") for ctx in all_contexts)


# ===================================================================
# Test Section 4.8: Cardinality Validation Tests
# ===================================================================


def test_cardinality_single_constraint_passes(context_manager, mock_state):
    """Single cardinality constraint passes when context is single."""
    pv_context = PVAddressesContext(pvs=["PV1"])
    context_manager.set_context("PV_ADDRESSES", "main", pv_context, skip_validation=True)

    step = {"inputs": [{"PV_ADDRESSES": "main"}]}

    # Should pass - single context, "single" constraint
    contexts = context_manager.extract_from_step(
        step, mock_state, constraints=[("PV_ADDRESSES", "single")], constraint_mode="hard"
    )

    assert "PV_ADDRESSES" in contexts
    assert not isinstance(contexts["PV_ADDRESSES"], list)


def test_cardinality_single_constraint_fails_on_multiple(context_manager, mock_state):
    """Single cardinality constraint fails when context is multiple."""
    pv1 = PVAddressesContext(pvs=["PV1"])
    pv2 = PVAddressesContext(pvs=["PV2"])
    context_manager.set_context("PV_ADDRESSES", "set1", pv1, skip_validation=True)
    context_manager.set_context("PV_ADDRESSES", "set2", pv2, skip_validation=True)

    step = {"inputs": [{"PV_ADDRESSES": "set1"}, {"PV_ADDRESSES": "set2"}]}

    # Should fail - multiple contexts, "single" constraint
    with pytest.raises(ValueError) as exc_info:
        context_manager.extract_from_step(
            step, mock_state, constraints=[("PV_ADDRESSES", "single")], constraint_mode="hard"
        )

    assert "expected single instance" in str(exc_info.value).lower()
    assert "but got 2" in str(exc_info.value)


def test_cardinality_multiple_constraint_passes(context_manager, mock_state):
    """Multiple cardinality constraint passes when context is multiple."""
    pv1 = PVAddressesContext(pvs=["PV1"])
    pv2 = PVAddressesContext(pvs=["PV2"])
    context_manager.set_context("PV_ADDRESSES", "set1", pv1, skip_validation=True)
    context_manager.set_context("PV_ADDRESSES", "set2", pv2, skip_validation=True)

    step = {"inputs": [{"PV_ADDRESSES": "set1"}, {"PV_ADDRESSES": "set2"}]}

    # Should pass - multiple contexts, "multiple" constraint
    contexts = context_manager.extract_from_step(
        step, mock_state, constraints=[("PV_ADDRESSES", "multiple")], constraint_mode="hard"
    )

    assert "PV_ADDRESSES" in contexts
    assert isinstance(contexts["PV_ADDRESSES"], list)
    assert len(contexts["PV_ADDRESSES"]) == 2


def test_cardinality_multiple_constraint_fails_on_single(context_manager, mock_state):
    """Multiple cardinality constraint fails when context is single."""
    pv_context = PVAddressesContext(pvs=["PV1"])
    context_manager.set_context("PV_ADDRESSES", "main", pv_context, skip_validation=True)

    step = {"inputs": [{"PV_ADDRESSES": "main"}]}

    # Should fail - single context, "multiple" constraint
    with pytest.raises(ValueError) as exc_info:
        context_manager.extract_from_step(
            step, mock_state, constraints=[("PV_ADDRESSES", "multiple")], constraint_mode="hard"
        )

    assert "expected multiple instances" in str(exc_info.value).lower()
    assert "but got single" in str(exc_info.value).lower()


def test_cardinality_mixed_constraints(context_manager, mock_state):
    """Mixed cardinality constraints work correctly."""
    # Single TIME_RANGE, multiple PV_ADDRESSES
    context_manager.set_context(
        "TIME_RANGE", "time", TimeRangeContext(start="2025-01-01"), skip_validation=True
    )
    context_manager.set_context(
        "PV_ADDRESSES", "set1", PVAddressesContext(pvs=["PV1"]), skip_validation=True
    )
    context_manager.set_context(
        "PV_ADDRESSES", "set2", PVAddressesContext(pvs=["PV2"]), skip_validation=True
    )

    step = {"inputs": [{"TIME_RANGE": "time"}, {"PV_ADDRESSES": "set1"}, {"PV_ADDRESSES": "set2"}]}

    # Should pass - TIME_RANGE is single, PV_ADDRESSES is multiple
    contexts = context_manager.extract_from_step(
        step,
        mock_state,
        constraints=[("TIME_RANGE", "single"), ("PV_ADDRESSES", "multiple")],
        constraint_mode="hard",
    )

    assert not isinstance(contexts["TIME_RANGE"], list)
    assert isinstance(contexts["PV_ADDRESSES"], list)


def test_cardinality_no_restriction_allows_both(context_manager, mock_state):
    """Constraints without cardinality work for both single and multiple."""
    # Single TIME_RANGE
    context_manager.set_context(
        "TIME_RANGE", "time", TimeRangeContext(start="2025-01-01"), skip_validation=True
    )

    step = {"inputs": [{"TIME_RANGE": "time"}]}

    # Should pass - no cardinality restriction
    contexts = context_manager.extract_from_step(
        step,
        mock_state,
        constraints=["TIME_RANGE"],  # String constraint - no cardinality
        constraint_mode="hard",
    )

    assert "TIME_RANGE" in contexts

    # Now try with multiple contexts
    context_manager.set_context(
        "PV_ADDRESSES", "set1", PVAddressesContext(pvs=["PV1"]), skip_validation=True
    )
    context_manager.set_context(
        "PV_ADDRESSES", "set2", PVAddressesContext(pvs=["PV2"]), skip_validation=True
    )

    step2 = {"inputs": [{"PV_ADDRESSES": "set1"}, {"PV_ADDRESSES": "set2"}]}

    # Should also pass - no cardinality restriction
    contexts2 = context_manager.extract_from_step(
        step2,
        mock_state,
        constraints=["PV_ADDRESSES"],  # String constraint - no cardinality
        constraint_mode="hard",
    )

    assert "PV_ADDRESSES" in contexts2
    assert isinstance(contexts2["PV_ADDRESSES"], list)


def test_cardinality_invalid_value_raises_error(context_manager, mock_state):
    """Invalid cardinality value raises clear error."""
    pv_context = PVAddressesContext(pvs=["PV1"])
    context_manager.set_context("PV_ADDRESSES", "main", pv_context, skip_validation=True)

    step = {"inputs": [{"PV_ADDRESSES": "main"}]}

    # Should fail - invalid cardinality value
    with pytest.raises(ValueError) as exc_info:
        context_manager.extract_from_step(
            step, mock_state, constraints=[("PV_ADDRESSES", "invalid")], constraint_mode="hard"
        )

    assert "invalid cardinality" in str(exc_info.value).lower()
    assert "must be 'single' or 'multiple'" in str(exc_info.value).lower()


def test_cardinality_archiver_use_case(context_manager, mock_state):
    """Real-world archiver capability use case."""
    # Setup: archiver needs single PV_ADDRESSES and single TIME_RANGE
    context_manager.set_context(
        "PV_ADDRESSES", "pvs", PVAddressesContext(pvs=["PV1", "PV2"]), skip_validation=True
    )
    context_manager.set_context(
        "TIME_RANGE",
        "range",
        TimeRangeContext(start="2025-01-01", end="2025-01-02"),
        skip_validation=True,
    )

    step = {"inputs": [{"PV_ADDRESSES": "pvs"}, {"TIME_RANGE": "range"}]}

    # Archiver pattern: both must be present and single
    contexts = context_manager.extract_from_step(
        step,
        mock_state,
        constraints=[("PV_ADDRESSES", "single"), ("TIME_RANGE", "single")],
        constraint_mode="hard",
    )

    # Both contexts present and guaranteed to be single objects
    assert "PV_ADDRESSES" in contexts
    assert "TIME_RANGE" in contexts
    assert not isinstance(contexts["PV_ADDRESSES"], list)
    assert not isinstance(contexts["TIME_RANGE"], list)

    # Can access directly without isinstance checks
    pv_context = contexts["PV_ADDRESSES"]
    time_context = contexts["TIME_RANGE"]
    assert pv_context.pvs == ["PV1", "PV2"]
    assert time_context.start == "2025-01-01"


def test_cardinality_with_soft_constraint_mode(context_manager, mock_state):
    """Cardinality works with soft constraint mode."""
    # Only provide ARCHIVER_DATA, not PV_VALUES
    context_manager.set_context(
        "ARCHIVER_DATA", "data", ArchiverDataContext(data=[1, 2, 3]), skip_validation=True
    )

    step = {"inputs": [{"ARCHIVER_DATA": "data"}]}

    # Soft mode: at least one required, single cardinality
    contexts = context_manager.extract_from_step(
        step,
        mock_state,
        constraints=[("PV_VALUES", "single"), ("ARCHIVER_DATA", "single")],
        constraint_mode="soft",  # Only need one of them
    )

    # Should pass - ARCHIVER_DATA present and single
    assert "ARCHIVER_DATA" in contexts
    assert not isinstance(contexts["ARCHIVER_DATA"], list)

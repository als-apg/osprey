"""Tests for execution_step_results reducer for parallel execution support."""

from osprey.state import merge_execution_step_results


def test_merge_execution_step_results_with_none_existing():
    """Test merging when existing is None."""
    new = {
        "step_0": {"result": "data1", "step_index": 0},
        "step_1": {"result": "data2", "step_index": 1},
    }

    result = merge_execution_step_results(None, new)

    assert len(result) == 2
    assert result["step_0"]["result"] == "data1"
    assert result["step_1"]["result"] == "data2"


def test_merge_execution_step_results_basic():
    """Test basic merging of step results."""
    existing = {"step_0": {"result": "data1", "step_index": 0}}
    new = {"step_1": {"result": "data2", "step_index": 1}}

    result = merge_execution_step_results(existing, new)

    assert len(result) == 2
    assert result["step_0"]["result"] == "data1"
    assert result["step_1"]["result"] == "data2"


def test_merge_execution_step_results_parallel():
    """Test merging multiple parallel step results."""
    existing = {"step_0": {"result": "PV1 data", "step_index": 0}}
    new = {
        "step_1": {"result": "PV2 data", "step_index": 1},
        "step_2": {"result": "PV3 data", "step_index": 2},
        "step_3": {"result": "PV4 data", "step_index": 3},
    }

    result = merge_execution_step_results(existing, new)

    assert len(result) == 4
    assert result["step_0"]["result"] == "PV1 data"
    assert result["step_1"]["result"] == "PV2 data"
    assert result["step_2"]["result"] == "PV3 data"
    assert result["step_3"]["result"] == "PV4 data"


def test_merge_execution_step_results_override():
    """Test that new results override existing for same key."""
    existing = {"step_0": {"result": "old_data", "step_index": 0}}
    new = {"step_0": {"result": "new_data", "step_index": 0, "extra": "field"}}

    result = merge_execution_step_results(existing, new)

    assert len(result) == 1
    assert result["step_0"]["result"] == "new_data"
    assert result["step_0"]["extra"] == "field"


def test_merge_execution_step_results_immutability():
    """Test that merge doesn't mutate input dictionaries."""
    existing = {"step_0": {"result": "data1", "step_index": 0}}
    new = {"step_1": {"result": "data2", "step_index": 1}}

    # Store original values
    existing_copy = existing.copy()
    new_copy = new.copy()

    result = merge_execution_step_results(existing, new)

    # Verify inputs weren't mutated
    assert existing == existing_copy
    assert new == new_copy

    # Verify result is independent
    result["step_0"]["result"] = "modified"
    assert existing["step_0"]["result"] == "data1"


def test_merge_execution_step_results_empty_new():
    """Test merging with empty new dictionary."""
    existing = {"step_0": {"result": "data1", "step_index": 0}}
    new = {}

    result = merge_execution_step_results(existing, new)

    assert len(result) == 1
    assert result["step_0"]["result"] == "data1"


def test_merge_execution_step_results_complex_data():
    """Test merging with complex nested data structures."""
    existing = {
        "step_0": {"result": {"pvs": ["PV1", "PV2"], "metadata": {"count": 2}}, "step_index": 0}
    }
    new = {"step_1": {"result": {"pvs": ["PV3", "PV4"], "metadata": {"count": 2}}, "step_index": 1}}

    result = merge_execution_step_results(existing, new)

    assert len(result) == 2
    assert result["step_0"]["result"]["pvs"] == ["PV1", "PV2"]
    assert result["step_1"]["result"]["pvs"] == ["PV3", "PV4"]
    assert result["step_0"]["result"]["metadata"]["count"] == 2

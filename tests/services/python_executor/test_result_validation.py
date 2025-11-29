"""Unit tests for results dictionary validation.

Tests the static analysis validation of the 'results' variable assignment
to ensure generated code follows the expected pattern of creating a results
dictionary.
"""

import pytest

from osprey.services.python_executor.models import validate_result_structure


class TestValidateResultStructure:
    """Test suite for validate_result_structure function."""

    # =========================================================================
    # VALID CASES - Should return True
    # =========================================================================

    def test_empty_dict_literal(self):
        """Empty dict literal should be valid."""
        code = "results = {}"
        assert validate_result_structure(code) is True

    def test_dict_literal_with_values(self):
        """Dict literal with key-value pairs should be valid."""
        code = """
results = {
    "mean": 42.5,
    "count": 100,
    "values": [1, 2, 3]
}
"""
        assert validate_result_structure(code) is True

    def test_dict_constructor_empty(self):
        """dict() constructor should be valid."""
        code = "results = dict()"
        assert validate_result_structure(code) is True

    def test_dict_constructor_with_kwargs(self):
        """dict() with keyword arguments should be valid."""
        code = "results = dict(mean=42.5, count=100, status='complete')"
        assert validate_result_structure(code) is True

    def test_dict_comprehension(self):
        """Dict comprehension should be valid."""
        code = "results = {k: v for k, v in items}"
        assert validate_result_structure(code) is True

    def test_dict_comprehension_complex(self):
        """Complex dict comprehension should be valid."""
        code = """
data = [('a', 1), ('b', 2), ('c', 3)]
results = {key.upper(): value * 2 for key, value in data if value > 0}
"""
        assert validate_result_structure(code) is True

    def test_results_in_middle_of_code(self):
        """Results assignment in middle of code should be detected."""
        code = """
import numpy as np
data = [1, 2, 3, 4, 5]
mean_value = np.mean(data)
results = {"mean": mean_value, "count": len(data)}
print("Done!")
"""
        assert validate_result_structure(code) is True

    def test_multiline_dict_literal(self):
        """Multi-line dict literal should be valid."""
        code = """
results = {
    "statistics": {
        "mean": 42.5,
        "median": 40.0,
        "std": 5.2
    },
    "metadata": {
        "timestamp": "2024-01-01",
        "version": "1.0"
    }
}
"""
        assert validate_result_structure(code) is True

    # =========================================================================
    # INVALID CASES - Should return False
    # =========================================================================

    def test_no_results_variable(self):
        """Code without 'results' variable should return False."""
        code = """
x = 10
y = 20
print(x + y)
"""
        assert validate_result_structure(code) is False

    def test_results_commented_out(self):
        """Commented out results assignment should return False."""
        code = """
# results = {"value": 42}
x = 10
"""
        assert validate_result_structure(code) is False

    def test_results_in_string(self):
        """'results' in a string literal should not count."""
        code = """
message = "Please store results in a dictionary"
print(message)
"""
        assert validate_result_structure(code) is False

    def test_results_assigned_to_none(self):
        """Results assigned to None should return False."""
        code = "results = None"
        assert validate_result_structure(code) is False

    def test_results_assigned_to_list(self):
        """Results assigned to a list should return False."""
        code = "results = [1, 2, 3]"
        assert validate_result_structure(code) is False

    def test_results_assigned_to_string(self):
        """Results assigned to a string should return False."""
        code = 'results = "some value"'
        assert validate_result_structure(code) is False

    def test_results_assigned_to_number(self):
        """Results assigned to a number should return False."""
        code = "results = 42"
        assert validate_result_structure(code) is False

    # =========================================================================
    # EDGE CASES - Can't validate statically (returns False)
    # =========================================================================

    def test_results_from_function_call(self):
        """Function call can't be validated statically."""
        code = """
def get_results():
    return {"value": 42}

results = get_results()
"""
        # Can't validate function return type statically
        assert validate_result_structure(code) is False

    def test_results_from_variable(self):
        """Assignment from variable can't be validated statically."""
        code = """
data = {"value": 42}
results = data
"""
        # Can't validate variable type statically
        assert validate_result_structure(code) is False

    def test_results_from_method_call(self):
        """Method call can't be validated statically."""
        code = """
import pandas as pd
df = pd.DataFrame({"col": [1, 2, 3]})
results = df.to_dict()
"""
        # Can't validate method return type statically
        assert validate_result_structure(code) is False

    def test_results_from_json_loads(self):
        """JSON parsing can't be validated statically."""
        code = """
import json
data = '{"key": "value"}'
results = json.loads(data)
"""
        assert validate_result_structure(code) is False

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    def test_syntax_error_returns_false(self):
        """Code with syntax errors should return False (not raise exception)."""
        code = """
def broken(
    results = {}
"""
        # Should not raise exception, just return False
        assert validate_result_structure(code) is False

    def test_empty_code(self):
        """Empty code should return False."""
        assert validate_result_structure("") is False

    def test_only_whitespace(self):
        """Only whitespace should return False."""
        assert validate_result_structure("   \n\t  ") is False

    # =========================================================================
    # REAL-WORLD EXAMPLES
    # =========================================================================

    def test_realistic_data_analysis(self):
        """Realistic data analysis code."""
        code = """
import numpy as np
import pandas as pd

# Load and process data
data = pd.read_csv('data.csv')
mean_value = data['value'].mean()
std_value = data['value'].std()

# Store results
results = {
    "mean": mean_value,
    "std": std_value,
    "count": len(data),
    "summary": data.describe().to_dict()
}
"""
        assert validate_result_structure(code) is True

    def test_realistic_epics_read(self):
        """Realistic EPICS PV reading code."""
        code = """
from epics import caget

# Read PV values
pv_value = caget('SOME:PV:NAME')
pv_timestamp = caget('SOME:PV:NAME.TIME')

# Build results dictionary
results = {
    "pv_name": "SOME:PV:NAME",
    "value": pv_value,
    "timestamp": pv_timestamp,
    "status": "success"
}
"""
        assert validate_result_structure(code) is True

    def test_realistic_calculation(self):
        """Realistic calculation code."""
        code = """
import math

# Perform calculations
x = 10
y = 20
sum_val = x + y
product_val = x * y
sqrt_val = math.sqrt(x)

# Create results
results = dict(
    sum=sum_val,
    product=product_val,
    sqrt=sqrt_val,
    inputs={'x': x, 'y': y}
)
"""
        assert validate_result_structure(code) is True

    def test_realistic_dict_comprehension_aggregation(self):
        """Realistic aggregation using dict comprehension."""
        code = """
# Group data by category
data = [
    {'category': 'A', 'value': 10},
    {'category': 'B', 'value': 20},
    {'category': 'A', 'value': 15}
]

# Aggregate by category
from collections import defaultdict
grouped = defaultdict(list)
for item in data:
    grouped[item['category']].append(item['value'])

results = {
    category: sum(values)
    for category, values in grouped.items()
}
"""
        assert validate_result_structure(code) is True

    # =========================================================================
    # MULTIPLE ASSIGNMENTS
    # =========================================================================

    def test_results_reassigned(self):
        """If results is assigned multiple times, should still return True."""
        code = """
results = {}  # First assignment (dict)
results["key"] = "value"  # Modification
results = {"final": "value"}  # Reassignment (also dict)
"""
        # Should find at least one dict-like assignment
        assert validate_result_structure(code) is True

    def test_results_first_assignment_not_dict(self):
        """First assignment not dict, but later assignment is dict."""
        code = """
results = None  # First assignment (not dict)
if some_condition:
    results = {"value": 42}  # Later assignment (dict)
"""
        # Should find the dict assignment
        assert validate_result_structure(code) is True


class TestValidateResultStructureIntegration:
    """Integration tests with actual generator patterns."""

    def test_basic_generator_pattern(self):
        """Pattern typically used by BasicLLMCodeGenerator."""
        code = """
import numpy as np

# Perform analysis
data = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(data)
std_value = np.std(data)

# Store results for downstream use
results = {
    "mean": float(mean_value),
    "std": float(std_value),
    "count": len(data)
}
"""
        assert validate_result_structure(code) is True

    def test_claude_code_pattern(self):
        """Pattern typically used by ClaudeCodeGenerator."""
        code = """
#!/usr/bin/env python3
\"\"\"
Analysis script generated by Claude Code.
\"\"\"

import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# Analyze
summary_stats = df.describe()

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['value'])
plt.savefig('figures/plot.png')

# Store results
results = {
    "summary": summary_stats.to_dict(),
    "record_count": len(df),
    "columns": list(df.columns),
    "figure_path": "figures/plot.png"
}
"""
        assert validate_result_structure(code) is True

    def test_incremental_results_building(self):
        """Pattern where results is built incrementally."""
        code = """
# Initialize results
results = {}

# Add data incrementally
results['step1'] = compute_step1()
results['step2'] = compute_step2()
results['final'] = results['step1'] + results['step2']
"""
        assert validate_result_structure(code) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


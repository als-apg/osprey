"""Variable-value builders shared by the engine, environment-plugin and simulator test suites.

Each package owns its own ``VARIABLES`` table and exposes it to tests as a
``test_variable_bounds`` fixture (``{name: [min, max]}``). The tables differ;
the arithmetic over them does not, so it lives here instead of being copied
into three conftests.

Deliberately not re-exported from the package ``__init__``: this is test
scaffolding, not part of the shared-schema surface, and nothing in ``src``
imports it.
"""


def valid_variables(bounds: dict[str, list[float]]) -> dict[str, float]:
    """Every variable at the midpoint of its bounds."""
    return {name: (lower + upper) / 2.0 for name, (lower, upper) in bounds.items()}


def bounds_variables(bounds: dict[str, list[float]], *, use_lower: bool = True) -> dict[str, float]:
    """Every variable pinned to its lower bound (default) or its upper bound."""
    index = 0 if use_lower else 1
    return {name: pair[index] for name, pair in bounds.items()}


def relative_increment(bounds: dict[str, list[float]], name: str, fraction: float = 0.1) -> float:
    """A step for *name* sized as *fraction* of its full range."""
    lower, upper = bounds[name]
    return (upper - lower) * fraction

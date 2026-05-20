"""Tests for ARIEL extensible module system.

Validates:
- Capabilities shared parameters
"""


class TestCapabilitiesSharedParameters:
    """Test capabilities.py shared parameters."""

    def test_shared_parameters_intact(self):
        """SHARED_PARAMETERS are still present."""
        from osprey.services.ariel_search.capabilities import SHARED_PARAMETERS

        names = [p.name for p in SHARED_PARAMETERS]
        assert "max_results" in names
        assert "start_date" in names
        assert "end_date" in names

"""
Tests for hierarchical channel finder separator override feature.

Tests the _separator metadata field that allows nodes to override
default separators defined in the naming pattern.
"""

import json
import tempfile
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.hierarchical import (
    HierarchicalChannelDatabase,
)


@pytest.fixture
def basic_separator_override_db():
    """Basic database testing separator override at leaf level."""
    content = {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "device", "type": "instances"},
                {"name": "signal", "type": "tree"},
                {"name": "suffix", "type": "tree", "optional": True},
            ],
            "naming_pattern": "{system}-{device}:{signal}_{suffix}",
        },
        "tree": {
            "TEST": {
                "DEVICE": {
                    "_expansion": {"_type": "list", "_instances": ["DEV-01"]},
                    "StatusReg": {
                        "_separator": "_",  # Override: use underscore instead of default colon
                        "_is_leaf": True,
                        "_description": "Status register with custom separator",
                        "RB": {"_description": "Readback"},
                        "SP": {"_description": "Setpoint"},
                    },
                }
            }
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(content, f)
        db_path = f.name

    db = HierarchicalChannelDatabase(db_path)
    yield db
    Path(db_path).unlink()


@pytest.fixture
def midlevel_separator_override_db():
    """Database testing separator override at middle hierarchy levels."""
    content = {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "device", "type": "instances"},
                {"name": "subdevice", "type": "tree", "optional": True},
                {"name": "signal", "type": "tree"},
            ],
            "naming_pattern": "{system}-{device}:{subdevice}:{signal}",
        },
        "tree": {
            "TEST": {
                "DEVICE": {
                    "_expansion": {"_type": "list", "_instances": ["DEV-01"]},
                    "MOTOR": {
                        "_separator": ".",  # Override: use dot for MOTOR's children
                        "_description": "Motor subdevice with dot separator",
                        "Position": {"_description": "Motor position"},
                        "Velocity": {"_description": "Motor velocity"},
                    },
                    "ADC": {
                        # No _separator - uses default colon
                        "_description": "ADC subdevice with default separator",
                        "Value": {"_description": "ADC value"},
                    },
                }
            }
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(content, f)
        db_path = f.name

    db = HierarchicalChannelDatabase(db_path)
    yield db
    Path(db_path).unlink()


@pytest.fixture
def multiple_overrides_db():
    """Database with multiple separator overrides in one channel path."""
    content = {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "device", "type": "instances"},
                {"name": "subdevice", "type": "tree", "optional": True},
                {"name": "signal", "type": "tree"},
                {"name": "suffix", "type": "tree", "optional": True},
            ],
            "naming_pattern": "{system}-{device}:{subdevice}:{signal}_{suffix}",
        },
        "tree": {
            "TEST": {
                "DEVICE": {
                    "_expansion": {"_type": "list", "_instances": ["DEV-01"]},
                    "CTRL": {
                        "_separator": ".",  # First override: dot for subdevice→signal
                        "_description": "Controller with custom separators",
                        "Mode": {
                            "_is_leaf": True,
                            # Inherits parent's dot, uses default underscore for suffix
                            "_description": "Mode with RB/SP",
                            "RB": {"_description": "Readback"},
                            "SP": {"_description": "Setpoint"},
                        },
                    },
                }
            }
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(content, f)
        db_path = f.name

    db = HierarchicalChannelDatabase(db_path)
    yield db
    Path(db_path).unlink()


class TestSeparatorOverrideBasic:
    """Test basic separator override functionality."""

    def test_separator_override_at_leaf(self, basic_separator_override_db):
        """Separator override should work for leaf-level signals with suffixes."""
        db = basic_separator_override_db

        # Channels should use custom underscore separator
        assert "TEST-DEV-01:StatusReg" in db.channel_map
        assert "TEST-DEV-01:StatusReg_RB" in db.channel_map
        assert "TEST-DEV-01:StatusReg_SP" in db.channel_map

        # Should NOT use default colon separator
        assert "TEST-DEV-01:StatusReg:RB" not in db.channel_map
        assert "TEST-DEV-01:StatusReg:SP" not in db.channel_map

    def test_database_without_separator_uses_defaults(self):
        """Database without _separator should use pattern defaults."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                    {"name": "suffix", "type": "tree", "optional": True},
                ],
                "naming_pattern": "{system}:{signal}_{suffix}",
            },
            "tree": {
                "SYS": {
                    "SIG": {
                        "_is_leaf": True,
                        "RB": {},
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Should use default separators from pattern
            assert "SYS:SIG" in db.channel_map
            assert "SYS:SIG_RB" in db.channel_map
        finally:
            Path(db_path).unlink()


class TestSeparatorOverrideMidLevel:
    """Test separator override at middle hierarchy levels."""

    def test_midlevel_separator_override(self, midlevel_separator_override_db):
        """Separator override should work for middle levels (subdevice→signal)."""
        db = midlevel_separator_override_db

        # MOTOR should use dot separator
        assert "TEST-DEV-01:MOTOR.Position" in db.channel_map
        assert "TEST-DEV-01:MOTOR.Velocity" in db.channel_map

        # Should NOT use default colon separator
        assert "TEST-DEV-01:MOTOR:Position" not in db.channel_map
        assert "TEST-DEV-01:MOTOR:Velocity" not in db.channel_map

    def test_mixed_separators_in_same_db(self, midlevel_separator_override_db):
        """Different nodes can have different separators in same database."""
        db = midlevel_separator_override_db

        # MOTOR uses dot (custom)
        assert "TEST-DEV-01:MOTOR.Position" in db.channel_map

        # ADC uses colon (default) - should NOT be affected by MOTOR's separator
        assert "TEST-DEV-01:ADC:Value" in db.channel_map


class TestSeparatorOverrideMultiple:
    """Test multiple separator overrides in one channel path."""

    def test_multiple_separators_in_path(self, multiple_overrides_db):
        """Channel can have multiple different separators in its path."""
        db = multiple_overrides_db

        # Should have dot (CTRL override) then underscore (default for suffix)
        assert "TEST-DEV-01:CTRL.Mode" in db.channel_map
        assert "TEST-DEV-01:CTRL.Mode_RB" in db.channel_map
        assert "TEST-DEV-01:CTRL.Mode_SP" in db.channel_map

        # Should NOT have wrong separator combinations
        assert "TEST-DEV-01:CTRL:Mode_RB" not in db.channel_map  # Wrong first sep
        assert "TEST-DEV-01:CTRL.Mode:RB" not in db.channel_map  # Wrong second sep
        assert "TEST-DEV-01:CTRL:Mode:RB" not in db.channel_map  # Both wrong


class TestSeparatorOverrideWithInstances:
    """Test separator override with instance expansions."""

    def test_separator_on_instance_expansion_container(self):
        """Separator override on instance container affects all instances."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}-{device}:{signal}",
            },
            "tree": {
                "TEST": {
                    "DEVICE": {
                        "_expansion": {
                            "_type": "range",
                            "_pattern": "D{:02d}",
                            "_range": [1, 2],
                        },
                        "_separator": ".",  # All instances use dot for their signals
                        "Signal": {},
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Both instances should use dot separator
            assert "TEST-D01.Signal" in db.channel_map
            assert "TEST-D02.Signal" in db.channel_map
            # Should NOT use default colon
            assert "TEST-D01:Signal" not in db.channel_map
            assert "TEST-D02:Signal" not in db.channel_map
        finally:
            Path(db_path).unlink()

    def test_separator_with_nested_instances(self):
        """Separator override with tree-based subdevices (not nested instances)."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "subdevice", "type": "tree", "optional": True},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}-{device}:{subdevice}:{signal}",
            },
            "tree": {
                "TEST": {
                    "DEVICE": {
                        "_expansion": {"_type": "list", "_instances": ["DEV-01"]},
                        "PUMP": {
                            "_separator": ".",  # Override for PUMP's signals
                            "Pressure": {},
                        },
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Should use dot separator from PUMP
            assert "TEST-DEV-01:PUMP.Pressure" in db.channel_map
            # Should NOT use default colon
            assert "TEST-DEV-01:PUMP:Pressure" not in db.channel_map
        finally:
            Path(db_path).unlink()


class TestSeparatorOverrideWithOptionalLevels:
    """Test separator override interaction with optional hierarchy levels."""

    def test_separator_override_skipping_optional_level(self):
        """Separator override when optional level is skipped."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "device", "type": "instances"},
                    {"name": "subdevice", "type": "tree", "optional": True},
                    {"name": "signal", "type": "tree"},
                    {"name": "suffix", "type": "tree", "optional": True},
                ],
                "naming_pattern": "{device}:{subdevice}:{signal}_{suffix}",
            },
            "tree": {
                "DEVICE": {
                    "_expansion": {"_type": "list", "_instances": ["D01"]},
                    "DirectSignal": {
                        "_separator": "_",  # Skips subdevice, uses underscore for suffix
                        "_is_leaf": True,
                        "RB": {},
                    },
                    "SUBDEV": {
                        "SubSignal": {
                            "_is_leaf": True,
                            "SP": {},
                        },
                    },
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Direct signal with custom separator (skips optional subdevice)
            assert "D01:DirectSignal_RB" in db.channel_map

            # Subdevice signal uses default separators
            assert "D01:SUBDEV:SubSignal_SP" in db.channel_map
        finally:
            Path(db_path).unlink()

    def test_separator_across_skipped_optional_levels(self):
        """Separator lookup across multiple skipped optional levels."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "opt1", "type": "tree", "optional": True},
                    {"name": "opt2", "type": "tree", "optional": True},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}:{opt1}:{opt2}:{signal}",
            },
            "tree": {"SYS": {"SIG": {}}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Should use first separator encountered (system→opt1) which is ":"
            assert "SYS:SIG" in db.channel_map
            # Should NOT have multiple colons
            assert "SYS::SIG" not in db.channel_map
            assert "SYS:::SIG" not in db.channel_map
        finally:
            Path(db_path).unlink()


class TestSeparatorOverrideEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_separator_override(self):
        """Empty string separator should join without any separator."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}:{signal}",
            },
            "tree": {
                "SYS": {
                    "_separator": "",  # No separator
                    "SIG": {},
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Should join with no separator
            assert "SYSSIG" in db.channel_map
            # Should NOT use default colon
            assert "SYS:SIG" not in db.channel_map
        finally:
            Path(db_path).unlink()

    def test_complex_separator_characters(self):
        """Custom separators can be any string."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}:{signal}",
            },
            "tree": {
                "SYS": {
                    "_separator": " >> ",  # Multi-character separator (avoid : which gets cleaned)
                    "SIG": {},
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert "SYS >> SIG" in db.channel_map
        finally:
            Path(db_path).unlink()


class TestSeparatorOverrideBackwardCompatibility:
    """Ensure backward compatibility with databases not using _separator."""

    def test_existing_databases_unchanged(self):
        """Databases without _separator should work exactly as before."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}:{device}-{signal}",
            },
            "tree": {
                "MAG": {
                    "DEVICE": {
                        "_expansion": {
                            "_type": "range",
                            "_pattern": "D{:02d}",
                            "_range": [1, 2],
                        },
                        "Current": {},
                        "Voltage": {},
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Should use default separators from pattern
            assert "MAG:D01-Current" in db.channel_map
            assert "MAG:D02-Voltage" in db.channel_map
        finally:
            Path(db_path).unlink()

    def test_mixed_old_and_new_syntax(self):
        """Can mix nodes with and without _separator in same database."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}:{signal}",
            },
            "tree": {
                "OLD": {
                    # No _separator - uses default
                    "SIG1": {},
                },
                "NEW": {
                    "_separator": ".",  # Custom separator
                    "SIG2": {},
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # OLD uses default colon
            assert "OLD:SIG1" in db.channel_map
            # NEW uses custom dot
            assert "NEW.SIG2" in db.channel_map
        finally:
            Path(db_path).unlink()

"""A config on/off switch reads a resolved ``${VAR:-false}`` for what it spells."""

from __future__ import annotations

from typing import Any

import pytest

from osprey_connectors.config import config_flag


@pytest.mark.parametrize(
    "value", [False, None, 0, "false", "False", " FALSE ", "0", "no", "off", ""]
)
def test_off_spellings_read_as_off(value: Any) -> None:
    assert config_flag(value, key="k") is False


@pytest.mark.parametrize("value", [True, 1, "true", "True", " TRUE ", "1", "yes", "on"])
def test_on_spellings_read_as_on(value: Any) -> None:
    assert config_flag(value, key="k") is True


@pytest.mark.parametrize("value", ["maybe", "ture", "${USE_NS}", "2", 2, -1, 0.5, [], {}])
def test_anything_else_is_refused_and_names_the_key(value: Any) -> None:
    with pytest.raises(ValueError, match=r"gateways\.read_only\.use_name_server"):
        config_flag(value, key="gateways.read_only.use_name_server")

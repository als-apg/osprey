"""Unit tests for the shared ARIEL search-tool envelope helpers."""

import pytest

from osprey.mcp_server.ariel.tools.search_envelope import advanced_params


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, False])
def test_advanced_params_carries_an_explicit_rerank(value):
    """An explicit choice is an override and has to reach the service."""
    assert advanced_params(rerank=value)["rerank"] is value


@pytest.mark.unit
def test_advanced_params_omits_an_unset_rerank():
    """An absent key is how the service hears "no preference"."""
    assert "rerank" not in advanced_params()


@pytest.mark.unit
def test_advanced_params_keeps_the_other_filters_independent():
    """Adding rerank must not disturb what the other arguments emit."""
    params = advanced_params(author="chen", expand_query=False, rerank=True)

    assert params == {"author": "chen", "expand_query": False, "rerank": True}

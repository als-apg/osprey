"""A key no HTTP header can carry is refused before any request is built."""

from __future__ import annotations

import pytest

from osprey.models.jev.config import JevSettings, resolve_key
from osprey.models.jev.errors import DecisionAuthError


def test_a_non_ascii_key_raises_no_key_without_echoing_it(monkeypatch) -> None:
    secret = "kéy-secret"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setenv("TYPESAFE_API_KEY", secret)
    settings = JevSettings.from_config({})

    with pytest.raises(DecisionAuthError) as info:
        resolve_key(settings, {})

    assert info.value.error_type == "no_key"
    assert secret not in str(info.value)

"""The entry point owns the clock: a silent store ends an open at its deadline.

Each test here opens a store that takes the connection and never answers
(:mod:`._silent_store`) through one layer an entry point opens it through, with a
short deadline passed the way that entry point passes its own, and asserts two
bounds: the open does not end before the deadline, and it ends within the
deadline plus the time closing the pool takes.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.exceptions import StoreSilentError

from ._silent_store import MARGIN_S, SHORT_DEADLINE_S, silent_store, silent_uri


def _config(uri: str, **database: Any) -> ARIELConfig:
    return ARIELConfig.from_dict({"database": {"uri": uri, **database}})


def _assert_silent(raised: StoreSilentError, port: int, started: float) -> None:
    elapsed = time.monotonic() - started
    assert SHORT_DEADLINE_S <= elapsed < SHORT_DEADLINE_S + MARGIN_S
    assert str(raised) == (
        f"the ARIEL store at 127.0.0.1:{port}/ariel did not answer within {SHORT_DEADLINE_S:g} s"
    )


@pytest.fixture(autouse=True)
def no_facility_file(monkeypatch):
    """The project's facility file is not what these tests read."""
    monkeypatch.setattr(
        "osprey.services.ariel_search.ingestion.facility_references.load_facility_records",
        lambda: None,
    )


@pytest.mark.wallclock
async def test_the_service_ends_its_open_at_the_deadline_it_was_given():
    from osprey.services.ariel_search.service import create_ariel_service

    with silent_store() as port:
        started = time.monotonic()
        with pytest.raises(StoreSilentError) as raised:
            await create_ariel_service(_config(silent_uri(port)), store_deadline_s=SHORT_DEADLINE_S)
        _assert_silent(raised.value, port, started)


@pytest.mark.wallclock
async def test_a_silent_read_only_login_ends_the_service_open_and_closes_the_other_pool(
    monkeypatch,
):
    """The read-only pool reaches the same store under the same deadline; its
    silence is not a refused login to record, and the ingestion pool already
    open is closed on the way out."""
    import osprey.services.ariel_search.database.connection as conn_mod
    from osprey.services.ariel_search.service import create_ariel_service

    real_open = conn_mod.create_connection_pool
    closed: list[bool] = []

    class _OpenPool:
        async def close(self) -> None:
            closed.append(True)

    async def _ingestion_answers(config, *, uri=None, max_size=10, deadline_s=None):
        if uri is None:
            return _OpenPool()
        return await real_open(config, uri=uri, max_size=max_size, deadline_s=deadline_s)

    monkeypatch.setattr(conn_mod, "create_connection_pool", _ingestion_answers)
    monkeypatch.setattr(
        "osprey.services.ariel_search.database.repository.ARIELRepository",
        lambda pool, config, **_: object(),
    )

    with silent_store() as port:
        config = _config(
            "postgresql://ariel:owner-secret@127.0.0.1:1/ariel",
            readonly_uri=silent_uri(port).replace("ariel:", "ariel_ro:", 1),
        )
        started = time.monotonic()
        with pytest.raises(StoreSilentError) as raised:
            await create_ariel_service(
                config, serves_sql_tool=True, store_deadline_s=SHORT_DEADLINE_S
            )
        _assert_silent(raised.value, port, started)
    assert closed == [True]


@pytest.mark.wallclock
async def test_the_capability_caches_nothing_on_a_silent_store(monkeypatch):
    from osprey.services.ariel_search import capability

    capability.reset_ariel_service()
    monkeypatch.setattr(capability, "_STORE_DEADLINE_S", SHORT_DEADLINE_S)
    with silent_store() as port:
        monkeypatch.setattr(
            capability,
            "get_config_value",
            lambda key, default=None: (
                {"database": {"uri": silent_uri(port)}} if key == "ariel" else default
            ),
        )
        for _ in range(2):  # the second call asks the store again
            started = time.monotonic()
            with pytest.raises(StoreSilentError) as raised:
                await capability.get_ariel_search_service()
            _assert_silent(raised.value, port, started)
            assert capability._ariel_service_instance is None

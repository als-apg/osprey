"""Shared fixtures for the CANOPY integration tests.

Some backend tests call ``start()`` with the real ``_arm_raw_put_block``, which
runs ``raw_put_block.install("armed", ...)`` in this process. That patches the
raw client modules in place and leaves a guard finder on ``sys.meta_path``, so
without a restore every later test in the worker runs against a patched
process.

:func:`raw_put_block_restored` puts it all back. It wraps
``raw_put_block._patch_rows`` so that every object a replacement is installed
on is tracked by :func:`tests.runtime._patch_restore.patch_restore` *before* it
is patched; on teardown each tracked namespace gets its attributes back,
the guard finders are dropped from ``sys.meta_path``; every other finder stays.

It is opt-in: a module that arms the block for real says so with
``pytestmark = pytest.mark.usefixtures("raw_put_block_restored")``. The arm
tests in ``test_backend_write``/``test_backend_read`` stub the arm and do not
use it.

:func:`recorder_archive` hands a recorder adapter test a fresh database in one
throwaway MongoDB per worker (``_recorder_store``); without a container engine
those tests skip.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any

import pytest
from tests.integrations.canopy import _recorder_store
from tests.runtime._patch_restore import patch_restore

from osprey.runtime import raw_put_block


@contextmanager
def tracking_raw_put_block(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Track every namespace ``raw_put_block`` patches inside the block; restore on exit."""
    with patch_restore() as snapshot:
        with monkeypatch.context() as patched:
            real = raw_put_block._patch_rows

            def tracked_patch_rows(label, eager_targets, deferred_targets, replacement):
                def tracked_replacement(dotted, attr, original, owner):
                    snapshot.track(owner)
                    return replacement(dotted, attr, original, owner)

                return real(label, eager_targets, deferred_targets, tracked_replacement)

            patched.setattr(raw_put_block, "_patch_rows", tracked_patch_rows)
            yield snapshot


@pytest.fixture
def raw_put_block_restored(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore whatever a real ``raw_put_block.install`` in this test patched."""
    with tracking_raw_put_block(monkeypatch):
        yield


@pytest.fixture
def raw_put_block_restorer(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], AbstractContextManager[Any]]:
    """The restore as a context manager, for a test that asserts what its exit restores."""
    return lambda: tracking_raw_put_block(monkeypatch)


@dataclass(frozen=True)
class RecorderArchive:
    """One test's archive: the adapter's URL, the collection, and the recorder writing it."""

    database: str
    url: str
    collection: Any
    writer: Any


@pytest.fixture(scope="session")
def recorder_store() -> Iterator[_recorder_store.Store]:
    yield from _recorder_store.started_store()


@pytest.fixture
def recorder_archive(
    recorder_store: _recorder_store.Store, monkeypatch: pytest.MonkeyPatch
) -> Iterator[RecorderArchive]:
    """A fresh database with the seeder's indexes, dropped again at teardown."""
    monkeypatch.setenv(_recorder_store.PASSWORD_ENV, _recorder_store.PASSWORD)
    database = f"recorder_{uuid.uuid4().hex[:12]}"
    client = recorder_store.client()
    collection = client[database][_recorder_store.COLLECTION]
    _recorder_store.prepare_collection(collection)
    writer = recorder_store.writer(database)
    try:
        yield RecorderArchive(database, recorder_store.url(database), collection, writer)
    finally:
        writer.close()
        client.drop_database(database)
        client.close()

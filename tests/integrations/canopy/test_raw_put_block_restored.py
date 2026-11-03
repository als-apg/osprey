"""The ``raw_put_block_restored`` fixture puts back what a real armed install patched."""

from __future__ import annotations

import sys
import types

import pytest
from tests.runtime._patch_restore import guard_finders

from osprey.runtime import raw_put_block

FAKE = "osprey_test_fake_raw_client"


def _original_put(*_args, **_kwargs):
    return "called through"


@pytest.fixture
def fake_row(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType(FAKE)
    module.put = _original_put
    monkeypatch.setitem(sys.modules, FAKE, module)
    return module


def _arm() -> None:
    raw_put_block.install(
        "armed",
        blocked_targets=[(FAKE, ("put",))],
        rpc_targets=[],
        refuse_rpc=False,
        marker="x",
        rpc_refusals={},
    )


def test_an_armed_fake_row_is_restored_on_exit(fake_row, raw_put_block_restorer) -> None:
    meta_path = list(sys.meta_path)
    with raw_put_block_restorer():
        _arm()
        assert fake_row.put is not _original_put, "the install patched the fake row"
    assert fake_row.put is _original_put
    assert [f for f in meta_path if f in sys.meta_path] == meta_path
    assert not guard_finders()


def test_the_home_module_of_a_patched_function_is_restored(
    fake_row, raw_put_block_restorer
) -> None:
    """``install`` follows ``put`` back to this module; that spelling is restored too."""
    this = sys.modules[__name__]
    with raw_put_block_restorer():
        _arm()
    assert this._original_put is _original_put
    assert fake_row.put() == "called through"


@pytest.mark.usefixtures("raw_put_block_restored")
def test_the_fixture_form_arms_inside_the_test(fake_row) -> None:
    _arm()
    assert fake_row.put is not _original_put


def test_after_the_fixture_form_the_row_is_back() -> None:
    """Runs after the test above in the same worker; nothing it patched leaked."""
    assert _original_put() == "called through"
    assert not guard_finders()

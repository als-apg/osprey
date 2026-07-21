"""CA-attribute audit for the connector-mediated device layer (task 2.7).

`ConnectorSettable`/`ConnectorReadable` (`devices/connector.py`) hold the live
`ControlSystemConnector` directly on the instance, because the in-process
Bluesky plan body IS handed the device object and must be able to call
`connector.read_channel`/`write_channel_checked` from `set()`/`read()`/
`describe()`. That makes this device object the load-bearing surface for the
"the connector is the sole reachable CA path" claim (CC-2): if a device's
*public* API leaked the raw connector, or an unchecked write method, any plan
author could bypass `write_channel_checked`'s refusal/verification path.

This module enumerates the PUBLIC attribute surface (no leading underscore)
of both device classes and asserts none of it is the connector itself, and
none of it is an unchecked write path. It passes as-is — the public surface
(`read`/`describe`/`set`/`name`/`connect`/`parent`/... — see
`_PUBLIC_ATTRS_ARE_STABLE` below) is already clean. The point of this test is
to LOCK that in as a regression: if a future edit to `connector.py` (or an
ophyd-async version bump that adds a new public `Device`/`StandardReadable`
attribute) exposes the connector or a bare `write_channel`, this test fails.

Known accepted residual (NOT closed by this test, and not closeable by
attribute hiding): the connector reference is stored as
`self._osprey_connector` (connector.py:98, 177) — single-underscore, i.e.
reachable-but-not-public. A plan body holding the device can still reach
`device._osprey_connector.write_channel(...)` (the UNCHECKED write, as
opposed to `write_channel_checked`) via ordinary Python attribute access;
Python has no enforced private attributes, so no naming convention closes
this off, only obfuscates it. The real defense against that path is NOT this
device layer — it's task 2.1's static plan validator (which flags
`write_channel(` and `_osprey_connector` as dangerous patterns before a plan
is ever exec'd), the load-time provenance gate (task 2.4), and human
approval (task 2.6). This test does not claim, and must not be read as
claiming, that the device layer contains an adversarial plan body — see
`connectors/control_system/base.py` (`write_channel` vs
`write_channel_checked`, lines ~270 and ~322) for the deeper audit target of
the unchecked-write surface itself.

Runs ONLY in a bluesky-capable environment, same as `test_connector_devices.py`
(bluesky/ophyd-async are never installed in the main worktree venv):

    <worktree>/.venv/bin/python -m pytest \
        tests/services/bluesky_bridge/test_device_no_raw_ca_attr.py -q
"""

from __future__ import annotations

from typing import Any

import pytest

bluesky = pytest.importorskip("bluesky")
ophyd_async = pytest.importorskip("ophyd_async")

from osprey.services.bluesky_bridge.devices.connector import (  # noqa: E402
    ConnectorReadable,
    ConnectorSettable,
)


class FakeConnector:
    """A minimal async double exposing both the checked and unchecked write paths.

    Mirrors `ControlSystemConnector`'s two write methods (`base.py:270`,
    `base.py:322`) just enough to make the asymmetry this test cares about
    checkable: `write_channel` is the raw/unchecked path, `write_channel_checked`
    is the refusal-raising path the device layer is supposed to use exclusively.
    """

    def __init__(self) -> None:
        self.checked_calls: list[tuple[str, Any]] = []
        self.unchecked_calls: list[tuple[str, Any]] = []

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        class _Reading:
            value = 1.0

        return _Reading()

    async def write_channel_checked(self, channel_address: str, value: Any, **kwargs: Any):
        self.checked_calls.append((channel_address, value))

    async def write_channel(self, channel_address: str, value: Any, **kwargs: Any):
        """The unchecked write path — never called by the device layer itself."""
        self.unchecked_calls.append((channel_address, value))


def _public_attrs(obj: Any) -> list[str]:
    """Public attribute names: no leading underscore, dunders excluded."""
    return sorted(name for name in dir(obj) if not name.startswith("_"))


# Locked-in public surface for each class (task 2.7 regression lock). A
# failure here means either `connector.py` grew a new public attribute, or an
# ophyd-async version bump added one to `StandardReadable`/`Device` — audit
# the new name against `base.py`'s write_channel/write_channel_checked before
# updating this set.
_EXPECTED_SETTABLE_PUBLIC_ATTRS = [
    "add_children_as_readables",
    "add_readables",
    "children",
    "connect",
    "describe",
    "describe_configuration",
    "hints",
    "log",
    "name",
    "parent",
    "read",
    "read_configuration",
    "set",
    "set_name",
    "stage",
    "unstage",
]

_EXPECTED_READABLE_PUBLIC_ATTRS = [
    "add_children_as_readables",
    "add_readables",
    "children",
    "connect",
    "describe",
    "describe_configuration",
    "hints",
    "log",
    "name",
    "parent",
    "read",
    "read_configuration",
    "set_name",
    "stage",
    "unstage",
]


def _assert_no_raw_ca_or_unchecked_write(device: Any, fake: FakeConnector) -> None:
    """Common assertion body for both device classes' public surface."""
    public_attrs = _public_attrs(device)

    assert "write_channel" not in public_attrs, (
        "the unchecked write path must never be a public device attribute"
    )
    assert "_osprey_connector" not in public_attrs, (
        "the connector reference is a known private residual, not public — "
        "see this module's docstring"
    )

    for attr_name in public_attrs:
        value = getattr(device, attr_name)

        assert value is not fake, (
            f"public attribute {attr_name!r} must not return the raw connector"
        )
        assert not hasattr(value, "write_channel"), (
            f"public attribute {attr_name!r} exposes an unchecked write_channel path"
        )
        assert not hasattr(value, "read_channel"), (
            f"public attribute {attr_name!r} exposes a raw read_channel/CA handle"
        )


def test_connector_settable_public_surface_has_no_raw_ca_or_unchecked_write() -> None:
    """`ConnectorSettable`'s public API never leaks the connector or `write_channel`."""
    fake = FakeConnector()
    device = ConnectorSettable(fake, "SP", name="motor")

    assert _public_attrs(device) == _EXPECTED_SETTABLE_PUBLIC_ATTRS
    _assert_no_raw_ca_or_unchecked_write(device, fake)


def test_connector_readable_public_surface_has_no_raw_ca_or_unchecked_write() -> None:
    """`ConnectorReadable`'s public API never leaks the connector or `write_channel`."""
    fake = FakeConnector()
    device = ConnectorReadable(fake, "PV", name="detector")

    assert _public_attrs(device) == _EXPECTED_READABLE_PUBLIC_ATTRS
    _assert_no_raw_ca_or_unchecked_write(device, fake)


def test_osprey_connector_residual_is_reachable_but_not_public() -> None:
    """Documents the accepted residual: `_osprey_connector` is reachable, not public.

    This is NOT a vulnerability this test asserts should be silently accepted
    in the abstract — it is the documented boundary of what this device layer
    can and cannot enforce in-process. Python has no attribute access control,
    so a plan body holding the device can always reach a single-underscore
    attribute; the containment claim rests on the plan NEVER containing this
    call in the first place (task 2.1's static validator flags exactly
    `write_channel(` and `_osprey_connector`), not on this attribute being
    hidden. See `connectors/control_system/base.py` for the unchecked
    `write_channel` this residual reaches.
    """
    fake = FakeConnector()
    device = ConnectorSettable(fake, "SP", name="motor")

    # Not part of the public surface...
    assert "_osprey_connector" not in _public_attrs(device)
    # ...but ordinary attribute access still reaches it. This is the residual.
    assert device._osprey_connector is fake
    assert hasattr(device._osprey_connector, "write_channel")

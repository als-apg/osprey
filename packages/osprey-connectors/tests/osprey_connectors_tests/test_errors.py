"""The package's exception vocabulary: one base, ``*Error`` names, kept builtin bases, context."""

import functools
import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import pytest

import osprey_connectors
from osprey_connectors import archiver_selection, call_timeout, connection, connector_types, factory
from osprey_connectors.config_sections import ArchiverSection, ControlSystemSection
from osprey_connectors.errors import (
    AlreadyApprovedError,
    ConfigurationError,
    ConnectorInternalError,
    ConnectorsError,
    ConnectorUnavailableError,
    ConnectRefusedError,
    StateLockedError,
    StateUnavailableError,
)
from osprey_connectors.simulation.records import AddressesDocument, VariablesDocument
from osprey_connectors.simulation.view import (
    NoSimulatorViewError,
    SimulatorView,
    SimulatorViewError,
    ViewSchemaError,
)


@functools.cache
def _public_exceptions() -> dict[str, type[BaseException]]:
    """Every public exception class defined anywhere in the package, by qualified name.

    A module whose optional third-party dependency is not installed (a lane
    without the simulation engines, say) is skipped; a failure to import one of
    the package's own modules is not.
    """
    found: dict[str, type[BaseException]] = {}
    for info in pkgutil.walk_packages(osprey_connectors.__path__, "osprey_connectors."):
        try:
            module = importlib.import_module(info.name)
        except ModuleNotFoundError as missing:
            if (missing.name or "").partition(".")[0] == "osprey_connectors":
                raise
            continue
        for name, item in vars(module).items():
            if (
                inspect.isclass(item)
                and issubclass(item, BaseException)
                and item.__module__ == module.__name__
                and not name.startswith("_")
            ):
                found[f"{module.__name__}.{name}"] = item
    return found


def test_every_public_exception_has_the_one_base_and_an_error_name():
    exceptions = _public_exceptions()
    assert exceptions
    off_base = [name for name, cls in exceptions.items() if not issubclass(cls, ConnectorsError)]
    misnamed = [name for name, cls in exceptions.items() if not cls.__name__.endswith("Error")]
    assert off_base == []
    assert misnamed == []


@pytest.mark.parametrize(
    ("qualified", "builtin"),
    [
        ("osprey_connectors.errors.ConnectorUnavailableError", ConnectionError),
        ("osprey_connectors.errors.ConnectorInternalError", RuntimeError),
        ("osprey_connectors.errors.ClientLibraryRefusedError", RuntimeError),
        ("osprey_connectors.errors.ChannelNotFoundError", LookupError),
        ("osprey_connectors.errors.ConnectRefusedError", ConnectionError),
        ("osprey_connectors.errors.ConfigurationError", ValueError),
        ("osprey_connectors.errors.StateUnavailableError", RuntimeError),
        ("osprey_connectors.errors.StateLockedError", RuntimeError),
        ("osprey_connectors.errors.AlreadyApprovedError", FileExistsError),
        ("osprey_connectors.settle.SettleTimeoutError", TimeoutError),
        ("osprey_connectors.ipc._proxy.ChildUnresponsiveError", TimeoutError),
        ("osprey_connectors.ipc._proxy.ChildLostError", ConnectionError),
        ("osprey_connectors.ipc._pool.ConnectorHostSlowCallError", TimeoutError),
        ("osprey_connectors.ipc._pool.ConnectorHostUnresponsiveError", TimeoutError),
        ("osprey_connectors.ipc._frames.FrameDecodeError", ValueError),
        ("osprey_connectors.ipc._frames.FrameEncodeError", TypeError),
        ("osprey_connectors.simulation.view.NoSimulatorViewError", LookupError),
        ("osprey_connectors.simulation.view.ViewSchemaError", ValueError),
        ("osprey_connectors.simulation.composite.WriteRefusedError", ValueError),
    ],
)
def test_a_documented_builtin_base_is_kept(qualified, builtin):
    assert issubclass(_public_exceptions()[qualified], builtin)


def test_the_bases_that_must_stay_apart_do():
    exceptions = _public_exceptions()
    # A stale gesture is not "the store is unavailable".
    assert not issubclass(
        exceptions["osprey_connectors.control_context.StoppedSinceError"], RuntimeError
    )
    # A connector bug is never read as an unreachable control system.
    assert not issubclass(ConnectorInternalError, OSError)
    # A policy refusal of connect() is not a lost connector.
    assert not issubclass(ConnectRefusedError, ConnectorUnavailableError)
    # A slow call is never a lost connection, and the link base is neither.
    link = exceptions["osprey_connectors.errors.ChildLinkError"]
    for slow in ("ipc._pool.ConnectorHostSlowCallError", "ipc._proxy.ChildUnresponsiveError"):
        cls = exceptions[f"osprey_connectors.{slow}"]
        assert issubclass(cls, link)
        assert not issubclass(cls, (ConnectionError, ConnectorUnavailableError))
    assert not issubclass(link, (ConnectionError, TimeoutError))
    lost = exceptions["osprey_connectors.ipc._proxy.ChildLostError"]
    assert issubclass(lost, link)
    assert issubclass(lost, ConnectorUnavailableError)


def test_the_write_path_errors_carry_their_context(tmp_path):
    locked = StateLockedError("held", path=tmp_path / "x.lock", waited_s=5.0)
    assert (locked.path, locked.waited_s) == (tmp_path / "x.lock", 5.0)
    assert StateUnavailableError("no root").path is None
    approved = AlreadyApprovedError(tmp_path / "a.json", authorization="ab" * 16)
    assert approved.authorization == "ab" * 16
    assert approved.path == tmp_path / "a.json"
    assert str(approved) == f"[Errno 17] {tmp_path / 'a.json'} exists; nothing was written"


# -- config readers -----------------------------------------------------------


@pytest.mark.parametrize(
    ("read", "key"),
    [
        (
            lambda: call_timeout.call_timeout_s({"timeout_s": 0}, connector_types.EPICS),
            "control_system.connector.epics.timeout_s",
        ),
        (
            lambda: call_timeout.call_timeout_s({"timeout": 3}, connector_types.EPICS),
            "control_system.connector.epics.timeout",
        ),
        (
            lambda: call_timeout.refuse_renamed_timeout_keys(
                {"control_system": {"connector": {"tango": {"timeout": 1}}}}
            ),
            "control_system.connector.tango.timeout",
        ),
        (
            lambda: connector_types.resolve_serving(
                ControlSystemSection.read(
                    {
                        "type": connector_types.VIRTUAL_ACCELERATOR,
                        "connector": {
                            connector_types.VIRTUAL_ACCELERATOR: {
                                connector_types.SERVING_LEAF: "nowhere"
                            }
                        },
                    }
                )
            ),
            connector_types.SERVING_KEY,
        ),
        (
            lambda: archiver_selection.resolve_archiver_settings(
                ArchiverSection.read({"settings": 3})
            ),
            "archiver.settings",
        ),
        (
            lambda: connection.read_connection_settings(
                {"timeout_s": -1}, where="archiver.settings"
            ),
            "archiver.settings.timeout_s",
        ),
        (
            lambda: connection.read_connection_settings(
                {"auth": {"token_env": "A", "username": "u"}}, where="x.settings"
            ),
            "x.settings.auth",
        ),
        (
            lambda: factory.ConnectorFactory.build_control_system_connector(
                ControlSystemSection.read({"type": "no_such_type"})
            ),
            "control_system.type",
        ),
    ],
)
def test_a_config_reader_names_the_key_it_refuses(read, key):
    with pytest.raises(ConfigurationError) as caught:
        read()
    assert caught.value.key == key
    # Still a ValueError, so every caller that caught the bare refusal catches it.
    assert isinstance(caught.value, ValueError)


# -- the simulator view -------------------------------------------------------


def _view_dir(tmp_path: Path) -> Path:
    root = tmp_path / "simulator"
    root.mkdir()
    return root


def test_a_place_with_no_view_names_the_file_it_lacks(tmp_path):
    root = _view_dir(tmp_path)
    with pytest.raises(NoSimulatorViewError) as caught:
        SimulatorView.open(root)
    assert caught.value.path == root / AddressesDocument.FILE
    assert isinstance(caught.value, SimulatorViewError)


def _opened(tmp_path: Path) -> SimulatorView:
    from osprey_connectors.simulation.records import AddressesDocument

    root = _view_dir(tmp_path)
    (root / AddressesDocument.FILE).write_text(
        json.dumps({"schema": AddressesDocument.SCHEMA, "channels": [], "status": []}),
        encoding="utf-8",
    )
    return SimulatorView.open(root)


def test_a_missing_view_file_is_no_view_naming_that_file(tmp_path):
    view = _opened(tmp_path)
    with pytest.raises(NoSimulatorViewError) as caught:
        view.document(VariablesDocument)
    assert caught.value.path == view.path / VariablesDocument.FILE
    assert str(view.path / VariablesDocument.FILE) in str(caught.value)


def test_a_view_file_that_is_not_json_is_a_schema_error_naming_it(tmp_path):
    view = _opened(tmp_path)
    (view.path / VariablesDocument.FILE).write_text("{", encoding="utf-8")
    with pytest.raises(ViewSchemaError) as caught:
        view.document(VariablesDocument)
    assert caught.value.path == view.path / VariablesDocument.FILE
    assert str(view.path / VariablesDocument.FILE) in str(caught.value)


def test_a_schema_refusal_carries_path_key_found_and_expected(tmp_path):

    view = _opened(tmp_path)
    (view.path / VariablesDocument.FILE).write_text(json.dumps({"schema": "old"}), encoding="utf-8")
    with pytest.raises(ViewSchemaError) as caught:
        view.document(VariablesDocument)
    error = caught.value
    assert (error.path, error.key, error.found, error.expected) == (
        view.path / VariablesDocument.FILE,
        "schema",
        "old",
        VariablesDocument.SCHEMA,
    )
    # The message is the one an operator already reads.
    assert "is schema 'old'; this OSPREY reads" in str(error)

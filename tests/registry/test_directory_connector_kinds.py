"""The registry initializer dispatches the directory kinds, for out-of-tree backends.

No provider names these kinds: the builtins live in the factory's own table. A
registration here is a facility's explicit choice, and on an installed
``osprey-connectors`` older than the kinds it fails naming what is missing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from osprey.registry.base import ConnectorRegistration
from osprey.registry.initializers import initialize_connectors
from osprey_connectors.errors import RegistryError
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors_tests._harness import isolated_connector_registries

KINDS = ("directory", "alarm_log", "alarm_state", "put_log")


def _stem(kind: str) -> str:
    """The factory's method and registry stem for *kind*; the directory kind keeps its old one."""
    return "pv_directory" if kind == "directory" else kind


_has_kinds = pytest.mark.skipif(
    not hasattr(ConnectorFactory, "register_pv_directory"),
    reason="connectors floor has no directory kinds",
)


class OutOfTreeBackend:
    """Only the class is read: the initializer registers it and builds nothing."""


def _initialize(connector_type: str) -> None:
    registration = ConnectorRegistration(
        name="out_of_tree",
        connector_type=connector_type,
        module_path=__name__,
        class_name=OutOfTreeBackend.__name__,
        description="An out-of-tree backend",
    )
    initialize_connectors(
        config=SimpleNamespace(connectors=[registration]),
        registries={},
        excluded_provider_names=[],
    )


@_has_kinds
@pytest.mark.parametrize("kind", KINDS)
def test_a_registration_reaches_its_kinds_registry(kind: str) -> None:
    with isolated_connector_registries():
        _initialize(kind)

        assert (
            getattr(ConnectorFactory, f"_{_stem(kind)}_connectors")["out_of_tree"]
            is OutOfTreeBackend
        )


@pytest.mark.parametrize("kind", KINDS)
def test_connectors_without_the_kind_refuse_it_by_name(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The floor's factory, on either tree: no ``register_<kind>`` to find.
    monkeypatch.delattr(ConnectorFactory, f"register_{_stem(kind)}", raising=False)

    with pytest.raises(RegistryError, match=rf"has no {kind} support"):
        _initialize(kind)


def test_an_unknown_kind_still_raises_naming_it() -> None:
    with pytest.raises(RegistryError, match="Unknown connector type: nonsense"):
        _initialize("nonsense")

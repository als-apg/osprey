"""Shared helpers for the simulation tests: a staged project and a stand-in
archive collection."""

import copy
from pathlib import Path
from typing import Any

#: The ``control_system`` block of a project serving the simulator in process:
#: it serves the simulator view under ``data/`` and takes no setting of its own.
IN_PROCESS_SIM_CONTROL_SYSTEM = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": {"serving": "in_process"}},
}


def stage_sim_project(
    root: Path, *, control_system: dict | None = None, **config_extra: Any
) -> Path:
    """Stage a project: ``config.yml`` beside the ``data/`` tree.

    The flat shape. The tests write the simulator view under ``data/``
    themselves (:func:`tests._simulator_view.write_scenarios_view`).
    ``control_system`` defaults to the simulator in process; any other top-level
    config sections (``ariel``, ``system``, ...) go in ``config_extra``.
    """
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    config = {
        "control_system": copy.deepcopy(
            IN_PROCESS_SIM_CONTROL_SYSTEM if control_system is None else control_system
        ),
        **config_extra,
    }
    (root / "config.yml").write_text(yaml.safe_dump(config))
    return root


class StubCollection:
    """The collection calls :func:`seed_base`, :func:`write_manifest` and
    :func:`compare_fingerprint` make, and no more.

    Lets the manifest contract run without a MongoDB container. What it cannot
    stand in for is BSON's round trip of the stored values; the container-backed
    ``TestSeedBase`` in ``test_archiver_seed.py`` covers that.
    """

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self._by_id: dict[Any, dict[str, Any]] = {}

    def create_index(self, *args: Any, **kwargs: Any) -> None:
        return None

    def insert_many(self, documents: list[dict[str, Any]], **kwargs: Any) -> None:
        self.documents.extend(documents)

    def replace_one(
        self, filter_: dict[str, Any], document: dict[str, Any], upsert: bool = False, **kwargs: Any
    ) -> None:
        # Like MongoDB: without ``upsert`` a replace of a missing document is a no-op.
        key = filter_["_id"]
        if upsert or key in self._by_id:
            self._by_id[key] = document

    def find_one(self, filter_: dict[str, Any], **kwargs: Any) -> dict[str, Any] | None:
        return self._by_id.get(filter_["_id"])

    @property
    def manifest(self) -> dict[str, Any] | None:
        """The manifest document, or ``None`` when none was written."""
        from osprey_connectors.simulation.archive import MANIFEST_ID

        return self._by_id.get(MANIFEST_ID)

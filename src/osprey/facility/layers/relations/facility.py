"""The facility a relations import resolves against.

:class:`FacilityIndex` is the facility as ``load_sources`` reads it with every
record of the layer being replaced dropped, so a re-import never resolves its
rows against its own previous output. It holds the devices (with class), the
places, the authored places, the groups (with members) and the known classes.

The index reads sources, not the built facility: ``fixes.yaml`` is not applied,
so a device that only a ``fixes.yaml`` operation adds is not in the index. A row
naming such a device stubs it (or is refused), and the build then reports the
clash between the stub and the fix.

Load errors in the replaced layer's own files are ignored, since the re-import
replaces those files; any other load error is one refusal naming the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from osprey.facility.errors import FacilityBuildError
from osprey.facility.layers import LayerSourceInvalid
from osprey.facility.places import resolve_place_name
from osprey.facility.sources import AUTHORED, load_sources

__all__ = ["FacilityIndex"]


def _in_layer(source: str, layer: str) -> bool:
    prefix = f"imported/{layer}"
    return source == prefix or source.startswith(prefix + "/")


@dataclass
class FacilityIndex:
    """The facility minus one layer, for resolving a relations table's cells.

    Attributes:
        devices: Device id to its class, or ``None`` when no layer states one.
            Where layers state different classes, the first stated (authored
            first, then imported layers by name) is kept; the build reports
            the conflict.
        places: Every place id.
        authored_places: The place ids the authored records state.
        groups: Group id to its members, the union of what the layers state.
        classes: Every class a device may carry (``validate.known_classes``).
    """

    devices: dict[str, str | None] = field(default_factory=dict)
    places: set[str] = field(default_factory=set)
    authored_places: set[str] = field(default_factory=set)
    groups: dict[str, set[str]] = field(default_factory=dict)
    classes: frozenset[str] = frozenset()

    @classmethod
    def load(cls, facility_dir: Path, layer: str) -> FacilityIndex:
        """Read the facility with ``layer``'s records dropped.

        Args:
            facility_dir: The ``data/facility`` directory.
            layer: The layer the import replaces.

        Returns:
            The index.

        Raises:
            LayerSourceInvalid: One ``source-invalid`` error naming the first
                load error outside ``imported/<layer>/``.
        """
        from osprey.facility.validate import known_classes

        result = load_sources(facility_dir)
        errors = [
            error
            for error in result.errors
            if not (error.sources and all(_in_layer(s, layer) for s in error.sources))
        ]
        if errors:
            first = errors[0]
            raise LayerSourceInvalid(
                [
                    FacilityBuildError(
                        "source-invalid",
                        first.record_id,
                        first.sources,
                        first.remedy,
                        record_kind=first.record_kind,
                        detail=f"the facility does not load: {first.detail}",
                    )
                ]
            )
        index = cls(classes=known_classes(result.sources.classes))
        for record in result.sources.records:
            if record.layer == layer:
                continue
            if record.kind == "device":
                stated = record.fields.get("class")
                stated = stated if isinstance(stated, str) else None
                if index.devices.get(record.id) is None:
                    index.devices[record.id] = stated
            elif record.kind == "place":
                index.places.add(record.id)
                if record.layer == AUTHORED:
                    index.authored_places.add(record.id)
            elif record.kind == "group":
                members = record.fields.get("members") or []
                index.groups.setdefault(record.id, set()).update(str(member) for member in members)
        return index

    def resolve_place(
        self, cell: str, extra: set[str], *, base: set[str] | None = None
    ) -> str | list[str] | None:
        """Resolve a table cell to a place id.

        A cell holding ``/`` is a path and must equal an id. A flat cell matches
        every id equal to it or ending in ``/<cell>``
        (:func:`~osprey.facility.places.resolve_place_name`).

        Args:
            cell: The cell's text.
            extra: The places the importing layer itself creates.
            base: The ids to match besides ``extra``; every place by default.

        Returns:
            The id for one match, the sorted candidates for several, ``None``
            for none.
        """
        return resolve_place_name(cell, (self.places if base is None else base) | extra)

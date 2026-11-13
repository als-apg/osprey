"""A package whose kind names a hook OSPREY does not offer."""

from __future__ import annotations

import dataclasses

from osprey.facility.kinds import ORM

HOOKED_KIND = {
    **{field.name: getattr(ORM, field.name) for field in dataclasses.fields(ORM)},
    "name": "hooked_kind",
    "on_measure": lambda *args: None,
}

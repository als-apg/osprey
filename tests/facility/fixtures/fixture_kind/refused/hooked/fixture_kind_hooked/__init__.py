"""A package whose kind names a hook OSPREY does not offer."""

from __future__ import annotations

import dataclasses

from fixture_kind_deployment import FIXTURE_KIND

HOOKED = {
    **{field.name: getattr(FIXTURE_KIND, field.name) for field in dataclasses.fields(FIXTURE_KIND)},
    "name": "hooked_kind",
    "on_trigger": lambda *args: None,
}

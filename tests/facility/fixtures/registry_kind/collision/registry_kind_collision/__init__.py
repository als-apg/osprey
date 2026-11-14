"""A package that registers a kind under a built-in kind's name."""

from __future__ import annotations

import dataclasses

from osprey.facility.kinds import ORM

ORM_AGAIN = dataclasses.replace(ORM, description="a second orm")

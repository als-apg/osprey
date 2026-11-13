"""A second package that registers a kind under the fixture kind's name."""

from __future__ import annotations

import dataclasses

from fixture_kind_deployment import FIXTURE_KIND

TWIN = dataclasses.replace(FIXTURE_KIND, description="a second fixture_kind")

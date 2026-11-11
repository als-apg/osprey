"""A package that registers a kind under a built-in kind's name."""

from __future__ import annotations

import dataclasses

from fixture_kind_deployment import FIXTURE_KIND

ORM_AGAIN = dataclasses.replace(FIXTURE_KIND, name="orm")

"""The words a wiring record's description uses are the schema's words.

The simulator reads ``role``, ``plane`` and ``refresh`` off each wiring entry
by the vocabulary its view module states; the facility schema states the same
three slots as enums. The two are one vocabulary.
"""

from __future__ import annotations

from osprey.facility.schema import core
from osprey_connectors.simulation.view import PLANES, REFRESH, ROLES


def test_the_roles_are_the_schemas_wiring_roles() -> None:
    assert set(ROLES) == {member.value for member in core.WiringRoleEnum}


def test_the_planes_are_the_schemas_planes() -> None:
    assert set(PLANES) == {member.value for member in core.PlaneEnum}


def test_the_refreshes_are_the_schemas_refreshes() -> None:
    assert set(REFRESH) == {member.value for member in core.RefreshEnum}

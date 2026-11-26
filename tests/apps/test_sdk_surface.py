"""``osprey.apps.sdk``'s public surface: every exported name is one the module carries.

Which names it exports is held by the contract guard,
``tests/apps/test_contract_surface.py``.
"""

from __future__ import annotations

import osprey.apps.sdk as sdk


def test_every_name_is_a_bare_identifier_the_module_carries():
    for name in sdk.__all__:
        assert name.isidentifier(), name
        assert hasattr(sdk, name), name

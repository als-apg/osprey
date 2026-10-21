"""The facility file's schema and the pydantic model generated from it.

``Facility`` validates one whole facility file; a key the schema does not
declare fails validation.
"""

from __future__ import annotations

import warnings

# The root class declares the facility file's ``schema`` header slot, which
# shadows pydantic's ``BaseModel.schema`` method; that one warning is expected.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r'Field name "schema" in "Facility" shadows an attribute in parent',
    )
    from osprey.facility.schema._generated import core

Facility = core.Facility

__all__ = ["Facility", "core"]

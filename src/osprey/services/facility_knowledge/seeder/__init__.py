"""Facility knowledge seeder subpackage.

Provides :func:`seed_from_ttl` for generating OKF stub documents from a
NARAD/als-ontology RDF/TTL file.  Import this module directly — it is
intentionally excluded from the parent ``facility_knowledge`` namespace to
avoid pulling in the ``knowledge`` extra at import time.

Example::

    from osprey.services.facility_knowledge.seeder import seed_from_ttl

    stubs = seed_from_ttl(Path("/path/to/als_gtb.ttl"))
"""

from .ttl_seeder import DeviceStub, local_name, seed_from_ttl

__all__ = [
    "DeviceStub",
    "local_name",
    "seed_from_ttl",
]

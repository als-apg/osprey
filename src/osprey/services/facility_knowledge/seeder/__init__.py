"""Facility knowledge seeder subpackage.

:func:`seed_from_ttl` (:mod:`.ttl_seeder`) turns a NARAD RDF/TTL file into OKF
stub *documents* — a text bundle on disk — driven offline from
``osprey knowledge``. Loading that same TTL into a Neo4j graph store is
:mod:`osprey.services.graph_store`'s job.

Import this subpackage directly — it is intentionally excluded from the parent
``facility_knowledge`` namespace so that the runtime read path does not pull in
``rdflib``.  It is a core dependency, so this is an import-graph boundary rather
than an optional-install one; it is imported lazily inside the functions that
need it.

Example::

    from osprey.services.facility_knowledge.seeder import seed_from_ttl

    stubs = seed_from_ttl(Path("/path/to/corpus.ttl"))
"""

from .ttl_seeder import DeviceStub, local_name, seed_from_ttl

__all__ = [
    "DeviceStub",
    "local_name",
    "seed_from_ttl",
]

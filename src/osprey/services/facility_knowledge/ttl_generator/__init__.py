"""Demo-machine TTL generator for the OSPREY graph store.

Derives a NARAD-convention knowledge graph from the control_assistant channel
database.  The subpackage is deliberately **not** re-exported from
:mod:`osprey.services.facility_knowledge` and this ``__init__`` re-exports
nothing, so importing the parent package never pulls in ``rdflib`` (the
optional ``knowledge`` extra).  Import the modules you need directly::

    from osprey.services.facility_knowledge.ttl_generator.model import build_model

:mod:`.model` is pure data — standard library only, no ``rdflib``, no I/O.
"""

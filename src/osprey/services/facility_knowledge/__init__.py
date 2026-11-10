"""
Facility Knowledge Service

Provides access to the OKF (Open Knowledge Format) document model and related
utilities for parsing, validating, and serializing facility knowledge documents.

Note: the seeder subpackage is intentionally excluded from this namespace — import it
directly from ``osprey.services.facility_knowledge.seeder``.  Keeping it out of the
package ``__init__`` keeps ``rdflib`` out of this module's import graph — the seeder
needs it, the document model does not.  The Neo4j graph-store primitives the TTL is
loaded with live in :mod:`osprey.services.graph_store`.
"""

from .okf import OKFDocument, OKFDocumentError

__all__ = [
    "OKFDocument",
    "OKFDocumentError",
]

"""
Facility Knowledge Service

Provides access to the OKF (Open Knowledge Format) document model and related
utilities for parsing, validating, and serializing facility knowledge documents.

Note: the seeder subpackage is intentionally excluded from this namespace — import it
directly from ``osprey.services.facility_knowledge.seeder``.  Keeping it out of the
package ``__init__`` ensures this module stays importable without the optional
``knowledge`` extra (rdflib), which the seeder needs but the document model does not.
"""

from .okf import OKFDocument, OKFDocumentError

__all__ = [
    "OKFDocument",
    "OKFDocumentError",
]

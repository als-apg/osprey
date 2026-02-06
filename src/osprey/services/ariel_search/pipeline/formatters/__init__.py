"""Formatter implementations for ARIEL RAP pipeline.

Formatters transform processed results into the final response format.

Available formatters:
    - CitationFormatter: Text with [#id] citations
    - JSONFormatter: Structured JSON for API responses
"""

from osprey.services.ariel_search.pipeline.formatters.citation import CitationFormatter
from osprey.services.ariel_search.pipeline.formatters.json import JSONFormatter

__all__ = [
    "CitationFormatter",
    "JSONFormatter",
]

"""Shared Typesense filter_by builder for AccelPapers tools."""


def build_filter_string(
    *,
    conference: str | None = None,
    year: int | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    author: str | None = None,
    document_type: str | None = None,
) -> str:
    """Build a Typesense ``filter_by`` string from optional filter params.

    Returns an empty string when no filters are active.
    """
    parts: list[str] = []
    if conference:
        parts.append(f"conference:={conference}")
    if year is not None:
        parts.append(f"year:={year}")
    if year_min is not None:
        parts.append(f"year:>={year_min}")
    if year_max is not None:
        parts.append(f"year:<={year_max}")
    if author:
        parts.append(f"all_authors:{author}")
    if document_type:
        parts.append(f"document_type:={document_type}")
    return " && ".join(parts)

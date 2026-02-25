"""JSON → Typesense indexer for AccelPapers.

Reads structured JSON files from INSPIRE downloads and builds a searchable
Typesense collection with BM25 full-text search and vector embeddings via
auto-embedding (Ollama nomic-embed-text).
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("osprey.mcp_server.accelpapers.indexer")

# --- Collection schema -------------------------------------------------------

PAPERS_SCHEMA = {
    "name": "papers",  # overridden at create time with get_collection_name()
    "fields": [
        # texkey is stored as the Typesense `id` field — not a separate field
        {"name": "title", "type": "string"},
        {"name": "abstract", "type": "string", "optional": True},
        {"name": "year", "type": "int32", "optional": True, "facet": True},
        {"name": "conference", "type": "string", "optional": True, "facet": True},
        {"name": "first_author", "type": "string", "optional": True, "facet": True},
        {"name": "all_authors", "type": "string", "optional": True},
        {"name": "affiliations", "type": "string", "optional": True},
        {"name": "keywords", "type": "string", "optional": True},
        {"name": "doi", "type": "string", "optional": True},
        {"name": "citation_count", "type": "int32", "facet": True},
        {"name": "document_type", "type": "string", "optional": True, "facet": True},
        {"name": "inspire_url", "type": "string", "optional": True},
        {"name": "pdf_url", "type": "string", "optional": True},
        {"name": "arxiv_id", "type": "string", "optional": True},
        {"name": "journal_title", "type": "string", "optional": True, "facet": True},
        {"name": "publisher", "type": "string", "optional": True},
        {"name": "num_pages", "type": "int32", "optional": True},
        {"name": "full_text", "type": "string", "optional": True},
        {"name": "json_path", "type": "string"},
        {"name": "earliest_date", "type": "string", "optional": True},
        {"name": "paper_id", "type": "string", "optional": True},
        {
            "name": "embedding",
            "type": "float[]",
            "num_dim": 768,
            "embed": {
                "from": ["title", "abstract", "keywords"],
                "model_config": {
                    "model_name": "openai/nomic-embed-text",
                    "api_key": "ollama",
                    # Typesense appends /v1/embeddings itself — pass the base URL only
                    "url": "http://localhost:11434",
                },
            },
        },
    ],
    "default_sorting_field": "citation_count",
}


def _build_papers_schema() -> dict:
    """Build the papers collection schema with env-var-configurable embedding settings.

    Environment variables:
        ACCELPAPERS_OLLAMA_URL: Ollama base URL (default: http://localhost:11434)
        ACCELPAPERS_EMBEDDING_MODEL: Model name (default: openai/nomic-embed-text)
        ACCELPAPERS_EMBEDDING_API_KEY: API key (default: ollama)
    """
    import copy

    schema = copy.deepcopy(PAPERS_SCHEMA)
    for field in schema["fields"]:
        if field["name"] == "embedding":
            cfg = field["embed"]["model_config"]
            cfg["url"] = os.environ.get("ACCELPAPERS_OLLAMA_URL", "http://localhost:11434")
            cfg["model_name"] = os.environ.get(
                "ACCELPAPERS_EMBEDDING_MODEL", "openai/nomic-embed-text"
            )
            cfg["api_key"] = os.environ.get("ACCELPAPERS_EMBEDDING_API_KEY", "ollama")
            break
    return schema


# --- Helpers ------------------------------------------------------------------


def normalize_year(raw: str | int | None) -> int | None:
    """Extract a 4-digit year from various formats."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Take first 4 digits
    digits = "".join(c for c in s[:10] if c.isdigit())
    if len(digits) >= 4:
        year = int(digits[:4])
        if 1900 <= year <= 2100:
            return year
    return None


def normalize_num_pages(raw: str | int | None) -> int | None:
    """Extract page count from string or int."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    digits = "".join(c for c in s if c.isdigit())
    if digits:
        return int(digits)
    return None


def extract_full_text(content: dict | None) -> str:
    """Extract concatenated full text from content.sections dict.

    ``content.sections`` is a dict keyed by string numbers, each with
    ``full_text`` (list of strings) and optionally ``clean_text``.
    """
    if not content or "sections" not in content:
        return ""
    sections = content["sections"]
    if not isinstance(sections, dict):
        return ""
    parts = []
    for _key in sorted(sections.keys(), key=lambda k: int(k) if k.isdigit() else 0):
        section = sections[_key]
        title = section.get("title", "")
        if title:
            parts.append(title)
        full_text = section.get("full_text", [])
        if isinstance(full_text, list):
            parts.extend(str(t) for t in full_text if t)
        elif isinstance(full_text, str):
            parts.append(full_text)
    return "\n".join(parts)


def get_conference(data: dict, subdir_name: str) -> str:
    """Determine conference name from paper data or parent directory."""
    conf = data.get("conf_acronym", "").strip()
    if conf:
        return conf
    # Fall back to subdir name if it looks like a conference
    if subdir_name and not subdir_name.startswith("ARXIV") and subdir_name != "book-chapter":
        return subdir_name
    return ""


def build_all_authors(data: dict) -> str:
    """Build a semicolon-separated author list."""
    authors = []
    first = data.get("first_author_full_name", "")
    if first:
        authors.append(first)
    others = data.get("other_authors_full_names", [])
    if isinstance(others, list):
        authors.extend(str(a) for a in others if a)
    return "; ".join(authors)


def _extract_affiliations(data: dict) -> str:
    """Extract affiliations from paper data."""
    affs = []
    first_aff = data.get("first_author_affiliations", "")
    if first_aff:
        affs.append(first_aff)
    # Also check metadata.authors for more affiliations
    meta_authors = data.get("metadata", {}).get("authors", [])
    for author in meta_authors:
        for aff in author.get("affiliations", []):
            val = aff.get("value", "")
            if val and val not in affs:
                affs.append(val)
    return "; ".join(affs)


def _parse_paper(data: dict, json_path: str, subdir_name: str) -> dict | None:
    """Parse a single paper JSON into a document dict. Returns None if unusable."""
    texkey = data.get("texkey", "").strip()
    title = data.get("title", "").strip()
    if not texkey or not title:
        return None

    keywords = data.get("keywords", [])
    kw_str = "; ".join(str(k) for k in keywords) if isinstance(keywords, list) else ""

    return {
        "id": texkey,  # Typesense id field = texkey
        "title": title,
        "abstract": data.get("abstract_value", "") or "",
        "year": normalize_year(data.get("year")),
        "earliest_date": data.get("earliest_date", "") or "",
        "conference": get_conference(data, subdir_name),
        "first_author": data.get("first_author_full_name", "") or "",
        "all_authors": build_all_authors(data),
        "affiliations": _extract_affiliations(data),
        "keywords": kw_str,
        "doi": data.get("doi", "") or "",
        "citation_count": data.get("citation_count", 0) or 0,
        "paper_id": data.get("paper_id", "") or "",
        "document_type": data.get("document_type", "") or "",
        "inspire_url": data.get("inspireHEP_url", "") or "",
        "pdf_url": data.get("pdf_url", "") or "",
        "arxiv_id": data.get("arxiv_eprints", "") or "",
        "journal_title": data.get("journal_title", "") or "",
        "publisher": data.get("publisher", "") or "",
        "num_pages": normalize_num_pages(data.get("number_of_pages")),
        "full_text": extract_full_text(data.get("content")) or "",
        "json_path": json_path,
    }


# --- Main indexer -------------------------------------------------------------


def _collect_json_files(data_dir: Path) -> list[tuple[Path, str]]:
    """Collect all JSON files with their parent subdir name."""
    files = []
    for entry in sorted(data_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".json":
            files.append((entry, ""))
        elif entry.is_dir():
            subdir_name = entry.name
            for f in sorted(entry.iterdir()):
                if f.is_file() and f.suffix == ".json":
                    files.append((f, subdir_name))
    return files


def build_index(
    data_dir: str | Path,
    batch_size: int = 200,
    recreate: bool = True,
) -> str:
    """Build the Typesense collection from INSPIRE JSON files.

    Args:
        data_dir: Directory containing JSON files (possibly in subdirectories).
        batch_size: Documents per import batch (default: 200).
        recreate: Drop and recreate the collection if it exists (default: True).

    Returns:
        The collection name.
    """
    from osprey.mcp_server.accelpapers.db import get_client, get_collection_name

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    client = get_client()
    collection_name = get_collection_name()

    logger.info("Building index: %s → Typesense collection '%s'", data_dir, collection_name)

    # Create or recreate collection
    if recreate:
        try:
            client.collections[collection_name].delete()
            logger.info("Deleted existing collection '%s'", collection_name)
        except Exception:
            pass  # Collection doesn't exist yet

    schema = {**_build_papers_schema(), "name": collection_name}
    client.collections.create(schema)
    logger.info("Created collection '%s'", collection_name)

    collection = client.collections[collection_name]

    json_files = _collect_json_files(data_dir)
    total = len(json_files)
    logger.info("Found %d JSON files to index", total)

    indexed = 0
    skipped = 0
    errors = 0
    failed_imports = 0
    batch: list[dict] = []
    t0 = time.time()

    for i, (fpath, subdir_name) in enumerate(json_files):
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", fpath.name, exc)
            errors += 1
            continue

        if not isinstance(data, dict):
            skipped += 1
            continue

        doc = _parse_paper(data, str(fpath), subdir_name)
        if doc is None:
            skipped += 1
            continue

        batch.append(doc)
        if len(batch) >= batch_size:
            results = collection.documents.import_(batch, {"action": "upsert"})
            failures = [r for r in results if not r.get("success", True)]
            failed_imports += len(failures)
            indexed += len(batch) - len(failures)
            if failures:
                logger.warning("Batch import: %d failures", len(failures))
            batch.clear()

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d/%d (%.0f files/sec) — indexed=%d skipped=%d errors=%d",
                i + 1, total, rate, indexed, skipped, errors,
            )

    # Flush remaining batch
    if batch:
        results = collection.documents.import_(batch, {"action": "upsert"})
        failures = [r for r in results if not r.get("success", True)]
        failed_imports += len(failures)
        indexed += len(batch) - len(failures)
        if failures:
            logger.warning("Final batch import: %d failures", len(failures))

    elapsed = time.time() - t0
    logger.info(
        "Indexing complete in %.1fs: %d indexed, %d skipped, %d errors, "
        "%d import failures (collection: %s)",
        elapsed, indexed, skipped, errors, failed_imports, collection_name,
    )

    return collection_name

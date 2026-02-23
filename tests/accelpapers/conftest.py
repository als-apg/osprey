"""Shared fixtures for AccelPapers MCP server tests.

Provides mock Typesense client fixtures and sample paper data for testing
tools and indexer in isolation.
"""

import json

import pytest


def get_tool_fn(tool_or_fn):
    """Extract the raw async function from a FastMCP FunctionTool."""
    if hasattr(tool_or_fn, "fn"):
        return tool_or_fn.fn
    return tool_or_fn


SAMPLE_PAPERS = [
    {
        "texkey": "Smith:2020abc",
        "first_author_full_name": "Smith, John",
        "first_author_affiliations": "CERN",
        "other_authors_full_names": ["Doe, Jane", "Wang, Li"],
        "title": "Beam position monitor calibration for synchrotron light sources",
        "year": "2020",
        "keywords": ["beam position monitor", "calibration", "synchrotron"],
        "earliest_date": "2020-03-15",
        "number_of_pages": "12",
        "pdf_url": "https://example.com/smith2020.pdf",
        "citation_count": 45,
        "abstract_value": "We present a novel calibration method for beam position monitors (BPMs) at synchrotron light sources. The method achieves sub-micron accuracy.",
        "doi": "10.1234/test.2020.001",
        "c_num": "",
        "paper_id": "001",
        "arxiv_eprints": "2003.12345",
        "inspireHEP_url": "https://inspirehep.net/literature/123456",
        "conf_acronym": "IPAC2020",
        "journal_title": "Phys.Rev.Accel.Beams",
        "document_type": "conference paper",
        "publisher": "",
        "content": {
            "tables": [],
            "figures": [],
            "footnotes": [],
            "sections": {
                "1": {
                    "title": "Abstract",
                    "full_text": ["We present a novel calibration method for BPMs."],
                },
                "2": {
                    "title": "Introduction",
                    "full_text": [
                        "Beam position monitors are essential instruments.",
                        "Accurate calibration is critical for orbit correction.",
                    ],
                },
            },
        },
    },
    {
        "texkey": "Chen:2019xyz",
        "first_author_full_name": "Chen, Wei",
        "first_author_affiliations": "SLAC",
        "other_authors_full_names": [],
        "title": "RF cavity design optimization using machine learning",
        "year": "2019",
        "keywords": ["RF cavity", "machine learning", "optimization"],
        "earliest_date": "2019-06-01",
        "number_of_pages": "8",
        "pdf_url": "https://example.com/chen2019.pdf",
        "citation_count": 120,
        "abstract_value": "Machine learning techniques are applied to optimize RF cavity geometry for particle accelerators.",
        "doi": "10.1234/test.2019.002",
        "c_num": "",
        "paper_id": "002",
        "arxiv_eprints": "1906.54321",
        "inspireHEP_url": "https://inspirehep.net/literature/789012",
        "conf_acronym": "",
        "journal_title": "Nucl.Instrum.Meth.A",
        "document_type": "article",
        "publisher": "Elsevier",
        "content": {
            "tables": [],
            "figures": [],
            "footnotes": [],
            "sections": {
                "1": {
                    "title": "Abstract",
                    "full_text": ["ML techniques for RF cavity optimization."],
                },
            },
        },
    },
    {
        "texkey": "Mueller:2021def",
        "first_author_full_name": "Mueller, Hans",
        "first_author_affiliations": "DESY",
        "other_authors_full_names": ["Smith, John"],
        "title": "Undulator radiation characterization at PETRA IV",
        "year": "2021",
        "keywords": ["undulator", "radiation", "PETRA IV", "synchrotron"],
        "earliest_date": "2021-09-20",
        "number_of_pages": "6",
        "pdf_url": "",
        "citation_count": 15,
        "abstract_value": "Undulator radiation properties at the upcoming PETRA IV facility are characterized through simulation and measurement.",
        "doi": "",
        "c_num": "",
        "paper_id": "003",
        "arxiv_eprints": "",
        "inspireHEP_url": "https://inspirehep.net/literature/345678",
        "conf_acronym": "IPAC2021",
        "journal_title": "",
        "document_type": "conference paper",
        "publisher": "",
        "content": None,
    },
    {
        "texkey": "Tanaka:2018ghi",
        "first_author_full_name": "Tanaka, Kenji",
        "first_author_affiliations": "KEK, Tsukuba",
        "other_authors_full_names": ["Yamamoto, Hitoshi"],
        "title": "Bunch length measurement using streak camera at SuperKEKB",
        "year": "2018",
        "keywords": ["bunch length", "streak camera", "SuperKEKB"],
        "earliest_date": "2018-01-10",
        "number_of_pages": "4",
        "pdf_url": "https://example.com/tanaka2018.pdf",
        "citation_count": 8,
        "abstract_value": "We report bunch length measurements at the SuperKEKB electron-positron collider using a streak camera system.",
        "doi": "10.1234/test.2018.004",
        "c_num": "",
        "paper_id": "004",
        "arxiv_eprints": "1801.98765",
        "inspireHEP_url": "https://inspirehep.net/literature/567890",
        "conf_acronym": "NAPAC2018",
        "journal_title": "",
        "document_type": "conference paper",
        "publisher": "",
        "content": {
            "tables": [],
            "figures": [],
            "footnotes": [],
            "sections": {
                "1": {
                    "title": "Introduction",
                    "full_text": ["SuperKEKB requires precise bunch length control."],
                },
            },
        },
    },
    {
        "texkey": "Jones:2022jkl",
        "first_author_full_name": "Jones, Alice",
        "first_author_affiliations": "Argonne",
        "other_authors_full_names": [],
        "title": "Beam position monitor electronics upgrade for APS-U",
        "year": "2022",
        "keywords": ["beam position monitor", "APS-U", "electronics"],
        "earliest_date": "2022-05-05",
        "number_of_pages": "10",
        "pdf_url": "https://example.com/jones2022.pdf",
        "citation_count": 3,
        "abstract_value": "The APS Upgrade requires new BPM electronics with improved resolution and faster data rates.",
        "doi": "10.1234/test.2022.005",
        "c_num": "",
        "paper_id": "005",
        "arxiv_eprints": "",
        "inspireHEP_url": "https://inspirehep.net/literature/901234",
        "conf_acronym": "IPAC2022",
        "journal_title": "",
        "document_type": "conference paper",
        "publisher": "",
        "content": {
            "tables": [],
            "figures": [],
            "footnotes": [],
            "sections": {
                "1": {
                    "title": "Abstract",
                    "full_text": ["New BPM electronics for APS-U."],
                },
            },
        },
    },
]


def _build_indexed_docs() -> list[dict]:
    """Build Typesense-format documents from SAMPLE_PAPERS using the indexer parser."""
    from osprey.mcp_server.accelpapers.indexer import _parse_paper

    docs = []
    for paper in SAMPLE_PAPERS:
        doc = _parse_paper(paper, f"/tmp/{paper['texkey']}.json", "")
        if doc:
            docs.append(doc)
    return docs


# Pre-built indexed documents for mock responses
_INDEXED_DOCS: list[dict] | None = None


def _get_indexed_docs() -> list[dict]:
    global _INDEXED_DOCS
    if _INDEXED_DOCS is None:
        _INDEXED_DOCS = _build_indexed_docs()
    return _INDEXED_DOCS


def _matches_filter(doc: dict, filter_by: str) -> bool:
    """Simple filter_by evaluator for mock Typesense searches."""
    if not filter_by:
        return True

    parts = filter_by.split(" && ")
    for part in parts:
        part = part.strip()
        if ":=" in part:
            field, value = part.split(":=", 1)
            if str(doc.get(field, "")) != value:
                return False
        elif ":>=" in part:
            field, value = part.split(":>=", 1)
            if (doc.get(field) or 0) < int(value):
                return False
        elif ":<=" in part:
            field, value = part.split(":<=", 1)
            if (doc.get(field) or 0) > int(value):
                return False
        elif ":" in part:
            # Substring/text match (e.g. all_authors:Smith)
            field, value = part.split(":", 1)
            if value.lower() not in str(doc.get(field, "")).lower():
                return False
    return True


def _matches_query(doc: dict, q: str, query_by: str) -> bool:
    """Simple text match for mock Typesense searches."""
    if q == "*":
        return True
    terms = q.lower().split()
    fields = [f.strip() for f in query_by.split(",")]
    for term in terms:
        found = False
        for field in fields:
            if term in str(doc.get(field, "")).lower():
                found = True
                break
        if found:
            return True
    return False


class MockDocumentProxy:
    """Mock for client.collections[name].documents[id]."""

    def __init__(self, docs: list[dict], doc_id: str):
        self._docs = {d["id"]: d for d in docs}
        self._doc_id = doc_id

    def retrieve(self) -> dict:
        if self._doc_id in self._docs:
            return dict(self._docs[self._doc_id])
        raise Exception(f"Document {self._doc_id} not found")


class MockDocuments:
    """Mock for client.collections[name].documents."""

    def __init__(self, docs: list[dict]):
        self._docs = docs

    def __getitem__(self, doc_id: str) -> MockDocumentProxy:
        return MockDocumentProxy(self._docs, doc_id)

    def search(self, params: dict) -> dict:
        q = params.get("q", "*")
        query_by = params.get("query_by", "title")
        filter_by = params.get("filter_by", "")
        per_page = params.get("per_page", 20)
        page = params.get("page", 1)
        sort_by = params.get("sort_by", "")
        facet_by = params.get("facet_by", "")

        # Filter matching docs
        matched = []
        for doc in self._docs:
            if not _matches_filter(doc, filter_by):
                continue
            if not _matches_query(doc, q, query_by):
                continue
            matched.append(doc)

        # Sort
        if sort_by:
            field_order = sort_by.split(":")
            field = field_order[0]
            reverse = len(field_order) > 1 and field_order[1] == "desc"
            matched.sort(
                key=lambda d: d.get(field) if d.get(field) is not None else 0,
                reverse=reverse,
            )

        # Build facets if requested
        facet_counts = []
        if facet_by:
            for facet_field in facet_by.split(","):
                facet_field = facet_field.strip()
                counts: dict[str, int] = {}
                for doc in self._docs:  # Facets are over ALL docs, not just matched
                    val = doc.get(facet_field)
                    if val is not None and str(val).strip():
                        key = str(val)
                        counts[key] = counts.get(key, 0) + 1
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                facet_counts.append({
                    "field_name": facet_field,
                    "counts": [{"value": k, "count": v} for k, v in sorted_counts],
                })

        total = len(matched)

        # Pagination
        if per_page > 0:
            start = (page - 1) * per_page
            matched = matched[start:start + per_page]

        # Build hits with highlight stubs
        hits = []
        for doc in matched:
            # Exclude fields
            exclude = set(params.get("exclude_fields", "").split(","))
            filtered_doc = {k: v for k, v in doc.items() if k not in exclude}

            hit: dict = {
                "document": filtered_doc,
                "text_match_info": {"score": 100},
                "highlight": {},
            }
            # Add abstract snippet highlight
            if "abstract" in doc:
                abstract = doc["abstract"] or ""
                snippet = abstract[:200] if abstract else ""
                hit["highlight"]["abstract"] = {"snippet": snippet}
            hits.append(hit)

        result: dict = {
            "found": total,
            "hits": hits,
        }
        if facet_counts:
            result["facet_counts"] = facet_counts
        return result

    def import_(self, docs: list[dict], params: dict | None = None) -> list[dict]:
        """Mock batch import — always succeeds."""
        self._docs.extend(docs)
        return [{"success": True} for _ in docs]


class MockCollection:
    """Mock for client.collections[name]."""

    def __init__(self, docs: list[dict]):
        self.documents = MockDocuments(docs)
        self._info = {
            "name": "papers",
            "num_documents": len(docs),
            "fields": [
                {"name": "title", "type": "string"},
                {"name": "abstract", "type": "string", "optional": True},
                {"name": "year", "type": "int32", "optional": True, "facet": True},
                {"name": "conference", "type": "string", "optional": True, "facet": True},
                {"name": "first_author", "type": "string", "optional": True, "facet": True},
                {"name": "all_authors", "type": "string", "optional": True},
                {"name": "citation_count", "type": "int32", "facet": True},
                {"name": "document_type", "type": "string", "optional": True, "facet": True},
                {"name": "journal_title", "type": "string", "optional": True, "facet": True},
            ],
        }

    def retrieve(self) -> dict:
        return dict(self._info)

    def delete(self) -> dict:
        return {"name": "papers"}


class MockCollections:
    """Mock for client.collections — supports both indexing and keyed access."""

    def __init__(self, docs: list[dict]):
        self._docs = docs

    def __getitem__(self, name: str) -> MockCollection:
        return MockCollection(self._docs)

    def create(self, schema: dict) -> dict:
        return schema


class MockTypesenseClient:
    """Mock Typesense client for testing."""

    def __init__(self, docs: list[dict] | None = None):
        self._docs = docs or []
        self.collections = MockCollections(self._docs)


@pytest.fixture
def sample_json_dir(tmp_path):
    """Create a temporary directory with sample paper JSON files."""
    # Create subdirectories mimicking INSPIRE structure
    ipac_dir = tmp_path / "IPAC2020"
    ipac_dir.mkdir()
    arxiv_dir = tmp_path / "ARXIV-acc"
    arxiv_dir.mkdir()
    ipac21_dir = tmp_path / "IPAC2021"
    ipac21_dir.mkdir()
    napac_dir = tmp_path / "NAPAC2018"
    napac_dir.mkdir()
    ipac22_dir = tmp_path / "IPAC2022"
    ipac22_dir.mkdir()

    # Write papers to appropriate subdirs
    target_dirs = [ipac_dir, arxiv_dir, ipac21_dir, napac_dir, ipac22_dir]
    for paper, target in zip(SAMPLE_PAPERS, target_dirs, strict=True):
        fpath = target / f"{paper['texkey'].replace(':', '-')}.json"
        fpath.write_text(json.dumps(paper))

    return tmp_path


@pytest.fixture
def mock_client():
    """Create a MockTypesenseClient pre-loaded with indexed sample docs."""
    docs = _get_indexed_docs()
    return MockTypesenseClient(docs)


@pytest.fixture
def patch_client(mock_client, monkeypatch):
    """Patch get_client and get_collection_name for tool tests."""
    monkeypatch.setattr(
        "osprey.mcp_server.accelpapers.db.get_client",
        lambda: mock_client,
    )
    monkeypatch.setattr(
        "osprey.mcp_server.accelpapers.db.get_collection_name",
        lambda: "papers",
    )
    return mock_client

"""Shared fixtures for AccelPapers MCP server tests.

Provides mock Typesense client fixtures and sample paper data for testing
tools and indexer in isolation.
"""

import json
from pathlib import Path

import numpy as np
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


# Pre-loaded embeddings fixture
_EMBEDDINGS: dict | None = None


def _get_embeddings() -> dict:
    """Load embeddings fixture file (cached at module level)."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        fixture_path = Path(__file__).parent.parent / "fixtures" / "accelpapers" / "embeddings.json"
        with open(fixture_path) as f:
            _EMBEDDINGS = json.load(f)
    return _EMBEDDINGS


def _attach_embeddings(docs: list[dict], doc_embeddings: dict[str, list[float]]) -> None:
    """Attach embedding vectors to indexed docs by matching on doc['id']."""
    for doc in docs:
        emb = doc_embeddings.get(doc["id"])
        if emb:
            doc["embedding"] = emb


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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    va, vb = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _text_match_score(doc: dict, q: str, text_fields: list[str]) -> float:
    """Compute a 0.0-1.0 text match score based on fraction of query terms found.

    Returns the ratio of query terms that appear in at least one of the given fields.
    """
    terms = q.lower().split()
    if not terms:
        return 0.0
    matched = 0
    for term in terms:
        for field in text_fields:
            if term in str(doc.get(field, "")).lower():
                matched += 1
                break
    return matched / len(terms)


def _score_query(
    doc: dict,
    q: str,
    query_by: str,
    query_embeddings: dict[str, list[float]],
) -> tuple[float, float | None] | None:
    """Score a document against a query, returning (blended_score, vector_distance) or None.

    Splits query_by fields into text fields and embedding, computes a blended score:
      score = (1 - alpha) * text_score + alpha * vector_score
    with alpha=0.3. Falls back to keyword-only (alpha=0) when no embedding available.

    Returns None if the document doesn't match at all.
    """
    if q == "*":
        return (1.0, None)

    all_fields = [f.strip() for f in query_by.split(",")]
    text_fields = [f for f in all_fields if f != "embedding"]
    has_embedding = "embedding" in all_fields

    text_score = _text_match_score(doc, q, text_fields)

    vector_score: float | None = None
    vector_distance: float | None = None

    if has_embedding:
        doc_emb = doc.get("embedding")
        query_emb = query_embeddings.get(q)
        if doc_emb and query_emb:
            sim = _cosine_similarity(doc_emb, query_emb)
            # Only count vector match if similarity is above a meaningful threshold.
            # Random/gibberish queries produce ~0.3-0.4 cosine similarity against real docs;
            # genuine semantic matches score > 0.5.
            if sim >= 0.5:
                # Convert cosine similarity ([-1, 1]) to a 0-1 score
                vector_score = max(0.0, (sim + 1.0) / 2.0)
                # Typesense vector_distance is cosine distance: 1 - sim (range [0, 2])
                vector_distance = 1.0 - sim

    if vector_score is not None:
        alpha = 0.3
        blended = (1.0 - alpha) * text_score + alpha * vector_score
    else:
        blended = text_score

    if blended <= 0.0:
        return None

    return (blended, vector_distance)


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

    def __init__(self, docs: list[dict], query_embeddings: dict[str, list[float]] | None = None):
        self._docs = docs
        self._query_embeddings = query_embeddings or {}

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

        has_embedding = "embedding" in query_by

        # Filter and score matching docs
        scored: list[tuple[dict, float, float | None]] = []
        for doc in self._docs:
            if not _matches_filter(doc, filter_by):
                continue
            result = _score_query(doc, q, query_by, self._query_embeddings)
            if result is None:
                continue
            score, vector_distance = result
            scored.append((doc, score, vector_distance))

        # Sort: explicit sort_by takes precedence, otherwise rank by score
        if sort_by:
            field_order = sort_by.split(":")
            field = field_order[0]
            reverse = len(field_order) > 1 and field_order[1] == "desc"
            scored.sort(
                key=lambda t: t[0].get(field) if t[0].get(field) is not None else 0,
                reverse=reverse,
            )
        else:
            scored.sort(key=lambda t: t[1], reverse=True)

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
                facet_counts.append(
                    {
                        "field_name": facet_field,
                        "counts": [{"value": k, "count": v} for k, v in sorted_counts],
                    }
                )

        total = len(scored)

        # Pagination
        if per_page > 0:
            start = (page - 1) * per_page
            scored = scored[start : start + per_page]

        # Build hits with highlight stubs and scores
        hits = []
        for doc, score, vector_distance in scored:
            # Exclude fields
            exclude = set(params.get("exclude_fields", "").split(","))
            filtered_doc = {k: v for k, v in doc.items() if k not in exclude}

            # Scale score to a Typesense-like text_match_info score
            text_match_score = int(score * 100)

            hit: dict = {
                "document": filtered_doc,
                "text_match_info": {"score": text_match_score},
                "highlight": {},
            }
            # Add vector_distance when embedding was used
            if has_embedding and vector_distance is not None:
                hit["vector_distance"] = vector_distance
            # Add abstract snippet highlight
            if "abstract" in doc:
                abstract = doc["abstract"] or ""
                snippet = abstract[:200] if abstract else ""
                hit["highlight"]["abstract"] = {"snippet": snippet}
            hits.append(hit)

        result_dict: dict = {
            "found": total,
            "hits": hits,
        }
        if facet_counts:
            result_dict["facet_counts"] = facet_counts
        return result_dict

    def import_(self, docs: list[dict], params: dict | None = None) -> list[dict]:
        """Mock batch import — always succeeds."""
        self._docs.extend(docs)
        return [{"success": True} for _ in docs]


class MockCollection:
    """Mock for client.collections[name]."""

    def __init__(self, docs: list[dict], query_embeddings: dict[str, list[float]] | None = None):
        self.documents = MockDocuments(docs, query_embeddings)
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
                {"name": "embedding", "type": "float[]", "num_dim": 768},
            ],
        }

    def retrieve(self) -> dict:
        return dict(self._info)

    def delete(self) -> dict:
        return {"name": "papers"}


class MockCollections:
    """Mock for client.collections — supports both indexing and keyed access."""

    def __init__(self, docs: list[dict], query_embeddings: dict[str, list[float]] | None = None):
        self._docs = docs
        self._query_embeddings = query_embeddings or {}
        self._cache: dict[str, MockCollection] = {}

    def __getitem__(self, name: str) -> MockCollection:
        if name not in self._cache:
            self._cache[name] = MockCollection(self._docs, self._query_embeddings)
        return self._cache[name]

    def create(self, schema: dict) -> dict:
        return schema


class MockTypesenseClient:
    """Mock Typesense client for testing."""

    def __init__(
        self,
        docs: list[dict] | None = None,
        query_embeddings: dict[str, list[float]] | None = None,
    ):
        self._docs = docs or []
        self.collections = MockCollections(self._docs, query_embeddings)


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
    """Create a MockTypesenseClient pre-loaded with indexed sample docs and embeddings."""
    docs = _get_indexed_docs()
    embeddings = _get_embeddings()
    _attach_embeddings(docs, embeddings["documents"])
    return MockTypesenseClient(docs, query_embeddings=embeddings["queries"])


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

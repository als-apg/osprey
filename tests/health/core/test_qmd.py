"""Tests for the core ``qmd`` health category.

The sidecars are stood in for by patching :meth:`QMDClient.status`, keyed on
the port each client was built for, so the tests also pin which port each
corpus's row asks.
"""

from __future__ import annotations

import pytest

from osprey.health.core.qmd import CATEGORY, qmd
from osprey.health.models import CheckResult, Status
from osprey.services.qmd import QMDClient, QMDIndexStatus, QMDUnavailableError

BOTH_CORPORA = {
    "services": {"qmd": {"port": 9000}},
    "facility_knowledge": {"bundle_path": "data/facility_knowledge"},
    "ariel": {"enhancement_modules": {"qmd_export": {"enabled": True, "mirror_path": "data/md"}}},
}


async def _run(config, monkeypatch, answers) -> dict[str, CheckResult]:
    """Run the category with sidecars answering *answers*, keyed by port."""

    def status(self: QMDClient) -> QMDIndexStatus:
        answer = answers[self._config.port]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(QMDClient, "status", status)
    results = await qmd(config)()
    assert all(r.category == CATEGORY for r in results)
    return {r.name: r for r in results}


async def test_one_row_per_corpus_sidecar_on_its_own_port(monkeypatch):
    rows = await _run(
        BOTH_CORPORA,
        monkeypatch,
        {9000: QMDIndexStatus(74, 0), 9001: QMDIndexStatus(135_348, 0)},
    )
    assert set(rows) == {"qmd_okf", "qmd_ariel"}
    assert rows["qmd_okf"].status is Status.OK
    assert rows["qmd_okf"].value == "74"
    assert "all embedded" in rows["qmd_ariel"].message


async def test_documents_without_vectors_are_a_warning_naming_the_count(monkeypatch):
    rows = await _run(
        BOTH_CORPORA,
        monkeypatch,
        {9000: QMDIndexStatus(74, 0), 9001: QMDIndexStatus(9984, 8726)},
    )
    row = rows["qmd_ariel"]
    assert row.status is Status.WARNING
    assert "8726 of 9984 documents have no vectors" in row.message
    assert "keyword search only" in row.details


async def test_a_sidecar_that_does_not_answer_is_a_warning(monkeypatch):
    rows = await _run(
        BOTH_CORPORA,
        monkeypatch,
        {9000: QMDUnavailableError("refused"), 9001: QMDIndexStatus(1, 0)},
    )
    assert rows["qmd_okf"].status is Status.WARNING
    assert rows["qmd_okf"].value == "offline"
    assert "127.0.0.1:9000" in rows["qmd_okf"].message


async def test_declared_corpora_get_rows_too(monkeypatch):
    config = {
        "services": {
            "qmd": {
                "port": 9000,
                "corpora": [{"name": "papers", "index": "prebuilt", "index_dir": "/i"}],
            }
        }
    }
    rows = await _run(config, monkeypatch, {9002: QMDIndexStatus(62_797, 0)})
    assert set(rows) == {"qmd_papers"}
    assert rows["qmd_papers"].value == "62797"


async def test_a_malformed_block_is_one_warning_row(monkeypatch):
    rows = await _run({"services": {"qmd": {"corpora": "papers"}}}, monkeypatch, {})
    assert set(rows) == {"qmd_config"}
    assert rows["qmd_config"].status is Status.WARNING


@pytest.mark.parametrize("config", [None, {}, {"services": {"postgresql": {}}}])
async def test_no_sidecar_configured_is_no_rows(config):
    assert await qmd(config)() == []

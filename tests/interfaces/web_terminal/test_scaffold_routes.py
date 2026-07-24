"""Tests for scaffold gallery routes — exception→HTTP translation contracts.

Each route wraps a ScaffoldGalleryService call and maps the service's typed
exceptions (KeyError/FileNotFoundError/FileExistsError/ValueError) onto the
right status code. The service is mocked so these tests pin the route layer's
translation rather than the service internals (covered elsewhere).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.scaffold import router

_SVC = "osprey.interfaces.web_terminal.routes.scaffold.ScaffoldGalleryService"


@pytest.fixture
def app(tmp_path):
    application = FastAPI()
    application.include_router(router)
    application.state.project_cwd = str(tmp_path)
    return application


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def svc():
    """A mock service; patched in for every request via the class constructor."""
    service = MagicMock()
    with patch(_SVC, return_value=service):
        yield service


class TestListScaffold:
    def test_summary_counts_by_status(self, client, svc):
        svc.list_artifacts.return_value = [
            {"status": "framework"},
            {"status": "framework"},
            {"status": "user-owned"},
        ]
        resp = client.get("/api/scaffold")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == {"total": 3, "framework": 2, "user_owned": 1}


class TestUntracked:
    def test_list_untracked_reports_count(self, client, svc):
        svc.scan_untracked.return_value = [{"canonical_name": "rules/x"}]
        resp = client.get("/api/scaffold/untracked")
        assert resp.json() == {"untracked": [{"canonical_name": "rules/x"}], "count": 1}

    def test_register_success(self, client, svc):
        svc.register_untracked.return_value = {"status": "registered"}
        resp = client.post("/api/scaffold/untracked/register", json={"name": "rules/x"})
        assert resp.status_code == 200
        svc.register_untracked.assert_called_once_with("rules/x")

    def test_register_missing_file_404(self, client, svc):
        svc.register_untracked.side_effect = FileNotFoundError("gone")
        resp = client.post("/api/scaffold/untracked/register", json={"name": "rules/x"})
        assert resp.status_code == 404

    def test_register_already_exists_409(self, client, svc):
        svc.register_untracked.side_effect = FileExistsError("dup")
        resp = client.post("/api/scaffold/untracked/register", json={"name": "rules/x"})
        assert resp.status_code == 409

    def test_delete_untracked_success(self, client, svc):
        svc.delete_untracked.return_value = {"status": "deleted"}
        resp = client.delete("/api/scaffold/untracked/rules/x")
        assert resp.status_code == 200

    def test_delete_untracked_missing_404(self, client, svc):
        svc.delete_untracked.side_effect = FileNotFoundError("gone")
        resp = client.delete("/api/scaffold/untracked/rules/x")
        assert resp.status_code == 404

    def test_delete_untracked_framework_artifact_400(self, client, svc):
        svc.delete_untracked.side_effect = ValueError("is framework")
        resp = client.delete("/api/scaffold/untracked/rules/x")
        assert resp.status_code == 400


class TestCreateArtifact:
    def test_create_success(self, client, svc):
        svc.create_artifact.return_value = {"status": "created"}
        resp = client.post(
            "/api/scaffold/create",
            json={"category": "rules", "name": "my-rule", "content": "x"},
        )
        assert resp.status_code == 200
        svc.create_artifact.assert_called_once_with("rules", "my-rule", "x")

    def test_create_invalid_category_400(self, client, svc):
        svc.create_artifact.side_effect = ValueError("bad category")
        resp = client.post("/api/scaffold/create", json={"category": "bogus", "name": "x"})
        assert resp.status_code == 400

    def test_create_conflict_409(self, client, svc):
        svc.create_artifact.side_effect = FileExistsError("exists")
        resp = client.post("/api/scaffold/create", json={"category": "rules", "name": "x"})
        assert resp.status_code == 409


class TestFrameworkAndDiff:
    def test_get_framework_success(self, client, svc):
        svc.get_framework_content.return_value = "rendered"
        resp = client.get("/api/scaffold/rules/x/framework")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"name": "rules/x", "content": "rendered", "source": "framework"}

    def test_get_framework_unknown_404(self, client, svc):
        svc.get_framework_content.side_effect = KeyError("unknown")
        resp = client.get("/api/scaffold/rules/x/framework")
        assert resp.status_code == 404

    def test_diff_success(self, client, svc):
        svc.compute_diff.return_value = {"has_diff": True}
        resp = client.get("/api/scaffold/rules/x/diff")
        assert resp.status_code == 200

    def test_diff_unknown_artifact_404(self, client, svc):
        svc.compute_diff.side_effect = KeyError("unknown")
        resp = client.get("/api/scaffold/rules/x/diff")
        assert resp.status_code == 404

    def test_diff_not_user_owned_404(self, client, svc):
        svc.compute_diff.side_effect = FileNotFoundError("not owned")
        resp = client.get("/api/scaffold/rules/x/diff")
        assert resp.status_code == 404


class TestClaimAndOverride:
    def test_claim_success(self, client, svc):
        svc.scaffold_override.return_value = {"status": "claimed"}
        resp = client.post("/api/scaffold/rules/x/claim")
        assert resp.status_code == 200

    def test_claim_unknown_404(self, client, svc):
        svc.scaffold_override.side_effect = KeyError("unknown")
        resp = client.post("/api/scaffold/rules/x/claim")
        assert resp.status_code == 404

    def test_claim_already_owned_409(self, client, svc):
        svc.scaffold_override.side_effect = FileExistsError("owned")
        resp = client.post("/api/scaffold/rules/x/claim")
        assert resp.status_code == 409

    def test_save_override_success(self, client, svc):
        svc.save_override.return_value = {"status": "saved"}
        resp = client.put("/api/scaffold/rules/x/override", json={"content": "new"})
        assert resp.status_code == 200
        svc.save_override.assert_called_once_with("rules/x", "new")

    def test_save_override_not_owned_404(self, client, svc):
        svc.save_override.side_effect = FileNotFoundError("not owned")
        resp = client.put("/api/scaffold/rules/x/override", json={"content": "new"})
        assert resp.status_code == 404

    def test_delete_override_default_keeps_file(self, client, svc):
        svc.unoverride.return_value = {"status": "removed"}
        resp = client.delete("/api/scaffold/rules/x/override")
        assert resp.status_code == 200
        svc.unoverride.assert_called_once_with("rules/x", delete_file=False)

    def test_delete_override_with_delete_file_flag(self, client, svc):
        svc.unoverride.return_value = {"status": "removed"}
        resp = client.delete("/api/scaffold/rules/x/override?delete_file=true")
        assert resp.status_code == 200
        svc.unoverride.assert_called_once_with("rules/x", delete_file=True)

    def test_delete_override_unknown_404(self, client, svc):
        svc.unoverride.side_effect = KeyError("unknown")
        resp = client.delete("/api/scaffold/rules/x/override")
        assert resp.status_code == 404


class TestGetScaffold:
    def test_get_content_stamps_name(self, client, svc):
        svc.get_content.return_value = {"content": "body", "source": "framework"}
        resp = client.get("/api/scaffold/rules/x")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "rules/x"
        assert data["content"] == "body"

    def test_get_content_unknown_404(self, client, svc):
        svc.get_content.side_effect = KeyError("unknown")
        resp = client.get("/api/scaffold/rules/x")
        assert resp.status_code == 404

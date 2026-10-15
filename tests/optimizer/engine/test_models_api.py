"""Tests for the models router in api/routers/models.py.

Covers: GET /models, POST /models.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey_optimizer.models.api import ModelFileInfo


class TestListModelsEndpoint:
    @pytest.mark.parametrize("empty", [False, True], ids=["success", "empty"])
    def test_lists_scanned_models(self, client: TestClient, empty: bool) -> None:
        real_models = (
            []
            if empty
            else [
                ModelFileInfo(label="env/2026-01-01", path="/abs/path/model.joblib"),
                ModelFileInfo(label="env/2026-01-02", path="/abs/path2/model.joblib"),
            ]
        )
        with patch(
            "osprey_optimizer.api.routers.models.scan_model_files",
            return_value=real_models,
        ):
            resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert [m["label"] for m in data] == [m.label for m in real_models]
        assert [m["path"] for m in data] == [m.path for m in real_models]

    def test_exception(self, client: TestClient) -> None:
        with patch(
            "osprey_optimizer.api.routers.models.scan_model_files",
            side_effect=OSError("disk error"),
        ):
            resp = client.get("/models")
        assert resp.status_code == 503


class TestUploadModelEndpoint:
    @pytest.mark.parametrize(
        ("filename", "content", "detail_token"),
        [
            pytest.param(
                "model.txt", b"x", "Only .joblib files are accepted", id="non_joblib_extension"
            ),
            pytest.param(
                "model.joblib", b"raw junk", "not a readable joblib file", id="non_bundle_content"
            ),
        ],
    )
    def test_rejected_with_422(
        self, client: TestClient, filename: str, content: bytes, detail_token: str
    ) -> None:
        """Both upload gates are client errors: a non-.joblib name is refused by
        extension, and .joblib bytes that are not a model bundle are refused on read."""
        resp = client.post(
            "/models", files={"file": (filename, content, "application/octet-stream")}
        )
        assert resp.status_code == 422
        assert detail_token in resp.json()["detail"]

    def test_oversize_upload_rejected(self, client: TestClient) -> None:
        """A file larger than the configured cap is refused with 413, before any save."""
        with (
            patch("osprey_optimizer.api.routers.models.MAX_MODEL_UPLOAD_BYTES", 10),
            patch("osprey_optimizer.api.routers.models.save_uploaded_model") as mock_save,
        ):
            resp = client.post("/models", files={"file": ("big.joblib", b"x" * 64)})
        assert resp.status_code == 413
        mock_save.assert_not_called()

    def test_save_failure_returns_500(self, client: TestClient) -> None:
        with patch(
            "osprey_optimizer.api.routers.models.save_uploaded_model",
            side_effect=RuntimeError("disk full"),
        ):
            resp = client.post(
                "/models",
                files={"file": ("model.joblib", b"payload", "application/octet-stream")},
            )
        assert resp.status_code == 500
        assert "Failed to save uploaded model" in resp.json()["detail"]

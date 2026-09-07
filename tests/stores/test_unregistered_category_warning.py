"""The unregistered-category warning names a config key, not a source file.

``type_registry.py`` is packaged code: an operator who follows a warning that
says "add to type_registry.py" is being told to patch an installed module,
which the next upgrade overwrites. The seam that really registers a category
for one deployment is ``artifact_server.categories``, read at startup by
:func:`osprey.stores.type_registry.load_categories_from_config`, so that is
what the three save paths must name.
"""

from __future__ import annotations

import logging

import pytest

from osprey.stores.artifact_store import ArtifactStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path, monkeypatch) -> ArtifactStore:
    repo_root = tmp_path / "repo"
    (repo_root / "build").mkdir(parents=True)
    config_path = repo_root / "build" / "config.yml"
    config_path.write_text(
        f"project_root: {repo_root}\nagent_data:\n  base_dir: state/agent\n",
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.chdir(repo_root)
    return ArtifactStore(workspace_root=repo_root / "state" / "agent")


def _warnings(caplog) -> str:
    return "\n".join(
        record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
    )


def test_save_file_names_the_config_key(store, caplog):
    with caplog.at_level(logging.WARNING):
        store.save_file(
            file_content=b"bytes",
            filename="thing.txt",
            artifact_type="document",
            title="Thing",
            category="not_a_category",
        )
    message = _warnings(caplog)
    assert "artifact_server.categories" in message
    assert "type_registry.py" not in message


def test_save_data_names_the_config_key(store, caplog):
    with caplog.at_level(logging.WARNING):
        store.save_data(
            data={"a": 1},
            tool="test",
            title="Thing",
            category="not_a_category",
        )
    message = _warnings(caplog)
    assert "artifact_server.categories" in message
    assert "type_registry.py" not in message


def test_save_channel_reading_names_the_config_key(store, caplog):
    with caplog.at_level(logging.WARNING):
        store.save_channel_reading(
            png_bytes=b"\x89PNG\r\n\x1a\n",
            npy_bytes=b"\x93NUMPY",
            title="Camera frame",
            summary={"shape": [2, 2]},
            access_details={"loader": "numpy.load"},
            metadata={"channel": "SIM:CAM1:ArrayData"},
            category="not_a_category",
        )
    message = _warnings(caplog)
    assert "artifact_server.categories" in message
    assert "type_registry.py" not in message

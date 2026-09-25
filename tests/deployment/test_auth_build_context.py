"""The auth sidecar's build context carries only this deploy's dev artifacts.

The context is materialized in place under ``build/`` and survives between
deploys, so anything a previous ``--dev`` run staged beside the Dockerfile is
still there the next time. Its Dockerfile installs every ``*.whl`` it finds, and
two wheels of one distribution at different versions are a resolution error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.deployment import compose_generator
from osprey.deployment.web_terminals import provision
from osprey.deployment.wheel_build import LOCAL_REQUIREMENTS_FILENAME

_STALE = (
    "osprey_framework-2026.9.0b3.post315+gb1c732b4e-py3-none-any.whl",
    "osprey_connectors-2026.9.0b3.post315+gb1c732b4e-py3-none-any.whl",
)
_FRESH = (
    "osprey_framework-2026.9.0b3.post317+g93cb73d92-py3-none-any.whl",
    "osprey_connectors-2026.9.0b3.post317+g93cb73d92-py3-none-any.whl",
)


@pytest.fixture
def stale_context(tmp_path: Path) -> Path:
    context = tmp_path / provision.AUTH_BUILD_CONTEXT
    context.mkdir(parents=True)
    for name in _STALE:
        (context / name).write_bytes(b"stale")
    (context / LOCAL_REQUIREMENTS_FILENAME).write_text("stale-requirement\n")
    return context


def _stage_fresh(out_dir: str) -> bool:
    for name in _FRESH:
        (Path(out_dir) / name).write_bytes(b"fresh")
    (Path(out_dir) / LOCAL_REQUIREMENTS_FILENAME).write_text("fresh-requirement\n")
    return True


@pytest.mark.usefixtures("stale_context")
def test_a_dev_deploy_replaces_the_wheels_an_earlier_one_staged(monkeypatch, tmp_path):
    monkeypatch.setattr(compose_generator, "_copy_local_framework_for_override", _stage_fresh)

    context, effective_dev = provision._materialize_auth_build_context(tmp_path, dev_mode=True)

    assert effective_dev is True
    assert sorted(p.name for p in context.glob("*.whl")) == sorted(_FRESH)
    assert (context / LOCAL_REQUIREMENTS_FILENAME).read_text() == "fresh-requirement\n"


@pytest.mark.usefixtures("stale_context")
def test_a_non_dev_deploy_leaves_no_wheel_behind(tmp_path):
    context, effective_dev = provision._materialize_auth_build_context(tmp_path, dev_mode=False)

    assert effective_dev is False
    assert list(context.glob("*.whl")) == []
    assert not (context / LOCAL_REQUIREMENTS_FILENAME).exists()


@pytest.mark.usefixtures("stale_context")
def test_the_bundled_files_are_still_written(tmp_path):
    context, _ = provision._materialize_auth_build_context(tmp_path, dev_mode=False)

    for name in provision._AUTH_CONTEXT_FILES:
        assert (context / name).is_file()

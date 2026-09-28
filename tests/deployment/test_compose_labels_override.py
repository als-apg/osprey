"""OSPREY's attribution labels, generated once for every rendered service.

The render writes ``build/osprey-labels.override.yml``: one entry per service
it rendered, each carrying the project name, the checkout identity, the
project path and the config digest. Every compose invocation passes it last,
so a service template does not have to spell those labels for ``osprey
status``, ``down``, ``reset`` and ``set`` to treat its containers as the
deployment's.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.cli.build_cmd import _copy_service_templates
from osprey.deployment import compose_generator, reset, status_display
from osprey.deployment.compose_generator import (
    CONFIG_DIGEST_LABEL,
    LABELS_OVERRIDE_FILENAME,
    PROJECT_LABEL,
    PROJECT_ROOT_LABEL,
    REPO_ID_LABEL,
    _inject_project_metadata,
    generated_label_entries,
    prepare_compose_files,
    project_label_values,
    repo_identity,
)
from osprey.deployment.runtime_helper import CONFIG_DIGEST_VAR


def test_status_reads_the_project_label_the_render_writes() -> None:
    """Every reader of a label key reads the key the render writes."""
    assert status_display.PROJECT_LABEL is compose_generator.PROJECT_LABEL
    assert compose_generator.PROJECT_ROOT_LABEL in reset.PATH_EVIDENCE_LABELS


def test_the_template_context_labels_are_project_label_values(tmp_path) -> None:
    """The template context and the override read one source of label values."""
    cfg = {"project_name": "site-fixture", "project_root": str(tmp_path)}
    assert _inject_project_metadata(cfg)["osprey_labels"] == project_label_values(cfg)


# ---------------------------------------------------------------------------
# The render writes the override
# ---------------------------------------------------------------------------

PROJECT = "site-fixture"

#: A facility template that spells none of OSPREY's labels, and its own in
#: list form.
SITE_PROBE_TEMPLATE = """\
services:
  site-probe:
    image: busybox
    labels:
      - site.owner=ops
"""


def _write_config(project: Path, deployed_services: list[str]) -> Path:
    """A config declaring the shipped ``openobserve`` and the facility ``site-probe``."""
    probe = project / "services" / "site-probe"
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "docker-compose.yml.j2").write_text(SITE_PROBE_TEMPLATE, encoding="utf-8")
    config_path = project / "config.yml"
    config = {
        "project_name": PROJECT,
        "project_root": str(project),
        "build_dir": str(project / "build"),
        "system": {"timezone": "UTC"},
        "services": {
            "openobserve": {"path": "./services/openobserve", "port": 5080},
            "site-probe": {"path": "./services/site-probe"},
        },
        "deployed_services": deployed_services,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _render(
    project: Path, monkeypatch: pytest.MonkeyPatch, deployed_services: list[str] | None = None
) -> list[str]:
    """Render the project from inside it and return the rendered compose files."""
    services = ["openobserve", "site-probe"] if deployed_services is None else deployed_services
    config_path = _write_config(project, services)
    _copy_service_templates(project)
    monkeypatch.chdir(project)
    _, compose_files = prepare_compose_files(str(config_path))
    return [str(project / path) for path in compose_files]


def _override(project: Path) -> dict:
    return yaml.safe_load((project / "build" / LABELS_OVERRIDE_FILENAME).read_text())


def _expected_labels(project: Path) -> dict[str, str]:
    return {
        PROJECT_LABEL: PROJECT,
        REPO_ID_LABEL: repo_identity(project),
        PROJECT_ROOT_LABEL: str(project),
        CONFIG_DIGEST_LABEL: "${OSPREY_CONFIG_DIGEST:-}",
    }


def test_the_digest_label_names_the_variable_the_runtime_sets(tmp_path: Path) -> None:
    """The override's digest value interpolates the variable ``runtime_env`` sets."""
    cfg = {"project_name": PROJECT, "project_root": str(tmp_path)}
    assert generated_label_entries(cfg)[CONFIG_DIGEST_LABEL] == f"${{{CONFIG_DIGEST_VAR}:-}}"


def test_every_rendered_service_gets_the_four_generated_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render(tmp_path, monkeypatch)

    services = _override(tmp_path)["services"]

    assert list(services) == ["openobserve", "site-probe"]
    for entry in services.values():
        assert entry == {"labels": _expected_labels(tmp_path)}


def test_a_template_that_spells_no_labels_is_named_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render(tmp_path, monkeypatch)

    services = _override(tmp_path)["services"]

    assert services["site-probe"] == services["openobserve"]


def test_the_override_leaves_the_env_digest_to_the_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render(tmp_path, monkeypatch)

    for entry in _override(tmp_path)["services"].values():
        assert "osprey.env.digest" not in entry["labels"]


def test_two_renders_write_identical_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _render(tmp_path, monkeypatch)
    first = (tmp_path / "build" / LABELS_OVERRIDE_FILENAME).read_bytes()

    _render(tmp_path, monkeypatch)

    assert (tmp_path / "build" / LABELS_OVERRIDE_FILENAME).read_bytes() == first


def test_a_render_with_no_services_removes_the_previous_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render(tmp_path, monkeypatch)
    assert (tmp_path / "build" / LABELS_OVERRIDE_FILENAME).is_file()

    _render(tmp_path, monkeypatch, deployed_services=[])

    assert not (tmp_path / "build" / LABELS_OVERRIDE_FILENAME).exists()


def test_a_container_labelled_only_by_the_override_is_this_checkouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``osprey status`` files a facility container under the deployment."""
    _render(tmp_path, monkeypatch)
    labels = _override(tmp_path)["services"]["site-probe"]["labels"]
    record = {"Names": [f"{PROJECT}-site-probe"], "State": "running", "Labels": dict(labels)}

    partition = status_display._partition_by_checkout(
        [record], repo_identity(tmp_path), PROJECT, ["site-probe"]
    )

    assert partition[0] == [record]

"""The framework hash of a claimed artifact, pinned to the formula stored hashes follow.

Every test builds a synthetic templates tree, so the digest is checked against
the formula a recorded claim already carries — file digest, render digest, or
tree digest — and not against whatever the packaged templates say today. A
moved formula would report false drift on every existing claim.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jinja2
import pytest

from osprey.build.manifest import MANIFEST_FILENAME, sha256_file
from osprey.services.build_artifacts.catalog import BuildArtifact, BuildArtifactCatalog
from osprey.services.build_artifacts.ownership import (
    framework_template_hash,
    sha256_directory,
    update_manifest_add_user_owned,
)


def _env(templates: Path, **kwargs) -> jinja2.Environment:
    return jinja2.Environment(loader=jinja2.FileSystemLoader(str(templates)), **kwargs)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _artifact(template_path: str, **kwargs) -> BuildArtifact:
    return BuildArtifact(
        canonical_name="x",
        template_path=template_path,
        output_path="x",
        description="x",
        **kwargs,
    )


def test_file_artifact_hash_is_the_packaged_file_digest(tmp_path: Path) -> None:
    path = _write(tmp_path / "claude_code" / "rules" / "plain.md", "plain\n")

    result = framework_template_hash(tmp_path, _artifact("rules/plain.md"), _env(tmp_path), {})

    assert result == "sha256:" + sha256_file(path)


def test_rendered_artifact_hash_is_the_digest_of_the_render(tmp_path: Path) -> None:
    """The digest is of the rendered text as written in UTF-8 text mode.

    On POSIX, text-mode writing leaves the newline as is, so the digest equals
    the SHA-256 of the rendered string's UTF-8 bytes.
    """
    _write(tmp_path / "claude_code" / "hello.md.j2", "Hello {{ who }}\n")

    result = framework_template_hash(
        tmp_path,
        _artifact("hello.md.j2"),
        _env(tmp_path, keep_trailing_newline=True),
        {"who": "ops"},
    )

    assert result == "sha256:" + hashlib.sha256(b"Hello ops\n").hexdigest()


def test_render_honours_the_artifact_template_root(tmp_path: Path) -> None:
    _write(tmp_path / "other_root" / "hello.md.j2", "Hello {{ who }}\n")

    result = framework_template_hash(
        tmp_path,
        _artifact("hello.md.j2", template_root="other_root"),
        _env(tmp_path, keep_trailing_newline=True),
        {"who": "ops"},
    )

    assert result == "sha256:" + hashlib.sha256(b"Hello ops\n").hexdigest()


def test_directory_artifact_hash_is_the_tree_digest(tmp_path: Path) -> None:
    directory = tmp_path / "services" / "x"
    _write(directory / "docker-compose.yml.j2", "{{ undefined_var.attr }}\n")
    _write(directory / "nested" / "extra.conf", "extra\n")
    env = _env(tmp_path, undefined=jinja2.StrictUndefined)

    result = framework_template_hash(
        tmp_path, _artifact("x", template_root="services", is_directory=True), env, {}
    )

    assert result == "sha256:" + sha256_directory(directory)


def test_missing_template_is_none(tmp_path: Path) -> None:
    env = _env(tmp_path)

    assert framework_template_hash(tmp_path, _artifact("absent.md"), env, {}) is None
    assert (
        framework_template_hash(
            tmp_path, _artifact("absent", template_root="services", is_directory=True), env, {}
        )
        is None
    )


def test_unrenderable_template_is_none_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    templates = tmp_path / "templates"
    _write(templates / "claude_code" / "broken.md.j2", '{% include "absent" %}\n')
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    result = framework_template_hash(templates, _artifact("broken.md.j2"), _env(templates), {})

    assert result is None
    assert [p for p in Path(tempfile.gettempdir()).iterdir() if p.name.endswith("broken.md")] == []


@pytest.mark.parametrize(
    ("name", "stub"),
    [
        ("rules/error-handling", "plain rule\n"),
        ("rules/safety", "Safety for {{ facility | default('any') }}\n"),
        ("services/postgresql", None),
    ],
)
def test_claim_records_the_unified_hash(tmp_path: Path, name: str, stub: str | None) -> None:
    artifact = BuildArtifactCatalog.default().get(name)
    assert artifact is not None
    templates = tmp_path / "templates"
    source = templates / artifact.template_root / artifact.template_path
    if stub is None:
        _write(source / "docker-compose.yml.j2", "services: {}\n")
        _write(source / "init" / "10-role.sh.j2", "echo role\n")
    else:
        _write(source, stub)
    env = _env(templates)
    project = tmp_path / "project"
    project.mkdir()
    (project / MANIFEST_FILENAME).write_text(json.dumps({}), encoding="utf-8")

    update_manifest_add_user_owned(
        project, SimpleNamespace(template_root=templates, jinja_env=env), {}, name
    )

    manifest = json.loads((project / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    expected = framework_template_hash(templates, artifact, env, {})
    assert expected is not None
    assert manifest["user_owned"][name]["framework_hash"] == expected


def test_every_context_free_catalog_artifact_hashes_the_packaged_template() -> None:
    from osprey.cli.templates.manager import TemplateManager

    manager = TemplateManager()
    templates = manager.template_root
    checked = 0
    for artifact in BuildArtifactCatalog.default().all_artifacts():
        source = templates / artifact.template_root / artifact.template_path
        if artifact.is_directory:
            expected = "sha256:" + sha256_directory(source)
        elif source.suffix != ".j2":
            expected = "sha256:" + sha256_file(source)
        else:
            continue
        result = framework_template_hash(templates, artifact, manager.jinja_env, {})
        assert result is not None, artifact.canonical_name
        assert result == expected, artifact.canonical_name
        checked += 1
    assert checked > 0

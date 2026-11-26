"""A render's ``data/`` is assembled from the deployment's declared sources.

Every source the layout declares a render path for
(:func:`osprey.repo_layout.rendered_sources`) is copied from the deployment
root to that path under the render; nothing else at the root reaches the
render, a declared source absent on disk renders nothing, and a mounted machine
root's facts still merge into the facility row's render path beside the
deployment's own content.
"""

from __future__ import annotations

from pathlib import Path

from osprey import repo_layout
from osprey.cli.templates.scaffolding import copy_template_data
from osprey_facility.roots import FacilityRoots


def _plant(root: Path, rel: str, text: str = "x\n") -> None:
    path = root.joinpath(*rel.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render(tmp_path: Path, repo: Path, facility_roots: FacilityRoots | None = None) -> Path:
    project = tmp_path / "build"
    project.mkdir()
    copy_template_data(
        tmp_path / "templates",
        project,
        "pkg",
        "demo",
        {},
        repo_root=repo,
        facility_roots=facility_roots,
    )
    return project


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_every_declared_source_renders_at_its_render_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    sources = repo_layout.rendered_sources()
    assert sources, "the layout declares no rendered source"
    for source in sources:
        _plant(repo, f"{source.source_rel}/marker.txt", f"{source.key}\n")

    project = _render(tmp_path, repo)

    for source in sources:
        landed = project.joinpath(*source.render_rel.split("/"), "marker.txt")
        assert landed.is_file(), source
        assert landed.read_text(encoding="utf-8") == f"{source.key}\n"
    assert all(source.render_rel.startswith("data/") for source in sources)


def test_top_level_content_outside_the_rows_is_not_rendered(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    facility = repo_layout.entry("facility")
    _plant(repo, f"{facility.path}/identity.yaml")
    for stray in (
        "other/keep.txt",
        "data/raw/staging.csv",
        "data/facility/old.yaml",
        "web/not-declared/page.html",
        "web/top.html",
        "notes.md",
    ):
        _plant(repo, stray)

    project = _render(tmp_path, repo)

    assert _files(project) == [f"{facility.render}/identity.yaml"]


def test_a_source_absent_on_disk_renders_nothing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    project = _render(tmp_path, repo)

    assert _files(project) == []
    assert not (project / "data").exists()


def test_version_control_metadata_is_never_rendered(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    facility = repo_layout.entry("facility")
    _plant(repo, f"{facility.path}/identity.yaml")
    _plant(repo, f"{facility.path}/.git/HEAD")
    _plant(repo, f"{facility.path}/knowledge/.git")
    _plant(repo, f"{facility.path}/.gitignore")

    project = _render(tmp_path, repo)

    assert _files(project) == [
        f"{facility.render}/.gitignore",
        f"{facility.render}/identity.yaml",
    ]


def test_a_mounted_machine_root_merges_into_the_facility_render(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    facility = repo_layout.entry("facility")
    _plant(repo, f"{facility.path}/knowledge/index.md", "deployment\n")
    machine = tmp_path / "machine"
    _plant(machine, "identity.yaml", "machine\n")
    _plant(machine, "devices/bpm.yaml", "machine\n")
    _plant(machine, "knowledge/index.md", "machine copy\n")
    _plant(machine, ".git/HEAD")

    project = _render(tmp_path, repo, FacilityRoots(machine=machine, content=repo / facility.path))

    rendered = project.joinpath(*facility.render.split("/"))
    assert _files(rendered) == ["devices/bpm.yaml", "identity.yaml", "knowledge/index.md"]
    assert (rendered / "knowledge" / "index.md").read_text(encoding="utf-8") == "deployment\n"
    assert (rendered / "identity.yaml").read_text(encoding="utf-8") == "machine\n"

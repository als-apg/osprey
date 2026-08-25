"""``resolve_render_relative_path`` — the one render-anchored config path.

``services.graphdb.ttl_path`` names an artifact of the render (the
``data/demo_machine.ttl`` the build assembled), so unlike every other config-relative key it
resolves against the ``config.yml`` directory itself, never the project root.
"""

from __future__ import annotations

from pathlib import Path

from osprey.utils.config_paths import resolve_config_relative_path, resolve_render_relative_path


def test_relative_value_resolves_against_the_render_not_the_repo(tmp_path):
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text("project_name: demo\n", encoding="utf-8")

    resolved = resolve_render_relative_path("./data/demo_machine.ttl", render)

    assert resolved == (render / "data" / "demo_machine.ttl").resolve()
    # The project-root rule, for the same inputs, would leave the render zone.
    assert (
        resolve_config_relative_path("./data/demo_machine.ttl", render)
        == (tmp_path / "data" / "demo_machine.ttl").resolve()
    )


def test_absolute_value_passes_through_unchanged(tmp_path):
    elsewhere = tmp_path / "corpus.ttl"
    assert resolve_render_relative_path(str(elsewhere), tmp_path / "build") == elsewhere


def test_tilde_value_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_render_relative_path("~/corpus.ttl", tmp_path) == Path(tmp_path, "corpus.ttl")


def test_without_config_dir_uses_the_resolved_config_file(tmp_path, monkeypatch):
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text("project_name: demo\n", encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(render / "config.yml"))

    assert (
        resolve_render_relative_path("data/demo_machine.ttl")
        == (render / "data" / "demo_machine.ttl").resolve()
    )


def test_without_any_config_at_all_resolves_against_the_cwd(tmp_path, monkeypatch):
    """No *config_dir* and no resolvable config file: the CWD fallback, the
    same last resort ``resolve_config_relative_path`` documents."""
    import osprey.utils.config_paths as config_paths

    monkeypatch.setattr(config_paths, "resolve_config_dir", lambda: None)
    monkeypatch.chdir(tmp_path)

    resolved = config_paths.resolve_render_relative_path("data/demo_machine.ttl")

    assert resolved == (tmp_path / "data" / "demo_machine.ttl").resolve()

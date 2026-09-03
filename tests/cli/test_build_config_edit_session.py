"""The build edits each rendered ``config.yml`` in one session, and reads see the edits.

``_render_project`` applies a few dozen comment-preserving edits to the config
it just rendered — the profile's overrides, the ownership registrations, the
MCP servers. Under :func:`~osprey.utils.config_writer.config_edit_session` they
share one ruamel load and one dump. The price is an ordering rule: whatever
reads the file's *bytes* while the session is open has to flush it first. These
tests pin that rule at each seam the render has — the shared parse
(``_rendered_config``), the limits reader, and the service injectors, which
rewrite the file with their own YAML instance — and pin the render's output
against the one-edit-at-a-time path, byte for byte.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import ruamel.yaml
from click.testing import CliRunner

from osprey.cli import build_cmd, build_limits_check
from osprey.cli.build_cmd import build
from osprey.utils.config_writer import config_edit_session, config_update_fields

SAMPLE = """\
control_system:
  type: mock
  limits_checking:
    enabled: false
container_runtime: docker
"""


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    build_cmd._rendered_config_cache.clear()


def _render(tmp_path: Path) -> Path:
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    (render_dir / "config.yml").write_text(SAMPLE, encoding="utf-8")
    return render_dir


def test_the_shared_parse_sees_pending_edits(tmp_path: Path) -> None:
    render_dir = _render(tmp_path)
    config_path = render_dir / "config.yml"

    with config_edit_session(config_path):
        config_update_fields(config_path, {"control_system.type": "epics"})
        assert build_cmd._rendered_config(render_dir)["control_system"]["type"] == "epics"
        # The read flushed: the file now carries the edit too.
        assert "type: epics" in config_path.read_text(encoding="utf-8")


def test_the_limits_reader_sees_pending_edits(tmp_path: Path) -> None:
    render_dir = _render(tmp_path)
    config_path = render_dir / "config.yml"

    with config_edit_session(config_path):
        config_update_fields(config_path, {"control_system.limits_checking.enabled": True})
        parsed = build_limits_check._rendered_config(render_dir)

    assert parsed["control_system"]["limits_checking"]["enabled"] is True


def test_the_injectors_run_on_a_flushed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Injectors rewrite config.yml with their own loader; they must see every edit."""
    render_dir = _render(tmp_path)
    config_path = render_dir / "config.yml"
    seen: list[str] = []

    def first_injector_step(project_path: Path) -> int:
        seen.append((project_path / "config.yml").read_text(encoding="utf-8"))
        return 0

    monkeypatch.setattr(build_cmd, "_copy_service_templates", first_injector_step)
    profile = SimpleNamespace(
        deploy_services=True,
        services=None,
        dispatch=None,
        nextcloud_bridge=None,
        gchat_bridge=None,
        bluesky=None,
        bluesky_web=None,
        virtual_accelerator=None,
        va_archiver=None,
    )

    with config_edit_session(config_path):
        config_update_fields(config_path, {"control_system.type": "epics"})
        build_cmd._inject_services(profile, tmp_path, render_dir)

    assert len(seen) == 1
    assert "type: epics" in seen[0]


class TestTheRenderedConfigIsUnchanged:
    """Byte-identical output: the session changes when the file is written, not what."""

    def test_exemplar_render_matches_the_one_edit_at_a_time_render(
        self, lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = CliRunner()

        def build_here() -> bytes:
            previous = Path.cwd()
            os.chdir(lifecycle_repo)
            try:
                result = runner.invoke(build, ["--skip-deps", "--skip-lifecycle"])
            finally:
                os.chdir(previous)
            assert result.exit_code == 0, result.output
            return (lifecycle_repo / "build" / "config.yml").read_bytes()

        batched = build_here()

        # The same repo, rendered again with every edit written as it is made.
        monkeypatch.setattr(build_cmd, "config_edit_session", contextlib.nullcontext)
        one_at_a_time = build_here()

        assert batched == one_at_a_time
        assert b"deployed_services" in batched  # a real render, not two empty files


def test_a_build_parses_each_renders_config_once(
    lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contract this module exists for, measured on a real build.

    Every writer that edits a rendered ``config.yml`` — the dotted-key writers,
    the service injectors, the ``auto`` resolvers — shares the render's one
    document, so the comment-preserving parse happens once per render. A
    second parse of the same file would mean a writer went around the session
    and the file had to be reloaded behind it. Dumps are not counted: a reader
    of the file's bytes flushes pending edits first, and how many times that
    happens is the readers' business, not the loader's.
    """
    loads: list[str] = []
    real_load = ruamel.yaml.YAML.load

    def counting_load(self: ruamel.yaml.YAML, stream: object) -> object:
        name = getattr(stream, "name", None)
        if isinstance(name, str) and name.endswith("config.yml"):
            loads.append(name)
        return real_load(self, stream)

    monkeypatch.setattr(ruamel.yaml.YAML, "load", counting_load)

    previous = Path.cwd()
    os.chdir(lifecycle_repo)
    try:
        result = CliRunner().invoke(build, ["--skip-deps", "--skip-lifecycle"])
    finally:
        os.chdir(previous)
    assert result.exit_code == 0, result.output

    # The renders: every config.yml the build wrote that is a project's own —
    # not a bundled service template's, which no writer here edits. The build
    # renders into a staging directory and swaps it into place, so the parsed
    # paths are compared by count, not by name: no file parsed twice, and as
    # many files parsed as there are renders.
    renders = [
        p for p in (lifecycle_repo / "build").rglob("config.yml") if "services" not in p.parts
    ]
    assert renders, "the exemplar build rendered nothing"
    render_loads = [name for name in loads if "/services/" not in name]
    twice = sorted({name for name in render_loads if render_loads.count(name) > 1})
    assert not twice, "a render's config.yml was parsed more than once:\n" + "\n".join(twice)
    assert len(render_loads) == len(renders), (
        f"{len(render_loads)} config.yml parse(s) for {len(renders)} render(s)"
    )

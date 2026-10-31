"""Two-anchor resolver-equality tests, against a real rendered project.

The audit trail and the hook config each have exactly one intended read path,
but each is *resolved* independently in three or two different places, none of
which imports another (a hook cannot import ``osprey`` at all — see
``osprey_hook_log.py``'s module docstring — and the writer/registry/reset
modules each re-derive rather than share a resolver for their own reasons).
Nothing here mocks a resolver: every path is computed by running the real
functions against a project this test actually renders with
:class:`~osprey.cli.templates.manager.TemplateManager`, the same entry point
``osprey build`` uses.

Two anchors, two groups of tests:

1. **The audit trail anchors on the REPO root.** The audit writer records into
   the agent record through :func:`osprey.agent_record.writer.spool_dir`, whose
   default is ``<repo root>/var/spool/<acting identity>``;
   :attr:`osprey.deployment.reset.ResetPlan.record_dirs` names the same spool
   root; and the hook's own ``audit_spool_dir()`` (``get_repo_root() /
   SPOOL_DIR_RELPATH / <identity>``) must name the writer's directory exactly.
   ``ResetPlan.audit_dir`` is the pre-upgrade audit tree the purge still
   deletes, which must be
   :data:`~osprey.utils.workspace.PRE_UPGRADE_AUDIT_RELPATH` under the
   same repo root; it is included in the equality on purpose.

2. **``hook_config.json`` anchors on the RENDER zone**, one level below the
   repo root: the path the MCP audit middleware reads it from
   (:func:`osprey.mcp_server.audit_middleware.hook_config_path`) must be the
   exact path ``osprey build`` wrote it to. That anchor is deliberately
   different from the audit trail's — see the middleware's own module docstring
   ("Never a project-root resolver... a server started with no OSPREY_CONFIG
   gets the degraded floor") — and the second class below asserts the two
   anchors are not just conceptually different but numerically different
   directories in a real two-zone render.
"""

from __future__ import annotations

from pathlib import Path

from osprey.agent_record import writer as record_writer
from osprey.agent_record.contract import SPOOL_DIR_ENV
from osprey.cli.templates.manager import TemplateManager
from osprey.deployment.reset import ResetPlan
from osprey.mcp_server import audit_middleware as am
from osprey.utils import workspace
from tests._preset_data import create_project_as_built
from tests.hooks.conftest import import_hook


def _render_two_zone_project(tmp_path: Path) -> tuple[Path, Path]:
    """Render a project the way ``osprey build`` lays one out on disk.

    ``osprey build`` renders into ``<repo_root>/build`` and stamps that render's
    ``config.yml`` with ``project_root: <repo_root>`` (see
    ``build_cmd._repo_render_context``'s ``"project_root": str(runtime_root or
    repo_root)``) — the repo root, never the render zone, so every relative path
    the render carries anchors one level up from the disposable tree. Reproduced
    here with :meth:`TemplateManager.create_project` directly (no profile.yml,
    no dependency install) by passing the same ``project_root`` override through
    ``context``, which the real render path threads through
    ``config.yml.j2``'s ``project_root: {{ project_root }}`` the same way.

    Returns ``(repo_root, build_dir)`` — ``build_dir`` is what
    ``create_claude_code_integration`` calls ``project_dir`` at build time, and
    is where it writes ``.claude/hooks/hook_config.json``.
    """
    repo_root = tmp_path / "deployment"
    manager = TemplateManager()
    build_dir = create_project_as_built(
        manager,
        project_name="build",
        output_dir=repo_root,
        data_bundle="hello_world",
        context={"project_root": str(repo_root)},
    )
    assert build_dir == repo_root / "build"
    return repo_root, build_dir


def _default_spool_dir(monkeypatch) -> Path:
    """Where the audit writer's entries land when no spool is configured."""
    monkeypatch.delenv(SPOOL_DIR_ENV, raising=False)
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "tester")
    return record_writer.spool_dir()


def _reset_plan(repo_root: Path) -> ResetPlan:
    return ResetPlan(
        repo_root=repo_root,
        project="deployment",
        identity="tester",
        project_name_source="test",
    )


class TestAuditDirResolverEquality:
    """The writer's spool, ``ResetPlan``'s dirs and the hook's own resolver agree."""

    def test_writer_resetplan_and_hook_name_the_same_audit_dirs(self, tmp_path, monkeypatch):
        repo_root, build_dir = _render_two_zone_project(tmp_path)
        monkeypatch.setenv("OSPREY_CONFIG", str(build_dir / "config.yml"))

        from_writer = _default_spool_dir(monkeypatch)
        plan = _reset_plan(repo_root)
        _record, from_reset_plan_spool = plan.record_dirs

        osprey_hook_log = import_hook("osprey_hook_log")
        from_hook = osprey_hook_log.audit_spool_dir(identity="tester")

        spool_root = repo_root / workspace.SPOOL_DIR_RELPATH
        assert from_writer == spool_root / "tester"
        assert from_reset_plan_spool == spool_root
        assert plan.audit_dir == repo_root / workspace.PRE_UPGRADE_AUDIT_RELPATH
        assert from_hook == from_writer

    def test_the_audit_dirs_are_anchored_on_the_repo_root_not_the_render_zone(
        self, tmp_path, monkeypatch
    ):
        """Same three resolvers, asserted against the render zone as a negative.

        A resolver that accidentally anchored on ``build/`` (the directory
        holding ``config.yml``) instead of walking up to the repo root would
        still produce *a* directory — this is the test that would catch that,
        since the two zones are genuinely different directories in this fixture
        (unlike a flat, zone-less layout where they would coincide either way).
        """
        repo_root, build_dir = _render_two_zone_project(tmp_path)
        monkeypatch.setenv("OSPREY_CONFIG", str(build_dir / "config.yml"))

        assert build_dir != repo_root
        wrong_spool = build_dir / workspace.SPOOL_DIR_RELPATH

        assert _default_spool_dir(monkeypatch).parent != wrong_spool
        plan = _reset_plan(repo_root)
        assert plan.record_dirs[1] != wrong_spool
        assert plan.audit_dir != build_dir / workspace.PRE_UPGRADE_AUDIT_RELPATH
        osprey_hook_log = import_hook("osprey_hook_log")
        assert Path(osprey_hook_log.get_repo_root()) != build_dir


class TestHookConfigPathResolverEquality:
    """The middleware's read path for ``hook_config.json`` equals the write path.

    The two zones ``osprey build`` produces are the render (``build/``, holding
    ``config.yml`` and ``.claude/``) and the repo root one level up (holding
    ``var/spool``). ``hook_config.json`` and the audit spool anchor on different
    ones of the two — stated explicitly, and checked below, because a resolver
    that quietly swapped one anchor for the other would still resolve to some
    existing directory rather than fail loudly.
    """

    def test_the_middleware_reads_exactly_where_build_wrote(self, tmp_path, monkeypatch):
        repo_root, build_dir = _render_two_zone_project(tmp_path)
        config_path = build_dir / "config.yml"
        monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
        am.reset_audit_state()

        write_site = build_dir / ".claude" / "hooks" / "hook_config.json"
        assert write_site.is_file(), "osprey build (create_project) must render hook_config.json"

        read_site = am.hook_config_path()

        assert read_site == write_site
        am.reset_audit_state()

    def test_hook_config_and_the_audit_spool_anchor_on_different_zones(self, tmp_path, monkeypatch):
        repo_root, build_dir = _render_two_zone_project(tmp_path)
        config_path = build_dir / "config.yml"
        monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
        am.reset_audit_state()

        hook_config_site = am.hook_config_path()
        audit_site = _default_spool_dir(monkeypatch).parent

        # hook_config.json is anchored on the RENDER zone (build/)...
        assert hook_config_site == build_dir / am.HOOK_CONFIG_RELPATH
        # ...the audit spool is anchored on the REPO root, one level up...
        assert audit_site == repo_root / workspace.SPOOL_DIR_RELPATH
        # ...and the two do not coincide in this real two-zone render.
        assert audit_site != build_dir / workspace.SPOOL_DIR_RELPATH
        assert hook_config_site != repo_root / am.HOOK_CONFIG_RELPATH
        am.reset_audit_state()

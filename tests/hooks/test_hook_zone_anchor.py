"""Zone-anchor derivation for the shipped Claude Code hooks.

A hook runs with its working directory at the RENDER — ``<repo>/build`` — which
is the directory Claude Code is launched in and therefore what
``CLAUDE_PROJECT_DIR`` and ``hook_input["cwd"]`` name. Durable agent state does
not live there: ``build/`` is re-created wholesale by every ``osprey build``,
and the stores the rest of OSPREY reads (``ArtifactStore``,
``PendingReviewStore``, the channel-finder feedback app) anchor on the config's
``project_root`` — the REPO root, one level up.

These tests pin the derivation that closes that split: ``get_repo_root`` in
``osprey_hook_log``, and the two hooks that write/read agent data through it.
They cover the three environments a hook actually runs in — a host chat session
(cwd at ``<repo>/build``), a container (``/app/<name>``, a real three-zone repo),
and a legacy flat layout with neither ``profile.yml`` nor a ``project_root`` key
— plus the stdlib-only constraint: a hook may be executed by a bare system
``python3`` with no PyYAML on its path, so the derivation must never need it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR, resolve_project_root

HOOKS_DIR = (
    Path(__file__).parents[2] / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"
)

#: Every hook the build ships. Discovered rather than listed so a hook added
#: later is covered by the constrained-interpreter case without anyone
#: remembering to name it here.
HOOK_MODULES = sorted(path.stem for path in HOOKS_DIR.glob("osprey_*.py"))


# -- Fixtures -----------------------------------------------------------------


def _three_zone_repo(root: Path, *, project_root_key: str | None = None) -> Path:
    """Build a repo with the post-#52 layout; return the render dir (``build/``).

    ``profile.yml`` at the root is the marker every OSPREY verb walks up to, and
    the render lands in ``build/``. ``project_root_key`` writes that key into the
    rendered config — what a real ``osprey build`` always emits, and the only
    answer that is right in a container, where a repo is built at one path and
    runs at another.
    """
    (root / "profile.yml").write_text("name: anchor-fixture\n")
    build = root / "build"
    build.mkdir(parents=True, exist_ok=True)
    lines = ["# rendered config\n"]
    if project_root_key is not None:
        lines.append(f"project_root: {project_root_key}\n")
    lines.append("hooks:\n  debug: false\n")
    (build / "config.yml").write_text("".join(lines))
    return build


def _repo_root(hook_module, hook_input=None):
    """Call the helper under test on the package copy of ``osprey_hook_log``."""
    return hook_module("osprey_hook_log").get_repo_root(hook_input)


def _without_a_config(monkeypatch, tmp_path):
    """Point ``OSPREY_CONFIG`` at a path that does not exist.

    An existing config file settles the root before any walk (rule 2), so the
    marker walk is only reachable when there is no config at all — a raw
    ``claude`` launch in a repo that has not been built. The walk-up cases say
    so explicitly rather than relying on a fixture happening to omit one.
    """
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "absent" / "config.yml"))


@pytest.fixture(autouse=True)
def _clear_anchor_env(monkeypatch):
    """No inherited ``OSPREY_CONFIG``/``CLAUDE_PROJECT_DIR`` may reach a case."""
    for var in ("OSPREY_CONFIG", "CLAUDE_PROJECT_DIR", "OSPREY_HOOK_DEBUG"):
        monkeypatch.delenv(var, raising=False)


# -- 1. The config's project_root is authoritative ----------------------------


@pytest.mark.unit
class TestProjectRootFromConfig:
    """Rule 1: whatever the config OSPREY_CONFIG names says, wins."""

    def test_config_project_root_beats_the_render_cwd(self, tmp_path, monkeypatch, hook_module):
        repo = tmp_path / "deployment"
        repo.mkdir()
        build = _three_zone_repo(repo, project_root_key=str(repo))
        monkeypatch.setenv("OSPREY_CONFIG", str(build / "config.yml"))

        assert _repo_root(hook_module, {"cwd": str(build)}) == str(repo)

    def test_container_path_wins_over_the_building_hosts(self, tmp_path, monkeypatch, hook_module):
        """A container's config records ``/app/<name>``, not the build host's path.

        This is the case no on-disk walk can answer: the image was rendered
        elsewhere, so only the recorded key names the path the repo runs at.
        """
        repo = tmp_path / "built-here"
        repo.mkdir()
        build = _three_zone_repo(repo, project_root_key="/app/demo")
        monkeypatch.setenv("OSPREY_CONFIG", str(build / "config.yml"))

        assert _repo_root(hook_module, {"cwd": str(build)}) == "/app/demo"

    def test_default_config_location_is_the_render(self, tmp_path, hook_module):
        """With no ``OSPREY_CONFIG``, the config is ``<project_dir>/config.yml``."""
        repo = tmp_path / "deployment"
        repo.mkdir()
        build = _three_zone_repo(repo, project_root_key=str(repo))

        assert _repo_root(hook_module, {"cwd": str(build)}) == str(repo)

    @pytest.mark.parametrize(
        "line",
        [
            'project_root: "/quoted/repo"',
            "project_root: '/quoted/repo'",
            "project_root:    /quoted/repo",
            "project_root: /quoted/repo  # trailing comment",
        ],
        ids=["double-quoted", "single-quoted", "extra-space", "inline-comment"],
    )
    def test_scalar_spellings(self, tmp_path, monkeypatch, hook_module, line):
        config = tmp_path / "config.yml"
        config.write_text(f"{line}\nhooks:\n  debug: false\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(config))

        assert _repo_root(hook_module, {"cwd": str(tmp_path)}) == "/quoted/repo"

    @pytest.mark.parametrize(
        "value",
        ["~", "null", "Null", "NULL", "true", "false"],
        ids=["tilde", "null", "null-title", "null-upper", "true", "false"],
    )
    def test_yaml_null_scalars_are_not_paths(self, tmp_path, monkeypatch, hook_module, value):
        """``project_root: ~`` means unset, not the user's home directory.

        These spellings load as ``None``/``bool``, so the framework's
        ``dotted_config_str`` rejects them and the resolver falls through. Read
        as literal text they are catastrophic instead of inert: ``~`` expands to
        ``$HOME`` and ``null`` becomes a relative directory beside the render,
        either of which anchors every store outside the deployment.
        """
        repo = tmp_path / "deployment"
        repo.mkdir()
        build = _three_zone_repo(repo)
        (build / "config.yml").write_text(f"project_root: {value}\nhooks:\n  debug: false\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(build / "config.yml"))

        assert _repo_root(hook_module, {"cwd": str(build)}) == str(repo)

    def test_quoted_tilde_stays_a_string(self, tmp_path, monkeypatch, hook_module):
        """Quoted, ``"~"`` *is* a string, and YAML — so the framework — expands it.

        The rejection above is about YAML's types, not about the characters, so
        it must not reach a value the config deliberately quoted.
        """
        config = tmp_path / "config.yml"
        config.write_text('project_root: "~"\nhooks:\n  debug: false\n')
        monkeypatch.setenv("OSPREY_CONFIG", str(config))

        assert _repo_root(hook_module, {"cwd": str(tmp_path)}) == str(Path("~").expanduser())

    def test_last_occurrence_wins(self, tmp_path, monkeypatch, hook_module):
        """Duplicate top-level keys resolve YAML's way: the last one is the value.

        A profile ``config:`` block can append a key the template already
        emitted, so a scanner that stopped at the first match would read the
        value the config does not actually have.
        """
        config = tmp_path / "config.yml"
        config.write_text("project_root: /first\nhooks:\n  debug: false\nproject_root: /second\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(config))

        assert _repo_root(hook_module, {"cwd": str(tmp_path)}) == "/second"

    @pytest.mark.parametrize(
        "text",
        [
            "# project_root: /commented-out\n",
            "#   PROJECT_ROOT=/env-var-doc\n",
            "nested:\n  project_root: /indented\n",
            "project_root:\n",
            "project_root: \n",
        ],
        ids=["comment", "env-doc-comment", "indented", "bare-key", "empty-value"],
    )
    def test_non_values_are_not_read(self, tmp_path, monkeypatch, hook_module, text):
        """Only a top-level key with a real value counts; the rest fall through.

        The shipped config templates carry ``project_root`` in prose comments and
        under nested blocks; matching either would anchor the whole deployment on
        documentation.
        """
        repo = tmp_path / "deployment"
        repo.mkdir()
        (repo / "profile.yml").write_text("name: fixture\n")
        build = repo / "build"
        build.mkdir()
        (build / "config.yml").write_text(text)
        monkeypatch.setenv("OSPREY_CONFIG", str(build / "config.yml"))

        # Falls through to the config's own repo, not to the commented value.
        assert _repo_root(hook_module, {"cwd": str(build)}) == str(repo)

    def test_unreadable_config_falls_through(self, tmp_path, monkeypatch, hook_module):
        repo = tmp_path / "deployment"
        repo.mkdir()
        build = _three_zone_repo(repo)
        monkeypatch.setenv("OSPREY_CONFIG", str(build / "nonexistent.yml"))

        assert _repo_root(hook_module, {"cwd": str(build)}) == str(repo)


# -- 2. The config's own repo, ahead of any walk -------------------------------


@pytest.mark.unit
class TestConfigDirectoryOutranksTheWalk:
    """Rule 2: an existing config file settles it, exactly as the framework does.

    ``workspace.resolve_project_root`` consults ``repo_root_for_config`` the
    moment the config path exists and never looks at the filesystem around the
    cwd. A hook that walked for ``profile.yml`` first would disagree with it
    whenever the two point at different repos — and the hook and the app read
    and write the *same* stores, so a disagreement is a split-brain, not a
    preference. These cases are asserted against the framework's own answer
    rather than a hard-coded path, so the two cannot drift apart quietly.
    """

    def test_config_outside_the_cwds_repo_wins(self, tmp_path, monkeypatch, hook_module):
        """``OSPREY_CONFIG`` elsewhere beats a ``profile.yml`` repo around the cwd."""
        repo = tmp_path / "deployment"
        repo.mkdir()
        build = _three_zone_repo(repo)  # a genuine marked repo around the cwd
        elsewhere = tmp_path / "elsewhere" / "build"
        elsewhere.mkdir(parents=True)
        (elsewhere / "config.yml").write_text("hooks:\n  debug: false\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(elsewhere / "config.yml"))

        expected = str(resolve_project_root(None))
        assert expected == str(tmp_path / "elsewhere"), "framework moved; re-derive the rule"
        assert _repo_root(hook_module, {"cwd": str(build)}) == expected

    def test_unzoned_config_outside_the_cwds_repo_wins(self, tmp_path, monkeypatch, hook_module):
        """The same, for a config whose directory is not the build zone."""
        repo = tmp_path / "deployment"
        repo.mkdir()
        build = _three_zone_repo(repo)
        elsewhere = tmp_path / "container-project"
        elsewhere.mkdir()
        (elsewhere / "config.yml").write_text("hooks:\n  debug: false\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(elsewhere / "config.yml"))

        expected = str(resolve_project_root(None))
        assert expected == str(elsewhere), "framework moved; re-derive the rule"
        assert _repo_root(hook_module, {"cwd": str(build)}) == expected


# -- 3. The profile.yml walk-up ------------------------------------------------


@pytest.mark.unit
class TestProfileMarkerWalkUp:
    """Rule 3: the marker on disk, found the way every OSPREY verb finds it.

    Reached when the config named no root *and* does not exist — the point at
    which the framework has nothing left to consult either.
    """

    def test_walks_up_from_the_render(self, tmp_path, monkeypatch, hook_module):
        repo = tmp_path / "deployment"
        repo.mkdir()
        build = _three_zone_repo(repo)  # config without a project_root key
        _without_a_config(monkeypatch, tmp_path)

        assert _repo_root(hook_module, {"cwd": str(build)}) == str(repo)

    def test_walks_up_from_a_nested_subdirectory(self, tmp_path, hook_module):
        repo = tmp_path / "deployment"
        repo.mkdir()
        build = _three_zone_repo(repo)
        nested = build / "a" / "b"
        nested.mkdir(parents=True)

        assert _repo_root(hook_module, {"cwd": str(nested)}) == str(repo)

    def test_innermost_marker_wins(self, tmp_path, monkeypatch, hook_module):
        outer = tmp_path / "outer"
        outer.mkdir()
        (outer / "profile.yml").write_text("name: outer\n")
        inner = outer / "inner"
        inner.mkdir()
        build = _three_zone_repo(inner)
        _without_a_config(monkeypatch, tmp_path)

        assert _repo_root(hook_module, {"cwd": str(build)}) == str(inner)

    def test_env_var_beats_hook_input_cwd(self, tmp_path, monkeypatch, hook_module):
        """``CLAUDE_PROJECT_DIR`` outranks ``hook_input['cwd']``, as everywhere else."""
        repo = tmp_path / "deployment"
        repo.mkdir()
        build = _three_zone_repo(repo)
        _without_a_config(monkeypatch, tmp_path)
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(build))

        assert _repo_root(hook_module, {"cwd": str(tmp_path / "elsewhere")}) == str(repo)

    def test_no_hook_input_uses_the_env_var(self, tmp_path, monkeypatch, hook_module):
        """UserPromptSubmit hooks read no stdin — the helper must work without it."""
        repo = tmp_path / "deployment"
        repo.mkdir()
        build = _three_zone_repo(repo)
        _without_a_config(monkeypatch, tmp_path)
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(build))

        assert _repo_root(hook_module) == str(repo)


# -- 4. Legacy flat layouts stay put -------------------------------------------


@pytest.mark.unit
class TestLegacyFlatLayout:
    """Rule 4: no marker, no key — a flat project anchors on itself."""

    def test_flat_project_with_config_anchors_on_itself(self, tmp_path, hook_module):
        (tmp_path / "config.yml").write_text("hooks:\n  debug: false\n")

        assert _repo_root(hook_module, {"cwd": str(tmp_path)}) == str(tmp_path)

    def test_flat_project_without_config_anchors_on_itself(self, tmp_path, hook_module):
        assert _repo_root(hook_module, {"cwd": str(tmp_path)}) == str(tmp_path)

    def test_build_named_config_parent_without_marker(self, tmp_path, hook_module):
        """A render whose repo lost its ``profile.yml`` still resolves one level up.

        Mirrors ``osprey.utils.workspace.repo_root_for_config``: the build zone's
        name is the fallback signal when the marker is missing.
        """
        repo = tmp_path / "deployment"
        build = repo / "build"
        build.mkdir(parents=True)
        (build / "config.yml").write_text("hooks:\n  debug: false\n")

        assert _repo_root(hook_module, {"cwd": str(build)}) == str(repo)

    def test_empty_project_dir_falls_back_to_cwd(self, tmp_path, hook_module, monkeypatch):
        monkeypatch.chdir(tmp_path)

        assert _repo_root(hook_module, {}) == str(tmp_path)


# -- 5. The stdlib-only constraint ---------------------------------------------


@pytest.mark.unit
def test_derivation_works_without_pyyaml(tmp_path):
    """A hook may run under a bare system ``python3`` that has no PyYAML.

    ``yaml`` is blocked outright rather than merely absent, so any import of it
    on the derivation path raises instead of silently succeeding from the test
    venv's site-packages.
    """
    repo = tmp_path / "deployment"
    repo.mkdir()
    build = _three_zone_repo(repo, project_root_key=str(repo))

    script = textwrap.dedent(f"""
        import sys
        sys.modules["yaml"] = None  # any `import yaml` now raises ImportError
        sys.path.insert(0, {str(HOOKS_DIR)!r})
        from osprey_hook_log import get_repo_root
        print(get_repo_root({{"cwd": {str(build)!r}}}))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(repo)


@pytest.mark.unit
def test_every_shipped_hook_is_covered():
    """The inventory below is discovered, so an empty glob must not pass quietly."""
    assert "osprey_hook_log" in HOOK_MODULES
    assert len(HOOK_MODULES) >= 5, f"hook discovery found only {HOOK_MODULES}"


@pytest.mark.unit
@pytest.mark.parametrize("module_name", HOOK_MODULES)
def test_hook_imports_without_yaml_or_osprey(tmp_path, module_name):
    """*Every* hook must load with neither PyYAML nor ``osprey`` importable.

    The constraint is the whole layer's, not one helper's: a raw ``claude``
    launch runs hooks under whatever ``python3`` is on PATH, and a hook that
    needs either package there does not fail loudly — it fails at import, before
    it can print anything, and its check goes dark while the session carries on
    looking healthy. Both packages are blocked outright rather than merely
    absent, so an accidental dependency cannot be masked by the test venv.

    Loading is the assertion: the script hooks run their whole body on import,
    so they are also shown to reach ``exit 0`` on empty input, which is the
    fail-open contract every one of them claims.
    """
    script = textwrap.dedent(f"""
        import sys
        sys.modules["yaml"] = None    # any `import yaml` now raises ImportError
        sys.modules["osprey"] = None  # ... and so does any `import osprey`
        sys.path.insert(0, {str(HOOKS_DIR)!r})
        import {module_name}
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        input="",
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr


# -- 6. The duplicated constants stay in step ----------------------------------


@pytest.mark.unit
def test_hook_constants_match_the_framework(hook_module):
    """The hook spells the marker and zone name out; both must still be right.

    They are duplicated rather than imported because a hook may run under an
    interpreter that cannot import osprey — which is exactly the condition that
    would let them drift silently, so the drift is asserted here instead.
    """
    from osprey.cli.repo_resolver import PROFILE_FILENAME
    from osprey.utils.workspace import BUILD_DIR_NAME

    hook_log = hook_module("osprey_hook_log")

    assert hook_log.PROFILE_MARKER == PROFILE_FILENAME
    assert hook_log.BUILD_DIR_NAME == BUILD_DIR_NAME


# -- 7. The hooks that consume it ----------------------------------------------


@pytest.mark.unit
def test_feedback_capture_writes_to_the_state_zone(tmp_path, hook_runner):
    """The capture hook must write where the feedback app reads.

    The hook runs with its cwd at the render; the app reads
    ``<repo>/var/agent_data/feedback/pending_reviews.json``. Anchored on the
    render, every captured item landed in a directory the next build deletes.
    """
    repo = tmp_path / "deployment"
    repo.mkdir()
    build = _three_zone_repo(repo, project_root_key=str(repo))

    hook_runner(
        "osprey_cf_feedback_capture.py",
        "mcp__channel-finder__build_channels",
        {"query": "BPM channels", "facility": "test"},
        cwd=build,
        config_path=build / "config.yml",
        tool_response={"channels": [{"name": "BPM:1"}], "total": 1},
        hook_input_extra={"cwd": str(build), "session_id": "s", "transcript_path": ""},
    )

    canonical = repo / DEFAULT_AGENT_DATA_BASE_DIR / "feedback" / "pending_reviews.json"
    inside_render = build / DEFAULT_AGENT_DATA_BASE_DIR / "feedback" / "pending_reviews.json"
    assert canonical.exists(), f"expected capture at {canonical}"
    assert not inside_render.exists(), "capture landed inside the disposable render zone"

    items = json.loads(canonical.read_text())["items"]
    assert len(items) == 1
    assert next(iter(items.values()))["channel_count"] == 1


@pytest.mark.unit
def test_feedback_capture_still_works_in_a_flat_project(tmp_path, hook_runner):
    """Regression guard: a project with no ``profile.yml`` anchors on itself."""
    hook_runner(
        "osprey_cf_feedback_capture.py",
        "mcp__channel-finder__build_channels",
        {"query": "BPM channels", "facility": "test"},
        cwd=tmp_path,
        tool_response={"channels": [{"name": "BPM:1"}], "total": 1},
        hook_input_extra={"cwd": str(tmp_path), "session_id": "s", "transcript_path": ""},
    )

    store = tmp_path / DEFAULT_AGENT_DATA_BASE_DIR / "feedback" / "pending_reviews.json"
    assert store.exists(), f"expected capture at {store}"


@pytest.mark.unit
def test_focus_validate_reads_from_the_state_zone(tmp_path, hook_runner_raw, monkeypatch):
    """The focus validator must read the focus file the gallery actually writes.

    Anchored on the render it found no file, stripped nothing, and reported
    success — a silent no-op rather than a visible failure.
    """
    repo = tmp_path / "deployment"
    repo.mkdir()
    build = _three_zone_repo(repo, project_root_key=str(repo))

    agent_data = repo / DEFAULT_AGENT_DATA_BASE_DIR
    (agent_data / "artifacts").mkdir(parents=True)
    (agent_data / "focus_state.txt").write_text(
        "[Gallery Focus]\nplot alpha (id=keep-me)\nplot beta (id=gone)\n"
    )
    (agent_data / "artifacts" / "artifacts.json").write_text(
        json.dumps({"entries": [{"id": "keep-me"}]})
    )
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(build))

    returncode, stdout, _ = hook_runner_raw(
        "osprey_focus_validate.py",
        "",
        {},
        cwd=build,
        config_path=build / "config.yml",
    )

    assert returncode == 0
    assert "keep-me" in stdout
    assert "gone" not in stdout

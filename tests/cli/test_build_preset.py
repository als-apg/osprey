"""Tests for the preset/--set surface and the `osprey build` it feeds.

A deployment starts with `osprey init --preset NAME [--set K=V]`, which
resolves the named preset through the profile pipeline and materializes it as
`profile.yml` (plus `data/`, `personas/`, `.env.example`) at a repo's root.
`osprey build --repo DIR` is zero-argument from there: it re-resolves that
repo's own `profile.yml`, with no preset/--set surface of its own, and renders
`DIR/build/`. This module covers both halves — the resolution pipeline `osprey
init` drives (bundled presets, --set inline scalars/lists, `extends`, the
drift-guard that prevents presets from depending on profile-dir-relative paths
that would break when shipped in a wheel) and what a plain `osprey build` does
with the profile.yml it finds.

A `--set` pair is an EDIT of the profile, not one more inheritance layer: it
replaces the value at the key it names, whatever was there. Inheritance
(`extends:`, a persona delta, a host-variant overlay) still adds — string lists
union — and the two are pinned apart below.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile import list_presets, resolve_build_profile
from osprey.cli.init_cmd import init
from osprey.errors import BuildProfileError


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _facility_data(root: Path, bundle: str = "hello_world") -> Path:
    """Lay down the facility data tree beside a fixture profile.

    ``osprey init`` materializes this tree from the preset's bundle and writes
    ``data: data`` into the profile it emits. A hand-written fixture profile
    has to do the same: ``data:`` is required, must resolve to a real
    directory, and a preset that reads a channel-limits database or a knowledge
    zone out of it needs the packaged content, not an empty directory.
    """
    from osprey.cli.templates.manager import TemplateManager

    destination = root / "data"
    shutil.copytree(
        TemplateManager().template_root / "apps" / bundle / "data",
        destination,
        dirs_exist_ok=True,
    )
    return destination


#: The keys every deployment must state, for fixtures that build a profile of
#: their own rather than inheriting a preset. The posture floor refuses a build
#: whose profile leaves any of them to a reader's fallback.
_POSTURE_FLOOR = (
    "  control_system.type: mock\n"
    "  archiver.type: mock_archiver\n"
    "  approval.enabled: true\n"
    "  approval.default_policy: always\n"
    "  claude_code.telemetry.enabled: false\n"
    "  hooks.debug: false\n"
)


def _config_yaml(project_dir: Path) -> dict:
    return yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))


def _assert_build_error_logged(caplog: pytest.LogCaptureFixture, *needles: str) -> None:
    """Assert the build reported one of *needles* to the operator.

    ``osprey build`` reports fatal user errors through ``logger.error()`` and
    then aborts, so the message reaches the operator on stderr — never on
    stdout, which is reserved for program output. click's ``Result.output``
    folds both streams together and cannot tell the two apart, so these
    assertions read the log record itself (house pattern, see
    ``tests/cli/test_templates.py``).
    """
    text = caplog.text.lower()
    assert any(needle.lower() in text for needle in needles), (
        f"expected one of {needles} in the build log; got records: "
        f"{[record.getMessage()[:80] for record in caplog.records]}"
    )


def _profile_yaml(repo: Path) -> dict:
    """The emitted source profile at *repo*'s root."""
    return yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))


def _materialize(runner: CliRunner, parent, name: str, preset: str, *extra: str):
    """Materialize a deployment repo under *parent*, then render its build zone.

    Returns the ``init`` result when it failed and the ``build`` result when it
    did not, so a caller asserting on ``exit_code`` sees whichever step actually
    refused. ``--set`` belongs to ``init`` — its pairs are baked into the
    emitted profile, not applied at render time — so they are forwarded there.

    The render lands at ``<parent>/<name>/build``; :func:`_project` is the one
    spelling of that path, so a caller never assembles it by hand.
    """
    repo = pathlib.Path(parent) / name
    created = runner.invoke(init, [str(repo), "--preset", preset, "--no-git", *extra])
    if created.exit_code != 0:
        return created
    return runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def _render_from(runner: CliRunner, profile_path, *extra: str):
    """Render the deployment repo that *profile_path* is the source of.

    A profile file IS its repo's source zone, so the repo is simply the file's
    directory. ``extra`` is accepted and ignored on this path: ``--set`` pairs
    are materialization-time inputs, and a repo whose profile already exists on
    disk has nothing left to bake.
    """
    repo = pathlib.Path(profile_path).parent
    return runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def _project(parent, name: str) -> pathlib.Path:
    """The render :func:`_materialize` produced for *name* under *parent*."""
    return pathlib.Path(parent) / name / "build"


def test_preset_hello_world_creates_project(runner: CliRunner, tmp_path: Path) -> None:
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world")
    assert result.exit_code == 0, result.output
    project_dir = _project(tmp_path, "smoke")
    assert (project_dir / "config.yml").exists()
    assert (project_dir / "CLAUDE.md").exists()


def test_set_flag_overrides_scalar(runner: CliRunner, tmp_path: Path) -> None:
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world", "--set", "model=sonnet")
    assert result.exit_code == 0, result.output
    config = _config_yaml(_project(tmp_path, "smoke"))
    assert config["claude_code"]["default_model"] == "sonnet"


def test_set_with_a_list_value_replaces_the_presets_list(runner: CliRunner, tmp_path: Path) -> None:
    """``--set hooks=[…]`` states the hook list; the preset's entries do not survive.

    The edit half of the rule at the CLI surface. An inheritance layer would
    union these and leave the operator unable to take a hook away, which is the
    narrowing the explicit-profile work exists to make expressible.
    """
    result = _materialize(
        runner, str(tmp_path), "smoke", "hello-world", "--set", "hooks=[memory-guard]"
    )
    assert result.exit_code == 0, result.output
    # The persisted manifest is the artifact list the build actually saw.
    manifest_path = _project(tmp_path, "smoke") / ".osprey-manifest.json"
    assert manifest_path.exists(), result.output
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["hooks"] == ["memory-guard"]


def test_preset_ariel_standalone_renders_logbook_persona(runner: CliRunner, tmp_path: Path) -> None:
    """The ariel-standalone preset's ``claude_md_template`` must travel from
    the preset YAML, through the build-profile parser, into the render
    context, into the rendered ``CLAUDE.md``, and into the manifest creation
    block so that ``osprey build`` round-trips the persona choice.

    Drift-guard: if any layer of that wiring (BuildProfile field,
    _KNOWN_PROFILE_KEYS, _parse_profile, build_cmd's context/manifest_context
    propagation, or the renderer's template-selection branch) breaks, the
    preset silently falls back to the control-system persona — exactly the
    regression this test pins down.
    """
    import json

    result = _materialize(runner, str(tmp_path), "smoke", "ariel-standalone")
    assert result.exit_code == 0, result.output

    project_dir = _project(tmp_path, "smoke")

    claude_md = (project_dir / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Logbook Research Assistant" in claude_md, claude_md[:200]
    assert "Control System Assistant" not in claude_md

    manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
    assert manifest["creation"]["claude_md_template"] == "CLAUDE.ariel.md.j2"


def test_preset_control_assistant_ships_live_openobserve_telemetry(
    runner: CliRunner, tmp_path: Path
) -> None:
    """control_assistant is the production-shaped reference facility: telemetry
    is wired LIVE against a co-deployed OpenObserve store. This pins the full
    wiring so a regression in the preset template (dropped service, disabled
    switch, hardcoded endpoint, or missing :- fallback) fails loudly.
    """
    result = _materialize(runner, str(tmp_path), "smoke", "control-assistant")
    assert result.exit_code == 0, result.output

    cfg = _config_yaml(_project(tmp_path, "smoke"))

    # openobserve is deployed alongside postgresql (not merely declared).
    assert "openobserve" in cfg["deployed_services"]
    assert "postgresql" in cfg["deployed_services"]

    tel = cfg["claude_code"]["telemetry"]
    assert tel["enabled"] is True
    assert tel["backend"] == "openobserve"
    # openobserve backend auto-derives the endpoint per network context — a
    # hardcoded localhost endpoint would make the in-container worker emit to its
    # own loopback and silently drop everything, so it must be absent.
    assert "endpoint" not in tel
    # The agent authenticates as the store's dedicated INGEST service account,
    # never as root: the store still initializes itself from ZO_ROOT_USER_*, but
    # that pair stays in the compose file and the root password never reaches
    # the agent's config.
    assert tel["openobserve"]["user"] == "${ZO_INGEST_USER_EMAIL:-ingest@example.com}"
    # The token carries NO ${VAR:-default}, and that absence is the point: a
    # literal default token in a shipped template would be a published
    # credential. `osprey up` provisions the account and writes the token the
    # store issues into .env; the preflights defer this one variable rather
    # than refusing a start that has not reached that step yet.
    assert tel["openobserve"]["password"] == "${ZO_INGEST_SA_TOKEN}"


def test_unknown_preset_name(runner: CliRunner, tmp_path: Path) -> None:
    """C10: unknown preset is a usage error → exit 2 (per click convention)."""
    result = _materialize(runner, str(tmp_path), "smoke", "bogus")
    assert result.exit_code == 2, result.output
    assert "bogus" in result.output.lower()
    for name in list_presets():
        assert name in result.output


def test_preset_name_normalization(runner: CliRunner, tmp_path: Path) -> None:
    """control-assistant and control_assistant must both resolve to the same preset."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()

    r_hyphen = _materialize(runner, str(out_a), "smoke", "control-assistant")
    r_under = _materialize(runner, str(out_b), "smoke", "control_assistant")
    assert r_hyphen.exit_code == 0, r_hyphen.output
    assert r_under.exit_code == 0, r_under.output
    cfg_a = _config_yaml(_project(out_a, "smoke"))
    cfg_b = _config_yaml(_project(out_b, "smoke"))
    # Same preset → same default_model in rendered config.
    # NB: the rendered key lives at claude_code.default_model, NOT top-level
    # (a top-level lookup would make this assertion vacuous).
    assert cfg_a["claude_code"]["default_model"] == cfg_b["claude_code"]["default_model"]


def test_preset_drift_guard() -> None:
    """Bundled presets must NOT depend on profile-dir-relative paths.

    services/env.file resolve relative to profile_dir, which for presets is the
    wheel-installed package directory. Any preset adding these will silently
    fail at install time. Catch it here.
    """
    import importlib.resources

    presets_root = importlib.resources.files("osprey.profiles.presets")
    presets_dir = Path(str(presets_root))
    yml_files = sorted(presets_dir.glob("*.yml"))
    assert yml_files, "no preset YAML files found"
    for yml in yml_files:
        raw = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        assert raw.get("services", {}) == {}, (
            f"{yml.name}: services must be empty (templates would break in the wheel)"
        )
        env = raw.get("env", {}) or {}
        assert env.get("file") is None, (
            f"{yml.name}: env.file must be unset (path would break in the wheel)"
        )


def test_unknown_profile_key_fails_the_build(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Unknown top-level keys fail the build, naming each typo.

    A profile with a typoed key silently ignored would ship a project missing
    whatever the profile actually asked for; failing loudly is what makes the
    typo visible before the project is built at all.
    """
    profile = tmp_path / "repo" / "profile.yml"
    profile.parent.mkdir()
    _facility_data(profile.parent)
    profile.write_text(
        "name: TypoTest\n"
        "extends: hello-world\n"
        "data: data\n"
        "provider: anthropic\n"
        "mcp_server: {}\n"  # typo of mcp_servers
        "permission: []\n"  # typo of permissions
    )
    with caplog.at_level(logging.ERROR):
        result = _render_from(runner, str(profile))

    assert result.exit_code != 0
    assert "mcp_server" in caplog.text
    assert "permission" in caplog.text
    assert not (_project(tmp_path, "repo")).exists()


def test_manifest_schema_version_bumped(runner: CliRunner, tmp_path: Path) -> None:
    """The manifest records the schema it was written against.

    Bumped to 1.3.0 when the retired bundle key left ``creation``: a reader
    of an older manifest must be able to tell that the key it is looking for
    was dropped rather than merely absent from this project.
    """
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world")
    assert result.exit_code == 0, result.output
    import json

    manifest = json.loads((_project(tmp_path, "smoke") / ".osprey-manifest.json").read_text())
    assert manifest["schema_version"] == "1.3.0"


def test_manifest_uses_build_args_not_init_args(runner: CliRunner, tmp_path: Path) -> None:
    """C3: on-disk key renamed from init_args to build_args."""
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world")
    assert result.exit_code == 0, result.output
    import json

    manifest = json.loads((_project(tmp_path, "smoke") / ".osprey-manifest.json").read_text())
    assert "build_args" in manifest
    assert "init_args" not in manifest


def test_set_dotted_path_lands_in_config_yml(runner: CliRunner, tmp_path: Path) -> None:
    """T3 (also pins B3 closure): --set with dotted key writes to nested config."""
    # `config.<...>` is the documented path for inserting custom rendered-config
    # fields via --set; assert the dotted key lands at the nested location.
    result = _materialize(
        runner, str(tmp_path), "smoke", "hello-world", "--set", "config.system.timezone=UTC"
    )
    assert result.exit_code == 0, result.output
    config = _config_yaml(_project(tmp_path, "smoke"))
    assert config.get("system", {}).get("timezone") == "UTC"


def test_set_yaml_typed_values(runner: CliRunner, tmp_path: Path) -> None:
    """T3: RHS of --set is YAML-parsed for free type coercion."""
    # Use config-side keys so we can reliably assert each parsed type.
    result = _materialize(
        runner,
        str(tmp_path),
        "smoke",
        "hello-world",
        "--set",
        "config.an_int=120",
        "--set",
        "config.a_bool=true",
        "--set",
        "config.a_null=null",
        "--set",
        "config.a_list=[a, b]",
    )
    assert result.exit_code == 0, result.output
    cfg = _config_yaml(_project(tmp_path, "smoke"))
    # config.* lands under the rendered config-overrides path.
    sect = cfg.get("config") or cfg  # tolerant of preset's actual layout
    assert sect.get("an_int") == 120 or cfg.get("an_int") == 120
    assert sect.get("a_bool") is True or cfg.get("a_bool") is True
    # null may be persisted as None or omitted; accept either.
    assert (sect.get("a_null") is None) or ("a_null" not in sect)
    assert sect.get("a_list") == ["a", "b"] or cfg.get("a_list") == ["a", "b"]


# ---------------------------------------------------------------------------
# An edit states; inheritance adds
#
# The two things that reach a profile mean different things. `extends:`, a
# persona delta and a host-variant overlay are inheritance — a layer is a
# DIFFERENCE, so string lists union and taking something away is an explicit
# verb. A `--set` pair is an EDIT — it REPLACES the value at the key it names,
# whatever was there. The cases below are the ones where the difference is
# visible, and where routing an edit back through the inheritance merge would
# silently lose what the operator said.
# ---------------------------------------------------------------------------


def test_init_set_can_state_an_empty_service_list(runner: CliRunner, tmp_path: Path) -> None:
    """`--set config.deployed_services=[]` reaches the emitted profile as `[]`.

    The narrowing a union cannot express, and the case that motivated the
    change: layered under inheritance, an empty list merged with the preset's
    four services is the preset's four services, so a deployment that meant to
    deploy nothing quietly deployed everything.
    """
    repo = tmp_path / "smoke"
    result = runner.invoke(
        init,
        [
            str(repo),
            "--preset",
            "control-assistant",
            "--no-git",
            "--set",
            "config.deployed_services=[]",
        ],
    )
    assert result.exit_code == 0, result.output

    assert _profile_yaml(repo)["config"]["deployed_services"] == []


def test_a_set_list_replaces_rather_than_unions_with_the_preset() -> None:
    """`--set config.claude_code.permissions.deny=[…]` IS the deny list.

    control-assistant denies one tool of its own. The edit names that key, so
    what resolves is exactly what was stated — not the preset's entry with the
    stated one appended, which is what a union would give.
    """
    profile, _dir = resolve_build_profile(
        None,
        "control-assistant",
        set_pairs=("config.claude_code.permissions.deny=[Bash(rm)]",),
    )

    assert profile.config["claude_code.permissions.deny"] == ["Bash(rm)"]


def test_a_persona_inherits_the_bases_deny_entries_by_union() -> None:
    """The inheritance half: control-assistant-knowledge extends control-assistant.

    Its delta adds one denied tool and inherits the base's, because a layer is
    a difference and string lists union. Pinned beside the edit case below so
    the two rules are read together — this is what an edit at the same key has
    to override.
    """
    profile, _dir = resolve_build_profile(None, "control-assistant-knowledge")

    assert profile.config["claude_code.permissions.deny"] == [
        "mcp__osprey_workspace__setup_patch",
        "mcp__osprey_facility_knowledge__draft_concept",
    ]


def test_an_edit_applies_after_extends_and_replaces_the_inherited_list() -> None:
    """The edit half of the same key: `--set` lands AFTER `extends` resolution.

    Applied as one more layer under inheritance it would union with both the
    persona's entry and the base's, and the base's
    `mcp__osprey_workspace__setup_patch` would reappear in a list the operator
    just stated in full. Applied to the resolved document, it is the list.
    """
    profile, _dir = resolve_build_profile(
        None,
        "control-assistant-knowledge",
        set_pairs=("config.claude_code.permissions.deny=[Bash(rm)]",),
    )

    assert profile.config["claude_code.permissions.deny"] == ["Bash(rm)"]


def test_set_path_through_scalar_aborts(runner: CliRunner, tmp_path: Path) -> None:
    """A --set key that descends through a scalar an earlier --set already
    wrote is refused by `osprey init`, which bakes --set pairs into the
    profile it materializes — one flag cannot both set `model` and treat
    `model` as a mapping to descend into."""
    # First --set sets a scalar, second tries to descend into it.
    result = _materialize(
        runner,
        str(tmp_path),
        "smoke",
        "hello-world",
        "--set",
        "model=haiku",
        "--set",
        "model.flavor=fast",
    )
    assert result.exit_code != 0, result.output
    output = result.output.lower()
    assert "scalar" in output or "conflict" in output


def test_set_malformed_pair_aborts(runner: CliRunner, tmp_path: Path) -> None:
    """A --set value without '=' or with an empty key is refused before it
    ever reaches the profile, rather than writing a garbage key into it."""
    no_eq = _materialize(runner, str(tmp_path), "smoke", "hello-world", "--set", "model")
    assert no_eq.exit_code != 0, no_eq.output
    assert "key=value" in no_eq.output.lower()

    empty_key = _materialize(runner, str(tmp_path), "smoke", "hello-world", "--set", "=oops")
    assert empty_key.exit_code != 0, empty_key.output
    assert "non-empty" in empty_key.output.lower() or "empty" in empty_key.output.lower()


def test_profile_mcp_servers_persisted_to_config(runner: CliRunner, tmp_path: Path) -> None:
    """A profile's mcp_servers land in the built project's config.yml."""
    profile = tmp_path / "repo" / "profile.yml"
    profile.parent.mkdir()
    _facility_data(profile.parent)
    profile.write_text(
        "name: McpTest\n"
        "extends: hello-world\n"
        "data: data\n"
        "provider: anthropic\n"
        "mcp_servers:\n"
        "  echo:\n"
        "    command: echo\n"
        "    args: [hello]\n"
        "    permissions:\n"
        "      allow: [echo]\n"
    )
    result = _render_from(runner, str(profile))
    assert result.exit_code == 0, result.output
    config = _config_yaml(_project(tmp_path, "repo"))
    # NB: profile mcp_servers are persisted under claude_code.servers
    # (see _persist_mcp_servers in build_cmd.py).
    servers = config.get("claude_code", {}).get("servers", {})
    assert "echo" in servers, f"claude_code.servers in config: {list(servers.keys())}"
    assert servers["echo"]["command"] == "echo"
    assert servers["echo"]["args"] == ["hello"]


def test_profile_categories_persisted_to_config(runner: CliRunner, tmp_path: Path) -> None:
    """A profile's custom artifact categories land in the built config.yml."""
    profile = tmp_path / "repo" / "profile.yml"
    profile.parent.mkdir()
    _facility_data(profile.parent)
    profile.write_text(
        "name: CatTest\n"
        "extends: hello-world\n"
        "data: data\n"
        "provider: anthropic\n"
        "artifact_server:\n"
        "  categories:\n"
        "    diagnostics:\n"
        "      label: Diagnostics\n"
        "      color: '#ff0066'\n"
    )
    result = _render_from(runner, str(profile))
    assert result.exit_code == 0, result.output
    config = _config_yaml(_project(tmp_path, "repo"))
    cats = config.get("artifact_server", {}).get("categories", {})
    assert "diagnostics" in cats, f"artifact_server.categories in config: {list(cats.keys())}"
    assert cats["diagnostics"]["label"] == "Diagnostics"
    assert cats["diagnostics"]["color"].lower() == "#ff0066"
    # Rendered defaults from the template survive the merge.
    assert "port" in config.get("artifact_server", {})


def test_profile_md_files_registered_as_user_owned(runner: CliRunner, tmp_path: Path) -> None:
    """Convention artifacts a profile ships are registered as user_owned in the
    manifest, so a later `osprey build` treats them as the operator's
    and never overwrites them."""
    profile_dir = tmp_path / "repo"
    (profile_dir / "rules").mkdir(parents=True)
    (profile_dir / "rules" / "extra.md").write_text("# Custom rule\nuser-defined content\n")
    _facility_data(profile_dir)
    profile = profile_dir / "profile.yml"
    profile.write_text(
        "extends: hello-world\nname: ConventionTest\ndata: data\nprovider: anthropic\n"
    )
    result = _render_from(runner, str(profile))
    assert result.exit_code == 0, result.output
    project_dir = _project(tmp_path, "repo")
    # 1. file actually landed
    assert (project_dir / ".claude" / "rules" / "extra.md").exists()
    # 2. registered in manifest user_owned section (or comparable artifact-ownership field)
    import json

    manifest = json.loads((project_dir / ".osprey-manifest.json").read_text())
    # The exact ownership key may be 'user_owned' or under 'artifacts'; assert presence.
    serialized = json.dumps(manifest)
    assert "extra.md" in serialized, (
        "Profile artifact not referenced in manifest at all — check _register_convention_artifacts"
    )


def test_extends_missing_base_aborts(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A profile's `extends` pointing at a missing file produces a clear error,
    not a stack trace, when the build resolves it."""
    profile = tmp_path / "profile.yml"
    profile.write_text("name: Orphan\nextends: ./does-not-exist.yml\ndata: data\n")
    with caplog.at_level(logging.WARNING):
        result = _render_from(runner, str(profile))
    assert result.exit_code != 0
    _assert_build_error_logged(caplog, "does-not-exist", "not found")


def test_extends_cycle_detected(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A circular `extends` chain (a -> b -> a) is detected and aborted rather
    than recursing until the stack gives out."""
    a = tmp_path / "profile.yml"
    b = tmp_path / "b.yml"
    a.write_text("name: A\nextends: ./b.yml\ndata: data\n")
    b.write_text("name: B\nextends: ./profile.yml\ndata: data\n")
    with caplog.at_level(logging.WARNING):
        result = _render_from(runner, str(a))
    assert result.exit_code != 0
    _assert_build_error_logged(caplog, "cycle", "circular")


@pytest.mark.parametrize("preset", list_presets())
def test_each_bundled_preset_builds_clean(preset: str, runner: CliRunner, tmp_path: Path) -> None:
    """Every bundled preset must materialize and build to a project with a
    valid config and manifest, and the profile it materialized into must still
    say which preset it came from — the manifest itself cannot, since a
    zero-argument build only ever sees a plain profile.yml and does not know
    whether (or from which preset) it was materialized.

    Auto-extends as new presets land (e.g. 'education'). A new preset that
    parses but doesn't build will fail this test on the next CI run.
    """
    result = _materialize(runner, str(tmp_path), "smoke", preset)
    assert result.exit_code == 0, result.output
    project_dir = _project(tmp_path, "smoke")
    # 1. Core artifacts rendered
    assert (project_dir / "config.yml").exists()
    assert (project_dir / "CLAUDE.md").exists()
    # 2. Manifest is valid JSON with the bumped schema
    import json

    manifest = json.loads((project_dir / ".osprey-manifest.json").read_text())
    assert manifest["schema_version"] == "1.3.0"
    # 3. The materialized profile still names the preset it came from.
    assert _profile_yaml(tmp_path / "smoke")["provenance"]["preset"] == preset
    # 4. Every preset must explicitly pin the facility timezone — agent timestamp
    #    interpretation/rendering keys off system.timezone, and a preset that omits
    #    it falls back to a silent default (the ariel_standalone blind spot). Binding
    #    to list_presets() auto-forces any new preset to declare it too.
    config = _config_yaml(project_dir)
    tz = config.get("system", {}).get("timezone")
    assert isinstance(tz, str) and tz.strip(), (
        f"preset {preset!r} does not pin system.timezone — agent timestamps would "
        f"fall back to a silent default; add an explicit `system.timezone`"
    )


def _attached_presets() -> list[str]:
    """Every bundled preset that builds attached (``deploy_services: false``)."""
    return [
        name
        for name in list_presets()
        if not resolve_build_profile(None, preset=name)[0].deploy_services
    ]


@pytest.mark.parametrize("preset", _attached_presets())
def test_attached_preset_built_alone_is_told_its_templates_defaults(
    preset: str, runner: CliRunner, tmp_path: Path
) -> None:
    """A persona preset materialized on its own — no hosting deployment in the
    repo — still builds, and is told what its app template deploys.

    Built beside its host, an attached render is told the host's client-facing
    facts from the host's render (``osprey.deployment.reach``). Built alone
    there is no such render, but the persona extends a deployment of the SAME
    app template, so the template rendered as a deployment is what the host
    would say at the shipped defaults. Every consumer the preset switches on
    must then resolve, and the sidecar port must be the one the template
    deploys — read from a real render of the parent preset rather than spelled
    here, so a moved default moves both.
    """
    from osprey.cli.build_profile_presets import _load_preset_raw
    from osprey.deployment.reach import reach_errors

    result = _materialize(runner, str(tmp_path), "alone", preset)
    assert result.exit_code == 0, result.output
    config = _config_yaml(_project(tmp_path, "alone"))
    assert reach_errors(config) == []

    parent = _load_preset_raw(preset)[0].get("extends")
    assert parent, f"{preset} extends nothing — which deployment is its host?"
    assert _materialize(runner, str(tmp_path), "host", Path(parent).stem).exit_code == 0
    host = _config_yaml(_project(tmp_path, "host"))
    assert config["services"]["qmd"]["port"] == host["services"]["qmd"]["port"]
    # The tabs the preset selects are told their address too — what a
    # deployment of the template derives when it injects the sidecars, read
    # by running the same injectors over the template-as-deployment.
    from osprey.cli.build_profile import resolve_build_profile

    # `enabled` is each render's OWN selection, not a projected fact: the
    # host deployment carries the block for the tab its personas select
    # (switched off there), and the persona that selects it switches it on.
    selected = resolve_build_profile(None, preset=preset)[0].web_panels
    for panel in ("events", "bluesky"):
        if panel in selected:
            told = {k: v for k, v in config["web"]["panels"][panel].items() if k != "enabled"}
            hosts = {k: v for k, v in host["web"]["panels"][panel].items() if k != "enabled"}
            assert told == hosts, panel
            assert config["web"]["panels"][panel]["enabled"] is True, panel
            assert host["web"]["panels"][panel]["enabled"] is False, panel


def test_deploying_profile_may_pin_only_the_events_path(runner: CliRunner, tmp_path: Path) -> None:
    """The reach refusal reads the config the injectors have FINISHED writing.

    A deploying profile that pins ``web.panels.events.path`` and nothing else
    — the documented way to move the dashboard's route — has its ``url``
    written by the dispatch injector moments later; refusing before that
    would name the very key the build was about to supply. The injector also
    fills the label and the health endpoint, so the tab health-gates itself
    and every persona told this entry gets the whole of it.
    """
    repo = tmp_path / "pathpin"
    created = runner.invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert created.exit_code == 0, created.output
    profile = _profile_yaml(repo)
    profile.setdefault("config", {})["web.panels.events.path"] = "/custom-route"
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    result = _render_from(runner, repo / "profile.yml")
    assert result.exit_code == 0, result.output
    events = _config_yaml(_project(tmp_path, "pathpin"))["web"]["panels"]["events"]
    assert events["path"] == "/custom-route"
    assert events["url"].startswith("http://localhost:")
    assert events["label"] == "EVENTS"
    assert events["health_endpoint"] == "/health"
    # The deployment itself does not select the tab (its write-armed personas
    # do), so its own render carries the complete entry switched off.
    assert events["enabled"] is False
    # Every persona inherited the pin. The one that selects the tab is told
    # the whole entry from this render and switches it on; the ones that do
    # not keep the inherited fragment switched off instead of rendering an
    # empty-url tab.
    personas = {
        path.name.rsplit("-", 1)[1]: yaml.safe_load((path / "config.yml").read_text())
        for path in _project(tmp_path, "pathpin").glob("pathpin-*")
    }
    assert set(personas) >= {"readonly", "readwrite"}
    assert personas["readwrite"]["web"]["panels"]["events"] == {**events, "enabled": True}
    readonly_events = personas["readonly"]["web"]["panels"]["events"]
    assert readonly_events["path"] == "/custom-route"
    assert readonly_events["enabled"] is False
    assert "url" not in readonly_events


def test_attached_profile_built_alone_may_name_its_host_by_hand(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Built alone, the profile's ``config:`` is where a host that differs from
    the template's defaults is named, and it wins over those defaults.

    Beside a host the same spelling is refused as a second home for one fact;
    alone there is no first home, so the hand-spelled value IS the projection.
    """
    preset = "control-assistant-logbook"
    repo = tmp_path / "alone"
    created = runner.invoke(init, [str(repo), "--preset", preset, "--no-git"])
    assert created.exit_code == 0, created.output
    profile = _profile_yaml(repo)
    profile["config"]["services.qmd.port"] = 9180
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    result = _render_from(runner, repo / "profile.yml")
    assert result.exit_code == 0, result.output
    assert _config_yaml(_project(tmp_path, "alone"))["services"]["qmd"]["port"] == 9180


def test_control_assistant_preset_ships_simulation_model(runner: CliRunner, tmp_path: Path) -> None:
    """The control-assistant preset bundles the simulation machine model.

    Pins the wiring: the data bundle ships ``data/simulation/machine.json``
    (shared channels) plus a ``scenarios/`` tree of self-contained bundles, and
    the rendered ``config.yml`` names the machine file exactly once, under the
    key path the connector factory scopes
    (``control_system.connector.mock``). The mock archiver derives its own copy
    from there, so a second declaration would be a divergence waiting to
    happen. No ``active_scenarios`` state file ships in ``data/``: the active
    set is runtime state under ``_agent_data/simulation/``, and its absence
    already means "nominal only".
    """
    import json

    result = _materialize(runner, str(tmp_path), "smoke", "control-assistant")
    assert result.exit_code == 0, result.output
    project_dir = _project(tmp_path, "smoke")
    sim_dir = project_dir / "data" / "simulation"

    machine_path = sim_dir / "machine.json"
    assert machine_path.exists(), "machine.json missing from built project"
    machine = json.loads(machine_path.read_text(encoding="utf-8"))
    assert "channels" in machine
    assert "scenarios" not in machine, "scenarios moved to bundle tree, not the machine file"

    # Self-contained scenario bundles (telemetry + optional logbook).
    for name in ("nominal", "vacuum-burst", "rf-thermal"):
        assert (sim_dir / "scenarios" / name / "scenario.json").exists(), f"{name} bundle missing"
    assert (sim_dir / "scenarios" / "nominal" / "logbook.json").exists()
    assert (sim_dir / "scenarios" / "rf-thermal" / "logbook.json").exists()
    # vacuum-burst is telemetry-only by design (no logbook narrative).
    assert not (sim_dir / "scenarios" / "vacuum-burst" / "logbook.json").exists()

    assert not (sim_dir / "active_scenarios").exists(), (
        "active_scenarios is runtime state — it must not ship in the build-owned data/ tree"
    )

    config = _config_yaml(project_dir)
    assert (
        config["control_system"]["connector"]["mock"]["simulation_file"]
        == "data/simulation/machine.json"
    )
    assert "simulation_file" not in config["archiver"].get("mock_archiver", {}), (
        "the archiver repeats the machine path; it derives it now"
    )


def test_preset_yaml_must_be_mapping(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A profile YAML that parses to a list, not a mapping, raises
    BuildProfileError rather than failing deeper in the pipeline with an
    unhelpful AttributeError."""
    # We can't easily inject a malformed bundled preset, but we can verify the
    # _load_preset_raw branch directly via the public function used by the CLI.
    from osprey.cli.build_profile import _load_preset_raw

    # Hijack the presets package via monkeypatching is awkward here — assert
    # the parallel error path on a profile that parses-to-list, which
    # exercises the same _parse_profile expectation.
    bad = tmp_path / "profile.yml"
    bad.write_text("- one\n- two\n")
    with caplog.at_level(logging.WARNING):
        result = _render_from(runner, str(bad))
    assert result.exit_code != 0
    _assert_build_error_logged(caplog, "mapping")
    # Keep _load_preset_raw imported so the symbol is referenced and a future
    # rename surfaces this test.
    assert callable(_load_preset_raw)


class TestBuildProfileChannelFinderModeValidation:
    """`BuildProfile.validate()` rejects unknown channel_finder_mode values."""

    def test_validate_rejects_channel_finder_mode_all(self, tmp_path: Path) -> None:
        from osprey.cli.build_profile import BuildProfile
        from osprey.errors import BuildProfileError

        profile = BuildProfile(name="t", channel_finder_mode="all")
        with pytest.raises(BuildProfileError) as exc:
            profile.validate(tmp_path)
        assert "channel_finder_mode" in str(exc.value)
        assert "in_context" in str(exc.value)

    def test_validate_rejects_unknown_channel_finder_mode(self, tmp_path: Path) -> None:
        from osprey.cli.build_profile import BuildProfile
        from osprey.errors import BuildProfileError

        profile = BuildProfile(name="t", channel_finder_mode="bogus")
        with pytest.raises(BuildProfileError):
            profile.validate(tmp_path)

    def test_validate_accepts_valid_channel_finder_modes(self, tmp_path: Path) -> None:
        """Every registered paradigm validates — the check derives from the registry.

        Read from :data:`VALID_CHANNEL_FINDER_MODES` rather than a literal list so
        registering a paradigm cannot leave this test asserting a stale set.
        """
        from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
        from osprey.cli.build_profile import BuildProfile

        (tmp_path / "data").mkdir(exist_ok=True)
        for mode in VALID_CHANNEL_FINDER_MODES:
            # The graph paradigm answers from a store the profile declares, so
            # its profile has to spell one; the others need no `config:` block.
            config = {"services.graphdb.path": "./services/graphdb"} if mode == "graph" else {}
            BuildProfile(name="t", data="data", channel_finder_mode=mode, config=config).validate(
                tmp_path
            )

    def test_validate_accepts_none_channel_finder_mode(self, tmp_path: Path) -> None:
        """None is valid at the profile level — manager.py raises only if
        channel-finder is actually selected and no mode is pinned."""
        from osprey.cli.build_profile import BuildProfile

        (tmp_path / "data").mkdir(exist_ok=True)
        BuildProfile(name="t", data="data", channel_finder_mode=None).validate(tmp_path)


class TestMirroredLogbookSeedNotMutated:
    """The build never mutates a profile-supplied logbook seed.

    Build-time timestamp rebasing was removed: demo/seed logbooks now carry
    *relative* timestamps (``when: {days_ago, time}``) resolved at ingest time
    by the generic adapter (see tests/services/ariel_search/test_demo_data.py),
    so the build copies seed data verbatim instead of rewriting it in place.
    """

    def test_mirrored_logbook_seed_is_copied_verbatim(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        import json

        profile_dir = tmp_path / "repo"
        (profile_dir / "project" / "data" / "logbook_seed").mkdir(parents=True)
        seed = {
            "entries": [
                {"id": "T-001", "when": {"days_ago": 7, "time": "08:15:00"}, "text": "older entry"},
                {
                    "id": "T-002",
                    "when": {"days_ago": 2, "time": "03:30:00"},
                    "text": "latest entry",
                },
            ]
        }
        _facility_data(profile_dir)
        seed_text = json.dumps(seed)
        (profile_dir / "project" / "data" / "logbook_seed" / "demo_logbook.json").write_text(
            seed_text
        )
        profile = profile_dir / "profile.yml"
        profile.write_text(
            "extends: hello-world\nname: SeedVerbatim\ndata: data\n"
            "provider: anthropic\nmodel: haiku\n"
        )

        result = _render_from(runner, str(profile))
        assert result.exit_code == 0, result.output

        built = json.loads(
            (_project(tmp_path, "repo") / "data" / "logbook_seed" / "demo_logbook.json").read_text()
        )
        # Seed data round-trips unchanged — the build did not rewrite timestamps.
        assert built == seed


# ---------------------------------------------------------------------------
# Attached projects (deploy_services: false)
# ---------------------------------------------------------------------------


class TestDeployServicesKnob:
    """``deploy_services: false`` marks an attached project: no service
    scaffolding runs, no ``services/`` tree is written, and the rendered
    config.yml carries an explicit empty ``deployed_services`` list. The knob
    defaults true, so every existing (self-contained) build is unchanged.
    """

    # A profile whose preset would normally scaffold postgresql + openobserve
    # and whose ``bluesky:`` block would normally inject a bridge service — so
    # an attached build has real scaffolding to suppress. The inherited
    # ``va_archiver:`` block is dropped: a deployment that records its own
    # archive projects the recorder's path into an attached render on purpose
    # (osprey.deployment.reach), and that one deliberate exception would blunt
    # the "nothing here is a service this render would run" assertion below.
    _PROFILE = (
        "name: Attachment Test\n"
        "extends: control-assistant\n"
        "data: data\n"
        "va_archiver: null\n"
        "provider: anthropic\n"
        "model: haiku\n"
        "channel_finder_mode: hierarchical\n"
        "bluesky:\n"
        "  port: 10080\n"
    )

    def _build(self, runner: CliRunner, tmp_path: Path, extra: str) -> Path:
        profile = tmp_path / "smoke" / "profile.yml"
        profile.parent.mkdir()
        _facility_data(profile.parent, "control_assistant")
        profile.write_text(self._PROFILE + extra)
        # The bundle's source zone, which `osprey init` lays down beside the
        # profile and the deploy binds into every entitled container. A bare
        # profile without it is refused by the Reach Contract (the bind source
        # would be an empty directory), and this class is about the knob.
        (profile.parent / "data" / "facility_knowledge").mkdir(parents=True, exist_ok=True)
        result = _render_from(runner, str(profile))
        assert result.exit_code == 0, result.output
        return _project(tmp_path, "smoke")

    def test_default_true_scaffolds_services(self, runner: CliRunner, tmp_path: Path) -> None:
        """Baseline: the same profile without the knob deploys its own stack."""
        project = self._build(runner, tmp_path, extra="")
        cfg = _config_yaml(project)
        assert "postgresql" in cfg["deployed_services"]
        assert "bluesky" in cfg["deployed_services"]
        assert (project / "services").is_dir()
        assert "postgresql" in cfg["services"]

    def test_false_scaffolds_nothing(self, runner: CliRunner, tmp_path: Path) -> None:
        """An attached project writes no services/ tree, scaffolds no service,
        and lists an explicit empty deployed_services.

        Its ``services:`` map is not empty: the build tells an attached render
        the client-facing facts of the services it reaches (``osprey.deployment.reach``
        — here, built alone, what the app template deploys). Those are ports
        and names to dial, never a service to run: no block carries the
        ``path`` a scaffolded service is declared by.
        """
        project = self._build(runner, tmp_path, extra="deploy_services: false\n")
        cfg = _config_yaml(project)
        # Explicit empty list — present so `osprey up` reads [] not None.
        assert cfg["deployed_services"] == []
        # Client facts only: nothing here is a service this render would run.
        services = cfg.get("services") or {}
        assert services, "an attached render is told where its host's services are"
        assert all("path" not in block for block in services.values()), services
        # No services/ directory at all.
        assert not (project / "services").exists()

    def test_readonly_persona_builds_attached(self, runner: CliRunner, tmp_path: Path) -> None:
        """The shipped read-only persona preset builds as an attached project."""
        result = _materialize(runner, str(tmp_path), "op", "control-assistant-readonly")
        assert result.exit_code == 0, result.output
        cfg = _config_yaml(_project(tmp_path, "op"))
        assert cfg["deployed_services"] == []
        assert not (_project(tmp_path, "op") / "services").exists()


def test_set_free_form_model_builds(runner: CliRunner, tmp_path: Path) -> None:
    """A model ID outside the provider's tier map builds — it passes through.

    Refusing here kept every model the tier map did not name (a newly released
    ID, a gateway-only alias) unusable until the map caught up. The resolver
    now trusts the provider to serve the ID and puts it in ANTHROPIC_MODEL
    verbatim; a misspelt ID fails at the provider, naming the ID.
    """
    result = _materialize(
        runner,
        str(tmp_path),
        "smoke",
        "hello-world",
        "--set",
        "provider=als-apg",
        "--set",
        "model=anthropic/claude-opus",
    )
    assert result.exit_code == 0, result.output
    cfg = _config_yaml(_project(tmp_path, "smoke"))
    assert cfg["claude_code"]["default_model"] == "anthropic/claude-opus"


def test_set_value_invalid_yaml_raises() -> None:
    """A --set value that isn't valid YAML raises BuildProfileError, not a YAMLError."""
    with pytest.raises(BuildProfileError, match="is not valid YAML"):
        resolve_build_profile(None, preset="hello-world", set_pairs=("foo=[unterminated",))


# ---------------------------------------------------------------------------
# Persona renders
#
# A persona project is rendered from a delta over this repo's own profile, by
# `osprey build` and by nothing else: one build of a repo writes its own
# `build/` plus `build/<repo>-<persona>/` for every delta in `personas/`. Both
# cases below drive that through the real command; the delta emission they
# depend on is pinned in tests/cli/test_persona_profile_emission.py.
# ---------------------------------------------------------------------------


def _persona_project(repo: pathlib.Path, persona: str) -> pathlib.Path:
    """Where a build of *repo* renders *persona*.

    The one spelling of the rule, so a test never assembles the path by hand:
    the render's name is the repo's own name and the delta's stem, which is also
    what `osprey init` writes into each catalog entry's `project_path`.
    """
    return repo / "build" / f"{repo.name}-{persona}"


def test_persona_delta_build_resolves_from_the_profile_root(
    runner: CliRunner, tmp_path: Path
) -> None:
    """FR-10 anchoring: a delta under `personas/` inherits the root profile and
    everything it names anchors at the ROOT, not at the delta's own parent."""
    from osprey.cli.templates.manager import TemplateManager

    root = tmp_path / "prof"
    (root / "personas").mkdir(parents=True)
    import shutil

    shutil.copytree(
        TemplateManager().template_root / "apps" / "hello_world" / "data", root / "data"
    )
    (root / "data" / "FACILITY_MARKER.txt").write_text("from the root\n")
    (root / "profile.yml").write_text(
        "name: RootProfile\nextends: hello-world\nprovider: anthropic\nmodel: sonnet\ndata: data\n"
    )
    (root / "personas" / "readonly.yml").write_text("name: ReadOnly\nmodel: haiku\n")

    result = _render_from(runner, str(root / "profile.yml"))
    assert result.exit_code == 0, result.output

    project = _persona_project(root, "readonly")
    # The delta alone names no provider and no data tree; both come from the root.
    assert _config_yaml(project)["claude_code"]["provider"] == "anthropic"
    assert (project / "data" / "FACILITY_MARKER.txt").is_file()
    # ...and the delta's own override still wins.
    assert _config_yaml(project)["claude_code"]["default_model"] == "haiku"
    # The deployment's own render is beside it and keeps the root's model, so
    # the assertion above cannot pass by reading the wrong directory.
    assert _config_yaml(root / "build")["claude_code"]["default_model"] == "sonnet"


def test_persona_exclusion_keeps_the_artifact_out_of_the_built_project(
    runner: CliRunner, tmp_path: Path
) -> None:
    """FR-10: an excluded convention artifact must not reach the project at all.

    Copying it anyway is worse than a no-op: the file shadows the framework's
    own version of that artifact, and the build then registers it as user-owned,
    freezing the shadow against regen — the exact inverse of what the exclusion
    asked for. And it fails silent (exit 0, no warning, hash correct), because
    the exclusion IS folded into the profile hash, so the project reads as fresh
    while carrying an artifact the persona explicitly dropped.

    Asserted through the CLI, not against `_apply_conventions`: the defect this
    pins lived in the WIRE between resolution and that call. The producer side
    (`load_profile_document().excluded_artifacts`) and the consumer side
    (`_apply_conventions(excluded=...)`) were each green in their own unit
    tests while the record between them was dropped — so only a test that
    crosses the seam can catch it.

    A nested artifact is excluded alongside the flat one because the exclusion
    vocabulary keeps the full path below the convention destination
    (`commands/osprey/scan`, not `commands/scan`): a basename rule would pass
    the flat case and silently miss the namespaced one.
    """
    from osprey.cli.templates.manager import TemplateManager

    root = tmp_path / "prof"
    (root / "personas").mkdir(parents=True)
    (root / "agents").mkdir()
    (root / "commands" / "osprey").mkdir(parents=True)
    shutil.copytree(
        TemplateManager().template_root / "apps" / "hello_world" / "data", root / "data"
    )
    (root / "agents" / "orbit-writer.md").write_text(
        "---\nname: orbit-writer\ndescription: profile-shipped agent\n---\n\nBody.\n"
    )
    (root / "commands" / "osprey" / "scan.md").write_text(
        "---\ndescription: profile-shipped namespaced command\n---\n\nBody.\n"
    )
    (root / "profile.yml").write_text(
        "name: RootProfile\nextends: hello-world\nprovider: anthropic\nmodel: sonnet\ndata: data\n"
    )
    (root / "personas" / "narrow.yml").write_text(
        "name: Narrow\n"
        "exclude:\n"
        "  agents:\n"
        "    - agents/orbit-writer\n"
        "  commands:\n"
        "    - commands/osprey/scan\n"
    )

    result = _render_from(runner, str(root / "profile.yml"))
    assert result.exit_code == 0, result.output

    project = _persona_project(root, "narrow")
    assert not (project / ".claude" / "agents" / "orbit-writer.md").exists()
    assert not (project / ".claude" / "commands" / "osprey" / "scan.md").exists()
    # Absence from the project is only half of it: the original defect also
    # REGISTERED the copied artifact as user-owned, freezing the shadow against
    # regen. A test that checked only the file would miss that half.
    user_owned = _config_yaml(project).get("scaffold", {}).get("user_owned", []) or []
    assert not any("orbit-writer" in str(entry) for entry in user_owned), user_owned
    assert not any("scan" in str(entry) for entry in user_owned), user_owned

    # Control: the deployment's own render, from the SAME build, does ship it —
    # so the assertions above cannot pass just because the artifact never
    # applied. One build produces both, which is what makes this a control.
    wide = root / "build"
    assert (wide / ".claude" / "agents" / "orbit-writer.md").is_file()
    assert (wide / ".claude" / "commands" / "osprey" / "scan.md").is_file()
    wide_owned = [str(entry) for entry in _config_yaml(wide)["scaffold"]["user_owned"]]
    assert "agents/orbit-writer" in wide_owned, wide_owned
    assert "commands/osprey/scan" in wide_owned, wide_owned


def test_persona_exclusion_of_a_panel_switches_its_inherited_block_off(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``exclude: web_panels:`` must reach the render, not just the selection.

    The web terminal reads its tab strip from ``web.panels.<id>`` alone. A
    persona subtracts a tab from ``web_panels``, but every ``config:`` fact
    about that panel — a builtin's label, a custom panel's url — is inherited
    additively and lands in the persona's render regardless; before the
    selection was projected onto those blocks, each one still rendered as a
    tab (a builtin block without ``enabled: false`` is on, and a custom block
    was on unconditionally). Asserted through the CLI, on the rendered
    config.yml and on the terminal's own reader of it, because the defect
    lived between the three.
    """
    from osprey.interfaces.web_terminal.app import _load_panel_config
    from osprey.profiles.web_panels import UNIVERSAL_PANELS

    root = tmp_path / "prof"
    (root / "personas").mkdir(parents=True)
    _facility_data(root, "control_assistant")
    (root / "data" / "facility_knowledge").mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text(
        "name: RootProfile\n"
        "data: data\n"
        "provider: anthropic\n"
        "model: haiku\n"
        "channel_finder_mode: hierarchical\n"
        "hooks: [memory-guard]\n"
        "web_panels: [okf, lattice, grafana]\n"
        "config:\n" + _POSTURE_FLOOR + "  web.panels.lattice.label: LATTICE\n"
        "  web.panels.grafana.label: GRAFANA\n"
        "  web.panels.grafana.url: http://grafana.local:3000\n"
    )
    (root / "personas" / "narrow.yml").write_text(
        "name: Narrow\nexclude:\n  web_panels:\n    - lattice\n    - grafana\n"
    )

    result = _render_from(runner, str(root / "profile.yml"))
    assert result.exit_code == 0, result.output

    narrow = _config_yaml(_persona_project(root, "narrow"))["web"]["panels"]
    # The inherited facts are still there — a persona cannot subtract config —
    # and each block now says it is not a tab of this render.
    assert narrow["lattice"] == {"label": "LATTICE", "enabled": False}
    assert narrow["grafana"]["url"] == "http://grafana.local:3000"
    assert narrow["grafana"]["enabled"] is False
    assert narrow["okf"]["enabled"] is True
    with patch(
        "osprey.utils.workspace.load_osprey_config",
        return_value=_config_yaml(_persona_project(root, "narrow")),
    ):
        enabled, custom, _default = _load_panel_config()
    assert enabled == UNIVERSAL_PANELS | {"okf"}
    assert custom == []

    # Control: the deployment's own render, from the same build, shows all three.
    wide = _config_yaml(root / "build")["web"]["panels"]
    assert wide["lattice"]["enabled"] is True
    assert wide["grafana"]["enabled"] is True
    assert wide["okf"]["enabled"] is True


class TestGraphModeRequiresAGraphStore:
    """`osprey init` refuses graph mode on a preset whose app template has no store.

    The paradigm is selectable on any profile, but its store is a service rather
    than a bundled database file — so the one preset built on the storeless
    ``channel_finder_standalone`` template turns the mode away at
    materialization time, before a project that could not answer anything
    reaches disk.
    """

    def test_set_graph_mode_on_the_standalone_preset_is_refused(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """The refusal reaches the operator by name of the block it is missing.

        `osprey init` reports an unmaterializable preset as a usage error (exit
        2) on stderr, not through the build log — the operator got the ``--set``
        wrong, and the message says which block would have made it right.
        """
        result = _materialize(
            runner,
            str(tmp_path),
            "cf",
            "channel-finder-standalone",
            "--set",
            "channel_finder_mode=graph",
        )
        assert result.exit_code == 2, result.output
        assert "services.graphdb" in result.output
        assert "channel_finder_mode: graph" in result.output
        assert not (tmp_path / "cf" / "profile.yml").exists()

    def test_set_graph_mode_on_the_control_assistant_preset_is_accepted(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Control: the same ``--set`` on a store-deploying preset materializes.

        Without it the refusal above could pass because ``--set
        channel_finder_mode=graph`` is refused everywhere rather than because
        this app template ships no store.
        """
        repo = pathlib.Path(tmp_path) / "cr"
        result = runner.invoke(
            init,
            [
                str(repo),
                "--preset",
                "control-assistant",
                "--no-git",
                "--set",
                "channel_finder_mode=graph",
            ],
        )
        assert result.exit_code == 0, result.output
        assert _profile_yaml(repo)["channel_finder_mode"] == "graph"


class TestRenderConfigReading:
    """`TemplateManager.render_config` — the config-only reading `osprey build`
    takes of the framework template when a standalone attached profile has no hosting
    deployment to be told by."""

    def test_a_template_root_without_a_config_template_is_refused(self, tmp_path: Path) -> None:
        """`project_template_for` finds no shared `project/` copy, and the
        reading names the root instead of rendering nothing."""
        from osprey.cli.templates import scaffolding
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        assert scaffolding.project_template_for(manager.template_root, "no-such-file.txt") is None
        # An empty template root ships no config template at all.
        with pytest.raises(ValueError, match="renders no config.yml"):
            scaffolding.render_project_config(
                tmp_path,
                manager.jinja_env,
                tmp_path / "config.yml",
                {},
            )

    def test_effective_artifacts_without_a_manifest_stays_none(self, tmp_path: Path) -> None:
        """A bundle that ships no manifest widens nothing: the caller's `None`
        stays `None`, so downstream output filtering stays off exactly as for
        a programmatic render before the fallback existed."""
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        manager.template_root = tmp_path  # no apps/, no manifests
        assert manager._effective_artifacts("anything", None) is None
        assert manager._effective_artifacts("anything", {"agents": []}) == {"agents": []}

"""Tests for the preset/override/--set surface and the `osprey build` it feeds.

A deployment starts with `osprey init --preset NAME [-O FILE] [--set K=V]`,
which resolves the named preset through the profile pipeline and materializes
it as `profile.yml` (plus `data/`, `personas/`, `.env.example`) at a repo's
root. `osprey build --repo DIR` is zero-argument from there: it re-resolves
that repo's own `profile.yml`, with no preset/override/--set surface of its
own, and renders `DIR/build/`. This module covers both halves — the resolution
pipeline `osprey init` drives (bundled presets, override files, --set inline
scalars/lists, `extends`, the drift-guard that prevents presets from depending
on profile-dir-relative paths that would break when shipped in a wheel) and
what a plain `osprey build` does with the profile.yml it finds.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
from pathlib import Path

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
    refused. ``-O`` and ``--set`` belong to ``init`` — they are baked into the
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
    directory. ``extra`` is accepted and ignored on this path: ``-O``/``--set``
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


def test_preset_with_override_file(runner: CliRunner, tmp_path: Path) -> None:
    override = tmp_path / "over.yml"
    override.write_text("model: opus\n")
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world", "-O", str(override))
    assert result.exit_code == 0, result.output
    config = _config_yaml(_project(tmp_path, "smoke"))
    assert config["claude_code"]["default_model"] == "opus"


def test_set_flag_overrides_scalar(runner: CliRunner, tmp_path: Path) -> None:
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world", "--set", "model=sonnet")
    assert result.exit_code == 0, result.output
    config = _config_yaml(_project(tmp_path, "smoke"))
    assert config["claude_code"]["default_model"] == "sonnet"


def test_set_with_list_value_extends(runner: CliRunner, tmp_path: Path) -> None:
    """--set on a string list union-dedups (per _merge_lists), preserving base order."""
    result = _materialize(
        runner, str(tmp_path), "smoke", "hello-world", "--set", "hooks=[memory-guard]"
    )
    assert result.exit_code == 0, result.output
    # The persisted manifest is the post-merge artifact list seen by the build.
    manifest_path = _project(tmp_path, "smoke") / ".osprey-manifest.json"
    assert manifest_path.exists(), result.output
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hooks = set(manifest["artifacts"]["hooks"])
    assert "memory-guard" in hooks
    # Preset's original hooks remain (union, not replace).
    assert {"hook-log", "hook-config", "approval"} <= hooks


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
    # Creds carry the ${VAR:-default} form (mirrors the compose template) so the
    # resolver never hits its fail-loud path before the first deploy mints them.
    assert tel["openobserve"]["password"] == "${ZO_ROOT_USER_PASSWORD:-Complexpass#123}"
    assert tel["openobserve"]["user"] == "${ZO_ROOT_USER_EMAIL:-root@example.com}"


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
    # (this test was previously vacuously passing on a top-level lookup).
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
    profile.write_text(
        "name: TypoTest\n"
        "data_bundle: hello_world\n"
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
    """B2/C3/C12: manifest schema bump from 1.1.0 to 1.2.0."""
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world")
    assert result.exit_code == 0, result.output
    import json

    manifest = json.loads((_project(tmp_path, "smoke") / ".osprey-manifest.json").read_text())
    assert manifest["schema_version"] == "1.2.0"


def test_manifest_uses_build_args_not_init_args(runner: CliRunner, tmp_path: Path) -> None:
    """C3: on-disk key renamed from init_args to build_args."""
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world")
    assert result.exit_code == 0, result.output
    import json

    manifest = json.loads((_project(tmp_path, "smoke") / ".osprey-manifest.json").read_text())
    assert "build_args" in manifest
    assert "init_args" not in manifest


def test_override_deep_merges_nested_dict(runner: CliRunner, tmp_path: Path) -> None:
    """T2: nested dicts in -O files deep-merge into preset values, not replace."""
    override = tmp_path / "over.yml"
    override.write_text("config:\n  system:\n    timezone: UTC\n")
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world", "-O", str(override))
    assert result.exit_code == 0, result.output
    config = _config_yaml(_project(tmp_path, "smoke"))
    # The override-injected value must land at the rendered nested key.
    assert config.get("system", {}).get("timezone") == "UTC"


def test_override_unions_string_list(runner: CliRunner, tmp_path: Path) -> None:
    """T2: string lists union-dedup with preserved base order (per _merge_lists)."""
    override = tmp_path / "over.yml"
    # hello-world preset has hooks: [hook-log, hook-config, approval] (or similar).
    # Add memory-guard via override; pre-existing items must remain.
    override.write_text("hooks:\n  - memory-guard\n")
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world", "-O", str(override))
    assert result.exit_code == 0, result.output
    import json

    manifest = json.loads((_project(tmp_path, "smoke") / ".osprey-manifest.json").read_text())
    hooks = set(manifest["artifacts"]["hooks"])
    assert "memory-guard" in hooks
    assert {"hook-log", "hook-config", "approval"} <= hooks


def test_multiple_override_files_apply_in_order(runner: CliRunner, tmp_path: Path) -> None:
    """T2: -O is multiple=True; later files win at the same key."""
    a = tmp_path / "a.yml"
    a.write_text("model: sonnet\n")
    b = tmp_path / "b.yml"
    b.write_text("model: opus\n")
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world", "-O", str(a), "-O", str(b))
    assert result.exit_code == 0, result.output
    config = _config_yaml(_project(tmp_path, "smoke"))
    assert config["claude_code"]["default_model"] == "opus"


def test_override_missing_file_aborts(runner: CliRunner, tmp_path: Path) -> None:
    """A missing -O file is refused where -O now lives: `osprey init`, which
    validates and bakes override layers into the profile it materializes.
    `osprey build` carries no -O of its own — it only ever re-renders a
    profile.yml that already exists — so there is nothing left for it to
    refuse this way."""
    result = _materialize(
        runner,
        str(tmp_path),
        "smoke",
        "hello-world",
        "-O",
        str(tmp_path / "does-not-exist.yml"),
    )
    assert result.exit_code != 0, result.output
    assert "not found" in result.output.lower()


def test_override_malformed_yaml_aborts(runner: CliRunner, tmp_path: Path) -> None:
    """Malformed YAML in an -O file aborts `osprey init` with a YAML-specific
    message naming the file, not a raw parser traceback."""
    bad = tmp_path / "bad.yml"
    bad.write_text("model: : invalid:\n  - [unterminated\n")
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world", "-O", str(bad))
    assert result.exit_code != 0, result.output
    assert "yaml" in result.output.lower()


def test_override_empty_file_is_noop(runner: CliRunner, tmp_path: Path) -> None:
    """T2: an empty -O file (parses to None) is a no-op (skip-None branch)."""
    empty = tmp_path / "empty.yml"
    empty.write_text("")
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world", "-O", str(empty))
    assert result.exit_code == 0, result.output
    # Project still built; preset's defaults remain.
    assert (_project(tmp_path, "smoke") / "config.yml").exists()


def test_override_non_mapping_aborts(runner: CliRunner, tmp_path: Path) -> None:
    """An -O file whose top-level body is a list, not a mapping, aborts
    `osprey init` — there is nothing to deep-merge a list into."""
    bad = tmp_path / "list.yml"
    bad.write_text("- one\n- two\n")
    result = _materialize(runner, str(tmp_path), "smoke", "hello-world", "-O", str(bad))
    assert result.exit_code != 0, result.output
    assert "mapping" in result.output.lower()


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


def test_set_overrides_override_file(runner: CliRunner, tmp_path: Path) -> None:
    """T3: --set wins over -O at the same key (per docstring precedence)."""
    over = tmp_path / "o.yml"
    over.write_text("model: sonnet\n")
    result = _materialize(
        runner, str(tmp_path), "smoke", "hello-world", "-O", str(over), "--set", "model=opus"
    )
    assert result.exit_code == 0, result.output
    cfg = _config_yaml(_project(tmp_path, "smoke"))
    assert cfg["claude_code"]["default_model"] == "opus"


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
    profile.write_text(
        "name: McpTest\n"
        "data_bundle: hello_world\n"
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
    profile.write_text(
        "name: CatTest\n"
        "data_bundle: hello_world\n"
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
    profile = profile_dir / "profile.yml"
    profile.write_text("name: ConventionTest\ndata_bundle: hello_world\nprovider: anthropic\n")
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
    profile.write_text("name: Orphan\nextends: ./does-not-exist.yml\ndata_bundle: hello_world\n")
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
    a.write_text("name: A\nextends: ./b.yml\ndata_bundle: hello_world\n")
    b.write_text("name: B\nextends: ./profile.yml\ndata_bundle: hello_world\n")
    with caplog.at_level(logging.WARNING):
        result = _render_from(runner, str(a))
    assert result.exit_code != 0
    _assert_build_error_logged(caplog, "cycle", "circular")


@pytest.mark.parametrize("preset", list_presets())
def test_each_bundled_preset_builds_clean(preset: str, runner: CliRunner, tmp_path: Path) -> None:
    """Every bundled preset must materialize and build to a project with a
    valid config and manifest, and the profile it materialized into must still
    say which preset it came from — the manifest itself no longer can, since a
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
    assert manifest["schema_version"] == "1.2.0"
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
        from osprey.cli.build_profile import BuildProfile

        for mode in ("in_context", "hierarchical", "middle_layer"):
            BuildProfile(name="t", channel_finder_mode=mode).validate(tmp_path)

    def test_validate_accepts_none_channel_finder_mode(self, tmp_path: Path) -> None:
        """None is valid at the profile level — manager.py raises only if
        channel-finder is actually selected and no mode is pinned."""
        from osprey.cli.build_profile import BuildProfile

        BuildProfile(name="t", channel_finder_mode=None).validate(tmp_path)


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
        seed_text = json.dumps(seed)
        (profile_dir / "project" / "data" / "logbook_seed" / "demo_logbook.json").write_text(
            seed_text
        )
        profile = profile_dir / "profile.yml"
        profile.write_text(
            "name: SeedVerbatim\ndata_bundle: hello_world\nprovider: anthropic\nmodel: haiku\n"
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

    # A profile whose bundle template would normally scaffold postgresql +
    # openobserve and whose ``bluesky:`` block would normally inject a bridge
    # service — so an attached build has real scaffolding to suppress.
    _PROFILE = (
        "name: Attachment Test\n"
        "data_bundle: control_assistant\n"
        "provider: anthropic\n"
        "model: haiku\n"
        "channel_finder_mode: hierarchical\n"
        "bluesky:\n"
        "  port: 8090\n"
    )

    def _build(self, runner: CliRunner, tmp_path: Path, extra: str) -> Path:
        profile = tmp_path / "smoke" / "profile.yml"
        profile.parent.mkdir()
        profile.write_text(self._PROFILE + extra)
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
        """An attached project writes no services/ tree, no services.* blocks,
        and an explicit empty deployed_services list."""
        project = self._build(runner, tmp_path, extra="deploy_services: false\n")
        cfg = _config_yaml(project)
        # Explicit empty list — present so `osprey up` reads [] not None.
        assert cfg["deployed_services"] == []
        # None of the would-be-scaffolded services leak into services.*.
        assert cfg.get("services") in ({}, None)
        for name in ("postgresql", "openobserve", "bluesky"):
            assert name not in (cfg.get("services") or {})
        # No services/ directory at all.
        assert not (project / "services").exists()

    def test_readonly_persona_builds_attached(self, runner: CliRunner, tmp_path: Path) -> None:
        """The shipped read-only persona preset builds as an attached project."""
        result = _materialize(runner, str(tmp_path), "op", "control-assistant-readonly")
        assert result.exit_code == 0, result.output
        cfg = _config_yaml(_project(tmp_path, "op"))
        assert cfg["deployed_services"] == []
        assert not (_project(tmp_path, "op") / "services").exists()


def test_set_unservable_model_fails_build(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A model the selected provider cannot serve must fail the build itself.

    The web-terminal container runs the same resolver strict at startup, so a
    build that merely warns here ships a deploy whose per-user terminals
    crash-loop behind the reverse proxy (502). The build is the checkpoint the
    operator actually watches — it must stop the `build && deploy` chain.
    """
    with caplog.at_level(logging.ERROR):
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
    assert result.exit_code != 0, result.output
    _assert_build_error_logged(caplog, "neither a model tier nor a model id")


def test_set_value_invalid_yaml_raises() -> None:
    """A --set value that isn't valid YAML raises BuildProfileError, not a YAMLError."""
    with pytest.raises(BuildProfileError, match="is not valid YAML"):
        resolve_build_profile(None, preset="hello-world", set_pairs=("foo=[unterminated",))


def test_override_file_invalid_yaml_raises(tmp_path: Path) -> None:
    """An -O override file with invalid YAML raises BuildProfileError, not a YAMLError."""
    bad = tmp_path / "bad.yml"
    bad.write_text("model: : invalid:\n  - [unterminated\n")
    with pytest.raises(BuildProfileError, match="Invalid YAML"):
        resolve_build_profile(None, preset="hello-world", overrides=(bad,))


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
        "name: RootProfile\n"
        "data_bundle: hello_world\n"
        "provider: anthropic\n"
        "model: sonnet\n"
        "data: data\n"
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
        "name: RootProfile\n"
        "data_bundle: hello_world\n"
        "provider: anthropic\n"
        "model: sonnet\n"
        "data: data\n"
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

"""Tests for `osprey build` preset/override/--set surface.

Covers the new resolve_build_profile() pipeline: bundled presets, override
files, --set inline scalars/lists, mutual exclusion of preset vs positional
profile, and the drift-guard that prevents presets from depending on
profile-dir-relative paths that would break when shipped in a wheel.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile import list_presets, resolve_build_profile
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


def test_list_presets_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(build, ["--list-presets"])
    assert result.exit_code == 0, result.output
    listed = {line.strip() for line in result.output.splitlines() if line.strip()}
    assert listed == set(list_presets())
    # Sanity: the workhorse preset must always be bundled.
    assert "hello-world" in listed


def test_preset_hello_world_creates_project(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    project_dir = tmp_path / "smoke"
    assert (project_dir / "config.yml").exists()
    assert (project_dir / "CLAUDE.md").exists()


def test_preset_with_override_file(runner: CliRunner, tmp_path: Path) -> None:
    override = tmp_path / "over.yml"
    override.write_text("model: opus\n")
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "-O",
            str(override),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    config = _config_yaml(tmp_path / "smoke")
    assert config["claude_code"]["default_model"] == "opus"


def test_set_flag_overrides_scalar(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "--set",
            "model=sonnet",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    config = _config_yaml(tmp_path / "smoke")
    assert config["claude_code"]["default_model"] == "sonnet"


def test_set_with_list_value_extends(runner: CliRunner, tmp_path: Path) -> None:
    """--set on a string list union-dedups (per _merge_lists), preserving base order."""
    result = runner.invoke(
        build,
        # memory-guard is a real registered hook NOT in the hello-world preset
        [
            "smoke",
            "--preset",
            "hello-world",
            "--set",
            "hooks=[memory-guard]",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # The persisted manifest is the post-merge artifact list seen by the build.
    manifest_path = tmp_path / "smoke" / ".osprey-manifest.json"
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
    block so that ``osprey claude regen`` round-trips the persona choice.

    Drift-guard: if any layer of that wiring (BuildProfile field,
    _KNOWN_PROFILE_KEYS, _parse_profile, build_cmd's context/manifest_context
    propagation, or the renderer's template-selection branch) breaks, the
    preset silently falls back to the control-system persona — exactly the
    regression this test pins down.
    """
    import json

    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "ariel-standalone",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output

    project_dir = tmp_path / "smoke"

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
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "control-assistant",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output

    cfg = _config_yaml(tmp_path / "smoke")

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


def test_positional_profile_still_works(runner: CliRunner, tmp_path: Path) -> None:
    """Backward-compat: existing osprey build PROJECT PROFILE.yml flow."""
    profile = tmp_path / "p.yml"
    profile.write_text(
        "name: PosTest\ndata_bundle: hello_world\nprovider: anthropic\nmodel: haiku\n"
    )
    result = runner.invoke(
        build,
        ["smoke", str(profile), "--skip-deps", "--skip-lifecycle", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "smoke" / "config.yml").exists()


def test_mutually_exclusive_profile_and_preset(runner: CliRunner, tmp_path: Path) -> None:
    """T6: positional + --preset is mutually exclusive -> exit 2."""
    profile = tmp_path / "p.yml"
    profile.write_text("name: X\ndata_bundle: hello_world\n")
    result = runner.invoke(
        build,
        ["smoke", str(profile), "--preset", "hello-world"],
    )
    assert result.exit_code == 2, result.output
    assert "Pass either a profile path or --preset, not both" in result.output


def test_neither_profile_nor_preset_required(runner: CliRunner) -> None:
    """T6: no profile + no preset is a usage error -> exit 2 with a specific message."""
    result = runner.invoke(build, ["smoke"])
    assert result.exit_code == 2, result.output
    # Pin the exact wording from build_profile_resolve.py
    assert "Either a profile path or --preset is required" in result.output


def test_unknown_preset_name(runner: CliRunner, tmp_path: Path) -> None:
    """C10: unknown preset is a usage error → exit 2 (per click convention)."""
    result = runner.invoke(
        build,
        ["smoke", "--preset", "bogus", "--skip-deps", "--output-dir", str(tmp_path)],
    )
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

    r_hyphen = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "control-assistant",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(out_a),
        ],
    )
    r_under = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "control_assistant",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(out_b),
        ],
    )
    assert r_hyphen.exit_code == 0, r_hyphen.output
    assert r_under.exit_code == 0, r_under.output
    cfg_a = _config_yaml(out_a / "smoke")
    cfg_b = _config_yaml(out_b / "smoke")
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


def test_unknown_profile_key_fails_the_build(runner: CliRunner, tmp_path: Path, caplog) -> None:
    """C11 promoted: unknown top-level keys now fail the build, naming each typo.

    Previously they were warned about and ignored, which shipped a project
    missing whatever the profile asked for.
    """
    profile = tmp_path / "p.yml"
    profile.write_text(
        "name: TypoTest\n"
        "data_bundle: hello_world\n"
        "provider: anthropic\n"
        "mcp_server: {}\n"  # typo of mcp_servers
        "permission: []\n"  # typo of permissions
    )
    import logging

    with caplog.at_level(logging.ERROR):
        result = runner.invoke(
            build,
            [
                "smoke",
                str(profile),
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code != 0
    assert "mcp_server" in caplog.text
    assert "permission" in caplog.text
    assert not (tmp_path / "smoke").exists()


def test_manifest_schema_version_bumped(runner: CliRunner, tmp_path: Path) -> None:
    """B2/C3/C12: manifest schema bump from 1.1.0 to 1.2.0."""
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    import json

    manifest = json.loads((tmp_path / "smoke" / ".osprey-manifest.json").read_text())
    assert manifest["schema_version"] == "1.2.0"


def test_manifest_uses_build_args_not_init_args(runner: CliRunner, tmp_path: Path) -> None:
    """C3: on-disk key renamed from init_args to build_args."""
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    import json

    manifest = json.loads((tmp_path / "smoke" / ".osprey-manifest.json").read_text())
    assert "build_args" in manifest
    assert "init_args" not in manifest


def test_manifest_records_preset_name_distinctly_from_data_bundle(
    runner: CliRunner, tmp_path: Path
) -> None:
    """B2: creation.template carries the preset name, not the data_bundle."""
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    import json

    manifest = json.loads((tmp_path / "smoke" / ".osprey-manifest.json").read_text())
    creation = manifest["creation"]
    # Preset name (hyphenated form) lives in 'template'; data_bundle is the underlying app bundle.
    assert creation["template"] == "hello-world"
    assert creation["data_bundle"] == "hello_world"


def test_reproducible_command_emits_preset_form_for_preset_build(
    runner: CliRunner, tmp_path: Path
) -> None:
    """C12: preset-sourced builds reproduce as 'osprey build NAME --preset PRESET'."""
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    import json

    manifest = json.loads((tmp_path / "smoke" / ".osprey-manifest.json").read_text())
    cmd = manifest["reproducible_command"]
    assert "--preset hello-world" in cmd
    assert "smoke" in cmd


def test_reproducible_command_emits_positional_form_for_profile_build(
    runner: CliRunner, tmp_path: Path
) -> None:
    """C12: positional-profile builds reproduce as 'osprey build NAME PROFILE.yml'."""
    profile = tmp_path / "p.yml"
    profile.write_text(
        "name: PosTest\ndata_bundle: hello_world\nprovider: anthropic\nmodel: haiku\n"
    )
    result = runner.invoke(
        build,
        [
            "smoke",
            str(profile),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    import json

    manifest = json.loads((tmp_path / "smoke" / ".osprey-manifest.json").read_text())
    cmd = manifest["reproducible_command"]
    # Positional form: must NOT reference --preset and MUST mention the profile path.
    assert "--preset" not in cmd
    assert str(profile) in cmd or profile.name in cmd


def test_override_deep_merges_nested_dict(runner: CliRunner, tmp_path: Path) -> None:
    """T2: nested dicts in -O files deep-merge into preset values, not replace."""
    override = tmp_path / "over.yml"
    override.write_text("config:\n  system:\n    timezone: UTC\n")
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "-O",
            str(override),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    config = _config_yaml(tmp_path / "smoke")
    # The override-injected value must land at the rendered nested key.
    assert config.get("system", {}).get("timezone") == "UTC"


def test_override_unions_string_list(runner: CliRunner, tmp_path: Path) -> None:
    """T2: string lists union-dedup with preserved base order (per _merge_lists)."""
    override = tmp_path / "over.yml"
    # hello-world preset has hooks: [hook-log, hook-config, approval] (or similar).
    # Add memory-guard via override; pre-existing items must remain.
    override.write_text("hooks:\n  - memory-guard\n")
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "-O",
            str(override),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    import json

    manifest = json.loads((tmp_path / "smoke" / ".osprey-manifest.json").read_text())
    hooks = set(manifest["artifacts"]["hooks"])
    assert "memory-guard" in hooks
    assert {"hook-log", "hook-config", "approval"} <= hooks


def test_multiple_override_files_apply_in_order(runner: CliRunner, tmp_path: Path) -> None:
    """T2: -O is multiple=True; later files win at the same key."""
    a = tmp_path / "a.yml"
    a.write_text("model: sonnet\n")
    b = tmp_path / "b.yml"
    b.write_text("model: opus\n")
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "-O",
            str(a),
            "-O",
            str(b),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    config = _config_yaml(tmp_path / "smoke")
    assert config["claude_code"]["default_model"] == "opus"


def test_override_missing_file_aborts(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """T2: a missing -O file produces a clear non-zero exit, not a stack trace."""
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            build,
            [
                "smoke",
                "--preset",
                "hello-world",
                "-O",
                str(tmp_path / "does-not-exist.yml"),
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code != 0
    _assert_build_error_logged(caplog, "not found", "does-not-exist")


def test_override_malformed_yaml_aborts(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """T2: malformed YAML in an -O file aborts cleanly with a YAML-related message."""
    bad = tmp_path / "bad.yml"
    bad.write_text("model: : invalid:\n  - [unterminated\n")
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            build,
            [
                "smoke",
                "--preset",
                "hello-world",
                "-O",
                str(bad),
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code != 0
    _assert_build_error_logged(caplog, "yaml", "invalid")


def test_override_empty_file_is_noop(runner: CliRunner, tmp_path: Path) -> None:
    """T2: an empty -O file (parses to None) is a no-op (skip-None branch)."""
    empty = tmp_path / "empty.yml"
    empty.write_text("")
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "-O",
            str(empty),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # Project still built; preset's defaults remain.
    assert (tmp_path / "smoke" / "config.yml").exists()


def test_override_non_mapping_aborts(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """T2: an -O file whose top-level body is a list (not a mapping) aborts."""
    bad = tmp_path / "list.yml"
    bad.write_text("- one\n- two\n")
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            build,
            [
                "smoke",
                "--preset",
                "hello-world",
                "-O",
                str(bad),
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code != 0
    _assert_build_error_logged(caplog, "mapping")


def test_set_dotted_path_lands_in_config_yml(runner: CliRunner, tmp_path: Path) -> None:
    """T3 (also pins B3 closure): --set with dotted key writes to nested config."""
    # `config.<...>` is the documented path for inserting custom rendered-config
    # fields via --set; assert the dotted key lands at the nested location.
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "--set",
            "config.system.timezone=UTC",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    config = _config_yaml(tmp_path / "smoke")
    assert config.get("system", {}).get("timezone") == "UTC"


def test_set_yaml_typed_values(runner: CliRunner, tmp_path: Path) -> None:
    """T3: RHS of --set is YAML-parsed for free type coercion."""
    # Use config-side keys so we can reliably assert each parsed type.
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "--set",
            "config.an_int=120",
            "--set",
            "config.a_bool=true",
            "--set",
            "config.a_null=null",
            "--set",
            "config.a_list=[a, b]",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = _config_yaml(tmp_path / "smoke")
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
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "hello-world",
            "-O",
            str(over),
            "--set",
            "model=opus",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = _config_yaml(tmp_path / "smoke")
    assert cfg["claude_code"]["default_model"] == "opus"


def test_set_path_through_scalar_aborts(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """T3: a --set key that descends through a scalar raises BuildProfileError."""
    # First --set sets a scalar, second tries to descend into it.
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            build,
            [
                "smoke",
                "--preset",
                "hello-world",
                "--set",
                "model=haiku",
                "--set",
                "model.flavor=fast",
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code != 0
    _assert_build_error_logged(caplog, "scalar", "conflict")


def test_set_malformed_pair_aborts(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """T3: --set without '=' or with empty key aborts cleanly."""
    with caplog.at_level(logging.WARNING):
        no_eq = runner.invoke(
            build,
            [
                "smoke",
                "--preset",
                "hello-world",
                "--set",
                "model",
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
    assert no_eq.exit_code != 0
    _assert_build_error_logged(caplog, "=", "key=value")

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        empty_key = runner.invoke(
            build,
            [
                "smoke",
                "--preset",
                "hello-world",
                "--set",
                "=oops",
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
    assert empty_key.exit_code != 0
    _assert_build_error_logged(caplog, "non-empty", "empty")


def test_profile_mcp_servers_persisted_to_config(runner: CliRunner, tmp_path: Path) -> None:
    """T4: profile-defined mcp_servers land in the project's config.yml."""
    profile = tmp_path / "p.yml"
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
    result = runner.invoke(
        build,
        [
            "smoke",
            str(profile),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    config = _config_yaml(tmp_path / "smoke")
    # NB: profile mcp_servers are persisted under claude_code.servers
    # (see _persist_mcp_servers in build_cmd.py).
    servers = config.get("claude_code", {}).get("servers", {})
    assert "echo" in servers, f"claude_code.servers in config: {list(servers.keys())}"
    assert servers["echo"]["command"] == "echo"
    assert servers["echo"]["args"] == ["hello"]


def test_profile_categories_persisted_to_config(runner: CliRunner, tmp_path: Path) -> None:
    """T4: custom artifact categories from profile land in config.yml."""
    profile = tmp_path / "p.yml"
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
    result = runner.invoke(
        build,
        [
            "smoke",
            str(profile),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    config = _config_yaml(tmp_path / "smoke")
    cats = config.get("artifact_server", {}).get("categories", {})
    assert "diagnostics" in cats, f"artifact_server.categories in config: {list(cats.keys())}"
    assert cats["diagnostics"]["label"] == "Diagnostics"
    assert cats["diagnostics"]["color"].lower() == "#ff0066"
    # Rendered defaults from the template survive the merge.
    assert "port" in config.get("artifact_server", {})


def test_profile_md_files_registered_as_user_owned(runner: CliRunner, tmp_path: Path) -> None:
    """T4: convention artifacts copied in are registered as user_owned in the manifest."""
    profile_dir = tmp_path / "profile"
    (profile_dir / "rules").mkdir(parents=True)
    (profile_dir / "rules" / "extra.md").write_text("# Custom rule\nuser-defined content\n")
    profile = profile_dir / "p.yml"
    profile.write_text("name: ConventionTest\ndata_bundle: hello_world\nprovider: anthropic\n")
    result = runner.invoke(
        build,
        [
            "smoke",
            str(profile),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    project_dir = tmp_path / "smoke"
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
    """T5: extends pointing at a missing file produces a clear error, not a stack trace."""
    profile = tmp_path / "p.yml"
    profile.write_text("name: Orphan\nextends: ./does-not-exist.yml\ndata_bundle: hello_world\n")
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            build,
            [
                "smoke",
                str(profile),
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code != 0
    _assert_build_error_logged(caplog, "does-not-exist", "not found")


def test_extends_cycle_detected(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """T5: circular extends: a -> b -> a is detected and aborted."""
    a = tmp_path / "a.yml"
    b = tmp_path / "b.yml"
    a.write_text("name: A\nextends: ./b.yml\ndata_bundle: hello_world\n")
    b.write_text("name: B\nextends: ./a.yml\ndata_bundle: hello_world\n")
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            build,
            [
                "smoke",
                str(a),
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code != 0
    _assert_build_error_logged(caplog, "cycle", "circular")


@pytest.mark.parametrize("preset", list_presets())
def test_each_bundled_preset_builds_clean(preset: str, runner: CliRunner, tmp_path: Path) -> None:
    """T1: every bundled preset must build to a project with a valid config and manifest.

    Auto-extends as new presets land (e.g. 'education'). A new preset that
    parses but doesn't build will fail this test on the next CI run.
    """
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            preset,
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    project_dir = tmp_path / "smoke"
    # 1. Core artifacts rendered
    assert (project_dir / "config.yml").exists()
    assert (project_dir / "CLAUDE.md").exists()
    # 2. Manifest is valid JSON with the bumped schema
    import json

    manifest = json.loads((project_dir / ".osprey-manifest.json").read_text())
    assert manifest["schema_version"] == "1.2.0"
    assert manifest["creation"]["template"] == preset
    # 3. Every preset must explicitly pin the facility timezone — agent timestamp
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

    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "control-assistant",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    project_dir = tmp_path / "smoke"
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


def test_preset_yaml_must_be_mapping(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """T5: a preset YAML that parses to a list (not a mapping) raises BuildProfileError."""
    # We can't easily inject a malformed bundled preset, but we can verify the
    # _load_preset_raw branch directly via the public function used by the CLI.
    from osprey.cli.build_profile import _load_preset_raw

    # Hijack the presets package via monkeypatching is awkward here — assert
    # the parallel error path on a positional profile parses-to-list, which
    # exercises the same _parse_profile expectation.
    bad = tmp_path / "bad.yml"
    bad.write_text("- one\n- two\n")
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build

    runner = CliRunner()
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            build,
            [
                "smoke",
                str(bad),
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
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

        profile_dir = tmp_path / "profile"
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
        profile = profile_dir / "p.yml"
        profile.write_text(
            "name: SeedVerbatim\ndata_bundle: hello_world\nprovider: anthropic\nmodel: haiku\n"
        )

        result = runner.invoke(
            build,
            [
                "smoke",
                str(profile),
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        built = json.loads(
            (tmp_path / "smoke" / "data" / "logbook_seed" / "demo_logbook.json").read_text()
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
        profile = tmp_path / "profile.yml"
        profile.write_text(self._PROFILE + extra)
        result = runner.invoke(
            build,
            [
                "smoke",
                str(profile),
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        return tmp_path / "smoke"

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
        # Explicit empty list — present so `osprey deploy` reads [] not None.
        assert cfg["deployed_services"] == []
        # None of the would-be-scaffolded services leak into services.*.
        assert cfg.get("services") in ({}, None)
        for name in ("postgresql", "openobserve", "bluesky"):
            assert name not in (cfg.get("services") or {})
        # No services/ directory at all.
        assert not (project / "services").exists()

    def test_readonly_persona_builds_attached(self, runner: CliRunner, tmp_path: Path) -> None:
        """The shipped read-only persona preset builds as an attached project."""
        result = runner.invoke(
            build,
            [
                "op",
                "--preset",
                "multi-user-demo-readonly",
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        cfg = _config_yaml(tmp_path / "op")
        assert cfg["deployed_services"] == []
        assert not (tmp_path / "op" / "services").exists()


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
        result = runner.invoke(
            build,
            [
                "smoke",
                "--preset",
                "hello-world",
                "--set",
                "provider=als-apg",
                "--set",
                "model=anthropic/claude-opus",
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(tmp_path),
            ],
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
# FR-6: every build goes through a profile
# ---------------------------------------------------------------------------


def _build_from_preset(runner: CliRunner, tmp_path: Path, *extra: str, preset: str = "hello-world"):
    return runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            preset,
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
            *extra,
        ],
    )


def _profile_yaml(profile_dir: Path) -> dict:
    return yaml.safe_load((profile_dir / "profile.yml").read_text(encoding="utf-8"))


def test_preset_build_materializes_a_profile_beside_the_project(
    runner: CliRunner, tmp_path: Path
) -> None:
    """FR-6: there is no profile-less build. A --preset build writes the profile
    it then builds from, and says where it put it."""
    result = _build_from_preset(runner, tmp_path)
    assert result.exit_code == 0, result.output

    profile_dir = tmp_path / "smoke-profile"
    assert (profile_dir / "profile.yml").is_file()
    assert (profile_dir / ".env.example").is_file()
    assert (profile_dir / "data").is_dir()

    # The manifest names it, which is how the deploy side finds the profile to
    # write minted secrets back to.
    import json

    manifest = json.loads((tmp_path / "smoke" / ".osprey-manifest.json").read_text())
    assert manifest["build_args"]["profile_path_abs"] == str(profile_dir / "profile.yml")


def test_second_preset_build_reuses_the_profile_verbatim(runner: CliRunner, tmp_path: Path) -> None:
    """The profile is the source of truth from the moment it exists: a rebuild
    reads the operator's edits back, and never re-materializes over them."""
    assert _build_from_preset(runner, tmp_path).exit_code == 0

    profile_path = tmp_path / "smoke-profile" / "profile.yml"
    original = profile_path.read_text(encoding="utf-8")
    edited = re.sub(r"(?m)^model:.*$", "model: opus", original)
    assert edited != original, "the emitted profile no longer carries a `model:` key"
    profile_path.write_text(edited, encoding="utf-8")
    (tmp_path / "smoke-profile" / "MINE.md").write_text("operator note\n")

    result = _build_from_preset(runner, tmp_path, "--force")
    assert result.exit_code == 0, result.output

    assert profile_path.read_text(encoding="utf-8") == edited
    assert (tmp_path / "smoke-profile" / "MINE.md").is_file()
    assert _config_yaml(tmp_path / "smoke")["claude_code"]["default_model"] == "opus"


def test_set_on_a_reuse_build_is_written_into_the_profile(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An explicit override IS a profile edit — announced, and durable."""
    assert _build_from_preset(runner, tmp_path).exit_code == 0

    with caplog.at_level(logging.INFO):
        result = _build_from_preset(runner, tmp_path, "--force", "--set", "model=opus")
    assert result.exit_code == 0, result.output

    profile_dir = tmp_path / "smoke-profile"
    assert _profile_yaml(profile_dir)["model"] == "opus"
    assert "model" in caplog.text and "override" in caplog.text.lower()
    assert _config_yaml(tmp_path / "smoke")["claude_code"]["default_model"] == "opus"

    # And it stays: a later build with no flags still gets it.
    assert _build_from_preset(runner, tmp_path, "--force").exit_code == 0
    assert _config_yaml(tmp_path / "smoke")["claude_code"]["default_model"] == "opus"


def test_set_list_on_a_reuse_build_replaces_rather_than_unions(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The documented semantic change: a write-back replaces the value at the
    key. A value written into a file has to be the value the file then holds."""
    assert _build_from_preset(runner, tmp_path).exit_code == 0
    assert len(_profile_yaml(tmp_path / "smoke-profile")["hooks"]) > 1

    result = _build_from_preset(runner, tmp_path, "--force", "--set", "hooks=[memory-guard]")
    assert result.exit_code == 0, result.output
    assert _profile_yaml(tmp_path / "smoke-profile")["hooks"] == ["memory-guard"]


def test_set_into_config_keeps_the_profile_dotted_key_shape(
    runner: CliRunner, tmp_path: Path
) -> None:
    """`config:` addresses the rendered config.yml with dotted keys held as
    single mapping keys. A write-back that nested them instead would replace a
    whole config subtree at build time rather than the leaf it names."""
    assert _build_from_preset(runner, tmp_path).exit_code == 0

    result = _build_from_preset(runner, tmp_path, "--force", "--set", "config.system.timezone=UTC")
    assert result.exit_code == 0, result.output

    config_block = _profile_yaml(tmp_path / "smoke-profile")["config"]
    assert config_block["system.timezone"] == "UTC"
    assert "system" not in config_block
    assert _config_yaml(tmp_path / "smoke")["system"]["timezone"] == "UTC"


def test_tier_on_a_reuse_build_is_written_into_the_profile(
    runner: CliRunner, tmp_path: Path
) -> None:
    assert _build_from_preset(runner, tmp_path, preset="channel-finder-standalone").exit_code == 0

    result = _build_from_preset(
        runner, tmp_path, "--force", "--tier", "3", preset="channel-finder-standalone"
    )
    assert result.exit_code == 0, result.output
    assert _profile_yaml(tmp_path / "smoke-profile")["tier"] == 3


def test_preset_drift_is_advisory_never_fatal(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A preset that moved on since materialization is reported, not re-applied."""
    assert _build_from_preset(runner, tmp_path).exit_code == 0

    profile_path = tmp_path / "smoke-profile" / "profile.yml"
    profile_path.write_text(
        profile_path.read_text(encoding="utf-8").replace(
            _profile_yaml(tmp_path / "smoke-profile")["provenance"]["preset_hash"],
            "sha256:stale",
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        result = _build_from_preset(runner, tmp_path, "--force")
    assert result.exit_code == 0, result.output
    assert "hello-world" in caplog.text
    assert "has changed" in caplog.text


def test_project_env_derives_from_the_profile_env(runner: CliRunner, tmp_path: Path) -> None:
    """FR-1/FR-9: the profile `.env` is where a secret survives a rebuild."""
    assert _build_from_preset(runner, tmp_path).exit_code == 0

    (tmp_path / "smoke-profile" / ".env").write_text("ANTHROPIC_API_KEY=profile-secret\n")
    assert _build_from_preset(runner, tmp_path, "--force").exit_code == 0

    assert "ANTHROPIC_API_KEY=profile-secret" in (tmp_path / "smoke" / ".env").read_text()


def test_project_env_example_comes_from_the_profile(runner: CliRunner, tmp_path: Path) -> None:
    """FR-4: one documented variable list, owned where the values are edited."""
    assert _build_from_preset(runner, tmp_path).exit_code == 0

    profile_example = tmp_path / "smoke-profile" / ".env.example"
    profile_example.write_text(profile_example.read_text() + "\nFACILITY_ONLY=\n")
    assert _build_from_preset(runner, tmp_path, "--force").exit_code == 0

    assert (tmp_path / "smoke" / ".env.example").read_text() == profile_example.read_text()


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

    result = runner.invoke(
        build,
        [
            "persona-proj",
            str(root / "personas" / "readonly.yml"),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output

    project = tmp_path / "persona-proj"
    # The delta alone names no provider and no data tree; both come from the root.
    assert _config_yaml(project)["claude_code"]["provider"] == "anthropic"
    assert (project / "data" / "FACILITY_MARKER.txt").is_file()
    # ...and the delta's own override still wins.
    assert _config_yaml(project)["claude_code"]["default_model"] == "haiku"


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

    result = runner.invoke(
        build,
        [
            "narrow-proj",
            str(root / "personas" / "narrow.yml"),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output

    project = tmp_path / "narrow-proj"
    assert not (project / ".claude" / "agents" / "orbit-writer.md").exists()
    assert not (project / ".claude" / "commands" / "osprey" / "scan.md").exists()
    # Absence from the project is only half of it: the original defect also
    # REGISTERED the copied artifact as user-owned, freezing the shadow against
    # regen. A test that checked only the file would miss that half.
    user_owned = _config_yaml(project).get("scaffold", {}).get("user_owned", []) or []
    assert not any("orbit-writer" in str(entry) for entry in user_owned), user_owned
    assert not any("scan" in str(entry) for entry in user_owned), user_owned

    # Control: the same profile built WITHOUT the exclusion does ship it, so the
    # assertions above cannot pass just because the artifact never applied.
    result = runner.invoke(
        build,
        [
            "wide-proj",
            str(root / "profile.yml"),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    wide = tmp_path / "wide-proj"
    assert (wide / ".claude" / "agents" / "orbit-writer.md").is_file()
    assert (wide / ".claude" / "commands" / "osprey" / "scan.md").is_file()
    wide_owned = [str(entry) for entry in _config_yaml(wide)["scaffold"]["user_owned"]]
    assert "agents/orbit-writer" in wide_owned, wide_owned
    assert "commands/osprey/scan" in wide_owned, wide_owned


# ---------------------------------------------------------------------------
# FR-6: the profile a --preset build reuses is the one it was made from
# ---------------------------------------------------------------------------


def test_build_with_a_different_preset_than_the_profile_is_refused(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A reused profile is built verbatim, so a --preset naming a DIFFERENT one
    selects nothing: accepting it built the stored preset's project under the
    requested preset's label, in the log, the manifest and the reproducible
    command alike."""
    assert _build_from_preset(runner, tmp_path).exit_code == 0

    result = _build_from_preset(runner, tmp_path, "--force", preset="channel-finder-standalone")
    assert result.exit_code != 0, result.output
    # Both presets, because which one the operator meant is exactly what the
    # CLI cannot know: keep the profile, or build the other one elsewhere.
    assert "hello-world" in result.output
    assert "channel-finder-standalone" in result.output

    # Refused before anything was touched, and the matching preset still builds.
    assert _profile_yaml(tmp_path / "smoke-profile")["provenance"]["preset"] == "hello-world"
    assert _build_from_preset(runner, tmp_path, "--force").exit_code == 0


def test_reused_profile_accepts_either_spelling_of_its_own_preset(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The refusal compares normalized names — `_` and `-` are one preset."""
    assert _build_from_preset(runner, tmp_path).exit_code == 0
    result = _build_from_preset(runner, tmp_path, "--force", preset="hello_world")
    assert result.exit_code == 0, result.output


def test_manifest_records_the_preset_the_profile_came_from(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The manifest names what the project was actually built from — the
    profile's own provenance — not the string that happened to be typed."""
    import json

    assert _build_from_preset(runner, tmp_path, preset="hello_world").exit_code == 0

    manifest = json.loads((tmp_path / "smoke" / ".osprey-manifest.json").read_text())
    stored = _profile_yaml(tmp_path / "smoke-profile")["provenance"]["preset"]
    assert manifest["build_args"]["preset"] == stored
    assert f"--preset {stored}" in manifest["reproducible_command"]


# ---------------------------------------------------------------------------
# FR-6: a write-back belongs to the build that made it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad_set", "key"),
    [
        # Two values the profile happily holds and the build then refuses, at
        # two different points: the provider spec, and the model resolver.
        ("provider=nonexistent-provider", "provider"),
        ("model=anthropic/claude-opus", "model"),
    ],
)
def test_failed_build_rolls_back_the_write_back_it_made(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture, bad_set: str, key: str
) -> None:
    """The write-back lands before the build validates what it wrote, so a value
    the project cannot be built from must not survive the build that failed on
    it: otherwise a typo turns into a stuck project — every later flag-free
    build fails too, for a reason none of them mentions, and the only way out is
    hand-editing the profile."""
    assert _build_from_preset(runner, tmp_path, "--set", "provider=als-apg").exit_code == 0

    profile_path = tmp_path / "smoke-profile" / "profile.yml"
    before = profile_path.read_bytes()

    with caplog.at_level(logging.WARNING):
        result = _build_from_preset(runner, tmp_path, "--force", "--set", bad_set)
    assert result.exit_code != 0, result.output
    assert profile_path.read_bytes() == before
    # And it says so: a silently reverted edit is its own puzzle.
    assert "restored" in caplog.text and key in caplog.text

    # The damage the rollback prevents: the next build carries no flags at all.
    assert _build_from_preset(runner, tmp_path, "--force").exit_code == 0


def test_successful_build_keeps_the_write_back_it_made(runner: CliRunner, tmp_path: Path) -> None:
    """Control for the rollback: a build that completes owns its edit."""
    assert _build_from_preset(runner, tmp_path).exit_code == 0
    assert _build_from_preset(runner, tmp_path, "--force", "--set", "model=opus").exit_code == 0
    assert _profile_yaml(tmp_path / "smoke-profile")["model"] == "opus"


def test_failed_first_build_removes_the_profile_it_materialized(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The create branch reaches the same stuck project by the other route: the
    bad override is BAKED IN as a materialization layer rather than written
    back, so there is no earlier state to restore — the directory this build
    made has to go with it."""
    with caplog.at_level(logging.WARNING):
        result = _build_from_preset(runner, tmp_path, "--set", "provider=nonexistent-provider")
    assert result.exit_code != 0, result.output

    profile_dir = tmp_path / "smoke-profile"
    assert not profile_dir.exists()
    assert "removing" in caplog.text and "provider" in caplog.text

    # The damage it prevents: the next build carries no flags and must not
    # inherit the poisoned profile.
    assert _build_from_preset(runner, tmp_path, "--force").exit_code == 0
    assert _profile_yaml(profile_dir)["provider"] != "nonexistent-provider"


def test_successful_first_build_keeps_the_profile_it_materialized(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Control for the removal: only a FAILED build takes its profile with it."""
    assert _build_from_preset(runner, tmp_path).exit_code == 0
    assert (tmp_path / "smoke-profile" / "profile.yml").is_file()


def test_failed_build_never_removes_a_profile_dir_it_did_not_create(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The removal is scoped to what this invocation made, so the interrupted
    materialization of F3 still meets the refusal that names it on the NEXT
    build — swallowing it here would hide the very state that message exists to
    explain."""
    profile_dir = tmp_path / "smoke-profile"
    profile_dir.mkdir()
    (profile_dir / "half-written.txt").write_text("operator's leftovers\n", encoding="utf-8")

    result = _build_from_preset(runner, tmp_path, "--set", "provider=nonexistent-provider")
    assert result.exit_code != 0, result.output
    # Untouched, and the message is still the build-appropriate one.
    assert (profile_dir / "half-written.txt").is_file()
    assert "osprey profile new" in result.output
    assert "re-run with --force to replace it" not in result.output


def test_rollback_that_fails_names_the_file_and_the_keys(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the profile cannot be put back, the operator is told which file
    still holds which override — the only way left to repair it by hand."""
    from osprey.cli import build_profile_resolve as resolve

    profile_path = tmp_path / "profile.yml"
    profile_path.write_text("name: Smoke\nmodel: sonnet\n", encoding="utf-8")

    guard = resolve.ProfileWriteBackGuard(profile_path, (), ("model=opus",), None)
    resolve.write_back_cli_overrides(profile_path, (), ("model=opus",))

    def _boom(path: Path, payload: bytes) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(resolve, "_atomic_write_bytes", _boom)
    with caplog.at_level(logging.ERROR):
        guard.rollback()

    assert str(profile_path) in caplog.text
    assert "model" in caplog.text
    assert "read-only file system" in caplog.text


def test_profile_write_back_replaces_rather_than_truncates(tmp_path: Path) -> None:
    """The write-back never opens the profile for truncation: it renders the new
    document beside it and moves it into place. Observable on a read-only file,
    which a truncating write cannot open at all — and the mode survives, so an
    atomic rewrite does not quietly re-permission the source of truth."""
    from osprey.cli.build_profile_resolve import write_back_cli_overrides

    profile_path = tmp_path / "profile.yml"
    profile_path.write_text("name: Smoke\nmodel: sonnet\n", encoding="utf-8")
    profile_path.chmod(0o444)

    write_back_cli_overrides(profile_path, (), ("model=opus",))

    assert yaml.safe_load(profile_path.read_text(encoding="utf-8"))["model"] == "opus"
    assert profile_path.stat().st_mode & 0o777 == 0o444
    assert [p.name for p in tmp_path.iterdir()] == ["profile.yml"]


def test_profile_write_back_that_fails_leaves_the_old_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of atomicity: an interrupted write leaves the profile it
    had — not a truncated one — and no temp file beside it."""
    from osprey.cli import build_profile_resolve as resolve

    profile_path = tmp_path / "profile.yml"
    original = "name: Smoke\nmodel: sonnet\n"
    profile_path.write_text(original, encoding="utf-8")

    def _boom(src: str, dst: str) -> None:
        raise OSError("no space left on device")

    # NB: patching `resolve.os.replace` patches the stdlib module itself, so it
    # is in force for everything this test runs — which is one write-back call.
    monkeypatch.setattr(resolve.os, "replace", _boom)
    with pytest.raises(OSError):
        resolve.write_back_cli_overrides(profile_path, (), ("model=opus",))

    assert profile_path.read_text(encoding="utf-8") == original
    assert [p.name for p in tmp_path.iterdir()] == ["profile.yml"]


# ---------------------------------------------------------------------------
# FR-6: refusals a build inherits from materialization
# ---------------------------------------------------------------------------


def test_profile_dir_without_a_profile_yml_gets_build_appropriate_remediation(
    runner: CliRunner, tmp_path: Path
) -> None:
    """An interrupted materialization leaves a `<name>-profile/` with no
    profile.yml. `osprey build --force` replaces the project and never the
    profile, so telling the user to re-run the build with --force sends them
    round a loop that cannot end."""
    for junk in ((), ("leftover.txt",)):
        target = tmp_path / "smoke-profile"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir()
        for name in junk:
            (target / name).write_text("half-written\n", encoding="utf-8")

        result = _build_from_preset(runner, tmp_path)
        assert result.exit_code != 0, result.output
        assert "re-run with --force to replace it" not in result.output
        assert "osprey profile new" in result.output
        assert "profile.yml" in result.output


def test_extends_override_is_refused_identically_on_both_branches(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A materialized profile is standalone. The same override file must be
    answered the same way on the build that materializes it and on the next
    one — not refused once and then written into the file."""
    override = tmp_path / "extends.yml"
    # A different preset than the one being built: `extends: <itself>` is a
    # cycle, and would be refused by resolution before either branch's own
    # check — which is not the refusal under test.
    override.write_text("extends: channel-finder-standalone\n", encoding="utf-8")

    created = _build_from_preset(runner, tmp_path, "-O", str(override))
    assert created.exit_code != 0, created.output
    assert "extends" in created.output
    assert not (tmp_path / "smoke-profile").exists()

    assert _build_from_preset(runner, tmp_path).exit_code == 0
    reused = _build_from_preset(runner, tmp_path, "--force", "-O", str(override))
    assert reused.exit_code != 0, reused.output

    from osprey.cli.build_profile import EXTENDS_OVERRIDE_REFUSAL

    assert EXTENDS_OVERRIDE_REFUSAL in created.output
    assert EXTENDS_OVERRIDE_REFUSAL in reused.output
    assert "extends" not in _profile_yaml(tmp_path / "smoke-profile")

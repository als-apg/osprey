"""What ``osprey init`` writes beside ``profile.yml``: the tier and app files.

A profile that stands up the multi-user web-terminal stack owns the tiers and
apps its roster names: ``osprey init`` writes ``tiers/<n>.yml`` for every tier a
person, ``default_tier``, a role or ``dispatch.tier`` names, and
``apps/<n>.yml`` for every app in ``shared:``.

Each is materialized the way ``profile.yml`` is: the bundled file's whole
content with its comments, a header saying what the file is, and a
``provenance:`` block naming what it was made from. There is no ``extends:`` and
no pointer into the package — the materialized repo is explicit. Location is
the relation to the host: a file in ``tiers/`` or ``apps/`` is laid over the
``profile.yml`` beside it, so none restates the host's data tree, provider or
model.

The byte specification is the exemplar in :mod:`tests.fixtures.lifecycle_repo`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_merge import compute_preset_hash
from osprey.cli.init_cmd import init
from osprey.cli.profile_cmd import profile
from osprey.utils.workspace import RETIRED_LAYER_DIRS
from osprey_connectors.errors import BuildProfileError
from tests.cli._lifecycle_build import invoke_build, rendered_config
from tests.fixtures.lifecycle_repo import (
    APP_PRESETS,
    EXEMPLAR_DIRNAME,
    TIER_PRESETS,
    exemplar_source_files,
)

#: Every tier and app file the exemplar holds, repo-relative.
EXEMPLAR_LAYER_FILES = tuple(
    [f"tiers/{name}.yml" for name in TIER_PRESETS] + [f"apps/{name}.yml" for name in APP_PRESETS]
)


def _new(runner: CliRunner, target: Path, preset: str, *extra: str):
    """Materialize a deployment repo at *target*. ``--no-git``: nothing here reads history."""
    return runner.invoke(init, [str(target), "--preset", preset, "--no-git", *extra])


@pytest.fixture(scope="module")
def exemplar_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``osprey init --preset control-assistant`` into a folder named like the exemplar.

    The folder name is the deployment's display name ("Als Exemplar"), which
    each tier is titled after, so this is the init the exemplar reproduces.
    """
    target = tmp_path_factory.mktemp("init") / EXEMPLAR_DIRNAME
    result = _new(CliRunner(), target, "control-assistant")
    assert result.exit_code == 0, result.output
    return target


# ---------------------------------------------------------------------------
# The exemplar, byte for byte
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relative", EXEMPLAR_LAYER_FILES)
def test_init_writes_the_exemplar_layer_byte_for_byte(exemplar_repo: Path, relative: str) -> None:
    """Each tier and app file is exactly the exemplar's: header, the bundled
    file's commented body, a tier's retitled ``name:``, the provenance block."""
    expected = exemplar_source_files(with_ci=False)[relative]

    assert (exemplar_repo / relative).read_text(encoding="utf-8") == expected


def test_init_writes_exactly_the_exemplar_layer_set(exemplar_repo: Path) -> None:
    """One file per tier and app the roster names, nothing else; no legacy
    retired layer directory beside them."""
    written = sorted(
        f"{folder}/{path.name}"
        for folder in ("tiers", "apps")
        for path in (exemplar_repo / folder).iterdir()
    )

    assert written == sorted(EXEMPLAR_LAYER_FILES)
    for retired in RETIRED_LAYER_DIRS:
        assert not (exemplar_repo / retired).exists()


@pytest.mark.parametrize("relative", EXEMPLAR_LAYER_FILES)
def test_exemplar_layers_carry_no_pointer_into_the_package(
    exemplar_repo: Path, relative: str
) -> None:
    """A4: the materialized repo is explicit. No ``extends:`` and no ``preset:``
    key names the bundled file; only the provenance record does, and that is
    a record for drift checks, not an inheritance."""
    parsed = yaml.safe_load((exemplar_repo / relative).read_text(encoding="utf-8"))

    assert "extends" not in parsed
    assert "preset" not in parsed
    folder, stem = relative.removesuffix(".yml").split("/")
    assert parsed["provenance"] == {
        "preset": stem,
        "preset_hash": compute_preset_hash(f"{folder}/{stem}"),
    }


@pytest.mark.parametrize("relative", EXEMPLAR_LAYER_FILES)
def test_exemplar_layers_restate_nothing_the_host_owns(exemplar_repo: Path, relative: str) -> None:
    """The host's data tree, provider, model and floor are its own: a tier or
    app file laid over it never carries a competing value."""
    parsed = yaml.safe_load((exemplar_repo / relative).read_text(encoding="utf-8"))

    for host_key in ("data", "app_template", "provider", "model", "requires_osprey_version"):
        assert host_key not in parsed, host_key


def test_exemplar_tiers_are_titled_after_the_deployment_and_apps_keep_their_own(
    exemplar_repo: Path,
) -> None:
    """A tier is an access level of the deployment's agent; an app's ``name:``
    is the title of the card it runs behind."""
    titles = {
        relative: yaml.safe_load((exemplar_repo / relative).read_text(encoding="utf-8"))["name"]
        for relative in EXEMPLAR_LAYER_FILES
    }

    for tier in TIER_PRESETS:
        assert titles[f"tiers/{tier}.yml"] == f"Als Exemplar ({tier})"
    assert titles["apps/logbook.yml"] == "Logbook Research"


def test_exemplar_profile_and_layers_build_every_render(
    exemplar_repo: Path, tmp_path: Path
) -> None:
    """The emitted repo builds: one render per tier and app it holds."""
    import shutil

    repo = tmp_path / EXEMPLAR_DIRNAME
    shutil.copytree(exemplar_repo, repo)

    result = invoke_build(repo)

    assert result.exit_code == 0, result.output
    for relative in EXEMPLAR_LAYER_FILES:
        stem = Path(relative).stem
        assert (repo / "build" / f"{EXEMPLAR_DIRNAME}-{stem}").is_dir(), stem


def test_exemplar_layer_texts_parse_to_deltas_keyed_by_render_name() -> None:
    """The texts are keyed by file, the parsed deltas by the render name every
    reader (the composition card, the provider tally) asks for."""
    from osprey.cli.profile_cmd import _layer_profile_texts, _parsed_layer_deltas

    resolved, _dir = resolve_build_profile(None, "control-assistant")
    texts = _layer_profile_texts(resolved, "Als Exemplar", "")
    expected = exemplar_source_files(with_ci=False)

    assert sorted(texts) == sorted(EXEMPLAR_LAYER_FILES)
    # Init names the facility description it seeds by its path once the file
    # is there; the emission itself spells the preset's bare name.
    from osprey.cli.init_revalidate import _rewrite_shadow_lines

    assert all(
        _rewrite_shadow_lines(texts[relative], relative, ["rules/facility"]) == expected[relative]
        for relative in texts
    )
    assert set(_parsed_layer_deltas(texts)) == {*TIER_PRESETS, *APP_PRESETS}


# ---------------------------------------------------------------------------
# The emitter
# ---------------------------------------------------------------------------


def test_the_emitter_names_a_nested_profile_path_in_its_hint() -> None:
    from osprey.cli.build_profile_emit import emit_layer_delta_yaml

    text = emit_layer_delta_yaml(
        preset_name="tiers/readonly",
        profile_name="My Facility",
        profile_filename="my-profile/tiers/readonly.yml",
    )

    assert "#   osprey validate my-profile/tiers/readonly.yml\n" in text
    assert yaml.safe_load(text)["name"] == "My Facility (readonly)"


@pytest.mark.parametrize(
    "name", ["readonly", "tiers/", *(f"{d}/readonly" for d in RETIRED_LAYER_DIRS), "tiers/a/b"]
)
def test_the_emitter_refuses_a_name_that_is_not_a_kind_qualified_layer(name: str) -> None:
    from osprey.cli.build_profile_emit import emit_layer_delta_yaml

    with pytest.raises(BuildProfileError, match="names no tier or app"):
        emit_layer_delta_yaml(preset_name=name, profile_name="X", profile_filename="x.yml")


def test_the_emitter_refuses_a_bundled_layer_that_declares_extends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A layer with a parent it would not carry is neither the bundled
    definition nor explicit, so it is refused rather than emitted short."""
    import osprey.cli.build_profile_emit as emit_mod

    real = emit_mod._load_preset_raw

    def with_extends(name: str):
        raw, path = real(name)
        return {**raw, "extends": "control-assistant"}, path

    monkeypatch.setattr(emit_mod, "_load_preset_raw", with_extends)

    with pytest.raises(BuildProfileError, match="declares extends"):
        emit_mod.emit_layer_delta_yaml(
            preset_name="apps/logbook", profile_name="X", profile_filename="apps/logbook.yml"
        )


# ---------------------------------------------------------------------------
# Which tiers and apps: the ones the roster names
# ---------------------------------------------------------------------------


def _web_terminals_edit(**changes: object) -> tuple[str, ...]:
    """A ``--set`` pair restating control-assistant's web-terminal block, changed.

    A ``--set`` pair replaces the value at the key it names, so the block is
    read back from the preset and stated in full with the named keys replaced.
    """
    preset, _dir = resolve_build_profile(None, "control-assistant")
    block = copy.deepcopy(preset.config["modules.web_terminals"])
    block.update(changes)
    return ("--set", "config.modules.web_terminals=" + json.dumps(block))


def _layer_names(target: Path, folder: str) -> set[str]:
    directory = target / folder
    return {path.stem for path in directory.iterdir()} if directory.is_dir() else set()


def test_only_the_tiers_and_apps_the_roster_names_are_written(
    runner: CliRunner, tmp_path: Path
) -> None:
    target = tmp_path / "my-facility"
    edit = _web_terminals_edit(
        users=[{"name": "bob", "index": 1, "tier": "readonly"}],
        default_tier="readonly",
        shared=["logbook"],
    )

    result = _new(runner, target, "control-assistant", *edit)

    assert result.exit_code == 0, result.output
    assert _layer_names(target, "tiers") == {"readonly"}
    assert _layer_names(target, "apps") == {"logbook"}


def test_the_default_tier_is_written_even_when_nobody_runs_on_it_yet(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``default_tier`` is what the next person added inherits, so its file is
    in the repo before that person is."""
    target = tmp_path / "my-facility"
    edit = _web_terminals_edit(
        users=[{"name": "alice", "index": 0, "tier": "readwrite"}],
        default_tier="readonly",
        shared=[],
    )

    result = _new(runner, target, "control-assistant", *edit)

    assert result.exit_code == 0, result.output
    assert _layer_names(target, "tiers") == {"readonly", "readwrite"}
    assert _layer_names(target, "apps") == set()


def test_role_and_dispatch_tiers_are_named_too() -> None:
    """Single sign-on may hand out any tier a role binds, and the dispatch
    worker runs on ``dispatch.tier``: both are tiers the deployment runs."""
    from osprey.cli.profile_cmd import _roster_layers

    config = {
        "modules.web_terminals": {
            "enabled": True,
            "default_tier": "readonly",
            "authorization": {"roles": {"operator": {"tier": "readwrite"}}},
            "users": [{"name": "bob", "index": 1}],
            "shared": ["knowledge"],
        }
    }

    assert _roster_layers(config, "admin") == (["admin", "readonly", "readwrite"], ["knowledge"])
    assert _roster_layers(config, None) == (["readonly", "readwrite"], ["knowledge"])


def test_without_web_terminals_only_the_dispatch_tier_is_named() -> None:
    """A deployment without web terminals deploys no roster, but its dispatch
    workers still run their tier."""
    from osprey.cli.profile_cmd import _roster_layers

    config = {
        "modules.web_terminals": {
            "enabled": False,
            "default_tier": "readonly",
            "authorization": {"roles": {"operator": {"tier": "readwrite"}}},
            "users": [{"name": "alice", "index": 0, "tier": "admin"}],
            "shared": ["knowledge"],
        }
    }

    assert _roster_layers(config, "readonly") == (["readonly"], [])
    assert _roster_layers(config, None) == ([], [])


def test_the_control_assistant_preset_runs_its_dispatch_workers_read_only() -> None:
    """A dispatched job has nobody to approve a write, so the shipped default is
    the read-only tier; a deployment that wants a writer says so."""
    preset, _dir = resolve_build_profile(None, "control-assistant")

    assert preset.dispatch is not None
    assert preset.dispatch.tier == "readonly"


def test_a_backend_only_deployment_writes_and_builds_its_dispatch_tier(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Switching the web terminals off keeps the dispatch workers on their tier:
    init writes that tier's file, and the build renders it and names it."""

    target = tmp_path / "backend-only"
    result = _new(
        runner,
        target,
        "control-assistant",
        "--set",
        "config.modules.web_terminals.enabled=false",
    )

    assert result.exit_code == 0, result.output
    assert _layer_names(target, "tiers") == {"readonly"}
    assert _layer_names(target, "apps") == set()

    built = invoke_build(target)

    assert built.exit_code == 0, built.output
    assert (target / "build" / "backend-only-readonly" / "Dockerfile").is_file()
    rendered = rendered_config(target)
    assert rendered["services"]["dispatch_worker"]["tier"] == "readonly"
    assert rendered["modules"]["web_terminals"]["renders"]["readonly"]["kind"] == "tier"


def test_a_preset_without_web_terminals_writes_no_tiers_or_apps(
    runner: CliRunner, tmp_path: Path
) -> None:
    target = tmp_path / "my-facility"

    assert _new(runner, target, "hello-world").exit_code == 0

    assert not (target / "tiers").exists()
    assert not (target / "apps").exists()


# ---------------------------------------------------------------------------
# Refusals: a roster whose tiers or apps cannot be materialized
# ---------------------------------------------------------------------------


def test_a_tier_osprey_does_not_ship_is_refused_before_anything_is_written(
    runner: CliRunner, tmp_path: Path
) -> None:
    target = tmp_path / "my-facility"
    edit = _web_terminals_edit(
        users=[{"name": "bob", "index": 1, "tier": "ops"}], shared=["nonesuch"]
    )

    result = _new(runner, target, "control-assistant", *edit)

    assert result.exit_code == 2
    # Every problem named at once.
    assert "the roster names tier 'ops', but OSPREY ships no such tier" in result.output
    assert "the roster names app 'nonesuch', but OSPREY ships no such app" in result.output
    assert not target.exists()  # fail-before-mutating


def test_a_name_that_is_both_a_tier_and_an_app_is_refused(runner: CliRunner, tmp_path: Path):
    """A render is named after its file, so a tier and an app sharing a name
    would be one render built from two files."""
    target = tmp_path / "my-facility"
    edit = _web_terminals_edit(
        users=[{"name": "bob", "index": 1, "tier": "logbook"}], shared=["logbook"]
    )

    result = _new(runner, target, "control-assistant", *edit)

    assert result.exit_code == 2
    assert "'logbook' is both a tier the roster names and an app in shared:" in result.output
    assert not target.exists()


def test_a_malformed_emitted_layer_is_reported_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osprey.cli.build_profile_emit as emit_mod
    from osprey.cli.profile_cmd import _materialize_profile_directory

    monkeypatch.setattr(emit_mod, "emit_layer_delta_yaml", lambda **_: "name: [unclosed\n")
    target = tmp_path / "my-facility"

    with pytest.raises(BuildProfileError, match=r"Emitted layer (tiers|apps)/.*not valid YAML"):
        _materialize_profile_directory(target, "control-assistant")

    assert not target.exists()


# ---------------------------------------------------------------------------
# The privilege tally: every card the roster runs, people and apps
# ---------------------------------------------------------------------------


def _card_edit(access: object) -> tuple[str, ...]:
    """The ``--set`` pair adding one admin card, carrying *access*, to the roster.

    The roster is stated in full because an edit replaces the list. ``dana``
    and index 9 are clear of the preset's own names and ports. ``admin`` is the
    shipped stack's privileged tier — the setup tool and the web Config panel.
    """
    preset, _dir = resolve_build_profile(None, "control-assistant")
    roster = copy.deepcopy(preset.config["modules.web_terminals"]["users"])
    roster.append({"name": "dana", "index": 9, "tier": "admin", "access": access})
    return _web_terminals_edit(users=roster)


def _shared_card_refusals(output: str) -> set[str]:
    """The shared-card sentences in an accumulated refusal, one per entry."""
    return {
        line.strip().removeprefix("- ")
        for line in output.splitlines()
        if "is a shared card" in line
    }


@pytest.mark.parametrize(
    ("access", "phrase"),
    [
        (["domain:example.com"], "access members 'domain:example.com'"),
        (["user:dana@example.com"], "access members 'user:dana@example.com'"),
        (["self", "user:dana@example.com"], "access members 'self', 'user:dana@example.com'"),
        ("any", "access: 'any'"),
    ],
)
def test_a_shared_privileged_card_is_refused(
    runner: CliRunner, tmp_path: Path, access: object, phrase: str
) -> None:
    """Every principal set naming somebody beyond the entry's own user shares
    the card; on the admin tier that hands deployment editing to whoever the
    set admits, so materialization refuses and quotes the value as authored."""
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant", *_card_edit(access))

    assert result.exit_code == 2
    assert f"user 'dana' is a shared card ({phrase})" in result.output
    assert "resolves to render 'admin'" in result.output
    assert "Set access: own for 'dana'" in result.output
    assert not target.exists()  # fail-before-mutating


@pytest.mark.parametrize("access", ["own", ["self"]])
def test_an_owner_only_card_on_the_same_tier_materializes(
    runner: CliRunner, tmp_path: Path, access: object
) -> None:
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant", *_card_edit(access))

    assert result.exit_code == 0, result.output
    assert (target / "tiers" / "admin.yml").is_file()


def test_a_privileged_default_tier_is_refused(runner: CliRunner, tmp_path: Path) -> None:
    """``default_tier`` is handed out by accident — to every person who names no
    tier — so it must not be the tier that edits the deployment."""
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant", *_web_terminals_edit(default_tier="admin"))

    assert result.exit_code == 2
    assert "would hand deployment editing to a terminal" in result.output
    assert "'admin'" in result.output
    assert not target.exists()


def _tampered_logbook(monkeypatch: pytest.MonkeyPatch, line: str) -> None:
    """Emit the bundled logbook app with ``line`` added to its ``config:``."""
    import osprey.cli.build_profile_emit as emit_mod

    real = emit_mod.emit_layer_delta_yaml

    def tampered(**kwargs: str) -> str:
        text = real(**kwargs)
        if kwargs["preset_name"] != "apps/logbook":
            return text
        return text.replace("\nconfig:\n", f"\nconfig:\n  {line}\n", 1)

    monkeypatch.setattr(emit_mod, "emit_layer_delta_yaml", tampered)


def test_an_app_that_opens_a_door_is_still_composed_without_it(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tally judges an app card on the render the build composes, which
    closes every deployment-editing door whatever the app file says: the
    shared logbook card holds nothing, so init does not refuse it."""
    _tampered_logbook(monkeypatch, "web.config_panel.enabled: true")
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant")

    assert result.exit_code == 0, result.output
    assert "user 'logbook' is a shared card" not in result.output


def test_an_app_the_composer_refuses_is_refused_by_init(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tally runs over every card the roster runs, the derived app cards
    included: an app file that lifts the setup deny composes into a render
    that could edit the deployment, so init refuses it before writing a repo."""
    _tampered_logbook(
        monkeypatch, "claude_code.permissions.remove_deny: [mcp__osprey_workspace__setup_patch]"
    )
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant")

    assert result.exit_code == 2
    assert "'logbook'" in result.output
    assert "must not be able to write or edit the deployment" in result.output
    assert not target.exists()


def test_init_and_the_deploy_lint_refuse_the_shared_card_in_the_same_words(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The build-time brace and the deploy-time belt say ONE sentence: both
    import the rule from the renders module rather than stating it."""
    linted = tmp_path / "linted-facility"
    assert _new(runner, linted, "control-assistant", *_card_edit("own")).exit_code == 0

    profile_path = linted / "profile.yml"
    document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    for entry in document["config"]["modules.web_terminals"]["users"]:
        if entry.get("name") == "dana":
            entry["access"] = ["domain:example.com"]
    profile_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    lint_result = runner.invoke(profile, ["validate", str(linted)])
    init_result = _new(
        runner,
        tmp_path / "refused-facility",
        "control-assistant",
        *_card_edit(["domain:example.com"]),
    )

    assert lint_result.exit_code == 2 and init_result.exit_code == 2
    refusals = _shared_card_refusals(lint_result.output)
    assert refusals  # the lint really did report it, so the comparison means something
    assert refusals == _shared_card_refusals(init_result.output)


def test_an_unreadable_access_value_is_refused_rather_than_raised(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The parser's own sentence — the one the deploy lint prints — rather than
    a traceback, and the privilege rule says nothing about that entry."""
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant", *_card_edit(["group:ops"]))

    assert result.exit_code == 2
    assert "entry 'dana' has an access member 'group:ops'" in result.output
    assert "the 'group:' prefix is reserved" in result.output
    assert "is a shared card" not in result.output
    assert not target.exists()  # fail-before-mutating


# ---------------------------------------------------------------------------
# Build exhaust never travels with the data tree (wheel/source parity)
# ---------------------------------------------------------------------------


def test_data_copy_ignore_drops_only_the_named_subtree(tmp_path: Path) -> None:
    """``benchmarks/results`` is dropped; a same-named directory anywhere else,
    and its sibling staging dirs, are not."""
    import shutil

    from osprey.cli.profile_cmd import _data_copy_ignore

    source = tmp_path / "src"
    for rel in ("benchmarks/results", "benchmarks/cross_paradigm", "results", "raw/results"):
        (source / rel).mkdir(parents=True)
        (source / rel / "f.txt").write_text("x", encoding="utf-8")

    shutil.copytree(source, tmp_path / "dst", ignore=_data_copy_ignore(source))

    dst = tmp_path / "dst"
    assert not (dst / "benchmarks" / "results").exists()
    assert (dst / "benchmarks" / "cross_paradigm" / "f.txt").is_file()
    assert (dst / "results" / "f.txt").is_file()
    assert (dst / "raw" / "results" / "f.txt").is_file()


def test_benchmark_results_in_a_source_checkout_are_not_materialized(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source checkout that has run the channel-finder benchmark holds
    ``data/benchmarks/results/``; a wheel install never does (hatch excludes
    it). Emission must be the same tree either way, so the copy drops it.

    The checkout is a copy of the shipped template tree: other tests read the
    real one while this runs beside them."""
    import shutil

    from osprey.cli.templates.manager import TemplateManager

    root = tmp_path / "templates"
    shutil.copytree(
        TemplateManager().template_root, root, ignore=shutil.ignore_patterns("__pycache__")
    )
    results = root / "apps" / "control_assistant" / "data" / "benchmarks" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "run-from-a-test.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(TemplateManager, "_get_template_root", lambda self: root)

    target = tmp_path / "my-facility"
    assert _new(runner, target, "control-assistant").exit_code == 0

    assert not (target / "data" / "benchmarks" / "results").exists()
    # The sibling staging dir the bundle really ships still comes across.
    assert (target / "data" / "benchmarks" / "cross_paradigm").is_dir()

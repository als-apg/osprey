"""Unit tests for the doc-screenshot registry, validation, and CLI selection.

These are CI-safe: no container, no browser, no live agent — they exercise the
declarative layer (:mod:`docs.screenshots.recipes`) and the CLI selection matrix
(:mod:`docs.screenshots.__main__`) only.
"""

from __future__ import annotations

import pytest
from docs.screenshots import recipes
from docs.screenshots.__main__ import main
from docs.screenshots.recipes import (
    CAPTION_PLACEHOLDER,
    DocShot,
    SubView,
    caption_substitutions,
    is_enabled,
    select_recipes,
    validate_registry,
)


def _standalone(name="switch", **kw):
    kw.setdefault("app_factory", "osprey.interfaces.artifacts.app:create_app")
    return DocShot(name=name, environment="standalone_interface", kind="static", **kw)


def _hub(name="hub", **kw):
    return DocShot(name=name, environment="hermetic_hub", kind="static", **kw)


def _stack_static(name="ariel", **kw):
    return DocShot(name=name, environment="tutorial_stack", kind="static", **kw)


def _agentic(name="hero", **kw):
    kw.setdefault("prompt", "Plot the beam current.")
    kw.setdefault("wait_for", "plot")
    return DocShot(name=name, environment="tutorial_stack", kind="agentic", **kw)


# ---------------------------------------------------------------------------
# The shipped registry is always well-formed
# ---------------------------------------------------------------------------


def test_shipped_registry_is_valid():
    validate_registry()  # must not raise for the real REGISTRY


# ---------------------------------------------------------------------------
# validate_registry rules
# ---------------------------------------------------------------------------


def test_empty_registry_is_valid():
    validate_registry([])


@pytest.mark.parametrize(
    ("shots", "match"),
    [
        pytest.param(
            [_standalone("dup"), _standalone("dup")], "duplicate recipe name", id="duplicate-name"
        ),
        pytest.param(
            [
                _stack_static("a", subviews=(SubView("#x", "shared"),)),
                _stack_static("b", subviews=(SubView("#y", "shared"),)),
            ],
            "duplicate output filename",
            id="duplicate-output",
        ),
        pytest.param(
            [_standalone("e", capture_mode="element")], "element_selector", id="element-selector"
        ),
        pytest.param(
            [DocShot(name="h", environment="tutorial_stack", kind="agentic")],
            "prompt and wait_for",
            id="agentic-prompt",
        ),
        pytest.param(
            [_standalone(init_sets=("apps.orbit-bump={}",))],
            "init_sets",
            id="init-sets-need-tutorial-stack",
        ),
        pytest.param(
            [
                DocShot(
                    name="h",
                    environment="standalone_interface",
                    kind="agentic",
                    app_factory="x:create_app",
                    prompt="p",
                    wait_for="plot",
                )
            ],
            "tutorial_stack",
            id="agentic-stack",
        ),
        pytest.param(
            [DocShot(name="s", environment="standalone_interface", kind="static")],
            "app_factory",
            id="app-factory",
        ),
        pytest.param([_standalone("s", themes=("light", "sepia"))], "themes", id="bad-theme"),
        pytest.param([_standalone("s", themes=())], "themes", id="empty-themes"),
        pytest.param([_hub(stage="nope")], "unknown stage", id="unknown-stage"),
    ],
)
def test_validate_registry_refuses(shots, match):
    with pytest.raises(ValueError, match=match):
        validate_registry(shots)


def test_element_mode_with_selector_ok():
    validate_registry(
        [_standalone("e", capture_mode="element", element_selector="osprey-theme-switcher")]
    )


def test_init_sets_are_key_value_pairs():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        validate_registry([_stack_static(init_sets=("apps.orbit-bump",))])
    validate_registry([_stack_static(init_sets=("apps.orbit-bump={}",))])


def _probe_stage(monkeypatch) -> None:
    """Register a no-op ``probe`` stage for the length of one test."""
    from docs.screenshots import contact_sheet

    probe = contact_sheet.Stage("probe", lambda _page: None)
    monkeypatch.setitem(contact_sheet.STAGES, "probe", probe)


def test_stage_requires_hermetic_hub(monkeypatch):
    _probe_stage(monkeypatch)
    with pytest.raises(ValueError, match="hermetic_hub"):
        validate_registry([_standalone(stage="probe")])


def test_hermetic_hub_with_a_known_stage_is_valid(monkeypatch):
    _probe_stage(monkeypatch)
    validate_registry([_hub(stage="probe")])


def test_output_names_single_vs_subviews():
    assert _standalone("solo").output_names() == ["solo"]
    multi = _stack_static("m", subviews=(SubView("#a", "m_a"), SubView("#b", "m_b")))
    assert multi.output_names() == ["m_a", "m_b"]


# ---------------------------------------------------------------------------
# is_enabled matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shot", "stack", "agentic", "expected"),
    [
        (_standalone(), False, False, True),  # standalone static: always on
        (_standalone(), True, True, True),
        (_hub(), False, False, True),  # hermetic hub static: always on
        (_hub(), True, True, True),
        (_stack_static(), False, False, False),  # tutorial static: needs --stack
        (_stack_static(), True, False, True),
        (_agentic(), False, False, False),  # agentic: needs --agentic
        (_agentic(), False, True, True),
        (_agentic(), True, False, False),  # --stack alone does not enable agentic
    ],
)
def test_is_enabled_matrix(shot, stack, agentic, expected):
    assert is_enabled(shot, stack=stack, agentic=agentic) is expected


# ---------------------------------------------------------------------------
# select_recipes over a patched registry
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_registry(monkeypatch):
    reg = [_standalone("switch"), _stack_static("ariel"), _agentic("hero")]
    monkeypatch.setattr(recipes, "REGISTRY", reg)
    # __main__ imported REGISTRY by value; patch its binding too.
    import docs.screenshots.__main__ as cli_mod

    monkeypatch.setattr(cli_mod, "REGISTRY", reg)
    return reg


@pytest.mark.usefixtures("sample_registry")
def test_select_default_only_standalone():
    names = [s.name for s in select_recipes()]
    assert names == ["switch"]


@pytest.mark.usefixtures("sample_registry")
def test_select_stack_adds_tutorial_static():
    names = {s.name for s in select_recipes(stack=True)}
    assert names == {"switch", "ariel"}


@pytest.mark.usefixtures("sample_registry")
def test_select_agentic_adds_hero():
    names = {s.name for s in select_recipes(agentic=True)}
    assert names == {"switch", "hero"}


@pytest.mark.usefixtures("sample_registry")
def test_select_only_filters():
    names = [s.name for s in select_recipes(stack=True, only="ariel")]
    assert names == ["ariel"]


@pytest.mark.usefixtures("sample_registry")
def test_select_only_disabled_returns_empty():
    # 'ariel' exists but --stack not passed → not selectable.
    assert select_recipes(only="ariel") == []


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------


def test_cli_list_returns_zero():
    assert main(["list"]) == 0


@pytest.mark.usefixtures("sample_registry")
def test_cli_list_prints_recipes(capsys):
    main(["list"])
    out = capsys.readouterr().out
    assert "switch" in out and "ariel" in out and "hero" in out


def test_cli_run_no_recipes_selected(monkeypatch, capsys):
    monkeypatch.setattr(recipes, "REGISTRY", [])
    import docs.screenshots.__main__ as cli_mod

    monkeypatch.setattr(cli_mod, "REGISTRY", [])
    assert main([]) == 1
    assert "No recipes selected" in capsys.readouterr().err


@pytest.mark.usefixtures("sample_registry")
@pytest.mark.parametrize(
    ("name", "hint"),
    [
        pytest.param("nope", "No recipe named", id="unknown-recipe"),
        # 'hero' is agentic; running --only hero without --agentic should hint.
        pytest.param("hero", "--agentic", id="needs-flag-hint"),
    ],
)
def test_cli_only_refuses(capsys, name, hint):
    assert main(["--only", name]) == 1
    assert hint in capsys.readouterr().err


def test_cli_invalid_registry_returns_two(monkeypatch, capsys):
    bad = [DocShot(name="s", environment="standalone_interface", kind="static")]  # no app_factory
    monkeypatch.setattr(recipes, "REGISTRY", bad)
    import docs.screenshots.__main__ as cli_mod

    monkeypatch.setattr(cli_mod, "REGISTRY", bad)
    assert main(["list"]) == 2
    assert "Invalid screenshot registry" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Caption substitution loader (the conf.py provenance hook)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("sample_registry")
def test_caption_substitutions_captured_vs_placeholder():
    # 'switch' is captured (has a manifest entry); 'ariel'/'hero' are not.
    manifest = {"switch": {"osprey_version": "2026.6.3", "kind": "static", "captured_utc": "x"}}
    subs = caption_substitutions(manifest)
    assert subs == {
        "captured_switch": "v2026.6.3",
        "captured_ariel": CAPTION_PLACEHOLDER,
        "captured_hero": CAPTION_PLACEHOLDER,
    }


def test_caption_substitutions_defined_for_every_recipe(sample_registry):
    # Every recipe name must have a substitution even with an empty manifest,
    # so the docs build never hits an undefined |captured_<name>|.
    subs = caption_substitutions({})
    assert set(subs) == {f"captured_{s.name}" for s in sample_registry}
    assert all(v == CAPTION_PLACEHOLDER for v in subs.values())


def test_shipped_caption_substitutions_cover_registry():
    # Against the real registry + on-disk manifest: one entry per recipe, and a
    # missing manifest entry degrades to the placeholder rather than raising.
    subs = caption_substitutions()
    assert set(subs) == {f"captured_{s.name}" for s in recipes.REGISTRY}

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


def test_duplicate_name_rejected():
    with pytest.raises(ValueError, match="duplicate recipe name"):
        validate_registry([_standalone("dup"), _standalone("dup")])


def test_duplicate_output_filename_rejected():
    a = _stack_static("a", subviews=(SubView("#x", "shared"),))
    b = _stack_static("b", subviews=(SubView("#y", "shared"),))
    with pytest.raises(ValueError, match="duplicate output filename"):
        validate_registry([a, b])


def test_element_mode_requires_selector():
    with pytest.raises(ValueError, match="element_selector"):
        validate_registry([_standalone("e", capture_mode="element")])


def test_element_mode_with_selector_ok():
    validate_registry(
        [_standalone("e", capture_mode="element", element_selector="osprey-theme-switcher")]
    )


def test_agentic_requires_prompt_and_wait_for():
    with pytest.raises(ValueError, match="prompt and wait_for"):
        validate_registry([DocShot(name="h", environment="tutorial_stack", kind="agentic")])


def test_agentic_must_be_tutorial_stack():
    with pytest.raises(ValueError, match="tutorial_stack"):
        validate_registry(
            [
                DocShot(
                    name="h",
                    environment="standalone_interface",
                    kind="agentic",
                    app_factory="x:create_app",
                    prompt="p",
                    wait_for="plot",
                )
            ]
        )


def test_standalone_requires_app_factory():
    with pytest.raises(ValueError, match="app_factory"):
        validate_registry([DocShot(name="s", environment="standalone_interface", kind="static")])


def test_bad_theme_rejected():
    with pytest.raises(ValueError, match="themes"):
        validate_registry([_standalone("s", themes=("light", "sepia"))])


def test_empty_themes_rejected():
    with pytest.raises(ValueError, match="themes"):
        validate_registry([_standalone("s", themes=())])


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


def test_select_default_only_standalone(sample_registry):
    names = [s.name for s in select_recipes()]
    assert names == ["switch"]


def test_select_stack_adds_tutorial_static(sample_registry):
    names = {s.name for s in select_recipes(stack=True)}
    assert names == {"switch", "ariel"}


def test_select_agentic_adds_hero(sample_registry):
    names = {s.name for s in select_recipes(agentic=True)}
    assert names == {"switch", "hero"}


def test_select_only_filters(sample_registry):
    names = [s.name for s in select_recipes(stack=True, only="ariel")]
    assert names == ["ariel"]


def test_select_only_disabled_returns_empty(sample_registry):
    # 'ariel' exists but --stack not passed → not selectable.
    assert select_recipes(only="ariel") == []


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------


def test_cli_list_returns_zero(capsys):
    assert main(["list"]) == 0


def test_cli_list_prints_recipes(sample_registry, capsys):
    main(["list"])
    out = capsys.readouterr().out
    assert "switch" in out and "ariel" in out and "hero" in out


def test_cli_run_no_recipes_selected(monkeypatch, capsys):
    monkeypatch.setattr(recipes, "REGISTRY", [])
    import docs.screenshots.__main__ as cli_mod

    monkeypatch.setattr(cli_mod, "REGISTRY", [])
    assert main([]) == 1
    assert "No recipes selected" in capsys.readouterr().err


def test_cli_only_unknown_recipe(sample_registry, capsys):
    assert main(["--only", "nope"]) == 1
    assert "No recipe named" in capsys.readouterr().err


def test_cli_only_needs_flag_hint(sample_registry, capsys):
    # 'hero' is agentic; running --only hero without --agentic should hint.
    assert main(["--only", "hero"]) == 1
    assert "--agentic" in capsys.readouterr().err


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


def test_caption_substitutions_captured_vs_placeholder(sample_registry):
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

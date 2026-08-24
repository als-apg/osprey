"""Pin the channel-finder paradigm registry to one source of truth.

:data:`osprey.build.build_tiers.VALID_CHANNEL_FINDER_MODES` is the registry.
Every other place in the tree that enumerates paradigms derives from it, and
the few places that deliberately do not are recorded here so that adding a
paradigm cannot quietly leave one of them behind.
"""

from __future__ import annotations

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES, default_tier_for_mode
from osprey.cli.templates import manager, scaffolding
from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE
from osprey.services.channel_finder.benchmarks.runner import PARADIGM_CONFIG_KEYS


def test_graph_is_a_registered_paradigm():
    """``graph`` is in the registry, which is what makes the subtractions bite.

    Three pins below are written as ``VALID_CHANNEL_FINDER_MODES - {"graph"}``.
    That form is vacuously true if ``graph`` is not registered at all, so it is
    asserted here directly: the exclusions are deliberate carve-outs from a
    paradigm that exists, not leftovers describing one that does not.
    """
    assert "graph" in VALID_CHANNEL_FINDER_MODES


def test_mcp_pipeline_tool_map_serves_graph():
    """The graph pipeline advertises the read-only knowledge-graph vocabulary.

    Pinned because the channel-finder server renders its ``permissions.allow``
    from this map: a missing or misspelled entry ships a graph-mode agent whose
    every tool call is denied, with nothing raising to say so.
    """
    assert CHANNEL_FINDER_TOOLS_BY_PIPELINE["graph"] == [
        "capabilities",
        "example_queries",
        "get_schema",
        "read_cypher",
    ]


def test_scaffolding_paradigms_is_the_registry_itself():
    """``scaffolding._ALL_PARADIGMS`` is an alias, not a second list."""
    assert scaffolding._ALL_PARADIGMS is VALID_CHANNEL_FINDER_MODES


def test_manager_enable_flags_cover_every_paradigm():
    """Template flags are one ``enable_<mode>`` per registered paradigm."""
    expected = {f"enable_{mode}" for mode in VALID_CHANNEL_FINDER_MODES}
    for mode in VALID_CHANNEL_FINDER_MODES:
        flags = manager._enable_flags(mode)
        assert set(flags) == expected
        # Exactly the selected paradigm is switched on.
        assert flags[f"enable_{mode}"] is True
        assert [name for name, on in flags.items() if on] == [f"enable_{mode}"]


def test_manager_enable_flags_all_false_for_unregistered_mode():
    """An unregistered mode switches nothing on rather than raising."""
    flags = manager._enable_flags("not_a_paradigm")
    assert set(flags) == {f"enable_{mode}" for mode in VALID_CHANNEL_FINDER_MODES}
    assert not any(flags.values())


def test_mcp_pipeline_tool_map_keys_are_registered_paradigms():
    """Every keyed pipeline in the MCP registry is a registered paradigm."""
    assert set(CHANNEL_FINDER_TOOLS_BY_PIPELINE) <= set(VALID_CHANNEL_FINDER_MODES)


def test_benchmark_paradigm_config_keys_exclude_only_graph():
    """The benchmark harness covers every paradigm that has a ``database.path``.

    ``graph`` is the one deliberate exclusion: its store is a graph service,
    not a file the harness can point a paradigm config key at. Written as a
    subtraction so the pin holds whether or not ``graph`` is registered.
    """
    assert set(PARADIGM_CONFIG_KEYS) == set(VALID_CHANNEL_FINDER_MODES) - {"graph"}


def test_default_tier_is_inherited_for_every_paradigm():
    """Inherited rule, pinned here only so a new paradigm can't change it silently.

    ``default_tier_for_mode`` is unchanged by the registry work: ``in_context``
    lands on tier 1, every other paradigm on tier 3. A paradigm added to the
    registry inherits the tier-3 branch, and this test says so out loud.
    """
    for mode in VALID_CHANNEL_FINDER_MODES:
        assert default_tier_for_mode(mode) == (1 if mode == "in_context" else 3)


def _cli_choices(command: str, param: str) -> tuple[str, ...]:
    """The ``click.Choice`` values a ``channel-finder`` option offers."""
    from osprey.cli.channel_finder_cmd import channel_finder

    (option,) = [p for p in channel_finder.commands[command].params if p.name == param]
    return tuple(option.type.choices)


def test_validate_pipeline_choice_is_the_file_backed_paradigms():
    """``validate --pipeline`` offers every paradigm except ``graph``.

    ``validate`` inspects a database file on disk, so it can offer only the
    paradigms whose store is a file. Written as a subtraction so the pin holds
    whether or not ``graph`` is registered.
    """
    expected = tuple(sorted(set(VALID_CHANNEL_FINDER_MODES) - {"graph"}))
    assert _cli_choices("validate", "pipeline") == expected


def test_generate_format_choice_is_the_file_backed_paradigms_plus_all():
    """``generate --format`` offers the same paradigms, plus the ``all`` fan-out.

    ``generate`` writes a database file per paradigm, so it shares
    ``validate``'s exclusion of ``graph`` and adds ``all`` as the
    write-every-format shorthand.
    """
    expected = tuple(sorted(set(VALID_CHANNEL_FINDER_MODES) - {"graph"})) + ("all",)
    assert _cli_choices("generate", "fmt") == expected

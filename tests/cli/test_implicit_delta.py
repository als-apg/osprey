"""A ``personas/<name>.yml`` file resolves as a delta over the profile it lives under.

Living in ``personas/`` beside a ``profile.yml`` IS the inheritance (FR-10): the
file carries only what makes it that persona, and the build merges it over the
host profile rather than reading it standalone. Nothing in the file says so —
the classification is made from its location — so the resolver is the only place
that knows, and everything downstream depends on it getting the merge and the
ANCHOR right.

The anchor is the half that is easy to get wrong. Every profile-relative path —
``data:``, the convention directories, ``.env``, the hash material — resolves
against the profile ROOT, never the delta's own parent. That is what makes one
facility data tree serve the whole stack.

``exclude:`` is the delta's subtractive half, and its SPELLING is the contract.
A bare entry (``channel-finder``) unselects a built-in library artifact; a
qualified one (``agents/channel-finder``) omits the profile's own file of that
name. The two namespaces are disjoint — ``validate_artifacts`` rejects any
top-level ``agents:`` entry the built-in library does not carry — so conflating
them would make excluding a shadow delete the built-in version along with it,
when the point of excluding a shadow is to get the built-in version BACK.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_profile import compute_profile_hash
from osprey.cli.build_profile_load import load_profile, load_profile_document
from osprey.errors import BuildProfileError

# The top-level ``agents:``/``skills:`` lists select BUILT-IN library artifacts
# and nothing else — ``validate_artifacts`` rejects any other name — so the
# fixture has to use real ones. A made-up name here would model a profile the
# build refuses, and would make the two ``exclude:`` spellings look
# interchangeable when they are not.
_HOST = """\
name: Facility
app_template: hello_world
data: data
provider: anthropic
model: sonnet
tier: 1
channel_finder_mode: in_context
skills: [diagnose, session-report]
agents: [channel-finder, data-visualizer]
config:
  a.b: 1
  a.c: 2
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def host(tmp_path: Path) -> Path:
    """A materialized profile root with a data tree, as ``profile new`` emits."""
    root = tmp_path / "facility"
    _write(root / "data" / "channels.json", '{"channels": []}\n')
    _write(root / "profile.yml", _HOST)
    return root


def _persona(host: Path, name: str, body: str) -> Path:
    return _write(host / "personas" / f"{name}.yml", body)


# ── The implicit merge ───────────────────────────────────────────────────────


def test_delta_inherits_every_key_it_does_not_name(host: Path) -> None:
    """The whole point: a persona restates nothing, and still gets everything."""
    persona = _persona(host, "reader", "name: Reader\n")

    profile = load_profile(persona)

    assert profile.name == "Reader"
    assert profile.data_bundle == "hello_world"
    assert profile.provider == "anthropic"
    assert profile.model == "sonnet"
    assert profile.channel_finder_mode == "in_context"
    assert profile.tier == 1


def test_delta_wins_over_the_root_key_by_key(host: Path) -> None:
    """A key the delta names is the persona's; the rest still come from the root."""
    persona = _persona(host, "writer", "name: Writer\ntier: 3\n")

    profile = load_profile(persona)

    assert profile.tier == 3
    assert profile.provider == "anthropic"


def test_root_edits_reach_the_persona(host: Path) -> None:
    """One edit to the host retints the stack — no hand-mirroring into personas.

    The emission side pins that no persona file carries a competing value; this
    is the other half, that the merge then applies the host's.
    """
    persona = _persona(host, "reader", "name: Reader\n")
    assert load_profile(persona).provider == "anthropic"

    _write(host / "profile.yml", _HOST.replace("provider: anthropic", "provider: cborg"))

    assert load_profile(persona).provider == "cborg"


def test_root_extends_chain_is_resolved_before_the_merge(tmp_path: Path) -> None:
    """The root's own inheritance is part of what a persona inherits.

    A delta merges over the root's RESOLVED content, so a value the root only
    has by way of its ``extends`` base still reaches the persona.
    """
    root = tmp_path / "facility"
    _write(root / "base.yml", "name: Base\napp_template: hello_world\nmodel: opus\n")
    _write(root / "profile.yml", "extends: ./base.yml\nname: Facility\nprovider: cborg\n")
    persona = _persona(root, "reader", "name: Reader\n")

    profile = load_profile(persona)

    assert profile.model == "opus"
    assert profile.provider == "cborg"


def test_dotted_config_keys_merge_per_key(host: Path) -> None:
    """``config:`` is a mapping, so a delta adds and overrides keys, not the block.

    A persona that sets one dotted key must not silently drop the rest of the
    facility's configuration.
    """
    persona = _persona(host, "reader", "name: Reader\nconfig:\n  a.b: 99\n  a.d: 4\n")

    config = load_profile(persona).config

    assert config["a.b"] == 99  # overridden
    assert config["a.c"] == 2  # inherited, not dropped
    assert config["a.d"] == 4  # added


def test_two_personas_under_one_root_resolve_independently(host: Path) -> None:
    """Resolving one delta must not mutate the root the next one merges over."""
    reader = _persona(host, "reader", "name: Reader\ntier: 1\n")
    writer = _persona(host, "writer", "name: Writer\ntier: 3\n")

    assert load_profile(reader).tier == 1
    assert load_profile(writer).tier == 3
    assert load_profile(reader).tier == 1


# ── Anchoring at the profile root ────────────────────────────────────────────


def test_relative_paths_anchor_at_the_root_not_at_personas(host: Path) -> None:
    """``data: data`` on the root means the root's tree, read from a persona too.

    Anchored at the delta's own parent it would resolve to
    ``personas/data`` — absent — and every persona would build without the
    facility's channel database.
    """
    persona = _persona(host, "reader", "name: Reader\n")

    loaded = load_profile_document(persona)

    assert loaded.profile_dir == host
    assert loaded.profile.resolved_data_root(loaded.profile_dir) == (host / "data").resolve()


def test_persona_and_root_read_the_same_data_tree(host: Path) -> None:
    """One facility tree for the whole stack — the shared-tree promise."""
    persona = _persona(host, "reader", "name: Reader\n")

    persona_loaded = load_profile_document(persona)
    root_loaded = load_profile_document(host / "profile.yml")

    assert persona_loaded.profile.resolved_data_root(
        persona_loaded.profile_dir
    ) == root_loaded.profile.resolved_data_root(root_loaded.profile_dir)


def test_a_standalone_profile_anchors_at_its_own_directory(tmp_path: Path) -> None:
    """Only ``personas/`` files re-anchor; every other file is unaffected."""
    root = tmp_path / "plain"
    _write(root / "data" / "channels.json", "{}\n")
    profile_path = _write(
        root / "profile.yml", "name: Plain\napp_template: hello_world\ndata: data\n"
    )

    loaded = load_profile_document(profile_path)

    assert loaded.profile_dir == root
    assert loaded.is_persona_delta is False


def test_a_bare_file_outside_any_profile_still_loads(tmp_path: Path) -> None:
    """A one-off YAML in a scratch directory is not a delta and needs no root."""
    profile_path = _write(tmp_path / "scratch.yml", "name: Scratch\napp_template: hello_world\n")

    loaded = load_profile_document(profile_path)

    assert loaded.is_persona_delta is False
    assert loaded.profile_dir == tmp_path


def test_a_personas_file_with_no_root_profile_errors(tmp_path: Path) -> None:
    """An orphan delta names a base that is not there, so it cannot be built.

    Building it standalone would quietly produce a project missing everything
    the delta expected to inherit, which is worse than refusing.
    """
    stray = _write(tmp_path / "elsewhere" / "personas" / "reader.yml", "name: Reader\n")

    with pytest.raises(BuildProfileError) as exc:
        load_profile_document(stray)

    assert "profile root is missing" in str(exc.value)


# ── Rejections ───────────────────────────────────────────────────────────────


def test_extends_inside_personas_is_rejected(host: Path) -> None:
    """A delta already has a base; a second one would be ambiguous.

    Silently honouring it would resolve the persona against something other
    than the profile it lives under — the one thing its location promises.
    """
    persona = _persona(host, "reader", "extends: ../profile.yml\nname: Reader\n")

    with pytest.raises(BuildProfileError) as exc:
        load_profile(persona)

    message = str(exc.value)
    assert "extends" in message
    assert str(persona) in message
    assert str(host / "profile.yml") in message  # says where to edit instead


def test_extends_naming_a_preset_inside_personas_is_rejected_too(host: Path) -> None:
    """Not just path-shaped bases — a preset name is rejected on the same ground."""
    persona = _persona(host, "reader", "extends: hello-world\nname: Reader\n")

    with pytest.raises(BuildProfileError, match="extends"):
        load_profile(persona)


def test_a_delta_whose_root_is_a_yaml_list_errors(tmp_path: Path) -> None:
    """An unusable root is named, not silently treated as an empty base."""
    root = tmp_path / "facility"
    _write(root / "profile.yml", "- not\n- a mapping\n")
    persona = _persona(root, "reader", "name: Reader\n")

    with pytest.raises(BuildProfileError, match="must be a YAML mapping"):
        load_profile(persona)


def test_a_missing_profile_file_errors(tmp_path: Path) -> None:
    with pytest.raises(BuildProfileError, match="Profile not found"):
        load_profile(tmp_path / "nope.yml")


# ── exclude: list subtraction ────────────────────────────────────────────────


def test_delta_exclude_subtracts_from_the_merged_result(host: Path) -> None:
    """``exclude:`` removes what the ROOT contributed — that is its whole job.

    Regression guard for the resolver's central foot-gun: resolving the delta
    through the ``extends`` machinery before merging consumes its ``exclude:``
    against the delta's own layer, where the entry is not present, so the
    exclusion silently vanishes and the persona keeps a skill it disowned.
    """
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  skills: [session-report]\n")

    profile = load_profile(persona)

    assert profile.skills == ["diagnose"]


def test_delta_exclude_across_several_fields(host: Path) -> None:
    persona = _persona(
        host,
        "reader",
        "name: Reader\nexclude:\n"
        "  skills: [diagnose, session-report]\n"
        "  agents: [channel-finder, data-visualizer]\n",
    )

    profile = load_profile(persona)

    assert profile.skills == []
    assert profile.agents == []


def test_exclude_is_consumed_and_never_reaches_the_model(host: Path) -> None:
    """``exclude`` is a resolution directive, not profile content."""
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  skills: [session-report]\n")

    assert not hasattr(load_profile(persona), "exclude")


def test_excluding_an_absent_entry_is_a_silent_noop(host: Path) -> None:
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  skills: [never-there]\n")

    assert load_profile(persona).skills == ["diagnose", "session-report"]


# ── exclude: convention-artifact records ─────────────────────────────────────


def test_excluding_a_shadow_keeps_the_built_in_selected(host: Path) -> None:
    """FR-10's binding criterion: excluding a shadow RESTORES the built-in version.

    A profile that ships ``agents/channel-finder.md`` shadows the built-in agent
    of the same name. The qualified spelling drops only that file, so the
    still-selected built-in agent renders again. Dropping the list entry too
    would make the agent vanish altogether — the opposite of restoring it.
    """
    _write(host / "agents" / "channel-finder.md", "facility override\n")
    persona = _persona(
        host, "reader", "name: Reader\nexclude:\n  agents: [agents/channel-finder]\n"
    )

    loaded = load_profile_document(persona)

    assert loaded.excluded_artifacts == frozenset({"agents/channel-finder"})
    assert "channel-finder" in loaded.profile.agents


def test_the_bare_spelling_unselects_the_built_in_without_omitting_a_file(host: Path) -> None:
    """Bare names live in the built-in library's namespace, not the profile's.

    ``validate_artifacts`` rejects any ``agents:`` entry the library does not
    carry, so a bare entry can only mean "stop selecting the built-in one".
    """
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  agents: [channel-finder]\n")

    loaded = load_profile_document(persona)

    assert loaded.profile.agents == ["data-visualizer"]
    assert loaded.excluded_artifacts == frozenset()


def test_the_two_spellings_are_not_interchangeable(host: Path) -> None:
    """Pinned directly: the split is the contract, not an implementation detail."""
    bare = _persona(host, "bare", "name: R\nexclude:\n  agents: [channel-finder]\n")
    qualified = _persona(host, "qual", "name: R\nexclude:\n  agents: [agents/channel-finder]\n")

    bare_loaded = load_profile_document(bare)
    qualified_loaded = load_profile_document(qualified)

    assert bare_loaded.excluded_artifacts != qualified_loaded.excluded_artifacts
    assert bare_loaded.profile.agents != qualified_loaded.profile.agents


def test_a_bare_non_library_name_records_a_file_omission(host: Path) -> None:
    """It cannot be a built-in selection, so the profile's own file is what it means."""
    _write(host / "agents" / "orbit-writer.md", "facility agent\n")
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  agents: [orbit-writer]\n")

    assert load_profile_document(persona).excluded_artifacts == frozenset({"agents/orbit-writer"})


def test_a_mismatched_category_prefix_is_rejected(host: Path) -> None:
    """The key and the prefix disagree about which artifact is meant.

    Guessing either way excludes the wrong file or nothing at all, both of
    which look like the exclusion worked.
    """
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  agents: [skills/diagnose]\n")

    with pytest.raises(BuildProfileError, match="skills"):
        load_profile(persona)


def test_a_qualified_prefix_under_a_non_convention_field_is_rejected_too(host: Path) -> None:
    """``hooks:`` has no convention directory, so the prefix is a mistake there too.

    Passing it through as a literal hook name would match nothing and read as a
    successful exclusion.
    """
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  hooks: [agents/channel-finder]\n")

    with pytest.raises(BuildProfileError, match="agents"):
        load_profile(persona)


def test_a_convention_dir_with_no_list_field_can_be_excluded(host: Path) -> None:
    """``commands/`` has no profile list, so exclusion is the only way to drop one."""
    _write(host / "commands" / "scan.md", "slash command\n")
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  commands: [scan]\n")

    assert load_profile_document(persona).excluded_artifacts == frozenset({"commands/scan"})


def test_the_hyphenated_directory_is_reachable_by_its_field_spelling(host: Path) -> None:
    """A profile writes ``output_styles``; the directory is ``output-styles/``."""
    _write(host / "output-styles" / "terse.md", "style\n")
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  output_styles: [terse]\n")

    assert load_profile_document(persona).excluded_artifacts == frozenset({"output-styles/terse"})


def test_an_underscored_directory_is_not_rewritten(host: Path) -> None:
    """``mcp_servers/`` really does carry an underscore — it must survive intact."""
    (host / "mcp_servers" / "facility").mkdir(parents=True)
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  mcp_servers: [facility]\n")

    assert load_profile_document(persona).excluded_artifacts == frozenset({"mcp_servers/facility"})


def test_a_dict_shaped_declaration_is_dropped_with_its_files(host: Path) -> None:
    """Omitting an MCP server's sources while keeping its declaration deploys a
    server whose code is not there — and there is no built-in one to fall back
    to, so both halves go together whichever spelling is used."""
    (host / "mcp_servers" / "facility").mkdir(parents=True)
    _write(
        host / "profile.yml",
        _HOST + "mcp_servers:\n  facility:\n    command: run-it\n  other:\n    command: keep-me\n",
    )
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  mcp_servers: [facility]\n")

    loaded = load_profile_document(persona)

    assert sorted(loaded.profile.mcp_servers) == ["other"]
    assert loaded.excluded_artifacts == frozenset({"mcp_servers/facility"})


def test_a_field_that_is_neither_list_nor_convention_dir_is_rejected(host: Path) -> None:
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  provider: [anthropic]\n")

    with pytest.raises(BuildProfileError, match="unknown or non-list field"):
        load_profile(persona)


def test_a_non_list_exclude_value_is_rejected(host: Path) -> None:
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  skills: not-a-list\n")

    with pytest.raises(BuildProfileError, match="must be a list"):
        load_profile(persona)


def test_a_non_string_exclude_entry_is_rejected(host: Path) -> None:
    """A mapping where a name belongs used to surface as a bare ``TypeError``."""
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  skills:\n    - a: 1\n")

    with pytest.raises(BuildProfileError, match="exclude.skills"):
        load_profile(persona)


def test_an_exclusion_matching_nothing_is_warned_about(host: Path, caplog) -> None:
    """A misspelled shadow exclusion is otherwise a silent no-op.

    A warning rather than an error: a root may exclude something only some of
    its personas carry.
    """
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  agents: [agents/typo-here]\n")

    with caplog.at_level("WARNING"):
        load_profile_document(persona)

    assert "typo-here" in caplog.text


def test_the_hash_path_does_not_repeat_the_warning(host: Path, caplog) -> None:
    """A build loads AND hashes the same profile; the diagnostic belongs to one.

    Both go through the same resolver, so without suppressing it on the hash
    side every build prints each stale exclusion twice — and a warning that
    always repeats is one people stop reading.
    """
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  agents: [agents/typo-here]\n")

    with caplog.at_level("WARNING"):
        compute_profile_hash(persona)

    assert "typo-here" not in caplog.text


def test_a_matching_exclusion_is_not_warned_about(host: Path, caplog) -> None:
    """The warning must not cry wolf on the case it exists to contrast with."""
    _write(host / "agents" / "channel-finder.md", "facility override\n")
    persona = _persona(
        host, "reader", "name: Reader\nexclude:\n  agents: [agents/channel-finder]\n"
    )

    with caplog.at_level("WARNING"):
        load_profile_document(persona)

    assert "channel-finder" not in caplog.text


def test_a_bare_exclusion_of_a_shadowed_name_is_warned_about(host: Path, caplog) -> None:
    """The one exclusion mistake nothing else can see.

    A bare name only unselects the built-in artifact. When the profile also
    ships its own file of that name, the file keeps rendering, so the project
    comes out byte-identical — and because a bare library selection records no
    file omission, ``_warn_unmatched_exclusions`` has nothing to look at either.
    Without this the author's intent fails in complete silence.
    """
    _write(host / "agents" / "channel-finder.md", "facility override\n")
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  agents: [channel-finder]\n")

    with caplog.at_level("WARNING"):
        load_profile_document(persona)

    assert "channel-finder" in caplog.text
    # The remedy, not just the diagnosis: the qualified spelling is what the
    # author almost certainly meant, so the warning has to name it.
    assert "'agents/channel-finder'" in caplog.text


def test_a_bare_exclusion_of_an_unshadowed_name_is_not_warned_about(host: Path, caplog) -> None:
    """Unselecting a built-in the profile does NOT shadow is the ordinary case."""
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  agents: [channel-finder]\n")

    with caplog.at_level("WARNING"):
        load_profile_document(persona)

    assert "channel-finder" not in caplog.text


def test_the_shadow_warning_is_suppressed_on_the_hash_path(host: Path, caplog) -> None:
    """Same reason as every other exclusion diagnostic: a build loads AND hashes."""
    _write(host / "agents" / "channel-finder.md", "facility override\n")
    persona = _persona(host, "reader", "name: Reader\nexclude:\n  agents: [channel-finder]\n")

    with caplog.at_level("WARNING"):
        compute_profile_hash(persona)

    assert "channel-finder" not in caplog.text


def test_a_root_exclusion_reaches_its_personas(host: Path) -> None:
    """The root's own exclusions are part of what a persona inherits."""
    _write(host / "agents" / "orbit-writer.md", "facility agent\n")
    _write(host / "profile.yml", _HOST + "exclude:\n  agents: [agents/orbit-writer]\n")
    persona = _persona(host, "reader", "name: Reader\n")

    assert load_profile_document(persona).excluded_artifacts == frozenset({"agents/orbit-writer"})


def test_a_profile_that_excludes_nothing_records_nothing(host: Path) -> None:
    """No exclusions means an empty record, not a missing one."""
    assert load_profile_document(host / "profile.yml").excluded_artifacts == frozenset()


# ── The loader and the hash resolve the same document ────────────────────────


def test_a_root_edit_the_loader_sees_also_moves_the_hash(host: Path) -> None:
    """Behavioral pin on the shared resolver: whatever changes what a persona
    BUILDS must change what its staleness advisory compares.

    Asserting the two agree on a merged dict would only pin that they call the
    same function today; this pins the consequence, which is what the advisory
    actually promises.
    """
    persona = _persona(host, "reader", "name: Reader\n")
    before_provider = load_profile_document(persona).profile.provider
    before_hash = compute_profile_hash(persona)

    _write(host / "profile.yml", _HOST.replace("provider: anthropic", "provider: cborg"))

    assert load_profile_document(persona).profile.provider != before_provider
    assert compute_profile_hash(persona) != before_hash


def test_a_convention_only_exclusion_moves_the_hash(host: Path) -> None:
    """Two personas differing only by an exclusion build different projects.

    ``commands:`` has no list to subtract from, so the exclusion lives entirely
    in the derived record — invisible to a hash over the resolved YAML. Without
    folding that record, neither persona would ever report itself stale against
    the other.
    """
    _write(host / "commands" / "scan.md", "slash command\n")
    plain = _persona(host, "plain", "name: Reader\n")
    excluding = _persona(host, "excl", "name: Reader\nexclude:\n  commands: [scan]\n")

    assert compute_profile_hash(excluding) != compute_profile_hash(plain)


def test_a_shadowed_bare_exclusion_still_moves_the_hash(host: Path) -> None:
    """Pinned as a decision, not an accident: the hash folds the DECLARED lists.

    This exclusion renders byte-identical output — the shadow file is copied
    either way — so the hash move is a false positive on staleness. It is the
    accepted direction to be wrong in: deriving the effective post-shadow
    selection would duplicate the build's shadow rule inside the hash, and drift
    between the two would make the hash miss a REAL divergence, which is the
    failure the advisory exists to catch. An extra idempotent rebuild is cheap;
    a project that silently stops matching its source is not.
    """
    _write(host / "agents" / "channel-finder.md", "facility override\n")
    plain = _persona(host, "plain", "name: Reader\n")
    excluding = _persona(host, "excl", "name: Reader\nexclude:\n  agents: [channel-finder]\n")

    assert compute_profile_hash(excluding) != compute_profile_hash(plain)


def test_a_profile_without_exclusions_hashes_as_it_did_before(host: Path) -> None:
    """Folding the record must not disturb the digest of everything else.

    The empty record folds nothing, so bundled presets and ordinary profiles
    keep the hashes their manifests were stamped with.
    """
    plain = _persona(host, "plain", "name: Reader\n")
    same = _persona(host, "same", "name: Reader\n")

    assert compute_profile_hash(plain) == compute_profile_hash(same)

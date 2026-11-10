"""``osprey init --revalidate``: a repo made before tiers and shared logins, rewritten in place.

The previous release's ``profile.yml`` is the current exemplar's with the
retired swaps put back (``_RETIRED_PROFILE_SWAPS``): the default-tier and
tier-pin keys and the render catalog in their retired spelling, the logbook and
knowledge cards as users pinned to a tier, and the retired write-posture keys,
hooks and stanzas. The fixture under ``tests/fixtures/previous_release_repo/``
holds the five retired layer-directory files exactly as that release emitted
them (header, no provenance block) and the logbook and knowledge context
slots. Every other file of the repo — README, data tree, provider catalog, the
``channel-finder`` shared login the previous release had no word for — is the current
exemplar's own, because the rewrite never touches it. So the gate is exact: the
rewritten repo is the new exemplar, byte for byte.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_profile_document import (
    RETIRED_CATALOG_KEY,
    RETIRED_DEFAULT_TIER_KEY,
    RETIRED_TIER_KEY,
)
from osprey.cli.init_cmd import init
from osprey.cli.init_revalidate import RETIRED_PRESETS, _content_hash, _text_hash
from tests._retired_layout import RETIRED_LAYER_DIRNAME
from tests.fixtures.lifecycle_repo import exemplar_source_files, expand_sentinels

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "previous_release_repo"

#: Exemplar files the previous release wrote differently; the fixture holds those. It held
#: no application, and revalidating adds none: a deployment holds what it chose to.
_MIGRATED = (
    "apps/",
    "profile.yml",
    "tiers/readonly.yml",
    "tiers/readwrite.yml",
    "tiers/admin.yml",
    "tiers/logbook.yml",
    "tiers/knowledge.yml",
)

_CHANNEL_FINDER_ITEM = (
    "      - channel-finder   # Channel Finder: channel addresses from the hierarchy\n"
)


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's git config (signing, templates) out of the rewrite's `git mv`."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "OSPREY")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "osprey@example.org")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout


def _previous_release_repo(dest: Path, *, git: bool = False) -> Path:
    """Lay the previous release's repo at *dest* (see the module docstring)."""
    dest.mkdir(parents=True, exist_ok=True)
    for rel, text in exemplar_source_files().items():
        if rel.startswith(_MIGRATED):
            continue
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for source in FIXTURE.rglob("*"):
        if source.is_file():
            path = dest / source.relative_to(FIXTURE)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expand_sentinels(source.read_text(encoding="utf-8")), encoding="utf-8")
    (dest / "profile.yml").write_text(previous_release_profile(), encoding="utf-8")
    if git:
        _git(dest, "init", "-q")
        _git(dest, "add", "-A")
        _git(dest, "commit", "-q", "-m", "previous release")
    return dest


#: The retired write keys the previous release's readwrite and logbook files spell.
_READWRITE_RETIRED_KEYS = (
    "  control_system.writes_enabled: false\n"
    "  control_system.connector.epics.writes_enabled: false\n"
    "  control_system.connector.virtual_accelerator.writes_enabled: true\n"
)
_LOGBOOK_RETIRED_KEYS = (
    "  control_system.writes_enabled: false\n"
    "  control_system.connector.epics.writes_enabled: false\n"
    "  control_system.connector.virtual_accelerator.writes_enabled: false\n"
)


def _without_channel_finder(root: Path) -> None:
    """Make the fixture's profile the previous release's exactly: no `shared:` at all."""
    path = root / "profile.yml"
    text = path.read_text(encoding="utf-8")
    block = re.search(
        r"    # Logins everyone above.*?" + re.escape(_CHANNEL_FINDER_ITEM), text, re.S
    )
    assert block is not None
    path.write_text(text[: block.start()] + text[block.end() :], encoding="utf-8")
    (root / "tiers" / "channel-finder.yml").unlink()


def _repo_files(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def _revalidate(root: Path, *extra: str) -> object:
    return CliRunner().invoke(init, [str(root), "--revalidate", *extra], catch_exceptions=False)


#: The previous release's root armed writes (``writes_enabled: true``), which
#: the parse point reads as ``allowed``; the rewrite keeps that ceiling (an
#: existing ``allowed`` is never lowered), where a fresh repo ships ``asked``.
_KEPT_CEILING = ("  control_system.writes: asked\n", "  control_system.writes: allowed\n")


def revalidated_exemplar() -> dict[str, str]:
    """The new exemplar as the previous release's repo revalidates to it."""
    files = exemplar_source_files()
    asked, allowed = _KEPT_CEILING
    assert files["profile.yml"].count(asked) == 1
    files["profile.yml"] = files["profile.yml"].replace(asked, allowed)
    return files


#: ``(exemplar text, previous-release text)``: each pair puts back one thing the
#: previous release's ``profile.yml`` wrote and the revalidate rewrite replaces.
_RETIRED_PROFILE_SWAPS: tuple[tuple[str, str], ...] = (
    #: The retired helper hooks.
    (
        "hooks:\n",
        "hooks:\n"
        "  - hook-log          # Append every tool call to a structured JSONL audit log\n"
        "  - target-state      # Stdlib reader for the control-target state file (helper, not wired to an event)\n"
        "  - hook-config       # Inject project config.yml path into every tool call env\n",
    ),
    #: The facility rule at its previous-release path, outside rules/.
    (
        "  - rules/facility      # Facility-level conventions (naming, units, logbook)\n",
        "  - facility            # Facility-level conventions (naming, units, logbook)\n",
    ),
    #: The retired write-posture keys and their stanzas.
    (
        "  - writes-check      # Kill switch: refuse every write while its machine's write switch is off\n",
        "  - writes-check      # Kill switch: refuse every write while writes_enabled is false\n",
    ),
    (
        "  # Write ceiling, the FIRST guard in the write-safety chain: `never` refuses\n"
        "  # every hardware write before the limits check or the approval prompt is\n"
        "  # consulted, `asked` lets a login's write switch reach the position that\n"
        "  # prompts, and `allowed` also the position that does not. This preset ships\n"
        "  # `asked`; the read-only tier pins `never`, the admin tier `allowed`. Each\n"
        "  # login's own switch for each machine decides, so a ceiling arms nothing.\n"
        "  control_system.writes: asked\n",
        "  # Master write switch, the FIRST guard in the write-safety chain: while\n"
        "  # false, every hardware write is refused before the limits check or the\n"
        "  # approval prompt is consulted. On here because the baseline is the\n"
        "  # simulator, which cannot move a magnet; the read-only tier pins it off.\n"
        "  # Write posture is per connector type: a `control_system.connector.<type>.\n"
        "  # writes_enabled` overrides this for that type alone, and only a literal\n"
        "  # `true` arms writes at either level.\n"
        "  control_system.writes_enabled: true\n",
    ),
    (
        "  control_system.connector.virtual_accelerator.timeout_s: 5.0\n",
        "  control_system.connector.virtual_accelerator.timeout_s: 5.0\n"
        "  # Write posture for the simulator alone. Uncomment to arm writes here while\n"
        "  # the master switch keeps the live machine read-only; the shipped\n"
        "  # `readwrite` and `admin` tiers in tiers/ write this key.\n"
        "  # control_system.connector.virtual_accelerator.writes_enabled: true\n",
    ),
    (
        "  # control_system.connector.epics.step_read_timeout_s: 2.0\n",
        "  # control_system.connector.epics.step_read_timeout_s: 2.0\n"
        "  # Write posture for the live machine. Stating it pins it: a type with its\n"
        "  # own posture never falls back to the master switch.\n"
        "  # control_system.connector.epics.writes_enabled: false\n",
    ),
    (
        "  # long a call is given and the probe channel every connector type answers.\n",
        "  # long a call is given and the two leaves every connector type answers.\n",
    ),
    (
        "  # control_system.connector.doocs.timeout_s: 5.0\n",
        "  # control_system.connector.doocs.timeout_s: 5.0\n"
        "  # Write posture for the DOOCS machine. Same tri-state as the epics leaf\n"
        "  # above: stating it pins it, and only a literal true arms writes.\n"
        "  # control_system.connector.doocs.writes_enabled: false\n",
    ),
    (
        "  # Device attribute the target switch reads to prove this machine is reachable.\n",
        "  # The same two leaves as every other connector type, for the TANGO machine.\n"
        "  # control_system.connector.tango.writes_enabled: false\n",
    ),
    #: The per-tool approval policies for the tools the write switch now governs.
    (
        "  # still pauses for a yes/no prompt, unless its slider stands at `allowed`.\n"
        "  # Applied by the approval hook, so it reaches only hook-wired tools; the\n"
        "  # health check, channel-finder queries and most workspace tools are gated by\n"
        "  # the rendered settings.json permissions.\n",
        "  # still pauses for a yes/no prompt. Applied by the approval hook, so it\n"
        "  # reaches only hook-wired tools; the health check, channel-finder queries and\n"
        "  # most workspace tools are gated by the rendered settings.json permissions.\n",
    ),
    (
        "  # Per-tool policies: always (prompt every time) or skip (no prompt). A tool\n"
        "  # the write switch governs, channel writes and Python execution among them,\n"
        "  # takes no entry here: the slider on the control-target chip decides for it.\n",
        "  # Per-tool policies: always (prompt every time), skip (no prompt), or\n"
        "  # selective (content-aware, for `execute` only; other tools read it as\n"
        "  # always).\n"
        "  approval.tools.channel_write: always\n",
    ),
    (
        "  approval.tools.archiver_read: skip\n",
        "  approval.tools.archiver_read: skip\n"
        "  # Python execution: content-aware, so a script that only reads runs unprompted.\n"
        "  approval.tools.execute: selective\n",
    ),
    #: The previous stand-in banner and control-target chip names.
    (
        "  # meets there is a real machine's behaviour — the write switch, approval\n"
        '  # prompts, limit refusals — under the "Rehearsal — nothing moves" banner,\n'
        "  # on something that cannot move a magnet.\n",
        "  # meets there is a real machine's behaviour — approval prompts, limit\n"
        "  # refusals, the LIVE MACHINE (stand-in) banner — on something that cannot\n"
        "  # move a magnet.\n",
    ),
    (
        '  # control-target chip. Defaults: "Live", "Rehearsal", "Simulator".\n'
        "  # control_system.target_display_names.live: Live\n",
        '  # control-target chip. Defaults: "Real machine", "Rehearsal", "Simulator".\n'
        "  # control_system.target_display_names.live: Real machine\n",
    ),
    #: The retired roster keys and render catalog; the previous release had no apps.
    (
        "    default_tier: readonly   # used for any user below with no tier\n",
        "    default_persona: readonly   # used for any user below with no tier\n",
    ),
    (
        "        # `shared` renders one card per entry in `shared:` below, titled by the\n"
        "        # login's own `name:` in tiers/. Everyone on the roster opens them.\n"
        "        - type: shared\n"
        "          label: Shared logins\n",
        "",
    ),
    (
        "        tier: readwrite\n",
        "        persona: readwrite\n",
    ),
    (
        "        tier: readonly\n",
        "        persona: readonly\n",
    ),
    (
        '        display_name: "Read-Only View (Bob)"\n',
        '        display_name: "Read-Only View (Bob)"\n'
        "      # Not a person: a second product running beside them. Same machinery as\n"
        "      # any other entry — its own container, ports and volumes — but what is\n"
        "      # behind the card is the ARIEL logbook assistant, not a control terminal.\n"
        "      - name: logbook\n"
        "        index: 3\n"
        "        persona: logbook\n"
        '        display_name: "Logbook Research"\n'
        "        # A shared card: everyone on this roster opens it with their own\n"
        "        # password. It has no password of its own, so none is set in .env.\n"
        "        access: any\n"
        "      # The second product beside them: the facility knowledge layer — the\n"
        "      # knowledge graph, the graph channel finder and the knowledge bundle —\n"
        "      # read-only, with no control system behind it. Shared like the card\n"
        "      # above; the page follows roster order, so it sits with its sibling.\n"
        "      - name: knowledge\n"
        "        index: 4\n"
        "        persona: knowledge\n"
        '        display_name: "Facility Knowledge"\n'
        "        access: any\n",
    ),
    (
        "        tier: admin\n",
        "        persona: admin\n",
    ),
    (
        '        display_name: "Deployment Admin (Carol)"\n',
        '        display_name: "Deployment Admin (Carol)"\n'
        "    personas:\n"
        "      # `osprey build` builds one of these per file in personas/, into build/.\n"
        "      # `osprey up` builds nothing: if one is missing it stops and says so.\n"
        "      readonly:\n"
        "        build_profile: personas/readonly.yml\n"
        "      readwrite:\n"
        "        build_profile: personas/readwrite.yml\n"
        "      admin:\n"
        "        build_profile: personas/admin.yml\n"
        "      logbook:\n"
        "        build_profile: personas/logbook.yml\n"
        "        # Puts this persona's users under their own landing-page heading\n"
        "        # instead of in with the people. Presentation only — it changes\n"
        "        # nothing about the container, its ports, or what it can do.\n"
        "        landing_group: Standalone deployments\n"
        "      knowledge:\n"
        "        build_profile: personas/knowledge.yml\n"
        "        landing_group: Standalone deployments\n",
    ),
    (
        "      - logbook          # Logbook Research: search and synthesise the logbook\n"
        "      - knowledge        # Facility Knowledge: how the machine is built\n",
        "",
    ),
    #: The previous version floor and provider-catalog verb.
    (
        "requires_osprey_version: '>=2026.10.0'\n",
        "requires_osprey_version: '>=2026.9.0'\n",
    ),
    (
        "# `osprey init --revalidate` adds the packaged entries and keys it lacks.\n",
        "# `osprey profile expand --providers` refreshes the packaged entries in it.\n",
    ),
)


def previous_release_profile() -> str:
    """The previous release's ``profile.yml``: the exemplar's, retired swaps put back."""
    exemplar = exemplar_source_files()["profile.yml"]
    for old, _ in _RETIRED_PROFILE_SWAPS:
        assert exemplar.count(old) == 1, old
    text = exemplar
    for old, new in _RETIRED_PROFILE_SWAPS:
        text = text.replace(old, new)
    return text


# ── The gate ────────────────────────────────────────────────────────────────


def test_revalidated_previous_release_repo_is_byte_equal_to_the_new_exemplar(
    tmp_path: Path,
) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar", git=True)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    expected = revalidated_exemplar()
    actual = _repo_files(root)
    assert sorted(actual) == sorted(expected)
    for rel, text in expected.items():
        assert actual[rel] == text, f"{rel} differs from the new exemplar"


def test_the_diff_is_printed(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar", git=True)

    output = _revalidate(root).output

    assert f"-    {RETIRED_DEFAULT_TIER_KEY}: readonly" in output
    assert "+    default_tier: readonly" in output
    assert "+++ b/tiers/logbook.yml" in output
    assert "+++ b/tiers/readonly.yml" in output
    assert "-requires_osprey_version: '>=2026.9.0'" in output
    assert "Left for you to do by hand" not in output


def test_retired_layer_dir_is_moved_with_git_mv(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar", git=True)

    _revalidate(root)

    staged = _git(root, "diff", "--cached", "--name-status", "-M")
    assert re.search(
        rf"^R\d*\t{RETIRED_LAYER_DIRNAME}/readonly.yml\ttiers/readonly.yml$", staged, re.M
    ), staged
    assert not (root / f"{RETIRED_CATALOG_KEY}").exists()


def test_previous_release_without_shared_gains_the_shared_block(tmp_path: Path) -> None:
    """The previous release had no `shared:`; the rewrite writes it as init does today."""
    root = _previous_release_repo(tmp_path / "als-exemplar")
    _without_channel_finder(root)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    expected = revalidated_exemplar()["profile.yml"].replace(_CHANNEL_FINDER_ITEM, "")
    assert (root / "profile.yml").read_text(encoding="utf-8") == expected
    assert not (root / "apps").exists()
    shared = sorted(p.name for p in (root / "tiers").iterdir() if "lists: own" in p.read_text())
    assert shared == ["knowledge.yml", "logbook.yml"]


def test_web_terminal_context_of_moved_cards_is_removed(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")

    _revalidate(root)

    slots = sorted(p.name for p in (root / "web-terminal-context").iterdir())
    assert slots == ["alice", "bob", "carol"]


def test_a_second_run_rewrites_nothing(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    _revalidate(root)
    before = _repo_files(root)

    result = _revalidate(root)

    assert result.exit_code == 0
    assert "nothing to rewrite" in result.output
    assert _repo_files(root) == before


def test_a_swap_whose_exemplar_text_moved_names_that_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = exemplar_source_files()
    tier_line = "    default_tier: readonly   # used for any user below with no tier\n"
    assert files["profile.yml"].count(tier_line) == 1
    files["profile.yml"] = files["profile.yml"].replace(tier_line, "    default_tier: operator\n")
    monkeypatch.setattr(sys.modules[__name__], "exemplar_source_files", lambda: files)

    with pytest.raises(AssertionError, match=re.escape(tier_line.strip())):
        _previous_release_repo(tmp_path / "als-exemplar")


# ── What the rewrite leaves alone ───────────────────────────────────────────


def test_an_edited_logbook_retired_tier_stays_a_tier_and_the_manual_step_is_printed(
    tmp_path: Path,
) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    layer_file = root / f"{RETIRED_CATALOG_KEY}" / "logbook.yml"
    layer_file.write_text(
        layer_file.read_text(encoding="utf-8").replace(
            "default_panel: ariel", "default_panel: workspace"
        ),
        encoding="utf-8",
    )

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert "lists: own" not in (root / "tiers" / "logbook.yml").read_text(encoding="utf-8")
    assert not (root / "apps").exists()
    assert "lists: own" in (root / "tiers" / "knowledge.yml").read_text(encoding="utf-8")
    profile = (root / "profile.yml").read_text(encoding="utf-8")
    assert "        tier: logbook\n" in profile
    assert "      - name: knowledge\n" not in profile
    assert "Left for you to do by hand" in result.output
    assert "tiers/logbook.yml was made from `control-assistant-logbook`" in result.output


def test_the_manual_step_for_an_edited_logbook_names_its_model(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    layer_file = root / f"{RETIRED_CATALOG_KEY}" / "logbook.yml"
    layer_file.write_text(
        layer_file.read_text(encoding="utf-8") + "model: claude-sonnet-5\n", encoding="utf-8"
    )

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert (root / "tiers" / "logbook.yml").is_file()
    assert "with your edits, its `model: claude-sonnet-5` among them, in it" in " ".join(
        result.output.split()
    )
    assert "model: claude-sonnet-5" in (root / "tiers" / "logbook.yml").read_text(encoding="utf-8")


def test_an_edited_tier_retired_tier_is_kept_as_written(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    layer_file = root / f"{RETIRED_CATALOG_KEY}" / "readwrite.yml"
    edited = layer_file.read_text(encoding="utf-8") + "# a local note\nweb_panels: [lattice]\n"
    layer_file.write_text(edited, encoding="utf-8")

    result = _revalidate(root)

    assert (root / "tiers" / "readwrite.yml").read_text(encoding="utf-8") == edited
    assert "tiers/readwrite.yml was made from `control-assistant-readwrite`" in result.output


def test_catalog_knobs_move_to_the_top_of_the_tier_file(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    profile = root / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            f"        build_profile: {RETIRED_LAYER_DIRNAME}/readwrite.yml\n",
            f"        build_profile: {RETIRED_LAYER_DIRNAME}/readwrite.yml\n"
            "        extra_mounts:\n"
            "          - /opt/facility/data:/data:ro\n"
            "        image: registry.example.org/rw:1\n",
        ),
        encoding="utf-8",
    )

    _revalidate(root)

    tier = (root / "tiers" / "readwrite.yml").read_text(encoding="utf-8")
    assert re.search(
        r"^name: .*\nextra_mounts:\n  - /opt/facility/data:/data:ro\nimage: registry.example.org/rw:1\n",
        tier,
        re.M,
    ), tier
    assert f"{RETIRED_CATALOG_KEY}:" not in (root / "profile.yml").read_text(encoding="utf-8")


def test_variant_overlays_are_rewritten(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    overlay = root / "profiles" / "night.yml"
    overlay.parent.mkdir()
    overlay.write_text(
        "config:\n"
        "  # who lands where at night\n"
        f"  modules.web_terminals.{RETIRED_DEFAULT_TIER_KEY}: readwrite  # night shift\n"
        "  modules.web_terminals.authorization:\n"
        "    roles:\n"
        "      operator:\n"
        f"        {RETIRED_TIER_KEY}: readwrite\n",
        encoding="utf-8",
    )

    _revalidate(root)

    assert overlay.read_text(encoding="utf-8") == (
        "config:\n"
        "  # who lands where at night\n"
        "  modules.web_terminals.default_tier: readwrite  # night shift\n"
        "  modules.web_terminals.authorization:\n"
        "    roles:\n"
        "      operator:\n"
        "        tier: readwrite\n"
    )


def test_one_key_spelled_both_ways_is_refused_and_nothing_is_touched(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    profile = root / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            f"        {RETIRED_TIER_KEY}: readonly\n", "        tier: readonly\n"
        ),
        encoding="utf-8",
    )
    before = _repo_files(root)

    result = CliRunner().invoke(init, [str(root), "--revalidate"])

    assert result.exit_code == 2
    assert f"users[].{RETIRED_TIER_KEY}" in result.output and "users[].tier" in result.output
    assert _repo_files(root) == before


# ── Refusals ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "flags",
    [["--force"], ["--reset"], ["-d"], ["--dev"]],
    ids=["force", "reset", "detach", "dev"],
)
def test_revalidate_refuses_the_flags_that_make_a_repo(tmp_path: Path, flags: list[str]) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    before = _repo_files(root)

    result = CliRunner().invoke(init, [str(root), "--revalidate", *flags])

    assert result.exit_code == 2
    assert flags[0] in result.output
    assert _repo_files(root) == before


def test_revalidate_refuses_a_directory_without_a_profile(tmp_path: Path) -> None:
    result = CliRunner().invoke(init, [str(tmp_path), "--revalidate"])

    assert result.exit_code == 2
    assert "holds no profile.yml" in result.output


# ── The frozen retired-preset hashes ────────────────────────────────────────


def test_fixture_retired_layer_files_hash_to_their_retired_presets() -> None:
    """The frozen hashes are what an unedited previous-release file hashes to."""
    import yaml

    for preset, retired in RETIRED_PRESETS.items():
        text = (FIXTURE / f"{RETIRED_CATALOG_KEY}" / f"{retired.successor}.yml").read_text(
            encoding="utf-8"
        )
        assert f"`{preset}`" in text
        assert _content_hash(yaml.safe_load(text)) == retired.content_hash, preset
        assert _text_hash(text) == retired.text_hash, preset


def test_the_rewrite_does_not_depend_on_the_retired_presets_shipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retired presets leave the package; the rewrite must not read them."""
    from osprey.cli import build_profile_presets

    real = build_profile_presets._load_preset_raw

    def refuse_retired(name: str):
        if name in RETIRED_PRESETS:
            raise AssertionError(f"read retired preset {name}")
        return real(name)

    monkeypatch.setattr(build_profile_presets, "_load_preset_raw", refuse_retired)
    root = _previous_release_repo(tmp_path / "als-exemplar")

    assert _revalidate(root).exit_code == 0


# ── A failure leaves the repo as it was; a re-run finishes the job ──────────


def _assert_is_the_new_exemplar(root: Path) -> None:
    expected = revalidated_exemplar()
    actual = _repo_files(root)
    assert sorted(actual) == sorted(expected)
    for rel, text in expected.items():
        assert actual[rel] == text, f"{rel} differs from the new exemplar"


def test_an_unreadable_retired_tier_file_is_refused_and_nothing_is_touched(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    layer_file = root / f"{RETIRED_CATALOG_KEY}" / "readwrite.yml"
    layer_file.write_text(
        layer_file.read_text(encoding="utf-8") + "web_panels: [oops\n", encoding="utf-8"
    )
    before = _repo_files(root)

    result = CliRunner().invoke(init, [str(root), "--revalidate"])

    assert result.exit_code == 2, result.output
    assert f"{RETIRED_LAYER_DIRNAME}/readwrite.yml is not valid YAML" in result.output
    assert _repo_files(root) == before


def test_a_run_stopped_after_the_app_was_written_is_finished_by_the_next(tmp_path: Path) -> None:
    """The logbook login exists, but the users pinned to it never moved: they move now."""
    root = _previous_release_repo(tmp_path / "als-exemplar")
    (root / f"{RETIRED_CATALOG_KEY}" / "logbook.yml").unlink()
    (root / "tiers" / "logbook.yml").write_text(
        exemplar_source_files()["tiers/logbook.yml"], encoding="utf-8"
    )

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert "Left for you to do by hand" not in result.output
    _assert_is_the_new_exemplar(root)


def test_a_user_pinned_to_a_tier_with_no_file_is_reported(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    profile = root / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            f"        {RETIRED_TIER_KEY}: readonly\n", f"        {RETIRED_TIER_KEY}: viewer\n"
        ),
        encoding="utf-8",
    )

    result = _revalidate(root)

    assert (
        "profile.yml: `modules.web_terminals.users[bob].tier` names 'viewer', which has no "
        "file in tiers/" in result.output
    )


def test_a_tier_that_spells_provider_is_a_manual_step_naming_the_file(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    layer_file = root / f"{RETIRED_CATALOG_KEY}" / "readonly.yml"
    layer_file.write_text(
        layer_file.read_text(encoding="utf-8") + "provider: cborg\n", encoding="utf-8"
    )

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert "tiers/readonly.yml spells `provider: cborg`" in result.output
    assert "the provider lives in profile.yml" in result.output


# ── Operator content the rewrite must not lose ──────────────────────────────


def test_a_comment_only_edit_keeps_the_file_and_prints_the_manual_step(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    layer_file = root / f"{RETIRED_CATALOG_KEY}" / "readwrite.yml"
    edited = (
        layer_file.read_text(encoding="utf-8") + "# OPERATOR NOTE: armed on the simulator only\n"
    )
    layer_file.write_text(edited, encoding="utf-8")

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    # Kept as written, but for the write posture: the three retired keys become
    # the tier key they amount to (simulator armed: `allowed`), and the comment
    # the previous release shipped above `web.ui_mode` leaves with the preset's.
    assert (root / "tiers" / "readwrite.yml").read_text(encoding="utf-8") == edited.replace(
        _READWRITE_RETIRED_KEYS, "  control_system.writes: allowed\n"
    ).replace(
        "  # Full split-pane terminal + workspace layout for the write-armed operator.\n"
        "  # Pinned on both sides of the tier boundary (readonly pins `simple`) rather\n"
        "  # than left to the server default, for the same reason writes_enabled is.\n",
        "",
    )
    assert "tiers/readwrite.yml was made from `control-assistant-readwrite`" in result.output
    assert "# The tier boundary, and it takes three keys." in result.output


def test_a_comment_only_edit_keeps_a_logbook_retired_tier_a_tier(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    layer_file = root / f"{RETIRED_CATALOG_KEY}" / "logbook.yml"
    edited = layer_file.read_text(encoding="utf-8") + "# OPERATOR NOTE\n"
    layer_file.write_text(edited, encoding="utf-8")

    result = _revalidate(root)

    assert (root / "tiers" / "logbook.yml").read_text(encoding="utf-8") == edited.replace(
        _LOGBOOK_RETIRED_KEYS, "  control_system.writes: never\n"
    )
    assert not (root / "apps" / "logbook.yml").exists()
    assert "        tier: logbook\n" in (root / "profile.yml").read_text(encoding="utf-8")
    assert "tiers/logbook.yml was made from `control-assistant-logbook`" in result.output


def test_a_higher_floor_is_left_as_written(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    profile = root / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            "requires_osprey_version: '>=2026.9.0'", "requires_osprey_version: '>=2027.3.0'"
        ),
        encoding="utf-8",
    )

    _revalidate(root)

    assert "requires_osprey_version: '>=2027.3.0'\n" in profile.read_text(encoding="utf-8")


def test_a_catalog_entry_built_from_another_file_renames_that_file(tmp_path: Path) -> None:
    """A retired catalog entry pointing ``readwrite`` at ``ops.yml`` becomes tiers/readwrite.yml."""
    root = _previous_release_repo(tmp_path / "als-exemplar", git=True)
    _git(root, "mv", f"{RETIRED_LAYER_DIRNAME}/readwrite.yml", f"{RETIRED_LAYER_DIRNAME}/ops.yml")
    profile = root / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            f"build_profile: {RETIRED_LAYER_DIRNAME}/readwrite.yml",
            f"build_profile: {RETIRED_LAYER_DIRNAME}/ops.yml",
        ),
        encoding="utf-8",
    )

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert "Left for you to do by hand" not in result.output
    _assert_is_the_new_exemplar(root)


def test_a_catalog_file_that_cannot_be_renamed_prints_the_manual_step(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    profile = root / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            f"build_profile: {RETIRED_LAYER_DIRNAME}/readwrite.yml",
            f"build_profile: {RETIRED_LAYER_DIRNAME}/readonly.yml",
        ),
        encoding="utf-8",
    )

    result = _revalidate(root)

    assert (
        f"the `{RETIRED_CATALOG_KEY}:` catalog built 'readwrite' from {RETIRED_LAYER_DIRNAME}/readonly.yml"
        in result.output
    )


def test_a_catalog_field_with_no_successor_prints_the_manual_step(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    profile = root / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            f"        build_profile: {RETIRED_LAYER_DIRNAME}/admin.yml\n",
            f"        build_profile: {RETIRED_LAYER_DIRNAME}/admin.yml\n        project: als-admin\n",
        ),
        encoding="utf-8",
    )

    result = _revalidate(root)

    assert (
        f"profile.yml set `project` for 'admin' in its `{RETIRED_CATALOG_KEY}:` catalog"
        in result.output
    )


def test_a_logbook_retired_tier_under_another_name_becomes_the_logbook_login(
    tmp_path: Path,
) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    (root / f"{RETIRED_CATALOG_KEY}" / "logbook.yml").rename(
        root / f"{RETIRED_CATALOG_KEY}" / "elog.yml"
    )
    profile = root / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8")
        .replace(f"        {RETIRED_TIER_KEY}: logbook\n", f"        {RETIRED_TIER_KEY}: elog\n")
        .replace(
            f"      logbook:\n        build_profile: {RETIRED_LAYER_DIRNAME}/logbook.yml\n",
            f"      elog:\n        build_profile: {RETIRED_LAYER_DIRNAME}/elog.yml\n",
        ),
        encoding="utf-8",
    )

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert "Left for you to do by hand" not in result.output
    _assert_is_the_new_exemplar(root)


# ── References to a card that became a shared login ─────────────────────────


def test_a_role_pinned_to_a_migrated_login_is_reported(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    overlay = root / "profiles" / "night.yml"
    overlay.parent.mkdir()
    overlay.write_text(
        "config:\n"
        "  # the logbook crew works nights\n"
        "  modules.web_terminals.authorization:\n"
        "    roles:\n"
        "      archivist:\n"
        f"        {RETIRED_TIER_KEY}: logbook\n",
        encoding="utf-8",
    )

    result = _revalidate(root)

    steps = result.output.split("Left for you to do by hand:", 1)[1]
    assert (
        "profiles/night.yml: `modules.web_terminals.authorization.roles.archivist.tier` "
        "names 'logbook', which is now the shared login `logbook`" in steps
    )
    assert steps.count("logbook") == 2, steps


def test_an_unrelated_mention_of_a_migrated_login_is_not_reported(tmp_path: Path) -> None:
    root = _previous_release_repo(tmp_path / "als-exemplar")
    overlay = root / "profiles" / "night.yml"
    overlay.parent.mkdir()
    overlay.write_text(
        "config:\n  # logbook and knowledge stay up all night\n"
        f"  modules.web_terminals.{RETIRED_DEFAULT_TIER_KEY}: readonly\n",
        encoding="utf-8",
    )

    result = _revalidate(root)

    assert "Left for you to do by hand" not in result.output


def test_dropping_the_catalog_keeps_a_flow_list_before_it_on_one_line(tmp_path: Path) -> None:
    """B25: a roster whose last entry ends in a flow list (`access: [roster]`).

    The comments after the dropped catalog belong to whatever follows it; they
    were handed to the flow list's last item and written inside its brackets.
    """
    root = _previous_release_repo(tmp_path / "als-exemplar")
    profile = root / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    carol = '        display_name: "Deployment Admin (Carol)"\n'
    catalog = f"    {RETIRED_CATALOG_KEY}:\n"
    assert text.count(carol + catalog) == 1
    profile.write_text(
        text.replace(carol + catalog, carol + "        access: [roster]\n" + catalog),
        encoding="utf-8",
    )

    _revalidate(root)

    written = profile.read_text(encoding="utf-8")
    assert "        access: [roster]\n" in written
    assert "access: [roster\n" not in written
    # The comment introducing `shared:` still introduces it.
    assert "    # Logins everyone above may open" in written
    lines = written.splitlines()
    shared = lines.index("    shared:")
    assert lines[shared - 5].lstrip().startswith("# Logins everyone above may open")
    assert lines[shared - 6] == "        access: [roster]"

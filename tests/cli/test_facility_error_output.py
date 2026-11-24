"""What the CLI prints when a facility stops a command: exit code, stdout and stderr.

These cases pin the output of the commands that read or build a facility
description as it is before the facility errors leave ``click``: a build error
is one line on stderr with no ``Error: `` prefix and exit 1; a loader refusal
of ``facility import mml`` is a ``click`` error, with the prefix; a usage error
is the leaf's Usage/Try/Error block and exit 2; a warning leaves the exit code 0.

Every case but one invokes the root group ``osprey.cli.main.cli``, as an
operator's command line does. The exception is the SDK's ``invoke_build``,
which calls the build command itself, so the non-root path is pinned too.

stdout is compared whole after one normalisation, phase durations masked to
``(<t>)``; nothing is compared by substring. The module imports nothing a
later change adds, so it runs unchanged on the tree it was written against.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import sys
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.main import cli
from tests._builds import init_project

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
MML_TREE = REPO_ROOT / "tests" / "fixtures" / "mml" / "synthetic"
MAPPING = "imported/mml/mapping.yaml"

#: A phase's duration as the reporter prints it, e.g. ``(0.1s)`` or ``(12ms)``.
_DURATION = re.compile(r"\(\d+(\.\d+)?(ms|s|m)\)")


def _normalised(text: str) -> str:
    """*text* with every phase duration masked to ``(<t>)``."""
    return _DURATION.sub("(<t>)", text)


#: What every build of the demo prints on stdout before its facility is built.
PREAMBLE = (
    "  profile Demo (bundle demo)\n"
    "  provider catalog providers.yml\n"
    "  ⚠ system.timezone is UTC, so operator times are read and shown in UTC. Set"
    " system.timezone under `config:` in profile.yml to your facility's zone, for example"
    " America/New_York.\n"
)

#: What a build chained from a start verb adds: the pre-build lifecycle step.
PRE_BUILD = (
    "  Running pre_build commands...\n"
    "  ✓ canopy-payload (<t>): canopy payload: skipped (OSPREY_CANOPY_PAYLOAD_CHECK=0)\n"
)

#: What an import that replaced the bundled example prints on stdout.
SEEDED = (
    "wrote data/facility/knowledge/index.md\n"
    "Put your facility's documentation here: one page per topic.\n"
)

FIX_MISSING = (
    "facility: fix-missing: device SR/Q9 — no device SR/Q9 exists;"
    " fix: remove the fix from fixes.yaml"
)

SERVERS = (
    "ariel, bluesky, canopy, channel-directory, channel-finder, controls, event_dispatcher,"
    " graph, health, optimizer, osprey_facility_knowledge, osprey_workspace, phoebus, python"
)

TURN = "sessions[0].parts[0].turns[0]"

UNKNOWN_SERVER = (
    f"facility: reference-missing: scenario warm — scenarios/warm.yaml `{TURN}.steps[0]"
    f".tool.server` names server nowhere, not an OSPREY server; fix: name one of {SERVERS}"
    f" in `{TURN}.steps[0].tool.server`"
)

UNKNOWN_CHANNEL = (
    f"facility: reference-missing: scenario warm — scenarios/warm.yaml `{TURN}.channels`"
    f" names channel NO:SUCH:CHANNEL; fix: add channel NO:SUCH:CHANNEL or correct"
    f" `{TURN}.channels`"
)

UNKNOWN_FACTORY = (
    "facility: value-invalid: measurement kind stepped_kind — its measure plan's unipolar"
    " names StepMeasurement.sweep, which pyaml_cs_osprey does not offer; declared by package"
    " stepped-kind-pkg; fix: use the measure-plan words OSPREY offers"
)

REPLACED = "replaced the bundled example facility (53 files)"

#: A demo channel the scenario's turn may name.
CHANNEL = "BR:DIAG:BPM:01:POSITION:X"


@pytest.fixture(autouse=True)
def _no_canopy_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The build's canopy lines read these variables; the cases pin them unset and off."""
    monkeypatch.delenv("OSPREY_CANOPY_PAYLOAD", raising=False)
    monkeypatch.setenv("OSPREY_CANOPY_PAYLOAD_CHECK", "0")


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A demo repo, initialised once for this module and never edited."""
    return init_project(tmp_path_factory.mktemp("ca"), "demo", "demo")


@pytest.fixture
def repo(initialised: Path, tmp_path: Path) -> Path:
    """A copy of the initialised repo this test may edit."""
    copy = tmp_path / initialised.name
    shutil.copytree(initialised, copy, symlinks=True)
    return copy


def _invoke(*args: str) -> Result:
    return CliRunner().invoke(cli, list(args))


def _build(repo: Path) -> Result:
    return _invoke("build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")


def _validate(repo: Path) -> Result:
    return _invoke("facility", "validate", "--repo", str(repo))


def _import(verb: str, source: Path, repo: Path) -> Result:
    return _invoke("facility", "import", verb, str(source), "--repo", str(repo))


def _assert_output(result: Result, code: int, stdout: str, stderr: list[str]) -> None:
    assert (result.exit_code, _normalised(result.stdout), result.stderr.splitlines()) == (
        code,
        stdout,
        stderr,
    ), result.output


def _break(repo: Path) -> None:
    """A fix that drops a device the facility does not hold: one build error."""
    document = {
        "schema": "osprey.facility.fixes/1",
        "fixes": [{"op": "drop", "kind": "device", "id": "SR/Q9", "why": "gone"}],
    }
    (repo / "data" / "facility" / "fixes.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def _scenario(
    repo: Path, *, server: str = "nowhere", channels: tuple[str, ...] = (CHANNEL,)
) -> None:
    """Scenario ``warm`` with one session whose one tool step names *server*."""
    turn = {
        "prompt": "Read it.",
        "at_offset_s": 0,
        "channels": list(channels),
        "steps": [
            {"tool": {"server": server, "name": "channel_read"}, "input": {}, "result": "ok"},
            {"answer": "Done."},
        ],
    }
    session = {
        "slug": "look",
        "title": "Look",
        "when": {"days_ago": 1, "time": "08:00"},
        "parts": [{"title": "Look", "turns": [turn]}],
    }
    (repo / "data" / "facility" / "scenarios" / "warm.yaml").write_text(
        yaml.safe_dump({"sessions": [session]}, sort_keys=False), encoding="utf-8"
    )


# --- a facility build error ---------------------------------------------------------


def test_a_build_error_is_one_line_without_the_error_prefix(repo: Path) -> None:
    _break(repo)

    _assert_output(_build(repo), 1, PREAMBLE, [FIX_MISSING])


def test_validate_prints_the_same_build_error_line(repo: Path) -> None:
    _break(repo)

    _assert_output(_validate(repo), 1, "", [FIX_MISSING])


def test_the_sdk_build_helper_prints_the_line_off_the_root_group(repo: Path) -> None:
    from osprey.apps.sdk.testing import invoke_build

    _break(repo)

    result = invoke_build(repo)

    assert result.exception is None or isinstance(result.exception, SystemExit)
    _assert_output(result, 1, PREAMBLE, [FIX_MISSING])


# --- an unknown MCP server ----------------------------------------------------------


def test_an_unknown_mcp_server_stops_the_build(repo: Path) -> None:
    _scenario(repo)

    _assert_output(_build(repo), 1, PREAMBLE, [UNKNOWN_SERVER])


def test_an_unknown_mcp_server_stops_validate(repo: Path) -> None:
    _scenario(repo)

    _assert_output(_validate(repo), 1, "", [UNKNOWN_SERVER])


def test_an_unknown_mcp_server_beside_an_unknown_channel_under_build(repo: Path) -> None:
    _scenario(repo, channels=(CHANNEL, "NO:SUCH:CHANNEL"))

    _assert_output(_build(repo), 1, PREAMBLE, [UNKNOWN_CHANNEL])


def test_an_unknown_mcp_server_beside_an_unknown_channel_under_validate(repo: Path) -> None:
    _scenario(repo, channels=(CHANNEL, "NO:SUCH:CHANNEL"))

    _assert_output(_validate(repo), 1, "", [UNKNOWN_CHANNEL, UNKNOWN_SERVER])


def test_an_unknown_mcp_server_beside_a_missing_fix_target(repo: Path) -> None:
    _scenario(repo)
    _break(repo)

    _assert_output(_build(repo), 1, PREAMBLE, [FIX_MISSING])
    _assert_output(_validate(repo), 1, "", [FIX_MISSING])


# --- an unknown step factory --------------------------------------------------------

#: The installed package that registers ``stepped_kind``: the built-in ``orm``
#: whose per-member step names a ``StepMeasurement`` factory that does not exist.
_KIND_PACKAGE = "stepped_kind_pkg"
_KIND_MODULE = """\
import dataclasses

from osprey_facility.kinds import ORM

STEPPED_KIND = dataclasses.replace(
    ORM,
    name="stepped_kind",
    measure_plan={
        **ORM.measure_plan,
        "unipolar": {**ORM.measure_plan["unipolar"], "factory": "sweep"},
    },
)
"""


def _read_entry_points_afresh() -> None:
    getattr(metadata.entry_points, "cache_clear", lambda: None)()


@pytest.fixture
def stepped_kind(tmp_path: Path) -> Iterator[None]:
    """A distribution on the path that registers ``stepped_kind`` for one test."""
    site = tmp_path / "site"
    (site / _KIND_PACKAGE).mkdir(parents=True)
    (site / _KIND_PACKAGE / "__init__.py").write_text(_KIND_MODULE, encoding="utf-8")
    info = site / f"{_KIND_PACKAGE}-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: stepped-kind-pkg\nVersion: 0.1\n", encoding="utf-8"
    )
    (info / "entry_points.txt").write_text(
        f"[osprey_facility.kinds]\nstepped_kind = {_KIND_PACKAGE}:STEPPED_KIND\n",
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(site))
        patch.delitem(sys.modules, _KIND_PACKAGE, raising=False)
        importlib.invalidate_caches()
        _read_entry_points_afresh()
        yield
    sys.modules.pop(_KIND_PACKAGE, None)
    importlib.invalidate_caches()
    _read_entry_points_afresh()


@pytest.mark.usefixtures("stepped_kind")
def test_an_unknown_step_factory_stops_the_build(repo: Path) -> None:
    _assert_output(_build(repo), 1, PREAMBLE, [UNKNOWN_FACTORY])


@pytest.mark.usefixtures("stepped_kind")
def test_an_unknown_step_factory_stops_validate(repo: Path) -> None:
    _assert_output(_validate(repo), 1, "", [UNKNOWN_FACTORY])


# --- facility import mml ------------------------------------------------------------


def test_an_unreadable_mat_file_is_the_loader_line_with_the_error_prefix(
    repo: Path, tmp_path: Path
) -> None:
    source = tmp_path / "broken.mat"
    source.write_bytes(b"not a MAT-file at all\n" * 10)

    _assert_output(
        _import("mml", source, repo),
        1,
        SEEDED,
        [
            f"import mml: {REPLACED}",
            f"Error: Cannot read MAT-file {source}: Unknown mat file type, version 32, 97",
        ],
    )


def test_a_v73_mat_file_is_the_loader_line_with_the_error_prefix(
    repo: Path, tmp_path: Path
) -> None:
    source = tmp_path / "v73.mat"
    source.write_bytes(b"MATLAB 7.3 MAT-file".ljust(124, b" ") + b"\x00\x02IM" + b"\x00" * 64)

    _assert_output(
        _import("mml", source, repo),
        1,
        SEEDED,
        [
            f"import mml: {REPLACED}",
            f"Error: MAT-file {source} is version 7.3 (HDF5), which cannot be read."
            " Re-save it in MATLAB with save('-v7', ...) and try again.",
        ],
    )


def test_a_json_file_with_no_family_is_the_loader_line_with_the_error_prefix(
    repo: Path, tmp_path: Path
) -> None:
    source = tmp_path / "nothing.json"
    source.write_text('{"Machine": "Quokka"}', encoding="utf-8")

    _assert_output(
        _import("mml", source, repo),
        1,
        SEEDED,
        [
            f"import mml: {REPLACED}",
            f"Error: No middle-layer family found in {source}: expected a map of families,"
            " an 'ao'/'AO' wrapper around one, or a map of systems holding families.",
        ],
    )


def test_a_deck_named_as_the_export_is_the_usage_block(repo: Path, tmp_path: Path) -> None:
    source = tmp_path / "quokka.sr.lattice.mat"
    shutil.copyfile(MML_TREE / source.name, source)

    _assert_output(
        _import("mml", source, repo),
        2,
        SEEDED,
        [
            f"import mml: {REPLACED}",
            "Usage: cli facility import mml [OPTIONS] EXPORT...",
            "Try 'cli facility import mml --help' for help.",
            "",
            f"Error: Cannot import {source}: it is a deck; name the export's AO file.",
        ],
    )


def test_deck_lines_then_mapping_problems(repo: Path, tmp_path: Path) -> None:
    """The deck line reaches stdout; the mapping's problems end the import, unprefixed."""
    export = tmp_path / "export"
    export.mkdir()
    for path in MML_TREE.glob("quokka.sr.*"):
        shutil.copyfile(path, export / path.name)
    facts_file = export / "quokka.sr.va.json"
    facts: dict[str, Any] = json.loads(facts_file.read_text(encoding="utf-8"))
    facts["lattice"] = {"refused": "no THERING in the workspace"}
    facts_file.write_text(json.dumps(facts), encoding="utf-8")
    mapping = repo / "data" / "facility" / MAPPING
    mapping.parent.mkdir(parents=True, exist_ok=True)
    text = (MML_TREE / MAPPING).read_text(encoding="utf-8")
    assert text.count("    name: SR\n") == 1
    mapping.write_text(text.replace("    name: SR\n", "    name: S R\n"), encoding="utf-8")

    _assert_output(
        _import("mml", export / "quokka.sr.ao.json", repo),
        1,
        SEEDED + "import mml: deck unchecked: SR: the export states no lattice fingerprint"
        " (no THERING in the workspace); the deck quokka.sr.lattice.mat is filed unchecked\n",
        [
            f"import mml: {REPLACED}",
            "models.SR.name: 'S R' is not PN_LOCAL",
            "section_order[0]: 'SR' is no model name",
            "section_order: leaves out the model S R",
            f"3 problems in data/facility/{MAPPING}; fix each and check again.",
        ],
    )


# --- facility import list -----------------------------------------------------------


def _role_refusal(name: str, row: int, address: str, role: str) -> str:
    return (
        f"facility: source-invalid: channel {address} — {name} row {row} states role {role};"
        " fix: write setpoint, readback or none, or leave it empty"
    )


def test_a_list_with_two_refused_rows_prints_two_lines(repo: Path, tmp_path: Path) -> None:
    listing = tmp_path / "two.csv"
    listing.write_text("address,role\nA:ONE,bogus\nA:TWO,other\n", encoding="utf-8")

    _assert_output(
        _import("list", listing, repo),
        1,
        SEEDED,
        [
            f"import list: {REPLACED}",
            _role_refusal("two.csv", 2, "A:ONE", "bogus"),
            _role_refusal("two.csv", 3, "A:TWO", "other"),
        ],
    )


def test_a_list_warning_then_a_refusal(repo: Path, tmp_path: Path) -> None:
    listing = tmp_path / "warn.csv"
    listing.write_text("address,device,model,role\nA:ONE,DEV1,SR,\nA:TWO,,,bogus\n", "utf-8")

    _assert_output(
        _import("list", listing, repo),
        1,
        SEEDED,
        [
            f"import list: {REPLACED}",
            "import list: warn.csv row 2 states model SR for device DEV1 and no row states its"
            " s; the model is ignored; fix: fill `s` on a row of the device, or empty `model`",
            _role_refusal("warn.csv", 3, "A:TWO", "bogus"),
        ],
    )


# --- profiles, start verbs and the command line -------------------------------------


def test_a_profile_error_under_the_nested_profile_group(repo: Path) -> None:
    profile = repo / "profile.yml"
    profile.write_text(profile.read_text(encoding="utf-8") + "tier: in_context\n", "utf-8")

    _assert_output(
        _invoke("profile", "validate", str(repo)),
        1,
        "",
        [
            "facility: profile-invalid: path tier — `tier` is not a profile key; the"
            " `in_context` tag on a channel selects the in_context subset; fix: remove `tier`"
            " from the profile"
        ],
    )


def test_up_with_build_on_a_broken_facility(repo: Path) -> None:
    _break(repo)

    _assert_output(
        _invoke("up", "--build", "--repo", str(repo)), 1, PREAMBLE + PRE_BUILD, [FIX_MISSING]
    )


def test_restart_with_build_on_a_broken_facility(repo: Path) -> None:
    """``restart`` re-raises the chained build's ``click`` error unchanged."""
    _break(repo)

    _assert_output(
        _invoke("restart", "--build", "--repo", str(repo)),
        1,
        PREAMBLE + PRE_BUILD,
        [FIX_MISSING],
    )


def test_a_command_line_usage_error(repo: Path) -> None:
    _assert_output(
        _invoke("build", "--repo", str(repo), "--no-such-flag"),
        2,
        "",
        [
            "Usage: cli build [OPTIONS]",
            "Try 'cli build --help' for help.",
            "",
            "Error: No such option '--no-such-flag'.",
        ],
    )


# --- a facility that builds with one warning ----------------------------------------

#: The demo's one ``relation-coarse`` warning, as ``facility validate`` prints it.
COARSE = (
    "facility: relation-coarse: relation type cabled_to — 1 claim states less than the"
    " facility knows (1 place end)"
)
COARSE_FIX = "name the device, port, unit or key attribute the claims leave out"


@pytest.mark.real_graph_index
def test_a_build_with_one_warning_exits_0_and_prints_it_on_stderr(repo: Path) -> None:
    """The whole build's stdout, its channel search index derived for real."""
    built = (
        PREAMBLE
        + "\n"
        + "→ Rendering the configuration\n"
        + "  · Channel search index built from facility.ttl: 4765 binding(s) over 1025"
        " device(s), 36 class(es).\n"
        "  · project files, agent artifacts and services (<t>)\n"
        "  · services injected: 1 profile service(s), event dispatch, bluesky bridge,"
        " bluesky web, virtual accelerator, archiver store, server fleet\n"
        "  · 6 render(s) (<t>)\n"
        "  · services bound to localhost only (127.0.0.1)\n"
        "  · bluesky plan devices: 677 settable / 4088 readable from data/bluesky_devices.yml\n"
        "  · compose files (<t>)\n"
        "  · 7 image build context(s) (<t>)\n"
        "  ✓ Rendering the configuration (<t>)\n"
        "\n"
        "demo — built\n"
        f"  command output   {repo / 'var' / 'logs'}\n"
        "  next             osprey up -d · osprey status  (no containers: osprey web)"
        " · osprey sim apply nominal\n"
    )

    _assert_output(
        _build(repo),
        0,
        built,
        [f"  ⚠ {COARSE}", f"    → {COARSE_FIX}", "  view bluesky not written: services.bluesky"],
    )


def test_validate_with_one_warning_exits_0_and_prints_it_on_stderr(repo: Path) -> None:
    _assert_output(
        _validate(repo),
        0,
        "",
        [
            f"{COARSE}; fix: {COARSE_FIX}",
            "limits: 460 setpoints have no limits record; the target serves them writable"
            " with no range",
            "  BR:DIAG:TV:01:PADDLE:SP",
            "  BR:DIAG:TV:02:PADDLE:SP",
            "  BR:DIAG:TV:03:PADDLE:SP",
            "  BR:DIAG:TV:04:PADDLE:SP",
            "  BR:DIAG:TV:05:PADDLE:SP",
            "  BR:MAG:DIPOLE-RAMP:01:CURRENT:SP",
            "  BR:MAG:DIPOLE-RAMP:01:GAIN:SP",
            "  BR:MAG:DIPOLE-RAMP:01:OFFSET:SP",
            "  BR:MAG:KICKER:01:VOLTAGE:SP",
            "  BR:MAG:QD-RAMP:01:CURRENT:SP",
            "  and 450 more; --all-limits lists every one",
            "relations: 18 claims (16 stated, 2 inferred); 17 accepted, 0 contested, 1 superseded",
        ],
    )

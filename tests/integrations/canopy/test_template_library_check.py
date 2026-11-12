"""The demo template ships its screen library together with its report.

A project scaffolded from the template holds ``data/canopy/library`` with the
orbit-feedback screen ``sr/orbit-feedback``, the generated landing page ``facility/overview``,
booster screen ``br/overview`` and storage-ring sector screens ``sr/sector-NN``, the
transfer-line panels ``gtl/GTL``
and ``bts/BTS`` with the screens and images they embed under ``common/``, and the two
``LIBRARY-REPORT`` files CANOPY generated for it. Two checks stand between that library and a running container, and a
fresh project has to pass both without anyone regenerating anything:

* ``osprey build`` compares the library with the report before it stages the
  payload (``library_is_fresh``, standard library only);
* the image build recomputes the whole report with the CANOPY it ships
  (``python -I -m canopy_mcp.report data/canopy/library --served-at library
  --check``).

The second runs here against the real ``canopy_mcp.report``, as a command under
this interpreter. A CANOPY release that changes what the report holds fails
that case until the template's report is regenerated with it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.cli.templates.manager import TemplateManager
from osprey.integrations.canopy import LIBRARY_SUBPATH, proven
from osprey.integrations.canopy.library import (
    KNOWN_SCHEMAS,
    REPORT_JSON,
    REPORT_MD,
    library_is_fresh,
    library_list,
)

ORBIT_FEEDBACK = "screens/sr/orbit-feedback.screen.json"
FACILITY_OVERVIEW = "screens/facility/overview.screen.json"
BOOSTER = "screens/br/overview.screen.json"
SECTORS = [f"sr/sector-{n:02d}" for n in range(1, 13)]
# Each sector's vacuum detail, and the absorber detail of each straight with thermocouples.
ABSORBER_STRAIGHTS = (2, 4, 5, 6, 7, 8, 9, 10, 11, 12)
DETAILS = [f"sr/_vacuum-{n:02d}" for n in range(1, 13)] + [
    f"sr/_absorbers-{n:02d}" for n in ABSORBER_STRAIGHTS
]
# The straights with two ion gauges; every other straight has one, every arc one.
TWO_GAUGE_STRAIGHTS = (1, 3, 4, 6, 11)
# The files the report lists: the facility overview, the booster, orbit feedback, the
# twelve sectors and their details, the two transfer-line panels, and the screens
# and images they embed.


def _sector_writes(n: int) -> list[str]:
    """What sector ``n``'s vacuum controls write: each valve's open setpoint (an Open and
    a Close button) and each gauge's trip reset."""
    gauges = (
        [f"ST{n:02d}A"] + ([f"ST{n:02d}B"] if n in TWO_GAUGE_STRAIGHTS else []) + [f"SR{n:02d}"]
    )
    writes = [f"SR:VAC:GAUGE:{g}:RESET:SP" for g in gauges]
    for valve in (n, 12 + n):
        writes += [f"SR:VAC:VALVE:{valve:02d}:OPEN:SP"] * 2
    return writes


PANELS = ("bts/BTS", "gtl/GTL")
EMBEDDED = ("_KnobDevice", "_bpm", "_tv")
IMAGES = ("Light off", "Light on", "Paddle in", "Paddle out", "TV off", "TV on")
LIBRARY_FILES = 3 + len(SECTORS) + len(DETAILS) + len(PANELS) + len(EMBEDDED) + len(IMAGES)


def _panel_files() -> list[str]:
    """Every path the transfer-line panels ship, directories included."""
    return [
        *(f"screens/{panel.split('/')[0]}" for panel in PANELS),
        *(f"screens/{panel}.screen.json" for panel in PANELS),
        "screens/common",
        "screens/common/embedded",
        *(f"screens/common/embedded/{name}.screen.json" for name in EMBEDDED),
        "screens/common/images",
        *(f"screens/common/images/{name}.jpg" for name in IMAGES),
    ]


@pytest.fixture(scope="module")
def library() -> Path:
    root = Path(TemplateManager().template_root) / "apps" / "demo"
    path = root / LIBRARY_SUBPATH
    assert path.is_dir(), path
    return path


def _check(library: Path) -> subprocess.CompletedProcess[str]:
    """CANOPY's check over *library*, isolated as the image build runs it."""
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "canopy_mcp.report",
            str(library),
            "--served-at",
            "library",
            "--check",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestWhatTheTemplateShips:
    def test_the_screens_the_two_panels_and_the_two_report_files_and_nothing_else(self, library):
        shipped = sorted(p.relative_to(library).as_posix() for p in library.rglob("*"))
        assert shipped == sorted(
            [
                REPORT_JSON,
                REPORT_MD,
                "screens",
                "screens/br",
                BOOSTER,
                "screens/facility",
                FACILITY_OVERVIEW,
                "screens/sr",
                ORBIT_FEEDBACK,
                *(f"screens/{sector}.screen.json" for sector in SECTORS),
                *(f"screens/{detail}.screen.json" for detail in DETAILS),
                *_panel_files(),
            ]
        )

    def test_no_shipped_file_is_ignored_by_git(self, library):
        """An ignored file never reaches a commit or a wheel, so the digest would not match."""
        if shutil.which("git") is None:
            pytest.skip("git is not installed")
        top = subprocess.run(
            ["git", "-C", str(library), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if top.returncode != 0:
            pytest.skip("the template is not in a git checkout")
        files = [str(p) for p in library.rglob("*") if p.is_file()]
        result = subprocess.run(
            ["git", "-C", top.stdout.strip(), "check-ignore", "--no-index", "--stdin"],
            input="\n".join(files) + "\n",
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.stdout == "", result.stdout

    def test_no_knob_panel_and_no_theme_file(self, library):
        """The template ships the panels with CANOPY's default theme and no KnobPanel."""
        names = [p.name for p in library.rglob("*")]
        assert not any("KnobPanel" in name for name in names), names
        assert not (library.parent / "theme").exists()

    def test_the_report_is_of_a_schema_this_release_reads(self, library):
        report = json.loads((library / REPORT_JSON).read_bytes())
        assert report["generator"]["schema"] in KNOWN_SCHEMAS
        assert report["manifest"] == [
            "br/overview",
            "bts/BTS",
            *(f"common/embedded/{name}" for name in EMBEDDED),
            "facility/overview",
            "gtl/GTL",
            *sorted(["sr/orbit-feedback", *SECTORS, *DETAILS]),
        ]

    def test_the_report_was_generated_by_the_proven_canopy(self, library):
        """The report is the one the proven CANOPY writes: its version names that commit."""
        version = json.loads((library / REPORT_JSON).read_bytes())["generator"]["canopy_version"]
        assert proven.version_names_pin("canopy", version), version

    def test_the_report_carries_no_provenance(self, library):
        """Nothing was copied in from a snapshot: the screen is the template's own."""
        assert "provenance" not in json.loads((library / REPORT_JSON).read_bytes())

    def test_the_report_names_what_the_sr_screens_can_write(self, library):
        """Orbit feedback its one corrector; each sector and its vacuum detail the valves'
        open setpoints and the gauges' resets; sector 1 also its injection screen."""
        report = json.loads((library / REPORT_JSON).read_bytes())
        written = sorted(
            (row["top"], row["pvResolved"])
            for row in report["writeBindings"]
            if row["top"].startswith("sr/")
        )
        expected = [("sr/orbit-feedback", "SR:MAG:HCM:01:CURRENT:SP")]
        for n in range(1, 13):
            for top in (f"sr/sector-{n:02d}", f"sr/_vacuum-{n:02d}"):
                expected += [(top, pv) for pv in _sector_writes(n)]
        expected += [
            ("sr/sector-01", "BTS:DIAG:TV:07:PADDLE:RB"),
            ("sr/sector-01", "BTS:DIAG:TV:07:PADDLE:SP"),
            ("sr/sector-01", "loc://srtv1"),
        ]
        assert written == sorted(expected)

    def test_the_report_counts_what_each_top_screen_can_write(self, library):
        report = json.loads((library / REPORT_JSON).read_bytes())
        tops = [row["top"] for row in report["writeBindings"]]
        vacuum = {n: len(_sector_writes(n)) for n in range(1, 13)}
        # The four at the end: _tv and _KnobDevice on their own, their macro-only
        # names left unresolved.
        assert len(tops) == 195 + 2 * sum(vacuum.values()) + 3 + 4
        assert {top: tops.count(top) for top in set(tops)} == {
            "gtl/GTL": 141,
            "bts/BTS": 53,
            "sr/orbit-feedback": 1,
            "common/embedded/_tv": 3,
            "common/embedded/_KnobDevice": 1,
            **{f"sr/sector-{n:02d}": w + (3 if n == 1 else 0) for n, w in vacuum.items()},
            **{f"sr/_vacuum-{n:02d}": w for n, w in vacuum.items()},
        }
        # Every vacuum write is one push button; BTS and GTL have one each.
        assert len(report["commandButtons"]) == 2 + 2 * sum(vacuum.values())
        for key in ("executeActions", "absoluteUrls"):
            assert report[key] == [], key

    def test_the_panels_open_each_other_and_the_overview_opens_the_sections(self, library):
        report = json.loads((library / REPORT_JSON).read_bytes())
        overview = [
            *((f"w-{697 + i}", sector) for i, sector in enumerate(SECTORS)),
            ("w-709", "bts/BTS"),
            ("w-710", "gtl/GTL"),
            ("w-711", "br/overview"),
        ]
        rows = [
            (row["topScreen"], row["nodeId"], row["file"], row["target"])
            for row in report["openDisplays"]
        ]
        details = sorted((top, target) for top, _, _, target in rows if top.startswith("sr/"))
        expected = []
        for n in range(1, 13):
            sector = f"sr/sector-{n:02d}"
            opened = [f"sr/_vacuum-{n:02d}"]
            if n in ABSORBER_STRAIGHTS:
                opened.append(f"sr/_absorbers-{n:02d}")
            expected += [(sector, d) for d in opened] + [(d, sector) for d in opened]
            expected.append((sector, "sr/orbit-feedback"))
        assert details == sorted(expected)
        assert [row for row in rows if not row[0].startswith("sr/")] == [
            ("bts/BTS", "w-210", "../gtl/GTL.screen.json", "gtl/GTL"),
            *(
                ("facility/overview", node, f"../{target}.screen.json", target)
                for node, target in overview
            ),
            ("gtl/GTL", "w-429", "../bts/BTS.screen.json", "bts/BTS"),
        ]


class TestTheBuildComparison:
    def test_library_is_fresh_passes_on_the_shipped_template(self, library):
        answer = library_is_fresh(library)
        assert answer, answer.lines
        assert answer.lines == (f"report is current ({LIBRARY_FILES} files)",)

    def test_the_report_lists_the_library_as_it_is(self, library):
        report = json.loads((library / REPORT_JSON).read_bytes())
        listed = "".join(f"{f['sha256']}  {f['path']}\n" for f in report["files"]["list"])
        assert listed.encode() == library_list(library)

    def test_an_edited_screen_is_refused_until_the_report_is_regenerated(self, library, tmp_path):
        """The comparison is live: the shipped pair passes because it matches."""
        copy = tmp_path / "library"
        shutil.copytree(library, copy)
        with (copy / ORBIT_FEEDBACK).open("ab") as handle:
            handle.write(b"\n")
        answer = library_is_fresh(copy)
        assert not answer
        assert any(line.startswith("digest mismatch") for line in answer.lines)
        assert f"changed: {ORBIT_FEEDBACK}" in answer.lines


class TestTheImageCheck:
    @pytest.mark.canopy
    def test_canopys_check_passes_on_the_shipped_template(self, library):
        result = _check(library)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout == f"report is current ({LIBRARY_FILES} files, --served-at library)\n"
        assert result.stderr == ""

    @pytest.mark.canopy
    def test_canopys_check_fails_once_a_screen_is_edited(self, library, tmp_path):
        copy = tmp_path / "library"
        shutil.copytree(library, copy)
        screen = json.loads((copy / ORBIT_FEEDBACK).read_bytes())
        screen["metadata"]["title"] = "edited after the report was written"
        (copy / ORBIT_FEEDBACK).write_text(json.dumps(screen), encoding="utf-8")
        result = _check(copy)
        assert result.returncode == 1, result.stdout + result.stderr
        assert f"stale: {REPORT_JSON} differs in: files, templates" in result.stderr
        assert f"stale: {REPORT_MD} differs" in result.stderr

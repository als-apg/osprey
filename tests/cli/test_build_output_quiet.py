"""What a build says at INFO, and what it now only says at DEBUG.

``osprey build`` renders the project six times — the deployment, one project per
persona, and a container copy of each. Every render used to re-emit its full
report at INFO, so an operator read the same injector block, the same provider
report and the same config-writer field dumps six times over for one build.

The demotion is the whole mechanism: no new log level, no change to
:func:`~osprey.utils.logger.configure_logging`. INFO stays the default filter,
and the per-render detail moves below it. These tests pin both halves — the
noise is gone from INFO, and it is still *there* at DEBUG, because a build that
cannot be debugged is not quieter, only less honest.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build as build_command
from osprey.cli.phase_reporter import PhaseReporter, install_reporter


class _Capture(logging.Handler):
    """Every record the build emits, kept whole so level and text stay paired."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []
        #: The repo the captured run built, for assertions about paths it named.
        self.repo: Path | None = None
        #: The phase-reporter lines the same run printed — the default view the
        #: demoted records used to be part of.
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, *, at_least: int) -> list[str]:
        """The formatted messages of every record at ``at_least`` or above."""
        return [r.getMessage() for r in self.records if r.levelno >= at_least]


class _RecordingReporter(PhaseReporter):
    """A real reporter that keeps its lines instead of printing them.

    Read off the reporter rather than out of ``result.output`` because reporter
    output goes through the Rich console singleton, which wraps to whatever
    terminal width a CliRunner presents.
    """

    def __init__(self, sink: list[str]) -> None:
        super().__init__(color=False)
        self._sink = sink

    def emit(self, text: str, style: str | None = None) -> None:
        self._sink.append(text)


@pytest.fixture(scope="module")
def build_records(tmp_path_factory: pytest.TempPathFactory) -> _Capture:
    """One real build of the exemplar, with every record it emitted.

    Module-scoped because the build is the expensive part and every assertion
    below reads the same run: what the six renders said is one fact, and
    re-running the build per assertion would only make it slower to check.

    ``--skip-deps``/``--skip-lifecycle`` drop the venv install and the profile's
    shell phases — neither emits the per-render reports under test, and both
    cost minutes.
    """
    from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

    repo = build_exemplar_repo(tmp_path_factory.mktemp("quiet") / EXEMPLAR_DIRNAME, seed_env=True)

    capture = _Capture()
    capture.repo = repo.resolve()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(capture)
    root.setLevel(logging.DEBUG)
    previous_cwd = Path.cwd()
    os.chdir(repo)
    # Pre-installed, so the verb reports into it — the same path a chained verb
    # puts `build` on — and its lines are captured beside the log records.
    previous_reporter = install_reporter(_RecordingReporter(capture.lines))
    try:
        result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    finally:
        install_reporter(previous_reporter)
        os.chdir(previous_cwd)
        root.removeHandler(capture)
        root.setLevel(previous_level)

    assert result.exit_code == 0, result.output
    return capture


#: Text that a build must no longer say at INFO, paired with what it is.
#:
#: Each fires once per render — six times for one build — except the provider
#: report, which the deployment's own render and its container copy each emit.
DEMOTED_NEEDLES = {
    "Injected": "the injector block's post-build hints",
    "service template(s)": "the service-template copy count",
    "Provider:": "the provider-credential report",
    "config_update_fields": "the config writer's field dump",
    "Rendering persona": "the per-persona render announcement",
    "satisfies": "the OSPREY-version check confirming it passed",
}


class TestLoggerDemotions:
    """The build/deploy path's noisy call sites, now at DEBUG."""

    @pytest.mark.parametrize("needle", sorted(DEMOTED_NEEDLES), ids=sorted(DEMOTED_NEEDLES))
    def test_logger_says_nothing_of_the_kind_at_info(
        self, build_records: _Capture, needle: str
    ) -> None:
        """INFO carries none of the per-render detail."""
        offenders = [m for m in build_records.messages(at_least=logging.INFO) if needle in m]
        assert not offenders, (
            f"{DEMOTED_NEEDLES[needle]} is still at INFO ({len(offenders)} record(s)): {offenders}"
        )

    @pytest.mark.parametrize("needle", sorted(DEMOTED_NEEDLES), ids=sorted(DEMOTED_NEEDLES))
    def test_logger_still_says_it_at_debug(self, build_records: _Capture, needle: str) -> None:
        """Demoted, not deleted — ``--debug`` still shows every line."""
        assert any(needle in m for m in build_records.messages(at_least=logging.DEBUG)), (
            f"{DEMOTED_NEEDLES[needle]} vanished entirely; demotion must keep it at DEBUG"
        )

    def test_logger_keeps_the_builds_one_success_line_at_info(
        self, build_records: _Capture
    ) -> None:
        """The sweep is a demotion, not a blanket silencing.

        A build that says nothing at all on success would pass every assertion
        above while being strictly worse than the noise it replaced.
        """
        assert any(
            m.startswith("✓ Rendered") for m in build_records.messages(at_least=logging.INFO)
        ), f"no success line survived: {build_records.messages(at_least=logging.INFO)}"

    def test_logger_spells_the_build_path_out_only_in_that_success_line(
        self, build_records: _Capture
    ) -> None:
        """One absolute path in the whole default view, and it is the useful one.

        Every demoted line above named the tree it had just written to, so a
        build spelled its own output path out once per render and each one wrapped
        across a normal terminal. Where the build landed is worth saying — once,
        at the end. Asserted as equality rather than as a count so a *different*
        line reacquiring a path cannot pass by replacing this one.
        """
        assert build_records.repo is not None
        carrying = [
            m for m in build_records.messages(at_least=logging.INFO) if str(build_records.repo) in m
        ]
        assert carrying == [f"✓ Rendered {build_records.repo / 'build'}"]

    def test_reporter_names_the_injected_services_once(self, build_records: _Capture) -> None:
        """The injector highlights survive the demotion, as one step line.

        Demoting the per-service hint blocks to DEBUG must not take the fact
        that services were injected down with them. The render phase names the
        components once — not once per service, and not once per render.
        """
        steps = [line for line in build_records.lines if "services injected:" in line]
        assert len(steps) == 1, f"expected one injector step line, got {len(steps)}: {steps}"
        # The exemplar declares dispatch, bluesky, bluesky panels, a virtual
        # accelerator and an archiver; two of them pin the line's shape.
        assert steps[0].startswith("  · services injected: "), steps[0]
        # Read as names rather than as substrings: the names are accumulated per
        # render into one list, so a second render feeding that list would repeat
        # every component *within* this single line and still pass the count
        # above. Asserting membership in the parsed list also proves the parse.
        listed = steps[0].split("services injected: ", 1)[1]
        names = [name.strip() for name in re.sub(r"\s*\([^()]*\)$", "", listed).split(",")]
        assert "event dispatch" in names and "archiver store" in names, names
        assert len(names) == len(set(names)), f"components repeated within the line: {names}"

    def test_logger_repeats_no_render_line_across_the_six_renders(
        self, build_records: _Capture
    ) -> None:
        """Not one duplicate at INFO from the renderer itself.

        Scoped to the ``build`` logger — the renderer's own voice, which is what
        the six renders share. Compose generation logs under its own name and
        repeats a line per service build context; that one is the compose
        generator's to quiet, and scoping here keeps this test reporting on the
        render loop rather than failing for a neighbour.
        """
        info = [
            r.getMessage()
            for r in build_records.records
            if r.levelno >= logging.INFO and r.name == "build"
        ]
        duplicates = {m for m in info if info.count(m) > 1}
        assert not duplicates, f"INFO still repeats across renders: {sorted(duplicates)}"


class TestLoggerConfigWriterQuiet:
    """``config.yml`` edits report at DEBUG, path echo and all."""

    def _config(self, tmp_path: Path) -> Path:
        path = tmp_path / "config.yml"
        path.write_text("modules:\n  web_terminals:\n    users:\n      - alice\n", encoding="utf-8")
        return path

    def test_logger_field_update_dump_is_debug(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from osprey.utils.config_writer import config_update_fields

        path = self._config(tmp_path)
        with caplog.at_level(logging.DEBUG):
            config_update_fields(path, {"modules.web_terminals.nginx_port": 8080})

        assert "config_update_fields" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.INFO]

    def test_logger_list_edits_are_debug(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from osprey.utils.config_writer import config_remove_from_list, config_replace_list

        path = self._config(tmp_path)
        key = ["modules", "web_terminals", "users"]
        with caplog.at_level(logging.DEBUG):
            config_remove_from_list(path, key, "alice")
            config_replace_list(path, key, ["bob"])

        assert "config_remove_from_list" in caplog.text
        assert "config_replace_list" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.INFO]


class _StubCollection:
    """The three calls :func:`seed_base` makes on a store, and nothing else."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []

    def create_index(self, *args: Any, **kwargs: Any) -> None:
        return None

    def insert_many(self, documents: list[dict[str, Any]], **kwargs: Any) -> None:
        self.documents.extend(documents)

    def replace_one(self, *args: Any, **kwargs: Any) -> None:
        return None


class TestLoggerArchiverSeedQuiet:
    """The seeder's own report line, off the deploy's INFO stream."""

    def test_logger_seed_report_is_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """A deploy seeds the archive; the seeder no longer narrates it at INFO.

        Seeded with no channels, so the run is the report line and the manifest
        write — the arithmetic is covered by the archiver seed suite, and what
        is under test here is only the level the summary goes out at.
        """
        from osprey.simulation.archiver_seed import SeedKnobs, seed_base

        knobs = SeedKnobs(
            retention_days=1, hot_span_hours=1, hot_cadence_sec=600, tail_cadence_sec=3600
        )
        with caplog.at_level(logging.DEBUG):
            report = seed_base(
                _StubCollection(),  # type: ignore[arg-type]
                [],
                knobs,
                t0=datetime(2026, 3, 14, 9, 26, 53, tzinfo=UTC),
            )

        assert report.describe() in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.INFO]


# ---------------------------------------------------------------------------
# stdout: the ``console.print`` chatter no log level can reach
# ---------------------------------------------------------------------------


_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def plain(text: str) -> str:
    """``text`` with ANSI styling gone and every whitespace run collapsed.

    Rich wraps to the console width, so a phrase that *is* on stdout can still
    be missing from a naive ``in`` check — split across a newline mid-sentence.
    An absence assertion against unnormalized Rich output is therefore prone to
    passing for the wrong reason; collapsing first makes it mean what it says.
    """
    return " ".join(_ANSI.sub("", text).split())


#: Console lines a render must no longer put on stdout, paired with what they are.
#:
#: All of them fire once per render — six times for one build — and each spells
#: out an absolute path, so the shortest still wraps to several lines in a normal
#: terminal.
#:
#: One demoted line is missing here: ``Copied machine data to <path>``, which
#: only runs for a bundle that ships ``machine_data/``. No bundled app does, so
#: a render cannot reach it and an assertion would pass vacuously. It is covered
#: by the same demotion in the same function as the web-terminal line below.
STDOUT_NEEDLES = {
    "Created": "Claude Code integration file(s)",
    "data files": "Copied template data files to",
    "tier artifacts": "Materialized tier",
    "web-terminal context": "Installed web-terminal context to",
}


@pytest.fixture(scope="module")
def rendered_project(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, _Capture]:
    """One real render, with everything it wrote to stdout and to the log.

    A render rather than a full build: these call sites live in the template
    layer, and :meth:`TemplateManager.create_project` is the shortest path that
    exercises all of them for real. Module-scoped for the same reason as
    :func:`build_records` — one render answers every assertion below.

    ``redirect_stdout`` rather than ``capsys`` because the fixture outlives a
    single test; the ``console`` singleton resolves ``sys.stdout`` per write, so
    the redirect catches it.
    """
    from osprey.cli.templates.manager import TemplateManager

    capture = _Capture()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(capture)
    root.setLevel(logging.DEBUG)
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            TemplateManager().create_project(
                project_name="quiet-render",
                output_dir=tmp_path_factory.mktemp("render"),
                data_bundle="control_assistant",
                context={"channel_finder_mode": "hierarchical"},
            )
    finally:
        root.removeHandler(capture)
        root.setLevel(previous_level)

    return plain(stdout.getvalue()), capture


class TestStdoutDemotions:
    """Per-render ``console.print`` chatter, moved off stdout onto DEBUG."""

    @pytest.mark.parametrize("case", sorted(STDOUT_NEEDLES), ids=sorted(STDOUT_NEEDLES))
    def test_stdout_render_says_nothing_of_the_kind(
        self, rendered_project: tuple[str, _Capture], case: str
    ) -> None:
        """Nothing on stdout, and the line still on DEBUG.

        The DEBUG half is what keeps this honest: it proves the call site ran at
        all, so the empty stdout is a demotion rather than a code path that
        never fired in this render.
        """
        out, records = rendered_project
        needle = STDOUT_NEEDLES[case]

        assert any(needle in m for m in records.messages(at_least=logging.DEBUG)), (
            f"{needle!r} never fired in this render — the stdout check below would be vacuous"
        )
        assert needle not in out, f"{needle!r} is still on stdout: {out}"


class TestStdoutDataCopyQuiet:
    """The two data-copy branches no build exercises together."""

    def _tree(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "channels.json").write_text("{}", encoding="utf-8")
        return root

    def test_stdout_profile_data_copy_is_quiet(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        """The profile branch — the one a facility build takes.

        Not reachable from :func:`rendered_project`: a profile's ``data:`` tree
        fully replaces the bundle's, so the two branches are alternatives and
        only the bundle one runs in a bundle render.
        """
        from osprey.cli.templates.scaffolding import copy_template_data

        data_root = self._tree(tmp_path / "profile" / "data")
        project = tmp_path / "project"
        project.mkdir()

        capsys.readouterr()
        with caplog.at_level(logging.DEBUG, logger="osprey.cli.templates"):
            copy_template_data(
                tmp_path / "templates",
                project,
                package_name="quiet_project",
                data_bundle="control_assistant",
                ctx={},
                data_root=data_root,
            )

        assert (project / "data" / "channels.json").exists()
        assert "Copied profile data files" in caplog.text
        assert "Copied profile data" not in plain(capsys.readouterr().out)

    def test_stdout_csv_prune_is_quiet(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pruning ``data/raw/`` reports the removal at DEBUG, not on stdout."""
        from osprey.cli.templates.scaffolding import prune_csv_build_artifacts

        raw = self._tree(tmp_path / "project" / "data" / "raw")

        capsys.readouterr()
        with caplog.at_level(logging.DEBUG, logger="osprey.cli.templates"):
            prune_csv_build_artifacts(tmp_path / "project", "hierarchical")

        assert not raw.exists()
        assert "no CSV build path" in caplog.text
        assert "no CSV build path" not in plain(capsys.readouterr().out)

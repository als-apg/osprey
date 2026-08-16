"""The BuildKit stream parser behind the live build table.

Most of these tests replay a real captured build. The two fixtures were taken
from actual runs and kept verbatim, because every shape that breaks a parser
here — reused vertex numbers, repeated headers, out-of-order step indices,
carriage-return progress captions — is one nobody would have thought to write
by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.deployment.build_progress import BuildModel, BuildRow

FIXTURES = Path(__file__).parent / "fixtures"

#: A compose build of four service images in one parallel invocation.
COLD_BUILD = FIXTURES / "buildkit_cold_build.log"

#: A single-image build, whose headers carry no service name at all.
PROJECT_IMAGE = FIXTURES / "buildkit_project_image.log"

#: BuildKit bookkeeping that is not a caption: the vertex stopwatch and the
#: vertex's own status. Before the parser told the two apart, six caption
#: observations in ten of the cold build replay matched this — the whole table
#: read ``DONE 0.0s`` for most of a quarter-hour build.
BOOKKEEPING = re.compile(r"^(\d+\.\d+\s|DONE\b|CACHED\b|ERROR\b|sha256:)")

#: Real captions from the cold build, pinned verbatim. Filtering bookkeeping is
#: only half the job: a filter that also ate these would leave a tidy table that
#: says nothing. Each one is chosen for a different shape the filters could trip
#: over — a stopwatch-prefixed line whose text starts with a word and a colon,
#: one carrying its own digits-and-dots, one ending in ``done``, and two that
#: never carried a stopwatch at all. All five arrive before their service's
#: image is named, so all three surfaces are still showing that row.
PINNED_CAPTIONS = [
    "Hit:1 https://deb.debian.org/debian trixie InRelease",
    "Fetched 10.0 MB in 1s (6811 kB/s)",
    "Reading package lists...",
    "transferring dockerfile: 10.59kB done",
    "exporting layers",
]


def replay(path: Path, model: BuildModel) -> list[dict[str, str | None]]:
    """Feed a fixture through a model, recording the step each line left set.

    The file is split on newlines only. Python's text mode would translate the
    carriage returns inside ``dpkg`` and ``pip`` progress captions into
    newlines, shredding one physical line into dozens and replaying a stream
    that ``run_captured`` never delivers.
    """
    lines = path.read_bytes().decode().split("\n")
    seen: list[dict[str, str | None]] = []
    for tick, line in enumerate(lines):
        model.feed(line, float(tick))
        seen.append({row.service: row.step for row in model.snapshot()})
    return seen


def steps_per_service(seen: list[dict[str, str | None]]) -> dict[str, set[str]]:
    """Every step value each service passed through during a replay.

    The final snapshot only shows where a service ended up; a build's progress
    is the sequence, so the assertions need the whole path.
    """
    steps: dict[str, set[str]] = {}
    for snapshot in seen:
        for service, step in snapshot.items():
            if step is not None:
                steps.setdefault(service, set()).add(step)
    return steps


def rows_by_service(model: BuildModel) -> dict[str, BuildRow]:
    return {row.service: row for row in model.snapshot()}


def caption_trace(path: Path) -> dict[str, list[str]]:
    """Every caption each service's row carried during a replay, in order.

    Consecutive repeats are collapsed: what a caption filter can get wrong is
    which strings reach a row, not how many ticks each one stayed there.
    """
    model = BuildModel()
    trace: dict[str, list[str]] = {}
    for tick, line in enumerate(path.read_bytes().decode().split("\n")):
        model.feed(line, float(tick))
        for row in model.snapshot():
            captions = trace.setdefault(row.service, [])
            if not captions or captions[-1] != row.caption:
                captions.append(row.caption)
    return trace


# ---------------------------------------------------------------------------
# The compose path: four services in one stream
# ---------------------------------------------------------------------------


@pytest.fixture
def cold_build() -> tuple[BuildModel, list[str], list[dict[str, str | None]]]:
    finished: list[str] = []
    model = BuildModel(on_finished_image=finished.append)
    return model, finished, replay(COLD_BUILD, model)


def test_a_parallel_build_gets_one_row_per_service(cold_build):
    model, _, _ = cold_build

    assert [row.service for row in model.snapshot()] == [
        "virtual-accelerator",
        "event-dispatcher",
        "bluesky-bridge",
        "bluesky-panels",
    ]


def test_build_infrastructure_never_becomes_a_service(cold_build):
    """`[va internal]` and `[bluesky-panels auth]` name a real service and are
    attributed to it — only the bare, nameless forms are infrastructure.

    This stream has no bare forms at all, so the absence assertions here would
    hold even with the filter removed. What it does prove is the other half:
    that the filter is not over-eager, because the two-word forms survive as
    steps on their services below. The filter itself is proved by the
    single-image replay, which is where the bare forms actually occur.
    """
    model, _, _ = cold_build

    services = {row.service for row in model.snapshot()}
    assert "internal" not in services
    assert "auth" not in services


def test_every_service_image_is_reported_once(cold_build):
    _, finished, _ = cold_build

    assert finished == [
        "my-control-assistant-va:local",
        "my-control-assistant-dispatch:local",
        "my-control-assistant-bluesky-bridge:local",
        "my-control-assistant-bluesky-panels:local",
    ]


def test_each_image_lands_on_the_service_that_built_it(cold_build):
    """The export vertices carry no step index and share their numbers with
    earlier sub-graphs; attribution has to come from their headers."""
    model, _, _ = cold_build

    rows = rows_by_service(model)
    assert rows["virtual-accelerator"].finished == "my-control-assistant-va:local"
    assert rows["event-dispatcher"].finished == "my-control-assistant-dispatch:local"
    assert rows["bluesky-bridge"].finished == "my-control-assistant-bluesky-bridge:local"
    assert rows["bluesky-panels"].finished == "my-control-assistant-bluesky-panels:local"


def test_each_service_walks_its_own_dockerfile_steps(cold_build):
    """Step indices are read from the headers, never counted from arrival
    order — they interleave across services and arrive out of sequence.

    `bluesky-bridge` genuinely reports `1/8` for its FROM vertex and `2/7`
    onwards for the rest: BuildKit shares that vertex with another service's
    sub-graph and it keeps that Dockerfile's total. The row reports what the
    stream said rather than repairing it.
    """
    _, _, seen = cold_build

    steps = steps_per_service(seen)
    assert steps["virtual-accelerator"] == {f"{n}/8" for n in range(1, 9)} | {
        "internal",
        "exporting",
    }
    assert steps["event-dispatcher"] == {f"{n}/8" for n in range(1, 9)} | {
        "internal",
        "exporting",
    }
    assert steps["bluesky-bridge"] == {"1/8"} | {f"{n}/7" for n in range(2, 8)} | {
        "internal",
        "exporting",
    }
    assert steps["bluesky-panels"] == {f"{n}/7" for n in range(1, 8)} | {
        "auth",
        "internal",
        "exporting",
    }


def test_a_service_carries_a_caption_for_what_it_is_doing(cold_build):
    """The caption is the whole point of the table: without it a long step is
    indistinguishable from a hung one."""
    _, _, seen = cold_build
    model, _, _ = cold_build

    rows = rows_by_service(model)
    assert all(row.caption for row in rows.values())
    assert rows["virtual-accelerator"].started_at < rows["bluesky-panels"].started_at


# ---------------------------------------------------------------------------
# Captions carry work, never bookkeeping
# ---------------------------------------------------------------------------
#
# Most of what a vertex prints is not a caption. BuildKit stamps its own
# stopwatch onto every line a step's command writes, and reports each vertex's
# status on lines of the same shape -- and both used to land in the caption
# column, where the stopwatch reads as a second step index (`6/8  21.51
# Collecting scikit-learn`) and `DONE 0.0s` reads as a finished build. The
# replay is the instrument: no hand-written line would have found sixty
# percent.


def test_no_caption_is_buildkit_bookkeeping():
    """The stopwatch and the vertex-status lines never reach a row."""
    trace = caption_trace(COLD_BUILD)

    polluted = [
        (service, caption)
        for service, captions in trace.items()
        for caption in captions
        if BOOKKEEPING.match(caption)
    ]
    assert polluted == []


def test_genuine_captions_survive_the_filter_verbatim():
    """The other half: a filter that ate the work too would pass the test above
    with an empty table."""
    trace = caption_trace(COLD_BUILD)

    seen = {caption for captions in trace.values() for caption in captions}
    assert [caption for caption in PINNED_CAPTIONS if caption not in seen] == []


def test_filtering_captions_changes_no_row(cold_build):
    """Bookkeeping lines still prove their service is alive, so the rows they
    belong to must all still be there — the filter drops text, not rows."""
    model, _, _ = cold_build

    assert list(caption_trace(COLD_BUILD)) == [row.service for row in model.snapshot()]


def test_the_vertex_stopwatch_is_not_part_of_the_caption():
    """`#12 0.322 Hit:1 ...` is one caption with a timestamp in front of it, and
    the timestamp is the vertex's, not the step's."""
    model = BuildModel()

    model.feed("#12 [virtual-accelerator 5/8] RUN apt-get update", 0.0)
    model.feed("#12 0.322 Hit:1 https://deb.debian.org/debian trixie InRelease", 1.0)

    assert model.snapshot()[0].caption == "Hit:1 https://deb.debian.org/debian trixie InRelease"


def test_the_stopwatch_is_stripped_from_the_frame_that_survived():
    """The stamp is per write, so a redrawn line carries one on every frame —
    including the last one, which is the only one anybody saw."""
    model = BuildModel()

    model.feed("#17 [virtual-accelerator 6/8] RUN pip install", 0.0)
    model.feed("#17 21.51 Collecting numpy\r21.52 Collecting scikit-learn", 1.0)

    assert model.snapshot()[0].caption == "Collecting scikit-learn"


@pytest.mark.parametrize(
    "echo",
    [
        "DONE 0.0s",
        "CACHED",
        "ERROR: process did not complete successfully: exit code 1",
        "sha256:00430733972c 14.40MB / 14.40MB 0.3s done",
    ],
)
def test_a_vertex_status_echo_leaves_the_row_s_work_on_screen(echo: str):
    """These close a vertex; they do not describe what the service is doing.
    They arrive last, so a row that showed them would show nothing else."""
    model = BuildModel()

    model.feed("#12 [virtual-accelerator 5/8] RUN pip install osprey-framework", 0.0)
    model.feed("#12 3.551 Collecting scikit-learn", 1.0)
    model.feed(f"#12 {echo}", 2.0)

    row = model.snapshot()[0]
    assert row.caption == "Collecting scikit-learn"
    assert row.step == "5/8"


def test_a_status_echo_still_proves_its_service_is_alive():
    """It is the one thing such a line does say, and the heartbeats read it:
    dropping the line outright would make a chatty service look stalled."""
    model = BuildModel()

    model.feed("#12 [virtual-accelerator 5/8] RUN pip install", 0.0)
    model.feed("#12 DONE 4.2s", 9.0)

    assert model.snapshot()[0].last_line_at == 9.0


def test_a_mid_build_vertex_done_does_not_finish_the_service():
    """A service's Dockerfile is a dozen vertices and every one of them reports
    `DONE`. Only `naming to ... done` means the image is built — reading the
    first `DONE` as one would drop a still-building service from the table and
    silence its heartbeats for the rest of the build."""
    model = BuildModel()

    model.feed("#12 [virtual-accelerator 5/8] RUN pip install", 0.0)
    model.feed("#12 DONE 4.2s", 1.0)

    assert model.snapshot()[0].finished is None


def test_a_status_echo_alone_opens_no_row():
    """Labelled, so the line could have bound to a row — it carries no step and
    no caption, so the row it would open is a blank line in the table."""
    model = BuildModel("proj")

    model.feed("#1 DONE 0.1s", 0.0)

    assert model.snapshot() == []


# ---------------------------------------------------------------------------
# The single-image path: headers with no service name
# ---------------------------------------------------------------------------


@pytest.fixture
def project_image() -> tuple[BuildModel, list[str], list[dict[str, str | None]]]:
    finished: list[str] = []
    model = BuildModel("my-control-assistant", on_finished_image=finished.append)
    return model, finished, replay(PROJECT_IMAGE, model)


def test_a_single_image_build_is_one_row_under_its_label(project_image):
    """Its headers are `[ 1/13]`, `[internal]` and `[auth]` — none of them name
    a service, so the label is the only thing that can name this row."""
    model, _, _ = project_image

    assert [row.service for row in model.snapshot()] == ["my-control-assistant"]


def test_a_nameless_header_does_not_invent_a_service(project_image):
    """`[auth]` and `[internal]` sit exactly where a service name sits here.
    Taken at face value each would open a phantom row beside the real one."""
    model, _, _ = project_image

    services = {row.service for row in model.snapshot()}
    assert "auth" not in services
    assert "internal" not in services


def test_the_single_image_reports_once_under_its_label(project_image):
    model, finished, _ = project_image

    assert finished == ["my-control-assistant:local"]
    assert model.snapshot()[0].finished == "my-control-assistant:local"


def test_nameless_step_indices_bind_to_the_label(project_image):
    """The indices arrive out of order — `[ 2/13]` lands after `[ 5/13]` —
    which is exactly why the row reads the header rather than counting."""
    _, _, seen = project_image

    steps = steps_per_service(seen)
    assert steps["my-control-assistant"] == {f"{n}/13" for n in range(1, 14)} | {
        "auth",
        "internal",
        "exporting",
    }


def test_a_bare_export_vertex_still_finds_its_row(project_image):
    """This build's export vertex never carries a header at all — it is bare
    `#22 exporting to image` — so it can only belong to the labelled row."""
    model, _, _ = project_image

    assert model.snapshot()[0].step == "exporting"


# ---------------------------------------------------------------------------
# Degrading on a stream that is not BuildKit's
# ---------------------------------------------------------------------------


def test_a_non_buildkit_stream_leaves_the_model_empty():
    """A provider with its own progress format must not be half-parsed into a
    misleading table; an empty model is the renderer's cue to stay out of the
    way."""
    finished: list[str] = []
    model = BuildModel("proj", on_finished_image=finished.append)

    for line in [
        "STEP 1/13: FROM docker.io/library/python:3.12-slim",
        "--> Using cache 4f3a1c0b9e77",
        "STEP 2/13: RUN apt-get update && apt-get install -y curl",
        "Successfully tagged localhost/proj:local",
        "",
    ]:
        model.feed(line, 0.0)

    assert model.snapshot() == []
    assert finished == []


def test_an_unrecognized_bracket_is_dropped_rather_than_guessed_at():
    """Labelled, so this pins the bracket as unrecognized rather than merely
    unbound: a nameless header would have opened the label's row here."""
    model = BuildModel("proj")

    model.feed("#1 [one two three] something", 0.0)
    model.feed("#2 [] something", 1.0)

    assert model.snapshot() == []


def test_a_nameless_header_with_no_label_is_dropped():
    """On the compose path every header names its service, so a nameless one
    has nothing to bind to — inventing a row for it would be a fabrication."""
    model = BuildModel()

    model.feed("#1 [ 2/13] RUN true", 0.0)
    model.feed("#1 DONE 0.1s", 1.0)

    assert model.snapshot() == []


# ---------------------------------------------------------------------------
# Shapes the fixtures prove, pinned in isolation
# ---------------------------------------------------------------------------


def test_a_reused_vertex_number_follows_its_latest_header():
    """`compose build` numbers each service's sub-graph independently, so the
    same `#7` belongs to a different service later in the same stream. Keyed on
    the number alone, captions cross silently between services."""
    model = BuildModel()

    model.feed("#7 [event-dispatcher 1/8] FROM python:3.12-slim", 0.0)
    model.feed("#7 extracting sha256:abc", 1.0)
    model.feed("#7 [bluesky-bridge 1/8] FROM python:3.11-slim", 2.0)
    model.feed("#7 resolving python:3.11-slim", 3.0)

    rows = rows_by_service(model)
    assert rows["event-dispatcher"].caption == "extracting sha256:abc"
    assert rows["bluesky-bridge"].caption == "resolving python:3.11-slim"


def test_a_repeated_infrastructure_header_stays_on_the_labelled_row():
    """The single-image build's bare `[internal]` headers occupy four distinct
    vertices but appear five times — one of them re-emits its header after a
    pause, interleaved with another vertex.

    Both halves of that shape can leak. Filtering the name but still mapping
    the vertex would send the continuation lines to a phantom row; counting
    occurrences rather than vertices would make the repeat look like a fifth
    vertex. Neither is visible from the final row unless the interleave is
    replayed, which is what this does.
    """
    model = BuildModel("proj")

    model.feed("#5 [internal] load metadata for docker.io/library/python:3.12-slim", 0.0)
    model.feed("#5 ...", 1.0)
    model.feed("#6 [auth] library/python:pull token for registry-1.docker.io", 2.0)
    model.feed("#6 DONE 0.0s", 3.0)
    model.feed("#5 [internal] load metadata for docker.io/library/python:3.12-slim", 4.0)
    model.feed("#5 DONE 0.6s", 5.0)

    rows = model.snapshot()
    assert [row.service for row in rows] == ["proj"]
    assert rows[0].caption == "load metadata for docker.io/library/python:3.12-slim"
    assert rows[0].started_at == 0.0
    # The `DONE` lines carry no caption, but they are still this row's output.
    assert rows[0].last_line_at == 5.0


def test_a_redrawn_progress_caption_keeps_only_its_last_frame():
    """`dpkg` redraws one line in place, so a single captured line holds a
    dozen overwrites; only the last was ever on screen."""
    model = BuildModel()

    model.feed("#17 [va 6/8] RUN apt-get install", 0.0)
    model.feed("#17 8.494 (Reading database ... \r(Reading database ... 45%\r", 1.0)

    assert model.snapshot()[0].caption == "(Reading database ... 45%"


def test_a_resumed_vertex_repeats_its_header_without_restarting_the_row():
    model = BuildModel()

    model.feed("#20 [va 6/8] RUN pip install", 0.0)
    model.feed("#20 [va 6/8] RUN pip install", 5.0)

    row = model.snapshot()[0]
    assert row.started_at == 0.0
    assert row.last_line_at == 5.0


def test_a_snapshot_does_not_change_underneath_its_reader():
    """The renderer's monitor thread snapshots while the main thread feeds."""
    model = BuildModel()
    model.feed("#20 [va 6/8] RUN pip install", 0.0)
    before = model.snapshot()

    model.feed("#20 [va 7/8] COPY . /app", 1.0)

    assert before[0].step == "6/8"
    assert model.snapshot()[0].step == "7/8"


# ---------------------------------------------------------------------------
# Finished-image reporting
# ---------------------------------------------------------------------------
#
# The callback is what container_lifecycle's step reporter is built on, and the
# fixtures cannot prove it: all four of the cold build's `naming to` lines
# arrive already `done`-suffixed and unrepeated. These pin the same behaviour
# the reporter's own tests pin, at the level the model owns it.


def test_each_finished_image_is_reported_once():
    finished: list[str] = []
    model = BuildModel(on_finished_image=finished.append)

    for line in [
        "#12 [va internal] load build context",
        "#37 naming to docker.io/library/proj-va:local",
        "#37 naming to docker.io/library/proj-va:local 0.1s done",
        "#41 naming to docker.io/library/proj-dispatch:local done",
    ]:
        model.feed(line, 0.0)

    assert finished == ["proj-va:local", "proj-dispatch:local"]


def test_an_image_named_twice_reports_once():
    """BuildKit repeats the `naming to` line (bare, then with a duration); an
    image must not appear to build twice."""
    finished: list[str] = []
    model = BuildModel(on_finished_image=finished.append)

    model.feed("#37 naming to docker.io/library/proj-va:local done", 0.0)
    model.feed("#37 naming to docker.io/library/proj-va:local 0.0s done", 1.0)

    assert finished == ["proj-va:local"]


def test_registry_qualified_tags_keep_their_registry():
    """Only the implicit docker.io/library/ prefix is noise; a real registry in
    the tag is the operator's own naming and stays."""
    finished: list[str] = []
    model = BuildModel(on_finished_image=finished.append)

    model.feed("#9 naming to ghcr.io/lab/proj-svc:2026.6 done", 0.0)

    assert finished == ["ghcr.io/lab/proj-svc:2026.6"]


def test_unfinished_naming_lines_report_nothing():
    """The bare `naming to` line appears while the export is still running; a
    step for it would announce an image that may yet fail."""
    finished: list[str] = []
    model = BuildModel(on_finished_image=finished.append)

    model.feed("#37 naming to docker.io/library/proj-va:local", 0.0)
    model.feed("#37 DONE 0.2s", 1.0)

    assert finished == []


def test_an_unattributable_image_is_still_reported():
    """A finished image is a fact about the build, not about one row: the step
    line must not go missing because the vertex was too anonymous to place."""
    finished: list[str] = []
    model = BuildModel(on_finished_image=finished.append)

    model.feed("#37 naming to docker.io/library/proj-va:local done", 0.0)

    assert finished == ["proj-va:local"]
    assert model.snapshot() == []

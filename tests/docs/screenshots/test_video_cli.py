"""Unit tests for the demo-video commands of the screenshot CLI.

``video``, ``upload-check``, ``upload`` and ``fetch-video``, the demo video in
``list``, and which options each command accepts.

CI-safe: no container, no browser, no ffmpeg. The preflight, the stack, the
recorder and the encoder are stubbed at the module attributes the CLI looks up
at call time.
"""

from __future__ import annotations

import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from docs.screenshots import __main__ as cli
from docs.screenshots import video_take
from docs.screenshots.__main__ import main
from docs.screenshots.video_timeline import Step, Timeline

import osprey

REPO_ROOT = Path(cli.__file__).resolve().parents[2]
VENV_BIN = Path(sys.executable).parent


def _which(missing: set[str] = frozenset(), osprey_at: Path | None = None):
    """A ``shutil.which`` stand-in: tools in *missing* are absent, osprey sits at *osprey_at*."""

    def which(name, *args, **kwargs):
        if name in missing:
            return None
        if name == "osprey":
            return str(osprey_at if osprey_at is not None else VENV_BIN / "osprey")
        return f"/usr/local/bin/{name}"

    return which


# ---------------------------------------------------------------------------
# video: benign absence is one ``skipped:`` line
# ---------------------------------------------------------------------------


def test_video_cli_ffmpeg_missing_prints_one_skipped_line(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", _which(missing={"ffmpeg"}))

    rc = main(["video"])

    out, err = capsys.readouterr()
    assert rc == 0
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1, err
    assert lines[0].startswith("skipped video:")
    assert "ffmpeg" in lines[0]
    assert "Traceback" not in out + err


def test_video_cli_ffprobe_missing_skips(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", _which(missing={"ffprobe"}))

    rc = main(["video", "--theme", "dark"])

    _, err = capsys.readouterr()
    assert rc == 0
    assert err.splitlines() == [err.strip()]
    assert err.startswith("skipped video:") and "ffprobe" in err


def test_video_cli_skip_raised_while_building_stack_is_one_line(monkeypatch, capsys, tmp_path):
    from docs.screenshots import capture

    monkeypatch.setattr(cli, "_preflight", lambda: None)
    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path / "demo-video")

    @contextmanager
    def no_browser(_work_dir):
        raise capture.ScreenshotSkip("playwright is not installed")
        yield  # pragma: no cover

    monkeypatch.setattr(cli, "_video_stack", no_browser)

    rc = main(["video"])

    out, err = capsys.readouterr()
    assert rc == 0
    assert out == ""
    assert err == "skipped video: playwright is not installed\n"


# ---------------------------------------------------------------------------
# video: the osprey-location preflight
# ---------------------------------------------------------------------------


def test_video_cli_preflight_passes_for_this_repo_venv(monkeypatch):
    monkeypatch.setattr(shutil, "which", _which())

    cli._preflight()  # no ScreenshotSkip


def test_video_cli_osprey_cli_outside_the_venv_skips(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", _which(osprey_at=Path("/opt/elsewhere/bin/osprey")))

    rc = main(["video"])

    _, err = capsys.readouterr()
    assert rc == 0
    assert err.startswith("skipped video:")
    assert len(err.splitlines()) == 1
    assert "/opt/elsewhere/bin/osprey" in err


def test_video_cli_osprey_cli_missing_skips(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", _which(missing={"osprey"}))

    rc = main(["video"])

    _, err = capsys.readouterr()
    assert rc == 0
    assert err.startswith("skipped video:") and "osprey" in err


def test_video_cli_osprey_package_outside_the_repo_skips(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", _which())
    monkeypatch.setattr(osprey, "__file__", "/opt/site-packages/osprey/__init__.py")

    rc = main(["video"])

    _, err = capsys.readouterr()
    assert rc == 0
    assert err.startswith("skipped video:")
    assert len(err.splitlines()) == 1
    assert "/opt/site-packages/osprey" in err


# ---------------------------------------------------------------------------
# video: recording and encoding against stubs
# ---------------------------------------------------------------------------


class _FakeTake:
    def __init__(self, theme):
        self.theme = theme
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def stubbed_video(monkeypatch, tmp_path):
    """Stub the preflight, the stack, the recorder and the encoder; return the call log."""
    calls: dict[str, list] = {"record": [], "encode": [], "stack": [], "order": []}
    out_dir = tmp_path / "demo-video"
    monkeypatch.setattr(cli, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(cli, "_preflight", lambda: None)

    @contextmanager
    def fake_stack(work_dir):
        stack = video_take.VideoStack(
            project_dir=tmp_path / "project",
            artifact_port=1,
            base_url="http://127.0.0.1:1",
            operator_secret="s",
            work_dir=work_dir,
        )
        calls["stack"].append(stack)
        yield stack

    def fake_record(stack, theme, **_kwargs):
        calls["record"].append(theme)
        calls["order"].append(f"record {theme}")
        take = _FakeTake(theme)
        stack.current = take
        return take, Timeline(theme=theme, version="1.2.3", recorded_at=0.0)

    durations = {"dark": 57.0, "light": 57.0}

    def fake_encode(theme, _take, _timeline, out):
        calls["encode"].append(theme)
        calls["order"].append(f"encode {theme}")
        return {
            "mp4": out / f"osprey-demo-{theme}.mp4",
            "poster": out / f"osprey-demo-{theme}-poster.jpg",
            "timeline": Timeline.path_for(out, theme),
            "size": 3_500_000,
            "duration": durations[theme],
        }

    monkeypatch.setattr(cli, "_video_stack", fake_stack)
    monkeypatch.setattr(video_take, "record_theme", fake_record)
    monkeypatch.setattr(cli, "_encode_theme", fake_encode)
    calls["durations"] = durations
    calls["out_dir"] = out_dir
    return calls


def test_video_cli_theme_dark_records_only_dark(stubbed_video, capsys):
    rc = main(["video", "--theme", "dark"])

    out, _ = capsys.readouterr()
    assert rc == 0
    assert stubbed_video["record"] == ["dark"]
    assert stubbed_video["encode"] == ["dark"]
    assert "osprey-demo-dark.mp4" in out
    assert "osprey-demo-light.mp4" not in out


def test_video_cli_default_records_each_theme_on_a_fresh_stack(stubbed_video, capsys):
    # A second take on the same stack finds the first take's cards in the
    # gallery and reuses them instead of doing the work on camera.
    rc = main(["video"])

    out, _ = capsys.readouterr()
    assert rc == 0
    assert stubbed_video["record"] == ["dark", "light"]
    assert stubbed_video["encode"] == ["dark", "light"]
    assert len(stubbed_video["stack"]) == 2
    first, second = stubbed_video["stack"]
    assert first is not second and first.work_dir != second.work_dir
    assert "WARNING" not in out


def test_video_cli_a_retried_take_gets_a_fresh_stack(stubbed_video, monkeypatch, capsys):
    # A retry on the failed take's stack would find that take's cards in the
    # gallery, and the agent would reuse them instead of doing the work.
    real = video_take.record_theme
    seen: list[str] = []

    def dark_fails_once(stack, theme, **kwargs):
        seen.append(theme)
        if theme == "dark" and len(seen) == 1:
            raise video_take.ProbeFailed("correlate", "no plot")
        return real(stack, theme, **kwargs)

    monkeypatch.setattr(video_take, "record_theme", dark_fails_once)
    rc = main(["video", "--theme", "dark"])

    out = capsys.readouterr().out
    assert rc == 0
    assert seen == ["dark", "dark"]
    assert len(stubbed_video["stack"]) == 2
    # One line per failed attempt.
    assert [line for line in out.splitlines() if line.startswith("WARNING")] == [
        "WARNING: dark take 1/3 failed at correlate: no plot"
    ]


def test_video_cli_a_theme_that_fails_every_take_names_it(stubbed_video, monkeypatch, capsys):
    def always(_stack, _theme, **_kwargs):
        raise video_take.ProbeFailed("post", "no approval")

    monkeypatch.setattr(video_take, "record_theme", always)
    rc = main(["video", "--theme", "dark"])

    out, err = capsys.readouterr()
    assert rc != 0
    assert len(stubbed_video["stack"]) == cli.TAKES_PER_THEME
    assert out.count("WARNING: dark take") == cli.TAKES_PER_THEME
    assert "theme dark" in err and "post" in err


def test_video_cli_a_take_outside_the_window_is_kept_with_a_warning(stubbed_video, capsys):
    # The length follows the agent's pace, which can change with OSPREY or the
    # models; an out-of-window take is handed back, never retaken.
    stubbed_video["durations"]["light"] = 85.2
    rc = main(["video", "--theme", "light"])
    out = capsys.readouterr().out
    assert rc == 0
    assert stubbed_video["record"] == ["light"]
    assert stubbed_video["encode"] == ["light"]
    assert len(stubbed_video["stack"]) == 1
    warnings = [line for line in out.splitlines() if line.startswith("WARNING")]
    assert len(warnings) == 1 and "85.2 s" in warnings[0]


def test_video_cli_each_theme_is_encoded_before_the_next_is_recorded(stubbed_video):
    main(["video"])
    assert stubbed_video["order"] == ["record dark", "encode dark", "record light", "encode light"]


def test_video_cli_a_failed_later_theme_keeps_the_earlier_video(stubbed_video, monkeypatch, capsys):
    real = video_take.record_theme

    def light_fails(stack, theme, **kwargs):
        if theme == "light":
            raise video_take.ProbeFailed("plot", "no plot")
        return real(stack, theme, **kwargs)

    monkeypatch.setattr(video_take, "record_theme", light_fails)
    rc = main(["video"])

    out, err = capsys.readouterr()
    assert rc != 0
    assert stubbed_video["encode"] == ["dark"]
    assert "osprey-demo-dark.mp4" in out
    assert "theme light" in err


def test_video_cli_stack_is_built_once_and_closes_the_open_take(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from docs.screenshots import capture

    events: list[str] = []

    def cm(name, value):
        @contextmanager
        def manager(*args, **kwargs):
            events.append(f"enter {name}")
            yield value
            events.append(f"exit {name}")

        return manager

    monkeypatch.setattr(capture, "rendered_artifact_port", lambda project_dir: 4321)
    monkeypatch.setattr(capture, "chromium_context", cm("browser", "BROWSER"))
    monkeypatch.setattr(capture, "tutorial_stack", cm("tutorial", tmp_path / "project"))
    web = SimpleNamespace(
        base_url="http://127.0.0.1:9",
        operator_secret="secret",
        claude_config_dir=tmp_path / "session-config",
    )
    monkeypatch.setattr(capture, "web_terminal", cm("terminal", web))

    take = _FakeTake("dark")
    with cli._video_stack(tmp_path) as stack:
        assert stack.browser == "BROWSER"
        assert stack.artifact_port == 4321
        assert stack.base_url == "http://127.0.0.1:9"
        assert stack.operator_secret == "secret"
        assert stack.claude_config_dir == tmp_path / "session-config"
        assert stack.work_dir == tmp_path
        stack.current = take
        events.append("record")

    assert take.closed
    assert stack.current is None
    assert events == [
        "enter browser",
        "enter tutorial",
        "enter terminal",
        "record",
        "exit terminal",
        "exit tutorial",
        "exit browser",
    ]


@pytest.mark.usefixtures("stubbed_video")
def test_video_cli_prints_paths_sizes_and_durations(capsys):
    main(["video", "--theme", "light"])

    out = capsys.readouterr().out
    assert "osprey-demo-light.mp4" in out
    assert "osprey-demo-light-poster.jpg" in out
    assert "timeline-light.json" in out
    assert "57.0 s" in out
    assert "3.5 MB" in out


@pytest.mark.parametrize("duration", [64.96, 75.4])
def test_video_cli_length_inside_the_window_does_not_warn(stubbed_video, capsys, duration):
    stubbed_video["durations"]["light"] = duration
    rc = main(["video", "--theme", "light"])
    assert rc == 0
    assert "WARNING" not in capsys.readouterr().out


def test_video_cli_length_window_is_fifty_to_eighty_seconds():
    assert (cli.VIDEO_MIN_S, cli.VIDEO_MAX_S) == (50.0, 80.0)


@pytest.mark.parametrize("duration", [45.9, 49.0, 80.5])
def test_video_cli_length_outside_window_warns(stubbed_video, capsys, duration):
    stubbed_video["durations"]["dark"] = duration

    rc = main(["video", "--theme", "dark"])

    out = capsys.readouterr().out
    assert rc == 0
    warnings = [line for line in out.splitlines() if line.startswith("WARNING")]
    assert len(warnings) == 1
    assert "50" in warnings[0] and "80" in warnings[0]


@pytest.mark.usefixtures("stubbed_video")
def test_video_cli_theme_failure_exits_non_zero_naming_theme_and_step(monkeypatch, capsys):
    def failing(_stack, _theme, **_kwargs):
        raise video_take.ProbeFailed("approve", "no approval card")

    monkeypatch.setattr(video_take, "record_theme", failing)

    rc = main(["video", "--theme", "dark"])

    out, err = capsys.readouterr()
    assert rc != 0
    assert "theme dark" in err and "approve" in err
    assert "Traceback" not in out + err


def test_video_cli_help_lists_theme(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["video", "--help"])
    assert exc.value.code == 0
    assert "--theme" in capsys.readouterr().out


def test_video_cli_rejects_unknown_theme():
    with pytest.raises(SystemExit) as exc:
        main(["video", "--theme", "sepia"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# upload-check
# ---------------------------------------------------------------------------


@pytest.fixture
def published(monkeypatch, tmp_path):
    """Both themes' takes: videos in OUTPUT_DIR, posters + manifest in the docs tree."""
    from docs.screenshots import video_manifest

    out, static = tmp_path / "out", tmp_path / "static"
    out.mkdir()
    static.mkdir()
    monkeypatch.setattr(cli, "OUTPUT_DIR", out)
    monkeypatch.setattr(cli, "STATIC_DEMO_DIR", static)

    def take(theme: str, version: str = "2026.9.0", duration: float = 58.2, size: int = 16):
        mp4 = out / f"osprey-demo-{theme}.mp4"
        mp4.write_bytes(theme.encode() * size)
        poster = static / f"osprey-demo-{theme}-poster.jpg"
        poster.write_bytes(b"poster-" + theme.encode())
        video_manifest.put_theme(
            static / "manifest.json",
            theme,
            video_manifest.entry(
                osprey_version=version,
                recorded_at=1_790_000_000.0,
                duration_s=duration,
                real_session_s=307.0,
                speed=16.0,
                mp4=mp4,
                poster=poster,
            ),
        )

    return SimpleNamespace(out=out, static=static, take=take, manifest=static / "manifest.json")


def test_video_cli_upload_check_passes_when_files_match_the_manifest(published, capsys):
    published.take("dark")
    published.take("light")

    rc = main(["upload-check"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "dark" in out and "light" in out and "2026.9.0" in out
    assert "2026-09-21" in out  # recorded_at of 1_790_000_000 in UTC
    assert "307" in out and "16" in out
    assert "not uploaded" in out
    assert "WARNING" not in out


def test_video_cli_upload_check_names_the_release_once_uploaded(published, capsys):
    from docs.screenshots import video_manifest

    published.take("dark")
    published.take("light")
    video_manifest.set_release(published.manifest, "docs-media-v2026.9.0")

    assert main(["upload-check"]) == 0
    out = capsys.readouterr().out
    assert "docs-media-v2026.9.0" in out and "not uploaded" not in out


@pytest.mark.parametrize("which", ["mp4", "poster"])
def test_video_cli_upload_check_fails_when_a_file_does_not_match(published, capsys, which):
    published.take("dark")
    published.take("light")
    target = (
        published.out / "osprey-demo-light.mp4"
        if which == "mp4"
        else published.static / "osprey-demo-light-poster.jpg"
    )
    target.write_bytes(b"re-encoded since")

    rc = main(["upload-check"])

    _, err = capsys.readouterr()
    assert rc != 0
    assert target.name in err and "sha256" in err


def test_video_cli_upload_check_fails_when_a_file_is_missing(published, capsys):
    published.take("dark")
    published.take("light")
    (published.out / "osprey-demo-light.mp4").unlink()

    assert main(["upload-check"]) != 0
    assert "osprey-demo-light.mp4" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["recorded_at", "osprey_version", "real_session_s"])
def test_video_cli_upload_check_fails_on_a_missing_manifest_field(published, capsys, field):
    import json

    published.take("dark")
    published.take("light")
    data = json.loads(published.manifest.read_text())
    del data["themes"]["dark"][field]
    published.manifest.write_text(json.dumps(data))

    rc = main(["upload-check"])

    out, err = capsys.readouterr()
    assert rc != 0
    assert field in err
    assert "1970" not in out + err


def test_video_cli_upload_check_fails_without_a_theme(published, capsys):
    published.take("dark")

    assert main(["upload-check"]) != 0
    assert "light" in capsys.readouterr().err


def test_video_cli_upload_check_warns_on_version_window_and_size(published, capsys):
    from docs.screenshots import video_encode

    published.take("dark", version="2026.9.0", duration=85.0)
    published.take("light", version="2026.9.1", size=video_encode.SIZE_BUDGET_BYTES)

    rc = main(["upload-check"])

    out = capsys.readouterr().out
    assert rc == 0
    warnings = [line for line in out.splitlines() if line.startswith("WARNING")]
    assert len(warnings) == 3
    assert any("2026.9.0" in w and "2026.9.1" in w for w in warnings)
    assert any("85.0 s" in w for w in warnings)
    assert any("osprey-demo-light.mp4" in w and "MB" in w for w in warnings)


# ---------------------------------------------------------------------------
# run and list are unchanged
# ---------------------------------------------------------------------------


def test_video_cli_list_still_works(capsys):
    assert main(["list"]) == 0
    assert capsys.readouterr().out


def test_video_cli_list_shows_the_demo_video(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    video = [line for line in out.splitlines() if line.startswith("video")]
    assert video and "dark,light" in video[0]
    assert "osprey-demo-<theme>.mp4" in out and "manifest.json" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--theme", "dark"],
        ["list", "--theme", "dark"],
        ["video", "--stack"],
        ["video", "--only", "ariel"],
        ["video", "--agentic"],
        ["upload-check", "--theme", "dark"],
        ["video", "--version", "2026.9.0"],
        ["upload"],
        ["run", "--strict"],
        ["fetch-video", "--version", "2026.9.0"],
    ],
)
def test_video_cli_rejects_flags_that_do_not_belong_to_the_command(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_video_cli_upload_checks_first_then_publishes(monkeypatch):
    from docs.screenshots import video_publish

    calls: list = []
    monkeypatch.setattr(cli, "_upload_check", lambda out_dir: calls.append("check") or 0)
    monkeypatch.setattr(
        video_publish,
        "upload",
        lambda version, out, static: calls.append((version, out, static)) or 0,
    )
    assert main(["upload", "--version", "2026.9.0"]) == 0
    assert calls == ["check", ("2026.9.0", cli.OUTPUT_DIR, cli.STATIC_DEMO_DIR)]


def test_video_cli_upload_stops_when_the_check_fails(monkeypatch):
    from docs.screenshots import video_publish

    monkeypatch.setattr(cli, "_upload_check", lambda out_dir: 1)
    monkeypatch.setattr(video_publish, "upload", lambda *a: pytest.fail("uploaded"))
    assert main(["upload", "--version", "2026.9.0"]) == 1


@pytest.mark.parametrize("strict", [True, False])
def test_video_cli_fetch_video_passes_strict_through(monkeypatch, strict):
    from docs.screenshots import video_publish

    seen: list = []
    monkeypatch.setattr(
        video_publish, "fetch", lambda static, strict: seen.append((static, strict)) or 0
    )
    argv = ["fetch-video", "--strict"] if strict else ["fetch-video"]
    assert main(argv) == 0
    assert seen == [(cli.STATIC_DEMO_DIR, strict)]


def test_video_cli_repo_root_is_the_checkout():
    assert (REPO_ROOT / "src" / "osprey").is_dir()


def test_video_cli_the_encoded_frames_carry_the_real_time_clock(monkeypatch, tmp_path):
    # The clock and the speed badge are burned into the frames the encoder reads.
    from types import SimpleNamespace

    from docs.screenshots import video_burn, video_encode
    from PIL import Image

    frames_dir = tmp_path / "work" / "dark" / "frames"
    frames_dir.mkdir(parents=True)
    frames = []
    for i, t in enumerate((0.0, 3.0, 50.0)):
        path = frames_dir / f"frame_{i:06d}.jpg"
        Image.new("RGB", video_burn.FRAME_SIZE, (90, 90, 90)).save(path)
        frames.append((path, t))
    take = SimpleNamespace(sink=SimpleNamespace(frames=frames), frames_dir=frames_dir)
    timeline = Timeline(
        theme="dark",
        version="1",
        recorded_at=0.0,
        steps=[
            Step(name="prompt-1", fast_forward=False, start=0.0, end=2.0),
            Step(name="plot", fast_forward=True, start=2.0, end=62.0),
            Step(name="close", fast_forward=False, start=62.0, end=64.0),
        ],
    )
    concat_lists: list[str] = []

    def encode(concat, mp4, _duration):
        concat_lists.append(Path(concat).read_text())
        Path(mp4).write_bytes(b"mp4")
        return 3

    monkeypatch.setattr(video_encode, "encode_to_budget", encode)
    monkeypatch.setattr(cli, "_publish_take", lambda *a, **k: {})
    cli._encode_theme("dark", take, timeline, tmp_path / "out")

    (text,) = concat_lists
    burned = [line[5:].strip("'") for line in text.splitlines() if line.startswith("file ")]
    assert burned and all(Path(p).parent.name == "burned-dark" for p in burned)
    assert any("_x" in Path(p).stem for p in burned)  # a sped-up piece carries the badge


def _rotating_timeline() -> Timeline:
    return Timeline(
        theme="dark",
        version="2026.9.0",
        recorded_at=1790500000.0,
        steps=[
            Step(name="focus-1", fast_forward=False, start=100.0, end=100.4),
            Step(name="prompt-1", fast_forward=False, start=100.4, end=106.0),
            Step(name="plot", fast_forward=True, start=106.0, end=206.0),
            Step(name="rotate", fast_forward=False, start=206.0, end=210.0),
            Step(name="close", fast_forward=False, start=210.0, end=213.0),
        ],
    )


def test_video_cli_publishing_a_take_writes_poster_manifest_and_a_local_copy(monkeypatch, tmp_path):
    # The poster (the 3D plot at the end of the rotation) and the manifest go
    # straight into the docs tree; the MP4 is copied there too, git-ignored,
    # so a local docs build plays it.
    from docs.screenshots import video_encode, video_manifest

    static = tmp_path / "static"
    monkeypatch.setattr(cli, "STATIC_DEMO_DIR", static)
    mp4 = tmp_path / "out" / "osprey-demo-dark.mp4"
    mp4.parent.mkdir()
    mp4.write_bytes(b"video")
    timeline = _rotating_timeline()
    commands: list[list[str]] = []

    def run(cmd, check):
        assert check
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"poster")

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(video_encode, "_binary", lambda name: name)
    monkeypatch.setattr(video_encode, "probe", lambda _p: {"size": 5, "duration": 58.24})

    result = cli._publish_take("dark", mp4, timeline)

    (cmd,) = commands
    assert f"select=eq(n\\,{video_encode.poster_frame(timeline)})" in cmd
    poster = static / "osprey-demo-dark-poster.jpg"
    assert result["poster"] == poster and poster.read_bytes() == b"poster"
    assert (static / "osprey-demo-dark.mp4").read_bytes() == b"video"
    manifest = video_manifest.load(static / "manifest.json")
    assert manifest["release"] is None
    entry = manifest["themes"]["dark"]
    assert entry["mp4"] == {"name": mp4.name, "sha256": video_manifest.sha256(mp4)}
    assert entry["poster"]["sha256"] == video_manifest.sha256(poster)
    assert entry["duration_s"] == 58.24
    assert entry["speed"] == round(video_encode.timeline_speed(timeline), 2)
    assert entry["real_session_s"] == pytest.approx(213.0 - 100.4, abs=0.1)
    assert entry["osprey_version"] == "2026.9.0"
    assert video_manifest.missing_fields(entry) == []

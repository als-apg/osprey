"""Unit tests for the docs-video registry and recording every registered video.

CI-safe: the browser, the tutorial stack, the recorder and the encoder are faked.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from docs.screenshots import __main__ as cli
from docs.screenshots import capture, video_manifest, video_take
from docs.screenshots.__main__ import main
from docs.screenshots.video_beats import hold
from docs.screenshots.video_layout import Layout
from docs.screenshots.video_timeline import Timeline
from docs.screenshots.videos import VIDEOS, validate_videos, video_named

from tests.docs.screenshots.video_fixture import SECOND

DOCS_SOURCE = Path(cli.__file__).resolve().parents[1] / "source"
LANDING = video_named("landing")

# The landing takes as committed before the manifest moved to one entry per video.
LANDING_RELEASE = "docs-media-v2026.9.0b4"
LANDING_SHA256 = {
    ("dark", "mp4"): "5d1063c2d03bc3dbeda01f08b10697becf33c2ac926bd5953776715a676b53ec",
    ("dark", "poster"): "85b2c9c6210c788f64284e58408e9a409d30c5a887ff6ab4404c88e5faf7d033",
    ("light", "mp4"): "f47bd6581915b103daa61111a07dc19b5372535697d5cefbb45d7b5aabd1675d",
    ("light", "poster"): "a2620736304dfd9698dab229606868c3dfd7aa8d8ce59dc8dd8fef388ea926f0",
}


# --- the registry -------------------------------------------------------------


def test_the_registry_validates() -> None:
    validate_videos()


def test_names_and_stems_are_unique() -> None:
    assert len({v.name for v in VIDEOS}) == len(VIDEOS)
    assert len({v.stem for v in VIDEOS}) == len(VIDEOS)


def test_video_named_finds_a_video_and_names_the_known_ones_otherwise() -> None:
    videos = (LANDING, SECOND)
    assert video_named("second", videos) is SECOND
    with pytest.raises(KeyError, match="landing, second"):
        video_named("nope", videos)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"themes": ()}, "themes"),
        ({"themes": ("dark", "sepia")}, "themes"),
        ({"storyboard": ()}, "empty storyboard"),
        ({"poster_step": "nope"}, "poster_step"),
        ({"init_sets": ("x.enabled",)}, "KEY=VALUE"),
        ({"layout": None}, "has no layout"),
        ({"layout": Layout(surface="tui", active="okf", tiles=())}, "layout surface"),
        ({"layout": Layout(surface="chat", active="", tiles=())}, "no active sidebar entry"),
    ],
)
def test_validate_videos_rejects_a_malformed_entry(change, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_videos([dataclasses.replace(SECOND, **change)])


def test_validate_videos_rejects_a_poster_step_that_is_only_a_plain_callable() -> None:
    video = dataclasses.replace(SECOND, storyboard=(lambda take: None,), poster_step="x")
    with pytest.raises(ValueError, match="poster_step"):
        validate_videos([video])
    validate_videos([dataclasses.replace(video, storyboard=(hold("x", None, 1),))])


def test_validate_videos_rejects_a_duplicate_name() -> None:
    with pytest.raises(ValueError, match="duplicate video name"):
        validate_videos([LANDING, dataclasses.replace(SECOND, name="landing")])


def test_validate_videos_rejects_a_duplicate_stem() -> None:
    with pytest.raises(ValueError, match="duplicate video stem"):
        validate_videos([LANDING, dataclasses.replace(SECOND, stem=LANDING.stem)])


@pytest.mark.parametrize(
    ("stem", "page"),
    [(v.stem, page) for v in VIDEOS for page in v.pages],
)
def test_every_page_a_video_names_embeds_it(stem, page) -> None:
    path = DOCS_SOURCE / page
    assert path.is_file(), path
    assert f"_static/demo/{stem}-light.mp4" in path.read_text(encoding="utf-8")


# --- the committed manifest ---------------------------------------------------


def test_the_committed_manifest_names_only_registered_videos_with_their_stems() -> None:
    manifest = video_manifest.load()
    assert set(manifest["videos"]) <= {v.name for v in VIDEOS}
    for video in VIDEOS:
        entry = manifest["videos"].get(video.name)
        if entry is None:
            continue  # pending: not recorded yet
        assert set(entry["themes"]) <= set(video.themes)
        for theme, take in entry["themes"].items():
            assert video_manifest.missing_fields(take) == []
            assert take["mp4"]["name"] == f"{video.stem}-{theme}.mp4"
            assert take["poster"]["name"] == f"{video.stem}-{theme}-poster.jpg"


def test_the_landing_takes_survived_the_manifest_migration() -> None:
    landing = video_manifest.video(video_manifest.load(), "landing")
    assert landing["release"] == LANDING_RELEASE
    assert set(landing["themes"]) == {"dark", "light"}
    for (theme, key), digest in LANDING_SHA256.items():
        assert landing["themes"][theme][key]["sha256"] == digest
        assert video_manifest.missing_fields(landing["themes"][theme]) == []


def test_the_committed_manifest_round_trips_byte_for_byte(tmp_path) -> None:
    committed = video_manifest.MANIFEST_PATH.read_bytes()
    copy = tmp_path / "manifest.json"
    video_manifest.save(video_manifest.load(), copy)
    assert copy.read_bytes() == committed
    assert video_manifest.load(copy) == video_manifest.load()


# --- recording every video ----------------------------------------------------


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    """Fake the browser, the stack, the recorder and the encoder; log what each saw."""
    log = SimpleNamespace(stacks=[], records=[], encodes=[])
    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path / "demo-video")
    monkeypatch.setattr(cli, "_preflight", lambda: None)

    @contextmanager
    def chromium_context():
        yield "BROWSER"

    @contextmanager
    def tutorial_stack(sets=()):
        log.stacks.append(tuple(sets))
        yield tmp_path / f"project-{len(log.stacks)}"

    @contextmanager
    def web_terminal(_project_dir):
        yield SimpleNamespace(
            base_url="http://127.0.0.1:1", operator_secret="s", claude_config_dir=None
        )

    monkeypatch.setattr(capture, "chromium_context", chromium_context)
    monkeypatch.setattr(capture, "tutorial_stack", tutorial_stack)
    monkeypatch.setattr(capture, "web_terminal", web_terminal)
    monkeypatch.setattr(capture, "rendered_artifact_port", lambda _project_dir: 1)

    def record_theme(stack, theme, storyboard, **_kw):
        video = next(v for v in cli.VIDEOS if v.storyboard is storyboard)
        log.records.append((video.name, theme, stack.project_dir))
        return SimpleNamespace(close=lambda: None), Timeline(
            theme=theme, version="1", recorded_at=0.0
        )

    def encode(video, theme, _take, _timeline, out):
        log.encodes.append((video.name, theme))
        return {
            "mp4": out / f"{video.stem}-{theme}.mp4",
            "poster": out / f"{video.stem}-{theme}-poster.jpg",
            "timeline": Timeline.path_for(out / video.name, theme),
            "size": 1_000_000,
            "duration": video.window_s[0] + 1,
        }

    monkeypatch.setattr(video_take, "record_theme", record_theme)
    monkeypatch.setattr(cli, "_encode_theme", encode)
    return log


@pytest.mark.usefixtures("second_registered")
def test_record_all_plays_every_video_in_registry_order_each_take_on_a_fresh_stack(
    recorder, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("DOCS_VIDEO_TEST_KEY", "dummy")

    assert main(["video"]) == 0

    assert [(name, theme) for name, theme, _ in recorder.records] == [
        ("landing", "dark"),
        ("landing", "light"),
        ("second", "dark"),
        ("second", "light"),
    ]
    assert recorder.encodes == [(n, t) for n, t, _ in recorder.records]
    assert recorder.stacks == [(), (), SECOND.init_sets, SECOND.init_sets]
    assert len({project for _, _, project in recorder.records}) == 4
    out, err = capsys.readouterr()
    assert "osprey-second-light.mp4" in out and "WARNING" not in out
    assert err == ""


@pytest.mark.usefixtures("second_registered")
def test_one_named_video_records_only_it(recorder, monkeypatch) -> None:
    monkeypatch.setenv("DOCS_VIDEO_TEST_KEY", "dummy")

    assert main(["video", "--video", "second"]) == 0

    assert [(n, t) for n, t, _ in recorder.records] == [("second", "dark"), ("second", "light")]
    assert recorder.stacks == [SECOND.init_sets] * 2


@pytest.mark.usefixtures("second_registered")
def test_an_unset_needs_env_skips_only_that_video(recorder, monkeypatch, capsys) -> None:
    monkeypatch.delenv("DOCS_VIDEO_TEST_KEY", raising=False)

    assert main(["video"]) == 0

    assert [(n, t) for n, t, _ in recorder.records] == [("landing", "dark"), ("landing", "light")]
    err = capsys.readouterr().err
    assert err.splitlines() == ["skipped video: second needs DOCS_VIDEO_TEST_KEY"]


def test_an_entry_cannot_be_declared_without_a_layout() -> None:
    fields = {f.name: getattr(SECOND, f.name) for f in dataclasses.fields(SECOND)}
    del fields["layout"]
    with pytest.raises(TypeError, match="layout"):
        type(SECOND)(**fields)


def test_the_watchdog_starts_in_the_chat_with_knowledge_open() -> None:
    watchdog = video_named("watchdog")
    assert watchdog.layout == Layout(surface="chat", active="okf", tiles=("okf",))
    assert "config.artifact_server.example_artifact=false" in watchdog.init_sets


def test_the_landing_layout_is_what_its_take_has_always_shown() -> None:
    assert LANDING.layout == Layout(surface="terminal", active="artifacts", tiles=("artifacts",))


def test_history_is_refused_on_the_terminal() -> None:
    with pytest.raises(ValueError, match="history needs the chat"):
        validate_videos([dataclasses.replace(LANDING, history=("p",))])


def test_the_code_review_video_shows_the_rule_before_the_mark_and_the_prompt() -> None:
    """Settings → Safety → the rules, spotlit, then the drawer closes before the arm."""
    beats = [b.name for b in video_named("code-review").storyboard]
    opening = ["more", "settings", "safety", "rules-listed", "rules", "rules-shown", "rule"]
    assert beats[: len(opening)] == opening
    assert beats.index("rule") < beats.index("close") < beats.index("arm") < beats.index("ask")

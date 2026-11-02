"""CLI entry point: ``python -m docs.screenshots [command] [options]``.

``run`` (the default) captures the container-free static recipes —
``standalone_interface``, ``static_page`` and ``hermetic_hub`` — zero container,
CI-safe locally. ``--stack`` opts into the tutorial-stack recipes (needs a
container runtime + the port layout's free postgres port); ``--agentic`` opts
into the live web-terminal hero (needs a live Claude session). ``list`` prints
the registry, the docs videos included, without capturing anything.

``video`` records every docs video in :data:`docs.screenshots.videos.VIDEOS`
(``--video NAME`` one of them, ``--theme dark|light`` one theme), each take on
its own tutorial stack, and encodes it into ``docs/demo-video/``; the posters
and ``manifest.json`` go into ``docs/source/_static/demo/``. It needs ffmpeg,
ffprobe, Playwright chromium, a container runtime, a live Claude session on a
provider that serves an Opus model, and the ``osprey`` CLI of this checkout; a
missing prerequisite prints one ``skipped video:`` line on stderr and exits 0,
and so does a video whose ``needs_env`` is not set, without stopping the
others. ``upload-check`` checks the takes against the manifest.
``upload --version YYYY.M.P[aN|bN|rcN]`` publishes the videos not uploaded yet
(or ``--video NAME``) to that version's ``docs-media`` release and names it in
the manifest. ``fetch-video`` downloads each video from the release the
manifest names, for the docs build.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from docs.screenshots.recipes import (
    REGISTRY,
    is_enabled,
    select_recipes,
    validate_registry,
)
from docs.screenshots.video_manifest import MANIFEST_PATH, STATIC_DEMO_DIR
from docs.screenshots.videos import VIDEOS, DocVideo, validate_videos, video_named

THEMES = ("dark", "light")

# Encoded videos and their timelines land here; ignored by git. The posters and
# the manifest go straight into the docs tree (STATIC_DEMO_DIR), with a
# git-ignored copy of each video so a local docs build plays it.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "demo-video"

# Takes tried per theme, each on its own fresh tutorial stack.
TAKES_PER_THEME = 3

REPO_ROOT = Path(__file__).resolve().parents[2]


COMMANDS = ("run", "list", "video", "upload-check", "upload", "fetch-video")

# The options each command takes; any other option set on it is an error.
_COMMAND_FLAGS = {
    "run": {"only", "stack", "agentic"},
    "list": set(),
    "video": {"theme", "video"},
    "upload-check": {"video"},
    "upload": {"version", "video"},
    "fetch-video": {"strict"},
}
_FLAG_NAMES = {
    "only": "--only",
    "stack": "--stack",
    "agentic": "--agentic",
    "theme": "--theme",
    "video": "--video",
    "version": "--version",
    "strict": "--strict",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m docs.screenshots",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=list(COMMANDS),
        help=(
            "'run' (default) captures the selected recipes; 'list' prints the registry; "
            "'video' records and encodes the docs videos; 'upload-check' checks the takes "
            "against the manifest; 'upload' publishes them to a version's release; "
            "'fetch-video' downloads the release the manifest names."
        ),
    )
    parser.add_argument(
        "--only", metavar="NAME", default=None, help="Capture only the named recipe."
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="Include tutorial_stack recipes (containers, on a port block of their own).",
    )
    parser.add_argument(
        "--agentic", action="store_true", help="Include agentic recipes (live Claude session)."
    )
    parser.add_argument(
        "--theme",
        choices=list(THEMES),
        default=None,
        help="video: record only this theme (default: both dark and light).",
    )
    parser.add_argument(
        "--video",
        choices=[v.name for v in VIDEOS],
        default=None,
        help="video, upload-check, upload: only this docs video (default: every one).",
    )
    parser.add_argument(
        "--version",
        metavar="VERSION",
        default=None,
        help="upload: the OSPREY version the videos belong to, YYYY.M.P or a pre-release such as 2026.9.0b4 (release docs-media-v<VERSION>).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fetch-video: fail on a hash mismatch or a gh error (a deploy, not a preview).",
    )
    return parser


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    allowed = _COMMAND_FLAGS[args.command]
    for dest, flag in _FLAG_NAMES.items():
        if getattr(args, dest) not in (None, False) and dest not in allowed:
            parser.error(f"{flag} does not apply to '{args.command}'")
    if args.command == "upload" and not args.version:
        parser.error("'upload' needs --version YYYY.M.P[aN|bN|rcN]")
    return args


def _print_registry() -> None:
    if not REGISTRY:
        print("(registry is empty)")
        return
    width = max(len(s.name) for s in REGISTRY)
    for shot in REGISTRY:
        outs = ", ".join(f"{o}.png" for o in shot.output_names())
        print(
            f"{shot.name:<{width}}  {shot.environment:<20} {shot.kind:<8} themes={','.join(shot.themes)}"
        )
        print(f"{'':<{width}}    → {outs}")
    for video in VIDEOS:
        print(
            f"video  {video.name}  themes={','.join(video.themes)}  pages={','.join(video.pages)}"
        )
        print(
            f"      → docs/demo-video/{video.stem}-<theme>.mp4, "
            f"docs/source/_static/demo/{video.stem}-<theme>-poster.jpg, manifest.json"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    # A malformed registry is a hard error for every command — surface it clearly.
    try:
        validate_registry()
        validate_videos()
    except ValueError as exc:
        print(f"Invalid screenshot registry: {exc}", file=sys.stderr)
        return 2

    if args.command == "list":
        _print_registry()
        return 0

    names = [args.video] if args.video else None

    if args.command == "upload-check":
        return _upload_check(OUTPUT_DIR, names)

    if args.command == "upload":
        from docs.screenshots import video_publish

        if _upload_check(OUTPUT_DIR, names) != 0:
            return 1
        return video_publish.upload(args.version, OUTPUT_DIR, STATIC_DEMO_DIR, names)

    if args.command == "fetch-video":
        from docs.screenshots import video_publish

        return video_publish.fetch(STATIC_DEMO_DIR, strict=args.strict)

    if args.command == "video":
        from docs.screenshots import capture

        themes = [args.theme] if args.theme else list(THEMES)
        videos = [video_named(args.video)] if args.video else list(VIDEOS)
        try:
            return _video(videos, themes, OUTPUT_DIR)
        except capture.ScreenshotSkip as exc:
            print(f"skipped video: {exc}", file=sys.stderr)
            return 0

    selected = select_recipes(stack=args.stack, agentic=args.agentic, only=args.only)

    if not selected:
        if args.only is not None:
            known = next((s for s in REGISTRY if s.name == args.only), None)
            if known is None:
                print(
                    f"No recipe named {args.only!r}. Run 'list' to see the registry.",
                    file=sys.stderr,
                )
            elif not is_enabled(known, stack=args.stack, agentic=args.agentic):
                flag = "--agentic" if known.kind == "agentic" else "--stack"
                print(f"Recipe {args.only!r} needs {flag}. Re-run with it added.", file=sys.stderr)
            return 1
        print(
            "No recipes selected. (Default runs standalone recipes; add --stack/--agentic.)",
            file=sys.stderr,
        )
        return 1

    # Import the runner lazily: 'list' and selection must work even where the
    # heavy capture dependencies (Playwright/chromium) are unavailable.
    from docs.screenshots import capture

    return 0 if capture.run(selected, stack=args.stack, agentic=args.agentic) else 1


def _preflight() -> None:
    """Raise ScreenshotSkip unless ffmpeg, ffprobe and this checkout's osprey are in reach.

    The recorder drives the ``osprey`` CLI to build the tutorial stack, so the
    CLI must be the one installed next to this interpreter and the package it
    imports must be this checkout's source; otherwise the video would show some
    other OSPREY than the one being documented.
    """
    from docs.screenshots import capture

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise capture.ScreenshotSkip(f"{tool} not found on PATH; the demo video needs it")

    venv_bin = Path(sys.executable).parent
    cli = shutil.which("osprey")
    if cli is None:
        raise capture.ScreenshotSkip(f"osprey CLI not found on PATH; expected it in {venv_bin}")
    if Path(cli).parent.resolve() != venv_bin.resolve():
        raise capture.ScreenshotSkip(
            f"osprey CLI at {cli} is not the one next to {sys.executable}; "
            "activate this checkout's environment"
        )

    import osprey

    package = Path(osprey.__file__).resolve()
    if not package.is_relative_to(REPO_ROOT):
        raise capture.ScreenshotSkip(
            f"osprey is imported from {package.parent}, not from this checkout ({REPO_ROOT}); "
            "install it in editable mode"
        )


@contextmanager
def _video_stack(work_dir: Path, video: DocVideo) -> Iterator[Any]:
    """Build the browser, *video*'s tutorial stack and web terminal; yield a VideoStack.

    The take still open when the block exits is closed before the stack is torn down.
    """
    from docs.screenshots import capture, video_take

    with (
        capture.chromium_context() as browser,
        capture.tutorial_stack(video.init_sets) as project_dir,
        capture.web_terminal(project_dir) as web,
    ):
        stack = video_take.VideoStack(
            project_dir=project_dir,
            artifact_port=capture.rendered_artifact_port(project_dir),
            base_url=web.base_url,
            operator_secret=web.operator_secret,
            claude_config_dir=web.claude_config_dir,
            work_dir=work_dir,
            browser=browser,
        )
        try:
            yield stack
        finally:
            if stack.current is not None:
                stack.current.close()
                stack.current = None


def _encode_theme(
    video: DocVideo, theme: str, take: Any, timeline: Any, out_dir: Path
) -> dict[str, Any]:
    """Save the timeline, encode the take to MP4, then publish its poster and manifest entry.

    The frames the encoder reads carry the burned-in real-time clock and, while
    sped up, the speed-up badge. Each video's timelines go in a folder of its own.
    """
    from docs.screenshots import video_burn, video_encode

    out_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = timeline.save(out_dir / video.name)
    burner = video_burn.Burner(take.frames_dir.parent / f"burned-{theme}")
    text, duration = video_encode.build_ffconcat(take.sink.frames, timeline, annotate=burner)
    concat = take.frames_dir.parent / f"frames-{theme}.ffconcat"
    concat.write_text(text)
    mp4 = out_dir / f"{video.stem}-{theme}.mp4"
    video_encode.encode_to_budget(concat, mp4, duration)
    return {"mp4": mp4, "timeline": timeline_path, **_publish_take(video, theme, mp4, timeline)}


def _publish_take(video: DocVideo, theme: str, mp4: Path, timeline: Any) -> dict[str, Any]:
    """Write *theme*'s poster, local video copy and manifest entry into the docs tree.

    The poster is the last frame of the video's ``poster_step`` beat. It is
    what the page shows before playback, without JavaScript, and when the
    video cannot play.
    """
    from docs.screenshots import video_encode, video_manifest

    STATIC_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    poster = STATIC_DEMO_DIR / f"{video.stem}-{theme}-poster.jpg"
    frame = video_encode.poster_frame(timeline, video.poster_step)
    subprocess.run(video_encode.poster_command(mp4, poster, frame=frame), check=True)
    local = STATIC_DEMO_DIR / mp4.name
    if local.resolve() != Path(mp4).resolve():
        shutil.copyfile(mp4, local)
    info = video_encode.probe(mp4)
    manifest = STATIC_DEMO_DIR / MANIFEST_PATH.name
    video_manifest.put_theme(
        manifest,
        video.name,
        theme,
        video_manifest.entry(
            osprey_version=timeline.osprey_version,
            recorded_at=timeline.recorded_at,
            duration_s=info["duration"],
            real_session_s=video_encode.real_session(timeline),
            speed=video_encode.timeline_speed(timeline),
            mp4=mp4,
            poster=poster,
        ),
    )
    return {
        "poster": poster,
        "manifest": manifest,
        "size": info["size"],
        "duration": info["duration"],
    }


def _window_warning(name: str, duration: float, video: DocVideo) -> str | None:
    low, high = video.window_s
    if low <= duration <= high:
        return None
    return f"{name} runs {duration:.1f} s, outside the {low:g}-{high:g} s window"


def _report(result: dict[str, Any], video: DocVideo) -> None:
    duration = result["duration"]
    print(f"{result['mp4']}  {result['size'] / 1e6:.1f} MB  {duration:.1f} s")
    print(f"{result['poster']}")
    print(f"{result['timeline']}")
    warning = _window_warning(result["mp4"].name, duration, video)
    if warning:
        print(f"WARNING: {warning}")


def _video(videos: list[DocVideo], themes: list[str], out_dir: Path) -> int:
    """Preflight, then record and encode each video's themes into *out_dir* in turn.

    Videos play in registry order, each theme on its own stack. A video whose
    ``needs_env`` is not set prints one ``skipped video:`` line and the next
    one goes on. A take that fails (a check, a stall, a dialog, an empty plot,
    a browser error) is retried, up to :data:`TAKES_PER_THEME` takes, each on a
    fresh tutorial stack: on a stack an earlier take has used, the agent finds
    that take's work and reuses it rather than doing it on camera. A take's
    length is never a failure; one outside the window is kept and reported with
    a WARNING. Each theme is encoded as soon as it is recorded, so a later
    failure keeps the earlier videos.
    """
    from docs.screenshots import video_take

    _preflight()
    for video in videos:
        unset = [name for name in video.needs_env if not os.environ.get(name)]
        if unset:
            print(f"skipped video: {video.name} needs {', '.join(unset)}", file=sys.stderr)
            continue
        for theme in (t for t in themes if t in video.themes):
            result = None
            last: video_take.ProbeFailed | None = None
            for attempt in range(1, TAKES_PER_THEME + 1):
                prefix = f"osprey-{video.name}-video-{theme}-"
                with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
                    try:
                        with _video_stack(Path(tmp), video) as stack:
                            take, timeline = video_take.record_theme(
                                stack,
                                theme,
                                video.storyboard,
                                layout=video.layout,
                                history=video.history,
                                write_switch=video.write_switch,
                                before_record=video.before_record,
                            )
                    except video_take.ProbeFailed as exc:
                        last = exc
                        print(
                            f"WARNING: {video.name} {theme} take {attempt}/{TAKES_PER_THEME} "
                            f"failed at {exc.step}: {exc.detail}"
                        )
                        continue
                    try:
                        result = _encode_theme(video, theme, take, timeline, out_dir)
                    except (subprocess.CalledProcessError, RuntimeError) as exc:
                        print(
                            f"error: encoding the {video.name} {theme} video failed: {exc}",
                            file=sys.stderr,
                        )
                        return 1
                break
            if result is None:
                assert last is not None
                failed = video_take.ThemeFailed(theme, last.step, last.detail, TAKES_PER_THEME)
                print(f"error: {video.name}: {failed}", file=sys.stderr)
                return 1
            _report(result, video)
    return 0


def _check_video(
    video: DocVideo, entry: dict[str, Any], out_dir: Path, errors: list[str], warnings: list[str]
) -> None:
    """Check one video's takes, appending what is wrong to *errors* and *warnings*.

    An uploaded video's MP4 lives in its release; it is checked here only
    when a local copy is present.
    """
    from docs.screenshots import video_encode, video_manifest

    release = entry.get("release")
    versions: dict[str, str] = {}
    for theme in video.themes:
        label = f"{video.name} {theme}"
        theme_entry = entry["themes"].get(theme)
        if theme_entry is None:
            errors.append(
                f"{label}: no take in the manifest; run 'python -m docs.screenshots video "
                f"--video {video.name}'"
            )
            continue
        missing = video_manifest.missing_fields(theme_entry)
        if missing:
            errors.append(f"{label}: manifest is missing {', '.join(missing)}")
            continue
        for key, folder in (("mp4", out_dir), ("poster", STATIC_DEMO_DIR)):
            path = folder / theme_entry[key]["name"]
            if not path.is_file():
                if not (key == "mp4" and release):
                    errors.append(f"{label}: missing {path}")
            elif video_manifest.sha256(path) != theme_entry[key]["sha256"]:
                errors.append(f"{label}: {path.name} does not match its sha256 in the manifest")
        versions[theme] = theme_entry["osprey_version"]
        print(
            f"{label}: osprey {theme_entry['osprey_version']}, recorded "
            f"{theme_entry['recorded_at']}, {theme_entry['duration_s']:.1f} s covering "
            f"{theme_entry['real_session_s']:.0f} s of real session at {theme_entry['speed']:g}x"
        )
        warning = _window_warning(theme_entry["mp4"]["name"], theme_entry["duration_s"], video)
        if warning:
            warnings.append(warning)
        mp4 = out_dir / theme_entry["mp4"]["name"]
        if mp4.is_file() and mp4.stat().st_size > video_encode.SIZE_BUDGET_BYTES:
            warnings.append(
                f"{mp4.name} is {mp4.stat().st_size / 1e6:.2f} MB, over the "
                f"{video_encode.SIZE_BUDGET_BYTES / 1e6:g} MB budget"
            )
    if len(set(versions.values())) > 1:
        detail = ", ".join(f"{theme} {version}" for theme, version in versions.items())
        warnings.append(
            f"{video.name}: the themes were recorded from different OSPREY versions ({detail})"
        )


def _upload_check(out_dir: Path, names: list[str] | None = None) -> int:
    """Check every registered video's takes (or the named ones) against the manifest.

    Fails when a theme or a manifest field is missing, or when an MP4 (in
    *out_dir*) or a poster (in the docs tree) is missing or does not match its
    SHA-256 in the manifest; a video already uploaded needs no local MP4.
    Warns, without failing, when a video falls outside its length window,
    exceeds the size budget, or its themes were recorded from different OSPREY
    versions. Reports whether each video is uploaded.
    """
    from docs.screenshots import video_manifest

    manifest = video_manifest.load(STATIC_DEMO_DIR / MANIFEST_PATH.name)
    videos = [video_named(n) for n in names] if names else list(VIDEOS)
    errors: list[str] = []
    warnings: list[str] = []
    for video in videos:
        _check_video(video, video_manifest.video(manifest, video.name), out_dir, errors, warnings)
    for line in warnings:
        print(f"WARNING: {line}")
    if errors:
        for line in errors:
            print(f"error: {line}", file=sys.stderr)
        return 1
    for video in videos:
        release = video_manifest.video(manifest, video.name).get("release")
        print(
            f"{video.name}: uploaded to release {release}"
            if release
            else f"{video.name}: not uploaded (no release named yet)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

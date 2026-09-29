# Documentation screenshots

The committed doc images are regenerated from recipes rather than captured by
hand, and a whole web-interface redesign can be reviewed as a single page.
(For building and link-checking the docs themselves, see
[LOCAL_TESTING.md](LOCAL_TESTING.md).)

## Refreshing documentation screenshots

The committed doc images under `docs/source/_static/screenshots/` are
regenerated from a declarative registry, not captured by hand. Each image is one
`DocShot` recipe in `docs/screenshots/recipes.py` — the authoritative list of
every doc screenshot and how it is produced. List them from the repository root
with:

```console
$ python -m docs.screenshots list
```

**The default is container-free.** `make screenshots` — run from `docs/`, where
the makefile lives — captures the container-free recipes: `standalone_interface`,
`static_page` and `hermetic_hub`. A `standalone_interface` recipe boots a single
interface `create_app()` on a throwaway port, so it needs neither a container
runtime nor seeded data. A `hermetic_hub` recipe boots the web terminal the
contact sheet uses. Regenerate one recipe with
`make screenshots-<name>`. The equivalent without the makefile is
`python -m docs.screenshots` from the repository root, which is what the target
runs for you.

**Two opt-in environments** cover the images that need real data:

- `SCREENSHOTOPTS=--stack` — the ARIEL search/browse/create/status views.
  Builds the `control-assistant` tutorial project, brings up Postgres
  (`osprey up -d`), and seeds the logbook with
  `osprey sim apply nominal --yes --now <anchor>`. Needs a container runtime
  and a free host port 5432. The `--now` anchor freezes the seeded dates, so
  repeat captures are byte-stable.
- `SCREENSHOTOPTS=--agentic` — the Web Terminal hero. Drives a live agent
  session to produce a real beam-current plot, so it needs a live Claude
  session on your subscription budget. Success is a structural check (non-blank
  image, correct viewport, plot present), not a byte comparison.

```console
$ cd docs
$ make screenshots                          # default: standalone only
$ make screenshots SCREENSHOTOPTS=--stack    # + ARIEL views (containers)
$ cd ..
$ python -m docs.screenshots --agentic --only web_terminal_hero
```

**Provenance is automatic.** Every capture stamps `manifest.json` with the
OSPREY version and UTC timestamp, and each figure's *"Captured with OSPREY
vX.Y.Z"* caption is generated from it — never hand-edit the version in a
caption.

This framework is **capture-only**: it is never a CI gate (the stack needs
Postgres; the hero needs a live agent). It is distinct from the CI visual-drift
guard — pixel diffs of each rendered interface against a committed baseline
live in the front-end **Visual** tests (regenerated with `--regen-baselines`),
and continue to run in CI unchanged.

Captures on the tutorial stack (the stack and agentic screenshots and the demo
video) share its settings: the tutorial is pinned to the provider's Opus model,
the agent session starts in auto permission mode, and the web terminal serves
its front-end libraries from this checkout (fetch them once with
`uv run osprey vendor fetch`).

## Reviewing a web-interface redesign (contact sheet)

When a web interface is being restyled, the **contact-sheet renderer** boots the
*real* interface in every theme/mode variant and folds the shots into one
self-contained page, so a whole redesign can be reviewed as a single artifact —
no live agent, provider, hardware, or network. It lives beside the screenshot
framework in `docs/screenshots/` but is a review tool, not a committed doc
image: nothing it produces is checked in or CI-gated.

```console
$ uv run python -m docs.screenshots.contact_sheet --out /tmp/sheet
```

That captures the Web Terminal's four shells — the **dark** and **light**
themes crossed with the **expert** and **simple** UI modes — writes one PNG per
cell into the output directory, composes them into `contact-sheet.html` there,
and prints its path. Open that one file to review every variant side by side.

**Comparing accent candidates.** Add `--accents` to render each of the four
variants twice, once under each accent candidate (blue vs teal), so a pending
accent decision can be made from real output rather than a mockup:

```console
$ uv run python -m docs.screenshots.contact_sheet --out /tmp/sheet --accents
```

To keep every cell looking like a working session with no live backend, the
renderer points the workspace panel at a pre-seeded demo store and replays a
canned terminal transcript. That transcript is width-guarded against the narrow
terminal card — the run fails fast if a line would overflow, before any browser
launches. Where no browser runtime is available the run skips with a one-line
notice instead of erroring.

**Extending it to another target.** The variant grid is the `VARIANTS` list of
`(theme, mode)` tuples near the top of `contact_sheet.py`, and the completeness
invariant `_FULL_MATRIX` mirrors it — add a cell to *both* to capture a new
theme/mode combination. To cover a new panel, seed its backing store the way
`seed_demo_workspace` seeds the workspace artifacts and wire it into
`hermetic_hub` so the panel renders populated; the capture loop and the
composed sheet then pick it up unchanged. A hub card in a particular UI state is
a `STAGED_VARIANTS` row naming a `STAGES` key, and the same key on a `DocShot`
makes it a committed doc image.

## Landing-page demo video

The documentation landing page plays a short demo video: three requests to the
agent on the `control-assistant` tutorial stack (a 3D plot, a correlation plot,
a logbook post), recorded in the real Web Terminal and cut to a 50–80 second
window. The agent's working time is sped up; a clock burned into every frame
shows the real elapsed time, and a badge shows the speed-up while it applies.
It is recorded once per theme (dark and light), and the page picks the one that
matches the reader's theme. Like the screenshots, it is capture-only and never a
CI gate.

**What it needs.** A container runtime (every take builds and starts its own
tutorial stack), a live Claude session on your subscription budget, `ffmpeg`
and `ffprobe` on the `PATH`, and the development environment of this checkout
(`uv sync --extra dev`, which provides Playwright and this checkout's `osprey`
CLI; install the browser once with `uv run playwright install chromium`, and
fetch the front-end libraries once with `uv run osprey vendor fetch`). The
provider must serve an Opus model: the tutorial is pinned to it. Pick the
provider with `OSPREY_DOCSHOTS_PROVIDER=<name>` (the preset's default
otherwise). A missing prerequisite prints one `skipped video:` line on stderr
and exits without recording.

**Recording a take.** From the repository root:

```console
$ uv run make -C docs demo-video                          # both themes
$ uv run make -C docs demo-video VIDEOOPTS="--theme dark" # one theme only
```

For each theme the run writes into `docs/demo-video/` (ignored by git) the video
`osprey-demo-<theme>.mp4` and `timeline-<theme>.json` (when each scene of the
take happened), and into `docs/source/_static/demo/` the poster
`osprey-demo-<theme>-poster.jpg` and its entry in `manifest.json`, plus a
git-ignored copy of the video so a local docs build plays it. The poster is the
last frame of the rotation: the 3D plot on screen after the agent has worked.
The page shows it before playback, without JavaScript, and whenever the video
cannot play. The manifest records per theme the OSPREY version, the recording
time, the video's length, the real session time it covers, the speed-up, and the
video's and the poster's SHA-256.

A take that fails (a check that does not pass, a stall, a dialog nobody can
answer, an empty plot on screen, a browser error) is retried on a fresh stack,
up to three takes per theme. A take's length is never a failure: the agent's
pace changes with OSPREY and the models, so a take outside the window is kept
and reported with a `WARNING`. Set `OSPREY_DOCSHOTS_KEEP=<dir>` to keep each
take's Claude Code session transcripts there for diagnosis.

**Reviewing a take.** Watch both MP4s end to end. Check that the gallery cards
open as they land, that the 3D plot rotates, that the correlation plot is shown
and is the plot attached to the logbook draft, that the approval step appears,
that the clock and the speed-up badge read correctly, and that the poster shows
the 3D plot. Then check both takes against the manifest:

```console
$ uv run python -m docs.screenshots upload-check
```

It fails when a take, a manifest field, a video or a poster is missing, or when
a file does not match its SHA-256 in the manifest; it warns when a video falls
outside the window, is over the 8 MB budget, or when the two themes were
recorded from different OSPREY versions; and it says whether the takes are
uploaded yet.

**Publishing.** Each OSPREY release has its own release of demo videos, named
`docs-media-vYYYY.M.P`, so every docs version plays the video recorded for it.
Upload both videos for the version being released, a final `YYYY.M.P` or a
pre-release such as `2026.9.0b4` (needs an authenticated `gh` CLI):

```console
$ uv run make -C docs upload-demo-video VERSION=YYYY.M.P
```

The target runs `upload-check`, creates that version's release with both videos
in one step (or replaces the videos of that same version's release on a
retake; it never touches another version's release), and names the release in
`manifest.json`. Commit `docs/source/_static/demo/` (the manifest and the two
posters); the videos themselves are never committed. The docs build downloads
exactly the release the committed manifest names and checks each video against
its SHA-256. When the manifest names no release yet, or the release or a video
is missing, the landing page shows the poster; a hash mismatch or another `gh`
error fails a deploy and is only a notice on a pull request. Between releases
`main` keeps showing the last release's video, whose manifest and posters are
the ones committed.

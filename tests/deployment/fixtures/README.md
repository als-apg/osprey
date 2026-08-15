# BuildKit progress fixtures

Two real BuildKit plain-progress spools, captured from OSPREY deploys and then
thinned. They are the inputs for `tests/deployment/test_build_progress.py`,
which parses the stream the lifecycle verbs render as a live build table.

They are captures, not constructions. A hand-written line would assert only what
its author already believed the format to be, which is the one thing these files
exist to check. Nothing here may be authored or edited by hand.

## Where they came from

Both were written by an OSPREY lifecycle verb into the project's
`var/logs/build-*.log` spool — the file the deploy already keeps, captured as
the build ran.

- `buildkit_cold_build.log` — a cold `compose build` of a four-service project
  (`bluesky-bridge`, `bluesky-panels`, `event-dispatcher`,
  `virtual-accelerator`). Every service name in the stream is a real service.
- `buildkit_project_image.log` — a single project-image build. It has 13
  nameless step headers, in both shapes BuildKit emits: space-padded below ten
  (`#10 [ 2/13]`) and unpadded from ten on (`#18 [10/13]`). It also carries the
  bare `[auth]` and `[internal]` vertices that have no service behind them at
  all.

## What may be trimmed

Only *repetitions* of a line type that is still represented elsewhere in the
file. Never the last instance of a type.

The bulk of a cold build is pip and apt output — timestamped caption lines of
the form `#17 3.551 <text>` under a long `RUN`. Those are safe to thin, and are
the only thing that has been thinned here. Each thinned vertex keeps its head
and tail captions, dropping the run of repeats between them.

The caption text, caption order, and each vertex's first/last timestamps are
real. The gap between the kept head slice and the kept tail slice is a
trimming artifact, not a real pause — at some vertices it is several times
larger than any genuine gap in the untrimmed source. **Do not assert on
inter-caption timing or caption rate**: doing so would encode a trimming
artifact as expected BuildKit behavior.

Left untouched, in stream order, byte for byte:

- every vertex header, in all of its shapes
- `DONE`, `CACHED`, and the `#N ...` pause marker
- `exporting`, `unpacking to`, `naming to`, and the pull/`sha256:` progress
  lines, including the bare-then-duration repeats a parser has to dedup
- the `auth` token vertices

Both fixtures capture successful builds. Neither contains an `#N ERROR`
terminator or a `CANCELED` line, so the failure path is not covered by either
one — a maintainer who needs it will have to capture a new fixture.

Do not reorder lines, renumber `#N` vertices, or split a physical line. Some
lines carry embedded carriage returns from `dpkg` progress bars; they are single
lines and must stay that way — the cold-build fixture has 188 such lines, a
count of its own trimmed content, not a count preserved from the untrimmed
source.

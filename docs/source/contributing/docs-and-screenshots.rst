.. _contributing-docs-and-screenshots:

=============================
Documentation and Screenshots
=============================

The documentation images are regenerated from recipes rather than captured by
hand, and a whole web-interface redesign can be reviewed as a single page.

Refreshing documentation screenshots
------------------------------------

The committed doc images under ``docs/source/_static/screenshots/`` are
regenerated from a declarative registry, not captured by hand. Each image is one
``DocShot`` recipe in ``docs/screenshots/recipes.py`` -- the authoritative list
of every doc screenshot and how it is produced. List them from the repository
root with:

.. code-block:: console

   $ python -m docs.screenshots list

**The default is container-free.** ``make screenshots`` -- run from ``docs/``,
where the makefile lives -- captures only the ``standalone_interface`` recipes.
Each boots a single interface ``create_app()`` on a throwaway port, so it needs
neither a container runtime nor seeded data. Regenerate one recipe with
``make screenshots-<name>``. The equivalent without the makefile is
``python -m docs.screenshots`` from the repository root, which is what the
target runs for you.

**Two opt-in environments** cover the images that need real data:

- ``SCREENSHOTOPTS=--stack`` -- the ARIEL search/browse/create/status views.
  Builds the ``control-assistant`` tutorial project, brings up Postgres
  (``osprey up -d``), and seeds the logbook with
  ``osprey sim apply nominal --yes --now <anchor>``. Needs a container runtime
  and a free host port 5432. The ``--now`` anchor freezes the seeded dates, so
  repeat captures are byte-stable.
- ``SCREENSHOTOPTS=--agentic`` -- the Web Terminal hero. Drives a live agent
  session to produce a real beam-current plot, so it needs a live Claude
  session on your subscription budget. Success is a structural check (non-blank
  image, correct viewport, plot present), not a byte comparison.

.. code-block:: console

   $ cd docs
   $ make screenshots                          # default: standalone only
   $ make screenshots SCREENSHOTOPTS=--stack    # + ARIEL views (containers)
   $ cd ..
   $ python -m docs.screenshots --agentic --only web_terminal_hero

**Provenance is automatic.** Every capture stamps ``manifest.json`` with the
OSPREY version and UTC timestamp, and each figure's *"Captured with OSPREY
vX.Y.Z"* caption is generated from it -- never hand-edit the version in a
caption.

This framework is **capture-only**: it is never a CI gate (the stack needs
Postgres; the hero needs a live agent). It is distinct from the CI visual-drift
guard -- pixel diffs of each rendered interface against a committed baseline
live in the front-end **Visual** tests above (regenerated with
``--regen-baselines``), and continue to run in CI unchanged.

Reviewing a web-interface redesign (contact sheet)
--------------------------------------------------

When a web interface is being restyled, the **contact-sheet renderer** boots the
*real* interface in every theme/mode variant and folds the shots into one
self-contained page, so a whole redesign can be reviewed as a single artifact --
no live agent, provider, hardware, or network. It lives beside the screenshot
framework in ``docs/screenshots/`` but is a review tool, not a committed doc
image: nothing it produces is checked in or CI-gated.

.. code-block:: console

   $ uv run python -m docs.screenshots.contact_sheet --out /tmp/sheet

That captures the Web Terminal's four shells -- the **dark** and **light** themes
crossed with the **expert** and **simple** UI modes -- writes one PNG per cell
into the output directory, composes them into ``contact-sheet.html`` there, and
prints its path. Open that one file to review every variant side by side.

**Comparing accent candidates.** Add ``--accents`` to render each of the four
variants twice, once under each accent candidate (blue vs teal), so a pending
accent decision can be made from real output rather than a mockup:

.. code-block:: console

   $ uv run python -m docs.screenshots.contact_sheet --out /tmp/sheet --accents

To keep every cell looking like a working session with no live backend, the
renderer points the workspace panel at a pre-seeded demo store and replays a
canned terminal transcript. That transcript is width-guarded against the narrow
terminal card -- the run fails fast if a line would overflow, before any browser
launches. Where no browser runtime is available the run skips with a one-line
notice instead of erroring.

**Extending it to another target.** The variant grid is the ``VARIANTS`` list of
``(theme, mode)`` tuples near the top of ``contact_sheet.py``, and the
completeness invariant ``_FULL_MATRIX`` mirrors it -- add a cell to *both* to
capture a new theme/mode combination. To cover a new panel, seed its backing
store the way ``seed_demo_workspace`` seeds the workspace artifacts and wire it
into ``hermetic_hub`` so the panel renders populated; the capture loop and the
composed sheet then pick it up unchanged.

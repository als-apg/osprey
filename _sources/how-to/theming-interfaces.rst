Theming the OSPREY Interfaces
==============================

Every browser interface OSPREY ships — the Web Terminal hub, the Artifacts
gallery, ARIEL, Channel Finder, Tuning, the Lattice dashboard, the event
dispatch dashboard, and the session activity/safety pages — draws its
colors, fonts, and a handful of layout constants from one generated design
system instead of hand-maintained, per-interface CSS. This guide covers the
three-tier token architecture, how to add a new theme ("skin"), and how to
wire a new interface into the system.

.. mermaid::

   flowchart LR
       core["tokens/core.json\n(primitives)"] --> themes["tokens/themes/*.json\n(semantic, per-theme)"]
       core --> ext["tokens/interfaces/*.json\n(extension, per-interface)"]
       themes --> build[generator/build.py]
       ext --> build
       build --> css["design_system/static/css/tokens.css"]
       build --> js["design_system/static/js/tokens.js"]
       build --> boot["design_system/static/js/theme-boot.js"]

The Three Token Tiers
----------------------

All token sources are hand-authored JSON in `DTCG format
<https://tr.designtokens.org/format/>`_ (Design Tokens Community Group) under
``src/osprey/interfaces/design_system/tokens/``. Nothing under ``tokens/`` is
hand-maintained CSS or JS — it is data, and the generator is the only thing
that reads it.

1. **Primitives** — ``tokens/core.json``. Color ramps (``color.slate.*``,
   ``color.teal.*``, ``color.amber.*``, ...), theme-independent. A primitive
   is never emitted as a CSS custom property directly; it only exists to be
   aliased. Each ramp step is a literal hex value that already existed
   somewhere in the fleet before migration — steps are not a synthetic
   mathematical scale, so gaps between step numbers are expected.

2. **Semantic tokens** — ``tokens/themes/dark.json`` and
   ``tokens/themes/light.json``. One JSON document per theme, each defining
   the same set of semantic names (``bg.primary``, ``text.secondary``,
   ``accent.base``, ``status.error``, ``border.default``, ``tint.accent.08``,
   ...) via one-hop aliases into ``core.json`` (``{color.teal.100}``). These
   are the names every interface's CSS should reach for first.

3. **Extension tokens** — ``tokens/interfaces/<name>.json``, one file per
   interface that needs tokens beyond the shared semantic set (a chart color
   bridge, a CRT scanline effect, a fixed layout dimension). Every extension
   token is namespaced with a short interface prefix — ``wt-`` (web
   terminal), ``art-`` (artifacts), ``ariel-``, ``cf-`` (channel finder),
   ``lat-`` (lattice) — enforced by
   ``generator/validate.py::check_namespace_collisions``. A bare, unprefixed
   extension name is a validation error: the whole point of the prefix
   convention is that two interfaces can each add a token called, say,
   ``header-height``, without colliding in the single flat CSS custom
   property namespace every token ultimately lives in.

The generator (``python -m osprey.interfaces.design_system.generator.build``)
loads all three tiers, resolves aliases, validates the whole tree, and
renders three generated artifacts under
``design_system/static/``:

``css/tokens.css``
    Every semantic and extension token as CSS custom properties, under
    ``:root, [data-theme="dark"]`` and ``[data-theme="light"]`` (and any
    further themes you add) blocks.

``js/tokens.js``
    The ``THEMES`` manifest (``[{id, label, mode}, ...]``) and ``DEFAULTS``
    (``{dark: <id>, light: <id>}``) that ``theme-manager.js`` uses to
    validate theme ids and resolve ``auto`` mode. Carries no color data —
    colors are read from computed CSS at runtime (see
    :ref:`theming-consuming-tokens`), not duplicated into JS.

``js/theme-boot.js``
    A tiny, dependency-free, non-module script that applies ``data-theme``
    before first paint (from ``localStorage``/``?theme=``/OS preference),
    so there is no flash of the wrong theme while the rest of the page
    loads.

All three are checked-in generated artifacts, not built at deploy time —
regenerate and commit them whenever you touch ``tokens/``.

.. note::

   ``core.json``, ``tokens.css``, and ``tokens.js`` are excluded from the
   fleet-wide hygiene scanner (``tests/interfaces/design_system/test_hygiene.py``):
   they *are* the color/token definitions, not consumers of them. Every
   other CSS/JS/HTML asset under ``src/osprey/interfaces/`` is scanned and
   must reference tokens via ``var(--name)`` rather than hardcoding colors.


Adding a Skin
--------------

"Adding a skin" means adding a new theme id beyond the two that ship today
(``dark``, ``light``) — for example a high-contrast variant. The generator
and runtime are both already N-theme-capable; there is nothing beyond the
token source tree to change.

Step 1: Copy a Theme Document
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cp src/osprey/interfaces/design_system/tokens/themes/light.json \
      src/osprey/interfaces/design_system/tokens/themes/high-contrast.json

Every interface extension file under ``tokens/interfaces/`` that defines
per-mode groups (``dark``/``light``) needs a value for your new theme's
*mode* too — see :ref:`theming-interface-mode` below.

Step 2: Edit the New Theme
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Update ``$extensions`` (the id the runtime and CSS selector will use) and
every semantic value:

.. code-block:: json

   {
     "$extensions": { "mode": "dark", "id": "high-contrast", "label": "High Contrast" },
     "bg": {
       "terminal": { "$value": "{color.slate.1100}", "$type": "color" },
       "primary": { "$value": "#000000", "$type": "color" }
     },
     "text": {
       "primary": { "$value": "#ffffff", "$type": "color" }
     }
   }

``mode`` is ``"dark"`` or ``"light"`` — it decides which of the two
``DEFAULTS`` slots (and which ``prefers-color-scheme`` bucket) this theme
can serve as the resolved value for ``auto``. It does **not** need to be
unique: nothing stops two themes from sharing a mode, but exactly the
themes tagged with a given mode are candidates for that mode's default and
for ``auto`` resolution.

Prefer aliasing into ``core.json`` primitives (``{color.slate.1100}``) over
a fresh literal wherever an existing ramp step is close enough — every
literal color you introduce is one more thing
``check_theme_completeness``/``check_wcag_gates`` have to validate and one
more color nobody else's theme can reuse. If you do need a genuinely new
primitive, add the ramp step to ``core.json`` first with a
``$description`` explaining where it comes from, following the existing
entries' style.

Step 3: Build
^^^^^^^^^^^^^^

.. code-block:: bash

   python -m osprey.interfaces.design_system.generator.build

This loads the whole ``tokens/`` tree, validates it, and rewrites
``tokens.css``/``tokens.js``/``theme-boot.js``. A validation failure prints
every error with its file and dot-path — fix all of them before
re-running; there is no partial-success mode.

The most common first-run failures when adding a theme:

* **Theme completeness** — every semantic token defined in ``dark.json``
  must also be defined in your new theme (and vice versa). A key present in
  one theme and missing in another is rejected outright.
* **Interface-mode completeness** — see :ref:`theming-interface-mode`.
* **WCAG contrast gates** — ``text.primary``/``text.secondary`` vs.
  ``bg.primary`` must clear 4.5:1, ``text.muted`` 3:1, ``accent`` vs.
  ``bg.primary`` 3:1. These are computed, not eyeballed; a theme that reads
  fine to you can still fail the gate.

Step 4: Run the Contract Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv run pytest tests/interfaces/design_system/ -v

``test_contract.py`` re-runs the full generator validation sweep against
the real, committed ``tokens/`` tree (not a fixture) — it is the
authoritative gate for shipped token data, distinct from
``test_model.py``/``test_validate.py``/``test_emit_*.py``, which exercise
the generator code itself against synthetic trees. It also checks
``tokens.js``'s ``THEMES``/``DEFAULTS`` against the theme sources, orphaned
color primitives, and literal colors that duplicate a ramp step where a
one-hop alias would do.

``test_freshness.py`` is the drift gate: it regenerates the artifacts into
a temp directory and diffs them byte-for-byte against what is committed
under ``static/``. If you built (Step 3) and forgot to ``git add`` the
regenerated files, this is what catches it in CI.

Step 5: Regenerate Baselines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A new theme id needs its own visual regression baselines (screenshots) —
``test_visual.py`` captures each interface's main view per theme at a fixed
viewport and diffs future runs against committed PNGs under
``tests/interfaces/design_system/baselines/``. Regenerate with the pytest
option the suite documents (``--regen-baselines``), review the new PNGs,
and commit them alongside your theme addition. Pixel-diff baselines are
Linux-rendered (CI) — macOS runs skip the byte-compare (AA/subpixel
rendering differs) and only verify the screenshots capture without
erroring.


.. _theming-interface-mode:

A Note on Interface-Mode Completeness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extension token files are keyed by **mode** (``dark``/``light``), not by
theme id, because an extension token is usually a visual effect ("CRT
scanline opacity") that only needs two states, not N. When you add a theme
whose ``mode`` already has an extension group (e.g. your new theme is
``"mode": "light"`` and ``tokens/interfaces/web_terminal.json`` already has
a ``light`` group), nothing changes — the new theme reuses that mode's
extension values automatically. You only need to touch extension files if
you introduce a genuinely new mode, which is not the common case.


.. _theming-consuming-tokens:

Consuming Tokens in a New Interface
-------------------------------------

Every interface's ``<head>`` follows the same three-line opener, in order:

.. code-block:: html

   <script src="/design-system/js/theme-boot.js"></script>
   <link rel="stylesheet" href="/design-system/css/tokens.css">
   <link rel="stylesheet" href="/design-system/css/base.css">

``theme-boot.js`` must load first and must **not** be a module script
(module scripts are deferred, which would let a pre-theme flash slip
through) — it sets ``data-theme`` synchronously before anything paints.
``tokens.css`` makes every custom property available. ``base.css`` is
optional (see below).

Mount the design system's static directory once per FastAPI app:

.. code-block:: python

   DESIGN_SYSTEM_STATIC_DIR = Path(__file__).parent.parent / "design_system" / "static"
   app.mount("/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system")

Then initialize the runtime — a small module script, placed after the head
links:

.. code-block:: html

   <script type="module">
     import { initTheme } from '/design-system/js/theme-manager.js';
     initTheme({ role: 'follower' });
   </script>

``role`` is almost always ``'follower'``: it applies whatever
``theme-boot.js`` already resolved pre-paint, applies a validated
``?theme=`` query parameter if present, and listens for
``'osprey-theme-change'`` broadcasts from the hub when embedded as an
iframe panel — but never persists a preference to ``localStorage`` and
never broadcasts itself. Only the Web Terminal hub uses
``role: 'hub'``: it is the one surface with a theme toggle button, the one
that persists the user's choice, and the one that broadcasts to every
embedded panel. There should only ever be one hub on a page.

app-shell vs. document pages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``base.css`` sets ``html, body { height: 100%; overflow: hidden; }`` — the
right default for a fixed-viewport, single-screen app shell (the Web
Terminal hub, the Artifacts gallery, Tuning, the Lattice dashboard). If
your interface is instead a normal scrolling *document* — a log viewer, a
guidelines page, a rendered markdown artifact — that default clips
everything below the fold. You have two options:

* **Link base.css and override the overflow back to auto**, after the
  ``base.css`` ``<link>`` so cascade order lets your rule win at equal
  specificity:

  .. code-block:: css

     /* base.css sets html,body { overflow: hidden } for the hub's fixed-
        viewport terminal shell. This page is a normal scrolling document,
        so it must override that back to auto or content below the fold
        is unreachable. */
     html, body {
       overflow: auto;
     }

  This is the right choice when you still want ``base.css``'s font,
  background, and reset rules — Channel Finder, ARIEL, and the session
  activity/safety pages all use this pattern.

* **Omit base.css entirely** if you don't need its app-shell layout rules
  at all — the artifacts markdown-rendering page does this: it links only
  ``tokens.css`` (colors) and writes its own minimal reset, since
  ``tokens.css`` itself carries no layout rules and can't clip anything.

Either way, decide this up front — a page that scrolls fine in local
testing without ``base.css`` in the picture can silently clip the moment
someone adds the link later.

Jinja-rendered vs. plain-static pages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most of the head wiring above is identical whether your page is served as
a plain static file (``FileResponse`` / ``StaticFiles``) or rendered
server-side (``Jinja2Templates.TemplateResponse``, or a hand-built HTML
string). The one place it diverges is **highlight.js's theme stylesheet**,
which must swap between a dark and a light vendor asset at runtime and
therefore needs a ``data-href-dark``/``data-href-light`` pair resolved
server-side, since the underlying URL depends on vendor mode
(CDN vs. self-hosted, chosen from config, not something a static file can
express):

.. code-block:: html

   <link id="hljs-theme" rel="stylesheet"
         href="{{ vendor_url('highlight.js atom-one-dark theme', '/static/vendor/atom-one-dark.min.css') }}"
         data-href-dark="{{ vendor_url('highlight.js atom-one-dark theme', '/static/vendor/atom-one-dark.min.css') }}"
         data-href-light="{{ vendor_url('highlight.js atom-one-light theme', '/static/vendor/atom-one-light.min.css') }}">
   <link rel="prefetch" href="{{ vendor_url('highlight.js atom-one-light theme', '/static/vendor/atom-one-light.min.css') }}">

``theme-manager.js`` reads ``data-href-dark``/``data-href-light`` off the
``#hljs-theme`` element and swaps ``href`` on every theme apply — it never
guesses a vendor path itself. If your interface doesn't render code blocks
with highlight.js, you don't need any of this. If it does and your page is
plain-static (no server-side templating available at all), you cannot use
``vendor_url()``; either hardcode a single vendor mode for that page or add
templating.

.. warning::

   A page that hardcodes a local vendor path directly (skipping
   ``vendor_url()``) will 404 in the default CDN vendor mode, where assets
   are served from a CDN rather than ``/static/vendor/``. This bit
   web-terminal and the artifacts pages before both were fixed to resolve
   the hljs stylesheet through ``vendor_url()`` — always route vendor
   asset paths through it rather than hardcoding either the CDN or the
   local form.


The Generator CLI Reference
------------------------------

.. code-block:: bash

   python -m osprey.interfaces.design_system.generator.build [--check]

Without ``--check``: loads ``tokens/``, validates it (aborting with every
error printed if invalid), renders all three artifacts, and writes them
under ``static/``.

With ``--check``: does the same load-validate-render, but never writes.
Instead it diffs each rendered artifact against what is already on disk
and exits non-zero with a unified diff for anything that has drifted. This
is the freshness gate CI runs — if you edit ``tokens/`` and forget to
regenerate, ``--check`` fails the build with a diff showing exactly what's
stale:

.. code-block:: bash

   python -m osprey.interfaces.design_system.generator.build --check

Exit codes: ``0`` on success (or a clean ``--check``), ``1`` on a
build/validation failure, or (``--check`` only) on detected drift.

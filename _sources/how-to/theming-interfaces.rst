Theming the OSPREY Interfaces
==============================

Every browser interface OSPREY ships — the Web Terminal hub, the Artifacts
gallery, ARIEL, Channel Finder, Tuning, the Lattice dashboard, the event
dispatch dashboard, and the session activity/safety pages — draws its
colors, fonts, and a handful of layout constants from one generated design
system instead of hand-maintained, per-interface CSS. This guide covers the
three-tier token architecture and how to wire a new interface into it. For
the theme contract itself (the semantic key set, WCAG gates, and the
theme-family model) see :doc:`osprey-themes`; for step-by-step instructions
to add a new theme, see :doc:`author-a-theme`.

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
    The ``THEMES`` manifest (``[{id, label, mode, family}, ...]``) and the
    per-family ``DEFAULTS`` map (``{family: {dark: <id>, light: <id>}}``)
    that ``theme-manager.js`` uses to validate theme ids and resolve
    ``auto`` mode *within the active family*. Carries no color data —
    colors are read from computed CSS at runtime (see
    :ref:`theming-consuming-tokens`), not duplicated into JS. See
    :doc:`osprey-themes` for the family model.

``js/theme-boot.js``
    A dependency-free, non-module script that applies ``data-theme``
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


Adding a Theme
---------------

Themes are grouped into **families** — a family is a ``{light, dark}``
pair, e.g. the built-in ``osprey`` family or the WCAG-AAA ``high-contrast``
family — and every theme must satisfy a metadata and completeness contract
beyond the token pipeline described above. See :doc:`osprey-themes` for
that contract (the semantic key set, ``$extensions`` metadata, the family
model, and the WCAG gates) and :doc:`author-a-theme` for the concrete,
step-by-step guide to authoring one, including the interface-extension
groups every ``tokens/interfaces/*.json`` file must carry (or opt out of)
for a new theme id.


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

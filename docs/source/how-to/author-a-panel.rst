Author an OSPREY Panel
========================

A **panel** is a directory bundling one HTML entry point plus a
``manifest.json`` — a self-contained, themed mini-app the Web Terminal can
mount alongside the chat surface. A compliant panel is **theme-aware** (it
boots the shared theme before first paint and honors ``?theme=``) and
**token-only** (every color, surface, border, and font resolves through a
``var(--…)`` design token — never a raw hex literal), so it renders correctly
under every theme family in both light and dark for free.

This guide is the developer/contributor workflow for **authoring and
validating** a panel — a developer writes the panel in the OSPREY source (or a
facility repo) and checks it against the panel validator. Runtime registration
and serving of a panel into the Web Terminal hub is forthcoming in a later
phase; today's authoring standard ends at author-and-validate.

Read :doc:`osprey-themes` for the theme contract the panel styles against, and
:doc:`embedding-osprey-panels` for the hub↔panel transport model a panel lives
under once served.

Step 1: Install the Authoring Skill
--------------------------------------

The guided path is the ``creating-an-osprey-panel`` skill. Install it, and an
agent or developer following it produces a panel that passes the validator:

.. code-block:: bash

   osprey skills install creating-an-osprey-panel

The skill inlines the full head markup, the real token names, the manifest
shape, and the validator self-check. The rest of this page is the contract the
skill implements — read on if you are authoring a panel by hand or want to
understand what the skill enforces.

Step 2: Author the Entry HTML
--------------------------------

Copy the reference panel as your starting point — it is the canonical exemplar
every panel is copied from:

.. code-block:: bash

   cp -r src/osprey/interfaces/design_system/panels/reference my-panel

The entry HTML's ``<head>`` carries a **load-bearing order** the validator
checks. The pre-paint boot script must come **first**, then the token
stylesheet, then a module script that wires up theming:

.. code-block:: html

   <head>
   <meta charset="UTF-8">
   <!-- Pre-paint theme boot FIRST: a plain (non-module) script that resolves
        and applies data-theme before first paint (it reads ?theme=), so the
        panel never flashes the wrong theme. -->
   <script src="/design-system/js/theme-boot.js"></script>
   <link rel="stylesheet" href="/design-system/css/tokens.css">
   <meta name="viewport" content="width=device-width, initial-scale=1.0">
   <title>My Panel</title>
   <script type="module">
     import { initTheme } from '/design-system/js/theme-manager.js';
     import { applyEmbedded } from '/design-system/js/frame-params.js';
     initTheme({ role: 'follower' });   // never the theme hub — the terminal is
     applyEmbedded();                   // ?embedded=true → body.embedded
   </script>
   </head>

Two rules the order encodes: ``theme-boot.js`` must stay a plain (non-module)
script so it runs synchronously before first paint, and it must precede
``tokens.css``. ``initTheme({ role: 'follower' })`` marks this surface as a
follower — it applies its own ``?theme=`` when a host appends one but never acts
as the theme hub. ``applyEmbedded()`` reads ``?embedded=true`` and adds
``body.embedded``; put any standalone-only chrome (a page header that only makes
sense opened on its own) behind ``body.embedded <selector> { display: none; }``
so it sits flush inside a host frame.

Step 3: Style Through Tokens Only
------------------------------------

Every color, background, border, and font must be a ``var(--…)`` token from
``tokens.css`` — a raw hex color literal anywhere in the HTML, or in any sibling
``.css``/``.js`` file, fails the validator (``RAW_HEX_COLOR``). This is exactly
what lets the panel theme correctly under every family for free.

.. code-block:: css

   body {
     background: var(--bg-primary);
     color: var(--text-primary);
     font-family: var(--font-display);
   }
   .card {
     background: var(--bg-elevated);
     border: 1px solid var(--border-default);
     box-shadow: var(--shadow-panel);
   }

If you need a color that isn't already in the reference panel, look it up in
``src/osprey/interfaces/design_system/static/css/tokens.css`` — never invent a
hex value. See :doc:`osprey-themes` for the semantic token set.

Step 4: Write the Manifest
-----------------------------

``manifest.json`` declares the panel's identity and entry point. Required
fields — ``id`` (a lowercase kebab slug matching ``^[a-z0-9][a-z0-9-]*$``),
``label`` (a non-empty display name), and ``entry`` (the HTML entry filename,
which must actually exist on disk). ``version`` is optional (integer,
defaults to ``1``); unknown keys are tolerated and preserved for forward
compatibility.

.. code-block:: json

   {
     "id": "my-panel",
     "label": "My Panel",
     "entry": "index.html",
     "version": 1
   }

An ``id`` with uppercase letters, spaces, or underscores (``My_Panel``,
``beam status``) is rejected — kebab slug only.

Step 5: Validate
-------------------

There is no CLI wrapper. Validate the panel directory with this one-liner — it
prints nothing and exits 0 when the panel is valid, and raises
``PanelValidationError`` listing every failure when it is not:

.. code-block:: bash

   uv run python -c "from osprey.interfaces.design_system.panels.validator import assert_valid_panel; assert_valid_panel('my-panel')"

The checks are cheap, decidable static rules — fix each reported rule and re-run
until it is silent:

- ``MANIFEST_MISSING`` — no ``manifest.json`` in the directory.
- ``MANIFEST_INVALID`` — bad JSON, or a missing/empty/wrong-typed field, or an
  ``id`` that isn't a kebab slug.
- ``ENTRY_MISSING`` — the ``entry`` file doesn't exist on disk.
- ``MISSING_DESIGN_SYSTEM_LINK`` — the entry HTML doesn't link ``tokens.css``.
- ``MISSING_THEME_BOOT`` — the entry HTML doesn't load ``theme-boot.js``.
- ``RAW_HEX_COLOR`` — a raw ``#rgb``/``#rrggbb``/… literal appears where a
  ``var(--…)`` token belongs (reported with its file and line).

The validator scans raw text, so a URL fragment whose name is all hex digits and
exactly 3/4/6/8 long (``href="#abc"``, ``href="#deadbeef"``) is flagged as a
color — rename the fragment rather than loosening the token-only rule. The panel
is complete when the one-liner raises nothing.

Cross-Origin Re-Theme Reality
--------------------------------

A panel and its host are served from different origins. A cross-origin panel
re-themes on **reload** when the host appends an updated ``?theme=`` — that path
works, and the pre-paint boot in Step 2 makes it flash-free. But
``postMessage``-based **live** re-theming is **same-origin only**: a cross-origin
panel must not expect to receive live theme broadcasts without a reload. Rely on
the ``?theme=`` boot plus reload, not on live cross-origin ``postMessage``. The
same-origin messaging model (for the hub's built-in panels) is the subject of
:doc:`embedding-osprey-panels`.

.. seealso::

   :doc:`osprey-themes`
      The theme contract: the semantic token set, family model, and WCAG gates
      a panel styles against.

   :doc:`author-a-theme`
      Adding a new theme or family to the design-token system.

   :doc:`embedding-osprey-panels`
      The hub↔panel transport contract — query/hash/postMessage, the
      same-origin model, and the standalone-vs-embedded chrome rules.

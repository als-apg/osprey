Design Language
===============

This page gives the visual rules of the OSPREY browser interfaces and the
shared parts that put them into practice. The token contract and the
enforcement details are in ``src/osprey/interfaces/design_system/DESIGN.md``.
Read both before you change a token, a shared primitive, or the web terminal
shell.

.. contents::
   :local:
   :depth: 1

Principles
----------

- The operator comes first. The chrome is quiet, and the content of a pane
  gets the space.
- Each pane has one header row. A panel never draws a second title row in its
  body.
- State is never shown by hue only. Add a shape or a word: a bar, an outline, a
  check mark, a label. The high-contrast themes have no hue.
- Use the shared tokens and the shared primitives. Do not make a local copy
  with other values.
- Popovers and tooltips sit at body level with fixed position, so a scrolled
  container does not cut them.

Token layers
------------

The tokens have three layers. The generator in
``design_system/generator/`` reads all three and writes ``tokens.css``,
``tokens.js`` and ``theme-boot.js``.

#. **Core primitives** (``tokens/core.json``): the color ramps and the scales
   for type, weight, leading, space, radius, z-index and duration. Color
   steps are never emitted directly.
#. **Themes** (``tokens/themes/*.json``): eight themes, four families in a
   dark and a light mode. Each theme aliases a core color step, one hop away,
   into a semantic name such as ``bg.primary`` or ``surface.card``.
#. **Interface extensions** (``tokens/interfaces/*.json``): values that only
   one interface needs, for example the web terminal's ``--wt-*`` sizes.

Never edit the generated files. Edit the JSON, then run the build (see
`Change process`_).

Surfaces
--------

There are four surfaces, from the deepest to the highest:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Surface
     - Token
     - Use
   * - Canvas
     - ``--surface-canvas``
     - The page under the cards. The sidebar sits on the canvas, with no card.
   * - Card
     - ``--surface-card``
     - A pane. Cards have a ``--surface-hairline`` border and the
       ``--radius-xl`` radius, with ``--wt-gutter`` (8 px) between them.
   * - Raised
     - ``--bg-elevated``
     - The tooltip and a small badge on a card.
   * - Floating
     - ``--surface-card`` with ``--shadow-dropdown``
     - Popovers and flyouts.

``--surface-well`` is a recessed area in a card, for example the body of the
artifact viewer. It is between the canvas and the card in every theme.

The washes mark state on these surfaces: ``--surface-hover``,
``--surface-press`` (pressed or open), ``--surface-selected`` and
``--surface-focus`` (the focus halo). They are alpha values, so they work on
every surface.

Use the surface tokens, not a ``--bg-*`` token, for the shell. The ``bg.*``
steps do not keep the same depth order in all eight themes. The surface group
does.

Type and space
--------------

- Size text with ``--text-*``, weight with ``--weight-*`` and line height with
  ``--leading-*``.
- Use ``--space-*`` for padding, margin and gap.
- Use ``--radius-*`` for corners: ``--radius-lg`` for buttons and rows,
  ``--radius-xl`` for cards and popovers.
- Use ``--duration-*`` for motion, with ``--wt-ease-out`` for the sidebar and
  for disclosure. ``--wt-transition-*`` is legacy. Do not use it in new CSS.
- Buttons use the display font (``--font-display``), never the mono font.
- Panel titles and sidebar labels use title case, not capital letters only.

Color and state
---------------

- Use the semantic color tokens: ``--text-*``, ``--color-accent``,
  ``--color-success``, ``--color-warning``, ``--color-error``.
- A selected row gets ``--surface-selected`` and an accent bar at its left
  edge.
- A status dot (``.osprey-dot``) shows the state by fill and shape. ``.ok``
  adds a halo. ``.hollow`` is an outline with no fill, for a closed pane.
- The validator applies WCAG contrast gates to the text tokens on
  ``bg.primary``. The high-contrast family must pass AAA. Text on a card is
  not gated, so keep it on the same text tokens.

Layout shell
------------

The web terminal uses one shell in the Simple view and in the Expert view:

- A sidebar on the canvas at the left. It is ``--wt-sidebar-width`` (232 px)
  with labels, 180–400 px by the handle on its right edge, or
  ``--wt-sidebar-collapsed-width`` (56 px) with icons only. The state is
  ``data-sidebar`` on ``<html>``. ``sidebar-boot.js`` sets it, and the
  stored width, before the first paint.
- From top to bottom, the sidebar holds the OSPREY mark and the collapse
  toggle, the control-system block, search, the ``Workspace`` and ``Tools``
  sections, and a footer with Docs, Feedback and More. Only the
  ``Workspace`` section gives up height.
- The panes are cards in the dock. Each pane has one header row,
  ``--wt-pane-head-height`` (40 px) high.
- The agent pane can close. Then ``<html>`` gets ``data-agent="closed"``, and
  the sidebar ``Agent`` row opens the pane again.

Primitives
----------

The shared primitives are in ``design_system/static/css/primitives.css``.
Load it after ``base.css``. Their behaviour is in five modules in
``design_system/static/js/``:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Part
     - What it gives
   * - ``.osprey-nav-row``, ``.osprey-section-label``
     - The sidebar row (icon, label, badge, disclosure) and the group heading.
   * - ``.osprey-list-row``, ``.osprey-list-head``
     - The child row of a sidebar list, with a title and a time.
   * - ``.osprey-icon-btn``, ``.osprey-btn``, ``.osprey-mini``,
       ``.osprey-mini-seg``
     - Icon button, text button, row action and segmented control.
   * - ``.osprey-card``, ``.osprey-pane-head``, ``.osprey-crumb``
     - The inset card, its header row and the breadcrumb.
   * - ``.osprey-popover``, ``popover.js``
     - Menus and flyouts. ``openPopover()`` places a popover beside the
       sidebar or under its opener.
   * - ``.osprey-tooltip``, ``tooltip.js``
     - One tooltip layer for the whole page. It shows a label and a one-line
       explanation (``data-tip``, ``title``, ``data-tip-detail``) 450 ms after
       the pointer arrives. A sidebar row that shows its label gets only the
       explanation.
   * - ``collapsible.js``, ``sidebar-boot.js``
     - The collapse state, its storage, and the pre-paint stamp.
   * - ``icons.js``
     - ``icon(name)`` builds a 16 px Lucide stroke icon (``.osprey-ic``).
   * - ``.osprey-dot``, ``.osprey-kbd``, ``.osprey-count``, ``.osprey-switch``,
       ``.osprey-search``
     - Status dot, keyboard hint, count pill, switch and search field.

``DESIGN.md`` section 9 lists every class and the API of each module. The
reference page, ``/design-system/reference.html``, shows each primitive in all
eight themes.

Keyboard and focus
------------------

- Every control shows ``:focus-visible`` with a 2 px accent outline.
- A popover moves focus to its first item when it opens. Escape closes it and
  moves focus back to the control that opened it.
- ``⌘\`` on macOS, or ``Ctrl+\`` on other systems, collapses and expands the
  sidebar. ``⌘K`` or ``Ctrl+K`` opens search.
- Get the modifier for a hint from ``platform.js`` (``isMacPlatform()``). Do
  not write ``⌘`` for all systems.
- A collapsed sidebar row has an ``aria-label``. The tooltip repeats the
  label but is not the accessible name.
- Use ``focus-trap.js`` in a dialog.

External panels
---------------

A panel that runs in its own process links the shared files by root-absolute
paths. The terminal proxy serves its own copy:

.. code-block:: html

   <link rel="stylesheet" href="/design-system/css/tokens.css">
   <link rel="stylesheet" href="/design-system/css/base.css">
   <link rel="stylesheet" href="/design-system/css/primitives.css">

When the terminal embeds a panel, the panel hides its own top bar and sends
its controls to the pane header through ``header-contrib.js`` (header
contract version 1). :doc:`/how-to/web-terminal/panels` gives the panel
contract, and :doc:`/how-to/web-terminal/theming` gives the theme families.

Enforcement
-----------

- ``generator.build --check`` fails when a generated file differs from the
  token JSON. The validator in ``generator/validate.py`` checks one-hop
  aliases, the same token set in each theme, the interface groups for each
  theme, and the WCAG gates.
- ``tests/interfaces/design_system/test_hygiene.py`` fails on a color literal,
  on a ``var()`` that no token defines, and on a local ``:root`` copy of a
  token. In ``web_terminal``, ``design_system``, ``ariel`` and
  ``channel_finder`` it also fails on a raw font size, weight, line height,
  radius or duration.
- ``tests/interfaces/test_js_module_tests.py`` fails when a module in
  ``static/js/`` has no Vitest file with the same name.
- ``tests/interfaces/design_system/test_visual.py`` compares screenshots with
  committed baselines for the dark and light themes.

Change process
--------------

#. Edit the token JSON in ``design_system/tokens/``. Add a new token to all
   eight themes.
#. Run the build from the repository root::

      uv run python -m osprey.interfaces.design_system.generator.build

#. Run the build again with ``--check``. It must exit with code 0.
#. Add a specimen to ``design_system/static/reference.html`` for a new
   primitive.
#. Update ``DESIGN.md``.
#. Add a fragment to ``changelog.d/``.

Adding a token name is safe. Renaming or removing a token breaks external
panels that use it, so keep the old name until every panel has moved.

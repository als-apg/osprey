OSPREY Themes
==============

Every OSPREY browser interface draws its colors from the shared design-token
system (see :doc:`theming-interfaces` for the token pipeline itself). A
**theme** is the formal contract a token document must satisfy to plug into
that system, and a **family** groups two themes — a dark one and a light
one — under one user-facing choice. This page defines that contract. To
author a new theme yourself, see :doc:`author-a-theme`.

What a Theme Is
----------------

A theme is one hand-authored `DTCG <https://tr.designtokens.org/format/>`_
JSON document under ``tokens/themes/<id>.json``. It must define:

- The **same semantic token key set** as every other theme — ``bg.*``,
  ``text.*``, ``accent.*``, ``status.*``, ``border.*``, ``tint.*``,
  ``terminal.*`` (the xterm ANSI palette), ``chart.*``, ``code.*``, and a
  handful of others. A key present in one theme and missing from another is
  a hard build error (``check_theme_completeness``) — themes are
  interchangeable, not independent designs.
- A root ``$extensions`` block identifying it to the runtime registry (see
  below).

Nothing about a theme is hand-maintained CSS or JS: the generator
(``python -m osprey.interfaces.design_system.generator.build``) is the only
thing that reads ``tokens/themes/*.json``, and ``tokens.css``/``tokens.js``/
``theme-boot.js`` are its checked-in, regenerated-not-hand-edited output.

Theme Metadata (``$extensions``)
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Field
     - Values
     - Meaning
   * - ``mode``
     - ``"dark"`` or ``"light"``
     - Which half of its family this theme is. Exactly two modes exist;
       nothing else is valid.
   * - ``id``
     - non-empty string
     - The theme's slug — the ``data-theme`` attribute value, the
       ``?theme=`` query value, and the ``THEMES`` manifest entry's key.
   * - ``label``
     - non-empty string
     - Display name shown in the switcher's mode toggle tooltip.
   * - ``family``
     - non-empty string
     - The ``{light, dark}`` pair this theme belongs to (see below). Also
       selects which WCAG gate this theme is held to.

All four fields are required on every theme document; a missing or
malformed one fails the build (``check_theme_metadata``).

The Family Model
------------------

A **family** is a ``{light, dark}`` pair presented to the user as one named
choice. OSPREY ships two built-in families:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Family
     - Theme id
     - Mode
     - WCAG gate
   * - ``osprey``
     - ``dark``
     - dark
     - AA
   * - ``osprey``
     - ``light``
     - light
     - AA
   * - ``high-contrast``
     - ``high-contrast-dark``
     - dark
     - AAA
   * - ``high-contrast``
     - ``high-contrast-light``
     - light
     - AAA

The switcher (``<osprey-theme-switcher>``) lists **families**, not
individual theme ids. Selecting a family applies that family's dark theme
by default; the separate mode toggle flips light/dark **within the active
family** (``toggleTheme()`` resolves the sibling mode of the active
family — never a global default); and ``auto`` follows the OS
``prefers-color-scheme`` preference, also within the active family. Picking
``high-contrast`` and toggling light/dark never leaves ``high-contrast``.

WCAG Contrast Gates
---------------------

Every theme's contrast is computed, not eyeballed, against a gate selected
by its declared ``family`` (fail-closed: an unrecognized or missing family
is held to AA, never something looser):

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Pair
     - ``osprey`` (AA)
     - ``high-contrast`` (AAA)
     - Basis
   * - ``text.primary``/``text.secondary`` vs ``bg.primary``
     - ≥ 4.5 : 1
     - ≥ 7 : 1
     - Body text
   * - ``text.muted`` vs ``bg.primary``
     - ≥ 3 : 1
     - ≥ 4.5 : 1
     - Large-scale/secondary text
   * - ``accent.base`` vs ``bg.primary``
     - ≥ 3 : 1
     - ≥ 4.5 : 1
     - Non-text UI

A theme that reads fine to the eye can still fail a gate; nudge the failing
token value in the source rather than weakening the gate.

Interface Extension Groups
-----------------------------

An interface that needs tokens beyond the shared semantic set (e.g. a chart
color bridge or a CRT scanline effect) declares them under
``tokens/interfaces/<name>.json``, with one top-level group **per theme
id** (``dark``, ``light``, ``high-contrast-dark``, ``high-contrast-light``,
...) — not per mode. Adding a new theme id therefore normally requires a
new group in *every* interface file, unless that file opts the id out via a
root ``$extensions.inherits`` map, borrowing a sibling id's group instead
(the right choice for purely decorative tokens that don't need an
AAA-specific value). See :doc:`author-a-theme` for the concrete steps.

Configuring the Default Theme (``web.theme``)
-------------------------------------------------

The top-level ``web:`` section of ``config.yml`` has a ``theme`` key:

.. code-block:: yaml

   web:
     theme: osprey   # a family (soft default) or a concrete theme id

``theme`` names either a **family** (e.g. ``osprey``, ``high-contrast`` —
resolved to that family's dark theme as the server-rendered default) or a
**concrete theme id** (e.g. ``high-contrast-light``) to pin an exact mode.
The web server resolves it once at startup and server-renders
``<html data-theme>`` from it, so the page first-paints in the configured
theme with no flash of the wrong one. A user's in-browser choice — the
switcher, ``?theme=``, or a previously stored preference — always overrides
this default and persists across reloads. An unknown value is logged as a
warning and falls back to ``osprey``.

.. note::

   ``web.theme`` is **separate** from the CLI's ``cli.theme`` (the Rich
   console theme for terminal output). The two are independent config keys
   that never share or influence each other's value.

.. seealso::

   :doc:`author-a-theme`
      Step-by-step guide to authoring a new theme.

   :doc:`theming-interfaces`
      The token pipeline (primitives → semantic → extension tiers) and how
      to wire a new interface into it.

   :doc:`use-web-terminal`
      Launching and configuring the Web Terminal.

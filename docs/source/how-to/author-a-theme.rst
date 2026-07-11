Author Your Own OSPREY Theme
==============================

This guide walks through adding a new theme that passes the design-token
generator's validation gates — either a new mode for an existing family or
an entirely new family. Read :doc:`osprey-themes` first for the contract
this theme must satisfy (the semantic key set, ``$extensions`` metadata,
and the WCAG gates).

Step 1: Decide What You're Adding
------------------------------------

- **Filling out an existing family** (it's missing a mode) — the common
  case. Your new theme inherits that family's WCAG gate (AA for ``osprey``,
  AAA for ``high-contrast``).
- **A brand-new family** — you'll author both its dark and light themes.
  A family not already listed in
  ``generator/validate.py``'s ``_WCAG_GATES_BY_FAMILY`` defaults to the AA
  gate (fail-closed); add an entry there if your family should be held to
  AAA instead.

Step 2: Author the Token Document
-------------------------------------

Copy a sibling theme as your starting point — a same-mode theme in your
target family if one exists, otherwise any existing theme:

.. code-block:: bash

   cp src/osprey/interfaces/design_system/tokens/themes/light.json \
      src/osprey/interfaces/design_system/tokens/themes/high-contrast-light.json

Update ``$extensions`` and every semantic value:

.. code-block:: json

   {
     "$extensions": {
       "mode": "light",
       "id": "high-contrast-light",
       "label": "High Contrast Light",
       "family": "high-contrast"
     },
     "bg": { "primary": { "$value": "#ffffff", "$type": "color" } },
     "text": { "primary": { "$value": "#000000", "$type": "color" } }
   }

Keep the identical set of semantic keys the theme you copied defines —
``check_theme_completeness`` rejects a theme that's missing (or adds) a
key relative to its siblings. Prefer aliasing into ``core.json`` primitives
(``{color.slate.1100}``) over a fresh literal wherever an existing ramp
step is close enough; add a new primitive to ``core.json`` first (with a
``$description``) only if you genuinely need one.

Step 3: Meet Your Family's WCAG Gate
----------------------------------------

``osprey``-family themes are held to AA (4.5:1 body text, 3:1 muted/large
text and non-text UI); ``high-contrast``-family themes to AAA (7:1 body
text, 4.5:1 muted/non-text UI) — see :doc:`osprey-themes` for the exact
pairs. These are computed by the generator, not eyeballed: a build failure
names the failing pair and its ratio. Nudge the failing token's value; never
weaken the gate to make a theme pass.

Step 4: Satisfy Interface-Extension Completeness
----------------------------------------------------

Every ``tokens/interfaces/<name>.json`` file needs a top-level group keyed
by your new theme's ``id`` (not its mode) — the same extension tokens
(chart bridges, decorative effects, ...) that file already defines for
other theme ids. For each interface file, either:

- **Author the group**, if the token values genuinely differ for your
  theme (e.g. a chart color that must itself meet the WCAG gate) — see
  ``tokens/interfaces/ariel.json`` for an interface that authors explicit
  ``high-contrast-dark``/``high-contrast-light`` groups; or
- **Opt out**, if your theme is a purely decorative variant that should
  reuse an existing group's values, via a root ``$extensions.inherits``
  entry:

  .. code-block:: json

     { "$extensions": { "inherits": { "high-contrast-light": "light" } } }

  See ``tokens/interfaces/web_terminal.json`` (and ``artifacts.json``,
  ``channel_finder.json``, ``lattice_dashboard.json``) for this pattern —
  each opts both new high-contrast ids out, inheriting the plain
  dark/light groups.

A theme id that is neither authored nor opted out in a given interface
file is a hard build error; opting out one file never excuses another.

Step 5: Regenerate and Verify
---------------------------------

.. code-block:: bash

   python -m osprey.interfaces.design_system.generator.build

This validates the whole ``tokens/`` tree and rewrites
``tokens.css``/``tokens.js``/``theme-boot.js``. Fix every reported error
(file + dot-path) before re-running — there is no partial-success mode.
Then confirm the freshness gate (what CI runs) is clean:

.. code-block:: bash

   python -m osprey.interfaces.design_system.generator.build --check

Finally, run the design-system test suite and regenerate visual baselines
for your new theme id (``test_visual.py --regen-baselines``, reviewed and
committed alongside your change):

.. code-block:: bash

   uv run pytest tests/interfaces/design_system/ -v

.. seealso::

   :doc:`osprey-themes`
      The theme contract: metadata, family model, WCAG gates.

   :doc:`theming-interfaces`
      The token pipeline and wiring a new interface into it.

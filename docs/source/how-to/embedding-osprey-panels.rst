Embedding OSPREY Panels
=========================

The Web Terminal hub embeds every other interface — Channel Finder, OKF,
ARIEL, Tuning, the Lattice dashboard, and the Artifacts gallery — as
same-origin iframe panels, and hosts its own session activity page
(``session.html``) the same way. This guide documents the shell↔panel
boundary as a versioned contract: three transports, a well-known set of
query params, per-panel deep-link forms, the same-origin postMessage model
that enforces the boundary, and the chrome contract that decides what a
panel shows of itself when it's running standalone versus inside the hub.
If you are wiring a new panel into the hub, or adding a new well-known param
or message type, this is the contract to extend rather than work around.

The Three Transports
----------------------

The contract splits along a simple line: **who decides the value, and when**.

query = creation-time config
    Read once, at panel boot, from the iframe's ``src`` URL. The hub decides
    these values when it *builds* the iframe URL — a panel embedded directly
    (not inside the hub) simply omits them and gets the default behavior.

hash = panel-owned deep-link
    Each panel owns its own hash grammar and reads it from its own
    ``location.hash`` — the hub does not construct or interpret it. See
    `Per-Panel Deep-Link Forms`_ below.

postMessage = live push
    Used for state changes that happen *after* a panel has already loaded —
    a theme toggle, a session switch — which query and hash, both read only
    at boot, cannot express. See `The Same-Origin Model`_.

Well-Known Query Params
--------------------------

Two query params are part of the contract today:

``embedded``
    ``"true"`` marks the page as running inside a host frame. Read by
    ``applyEmbedded()``, exported from the shared
    ``design_system/static/js/frame-params.js`` module (served at runtime as
    ``/design-system/js/frame-params.js``). It adds the ``embedded`` class to
    ``document.body`` and is a no-op for any other value, including absence.
    All six panels that can run under the hub — Channel Finder, OKF, ARIEL,
    Tuning, the Lattice dashboard, and Artifacts — plus the hub's own
    ``session.html``, import ``applyEmbedded()`` from this single shared
    module; there is no per-panel copy of this reader left to drift. See
    `Chrome Contract: Branding-Only, Standalone vs. Embedded`_ below for what
    each page actually does with the resulting ``embedded`` class.

``theme``
    Owned and read pre-paint by ``theme-boot.js`` / ``theme-manager.js``,
    **not** by ``frame-params.js``. ``theme-boot.js`` is a non-module inline
    script that resolves and applies ``data-theme`` before first paint;
    re-reading ``theme`` in the deferred ``frame-params.js`` ES module would
    just duplicate that read after the fact and risk a visible theme flash.
    See :doc:`theming-interfaces` for the full theme contract, including the
    ``follower``/``hub`` roles and the ``osprey-theme-change`` broadcast.

``CONTRACT_VERSION`` and the Bump Policy
-------------------------------------------

``frame-params.js`` exports a ``CONTRACT_VERSION`` string constant (currently
``'1'``). It exists purely as a documentation and coordination anchor —
**nothing on the wire carries it**: there is no ``?v=`` query param, no
per-message version field, and no code path branches on its value.

Bump ``CONTRACT_VERSION`` when either of these happens:

* a well-known param name or its semantics change (for example, ``embedded``
  started meaning something different, or a new well-known param joined the
  set with a name collision risk), or
* a postMessage message type or its payload shape changes.

Treat the bump as a signal for anyone reading the source, not a runtime
guard — there is no consumer today that inspects it programmatically. The
D15 chrome-contract work below (the ``<osprey-theme-switcher>`` component,
its embedded-hide rule, and the ``history.replaceState`` strip in
``setTheme()``) did **not** bump ``CONTRACT_VERSION``: it changed *how* a
page presents itself once ``embedded``/``theme`` are already applied, not
the wire semantics of either param.

Per-Panel Deep-Link Forms
-----------------------------

The hash transport is deliberately **panel-owned**: each panel defines its
own grammar, and the hub does not know or care what it means.

* **OKF** reads its own ``location.hash`` as ``#<conceptId>`` and routes to
  that concept on load. This is the one panel that currently uses the hash
  transport.
* **Web Terminal** (the hub) sets **no hash** on any iframe URL it builds. An
  earlier ``#/sessions?project=`` grammar existed on this codepath and has
  since been removed as vestigial — it had no reader.

Per-panel hash grammars diverge by design: they are documented here rather
than converged into one shared shape, because a hash is inherently
panel-internal navigation state. The seam a
future KNOWLEDGE concept-picker would use to deep-link the hub straight into
an OKF concept is exactly this transport — the hub would set the iframe's
initial ``#<conceptId>`` hash the same way a user's own OKF navigation does —
but no hub-side hash producer exists today.

The Same-Origin Model
------------------------

Every hub→panel and panel→hub ``postMessage`` call assumes the sender and
receiver share an origin — this is not a general cross-origin embedding
protocol, and the panels' iframe ``sandbox`` attribute includes
``allow-same-origin`` on that assumption.

**Senders** post with the target origin pinned to ``window.location.origin``,
never ``'*'``:

* ``theme-manager.js``'s ``_broadcast`` (theme changes to every embedded panel)
* Web Terminal's ``terminal.js`` ``notifySessionChange``
* Web Terminal's ``panel-manager.js``, two per-iframe resend call sites
  (theme and session-change repair on tab activation)

**Receivers** guard symmetrically, rejecting anything that isn't
same-origin before touching ``event.data``:

* ``theme-manager.js``'s ``_handleMessage``
* Artifacts ``gallery.js``'s session-change listener
* Web Terminal's ``session.html`` session-change listener
* Web Terminal's ``app.js`` paste-to-terminal listener

All four follow the same one-line shape:

.. code-block:: js

   window.addEventListener('message', (e) => {
     if (e.origin !== window.location.origin) return;
     // ... handle e.data
   });

Two Documented Cross-Origin Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two send sites in Artifacts deliberately keep ``'*'`` as the target origin
instead of ``window.location.origin``, each with a one-line reason recorded
at the call site:

* **paste→parent** (``sendToTerminal``, ``types.js``): posts
  ``osprey-paste-to-terminal`` to ``window.parent`` with ``'*'``, because the
  parent embedder may legitimately be cross-origin — Artifacts can be
  embedded outside the OSPREY hub entirely.
* **theme→nested-preview-iframe** (``_forwardThemeToPreviewFrames``,
  ``gallery.js``): forwards ``osprey-theme-change`` to nested preview
  iframes (rendered Plotly HTML artifacts) with ``'*'``, because that nested
  content may be ``null`` or cross-origin and there is no same-origin
  guarantee to pin to.

Do not treat these two as a precedent for a general "allow cross-origin"
escape hatch — every other sender/receiver pair pins to
``window.location.origin``, and there is no allowed-origins configuration
knob in the contract. If a genuine cross-origin consumer shows up beyond
these two cases, that is a new design decision, not a default to fall back
on.

Chrome Contract: Branding-Only, Standalone vs. Embedded
----------------------------------------------------------

Every page above draws its theme toggle from one shared component,
``<osprey-theme-switcher>`` (``design_system/static/js/components/
osprey-theme-switcher.js``), and every page decides what to show of its own
branding based on the same ``embedded`` class ``applyEmbedded()`` already
applies. The rule is **branding-only**: embedding hides a page's own logo or
title — the one thing that duplicates what the hub's own chrome already
shows — and nothing else. It does not hide navigation, a pipeline switcher,
or any other functional UI, because that stays useful even inside the hub.

``<osprey-theme-switcher>``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single light-DOM custom element renders the toggle button
``theme-manager.js`` already binds by id (``#theme-toggle``,
``#theme-icon-sun``, ``#theme-icon-moon``) — mounting it is one line,
``<osprey-theme-switcher></osprey-theme-switcher>``, plus a side-effect
import of the component module. It carries no theme-switching logic of its
own; see :doc:`theming-interfaces` for what ``initTheme()`` and the
``follower``/``hub`` roles actually do.

The component hides itself fleet-wide under ``embedded`` with one rule it
injects itself — ``body.embedded osprey-theme-switcher { display: none; }``
— so no page needs its own copy of that rule. What each page *is*
responsible for is hiding its own branding element, if it has one, with
exactly that same pattern:

.. code-block:: css

   body.embedded <your-branding-selector> { display: none; }

.. list-table:: Per-page branding selector
   :header-rows: 1
   :widths: 25 30 45

   * - Page
     - Branding selector
     - Notes
   * - Channel Finder
     - ``.app-logo``
     - Narrowed from an earlier whole-header rule (D15) — the header stays
       fixed and visible embedded so the pipeline switcher and nav remain
       usable; only the logo hides. See `A Non-Occlusion Anti-Regression
       Example`_.
   * - Artifacts
     - ``.logo``
     -
   * - ARIEL
     - ``.logo``
     -
   * - Tuning
     - ``.tuning-header``
     -
   * - Lattice Dashboard
     - ``.topbar-logo``
     -
   * - OKF
     - *(none)*
     - No pre-existing branding chrome to hide — only the switcher hides.
   * - session.html
     - ``header h1``
     - The refresh-status dot next to it stays visible embedded; it's a
       live status indicator, not branding.

A Non-Occlusion Anti-Regression Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Channel Finder's header is ``position: fixed`` and ``.app-main`` carries a
matching ``padding-top: 48px`` so page content clears it. When the
branding-only rule replaced an earlier whole-header hide, that padding had
to stay exactly as-is in both modes — a future change that reintroduced
whole-header hiding without also zeroing this padding would leave a silent
48px gap; one that kept the header but forgot the padding would occlude the
first 48px of content. ``test_channel_finder_embedded_non_occlusion`` in
``tests/interfaces/web_terminal/test_contract_params.py`` pins both
invariants together: the pipeline switcher's bounding box stays inside the
viewport, and ``.app-main``'s computed ``padding-top`` stays ``48px``
embedded.

D15: Session-Only Toggles, No Stale ``?theme=``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A standalone panel's theme toggle is deliberately **session-only**: clicking
it applies the new theme immediately, but a *follower* page never persists
that choice to ``localStorage`` and never broadcasts it (see
:doc:`theming-interfaces`) — a reload falls back to whatever
``?theme=``/``localStorage``/OS preference resolution ``theme-boot.js``
would have used anyway.

That only works cleanly if a leftover ``?theme=`` query param — say, from a
panel URL the hub built with an explicit theme, or from a link someone
bookmarked — can't out-rank that fallback on the next reload. So
``setTheme()`` (``theme-manager.js``) strips ``theme`` from the URL via
``history.replaceState`` on every explicit toggle, for **both** roles — this
also fixes a hub-side quirk where a stale ``?theme=`` in the address bar
could out-rank the user's already-persisted preference. The strip happens
only on the explicit-toggle path (clicking the switcher); a theme applied by
``initTheme()`` on load, or by a hub broadcast arriving via ``postMessage``,
never touches the URL.

New-Panel Recipe
^^^^^^^^^^^^^^^^^^

Wiring a new panel (or any standalone page that might one day run embedded)
into this contract:

#. Load ``theme-boot.js`` first in ``<head>`` (non-module — see
   :doc:`theming-interfaces`), then link ``tokens.css`` (and ``base.css`` if
   it's an app-shell page).
#. Add a side-effect import of
   ``/design-system/js/components/osprey-theme-switcher.js`` and mount
   ``<osprey-theme-switcher></osprey-theme-switcher>`` in the header. If the
   page already has hand-written ``#theme-toggle``/``#theme-icon-sun``/
   ``#theme-icon-moon`` markup, remove it first — a duplicate id would make
   ``document.getElementById('theme-toggle')`` resolve to whichever copy is
   first in document order.
#. Import ``initTheme`` from ``theme-manager.js`` and call
   ``initTheme({ role: 'follower' })`` — after the switcher's side-effect
   import, so its click handler wires up against a button that already
   exists.
#. Import ``applyEmbedded`` from ``frame-params.js`` and call it.
#. If the page has its own branding element, add exactly one rule:
   ``body.embedded <selector> { display: none; }`` — narrow to the branding
   itself, not the whole header, so functional chrome (nav, a pipeline
   switcher, a status indicator) stays usable embedded. Do **not** add a
   rule to hide the switcher yourself; it already hides itself fleet-wide.
#. Add the new page to ``_CHROME_CONTRACT_PANELS`` in
   ``tests/interfaces/web_terminal/test_contract_params.py`` — one tuple
   (launcher, path, branding selector or ``None``) exercises all three
   chrome-contract test functions (embedded-hide, standalone-toggle, D15
   reload-strip) for it automatically.

The Executable Contract Spec
--------------------------------

The whole contract above is exercised end-to-end, in a real browser, by
``tests/interfaces/web_terminal/test_contract_params.py``:

* query = creation-time config (``?embedded=true`` / ``?theme=``)
* hash = panel-owned deep-link (via the OKF standalone panel)
* postMessage = live push, including a synthetic foreign-origin rejection
  check against all four receivers listed above
* the chrome contract, for all six panels plus ``session.html``: branding
  and switcher hidden embedded, the switcher visible and functional
  standalone, the D15 reload-strip, and Channel Finder's non-occlusion pin

Treat that file as the up-to-date reference for exact assertions — it is the
contract's source of truth, not this prose. For the config-side view of
panels (enabling them, custom panel URLs, ``/api/panel-focus``), see
:doc:`use-web-terminal`.

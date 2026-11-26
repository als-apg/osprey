.. _reference-app-contract:

=========================
The Application Contract
=========================

An application declares in its ``app.yml``, with ``contract: N``, the version of the
application contract it is written for, and OSPREY refuses any manifest whose number
differs from the one it implements.

This OSPREY implements contract **1**, the value of
``osprey.apps.contract.APP_CONTRACT``.

What the contract covers
========================

One number covers four layers. The list is exhaustive: what this page does not list is
not promised.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Layer
     - What it holds
     - Where it is defined
   * - Manifest
     - The keys of ``app.yml``, with their types; the frontmatter keys of an app's
       ``agents/<name>.md``
     - ``AppManifest`` in ``src/osprey/apps/manifest.py``; ``AppAgent`` in
       ``src/osprey/apps/agents.py``
   * - Python
     - The names of ``osprey.apps.sdk.__all__``, ``osprey.apps.sdk.testing.__all__``
       and ``osprey_facility.kinds.__all__``; the public modules of
       ``osprey_connectors``
     - ``src/osprey/apps/sdk/``; ``packages/osprey-facility/src/osprey_facility/kinds.py``;
       ``packages/osprey-connectors``
   * - Panel
     - The panel rules; the exports of ``page-ready.js``; the ``osprey-*`` primitive
       classes; the design tokens; the ``type`` of each theme and session message a
       panel receives
     - ``osprey.interfaces.design_system.panels.validator``; ``page-ready.js``,
       ``primitives.css`` and ``tokens.css`` under
       ``src/osprey/interfaces/design_system/static/``; ``panel-iframe-sync.js`` under
       ``src/osprey/interfaces/web_terminal/static/js/``
   * - Process
     - The headers an app's server receives (``X-Osprey-Owner``,
       ``X-Osprey-Session``, ``X-Osprey-Caller``, ``X-Launch-Token``); the ``OSPREY_*``
       environment variables handed to its container and tool server
     - ``src/osprey/utils/owner_header.py``, ``src/osprey/utils/launch_token.py``;
       ``APP_ENVIRONMENT`` in ``src/osprey/apps/projection.py``

The ``OSPREY_*`` variables are ``OSPREY_APP_BIND``, ``OSPREY_APP_CONFIG``,
``OSPREY_APP_NAME``, ``OSPREY_APP_PORT``, ``OSPREY_APP_STATE_DIRS``,
``OSPREY_AUDIT_IDENTITY``, ``OSPREY_CONFIG``, ``OSPREY_CONTROL_CONTEXT_TREE``,
``OSPREY_GUARDED_RUN_DIR``, ``OSPREY_SESSION_ID`` (inherited by the tool server from the
agent's session), ``OSPREY_SPOOL_DIR``, ``OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS``,
``OSPREY_TERMINAL_ROSTER_OWNERS`` and ``OSPREY_TERMINAL_SECRET_<user>``, one per roster
user of a panel grant. Names an app declares under ``env:`` are its own.

Not promised:

- environment variables without the ``OSPREY_`` prefix (``CONFIG_FILE``,
  ``EPICS_CA_*``, ``TZ``, ``<APP>_LAUNCH_TOKEN``);
- mount paths and the shape of mounted files;
- the template context of a container's compose fragment;
- underscore-led modules.

The bump rule
=============

Removing, renaming or retyping anything this page lists, or changing what it means,
raises the number by one in the same change. Adding does not. A rule that newly refuses
something an app could do before is a breaking change. The change adds its row to the
`History`_. The rule is the one :doc:`facility-json` follows for its header.

The refusal
===========

``osprey build``, ``osprey app check`` and ``osprey validate`` read the number before
anything else in the manifest. A mismatch is the only finding reported for that
manifest, beside the profile's own errors. The line names the file, then
``contract:``, then one of four forms, each ending ``See <this page>.``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Declared
     - Message
   * - Nothing
     - ``not declared; this OSPREY implements contract N. Add "contract: N" to app.yml.``
   * - Not an integer
     - ``<value> is not an integer; this OSPREY implements contract N. Add "contract: N"
       to app.yml.`` ``true``, ``1.0`` and ``"1"`` are not integers.
   * - Lower
     - ``the app is written for contract M; this OSPREY implements N. Update the app to
       contract N.``
   * - Higher
     - ``the app is written for contract M; this OSPREY implements N. Upgrade OSPREY.``

For an installed app the file named is the distribution and its version.

What holds the contract
=======================

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Held by a test
     - Held by review
   * - Every manifest key keeps its name and type; every agent frontmatter key, SDK
       name, ``kinds`` name, ``page-ready.js`` export, primitive class, panel message
       type, header and ``OSPREY_*`` variable is still there
     - The ``osprey_connectors`` modules; the token names; the signature and meaning of
       a name that stays; a stricter rule

The build's import check
========================

``osprey build`` and ``osprey app check`` refuse an app that imports OSPREY outside the
modules above (rule ``import_boundary``), or imports from them a module that does not
exist (``import_missing_module``) or a name its module does not define
(``import_missing_name``). They judge the packages installed where the build runs.
The project image resolves ``osprey-connectors`` and ``osprey-facility`` on its own, so
an app can still fail at start in an image built from other versions.

History
=======

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Number
     - What it broke
   * - 1
     - Every app: ``contract`` is a required key of ``app.yml``.

Before contract 1
-----------------

These changes broke apps before the number existed:

- ``require.one_of`` accepts readback names only (2026-11-09).
- Hook code in an app is refused (2026-11-11).
- A panel must load ``page-ready.js`` and call ``markPageReady()``; a missing
  ``theme-boot.js``, a missing fonts link and raw hex colours are refused (2026.11.1).
- ``single_user_alias`` is removed; ``roster_grant: {}`` names the one operator
  (2026-11-21).
- The measurement-kinds module moved out of the framework to
  ``osprey_facility.kinds`` (2026-11-23).

=============
Web Interface
=============

ARIEL ships with a browser-based search interface that provides the same search capabilities as the CLI in a more approachable form. The interface is a FastAPI application serving a JavaScript single-page application (SPA). It connects to the same ``ARIELSearchService`` as the CLI and the ARIEL MCP tools, so any search module you register is automatically available in the UI.

.. raw:: html
   :file: ../../_diagrams/ariel-web-interface.html


Views
=====

The interface has four views, accessible via the navigation bar. All views are rendered client-side using hash-based routing (``#search``, ``#browse``, ``#create``, ``#status``).

.. tab-set::

   .. tab-item:: Search

      The primary view. A search bar with mode tabs (Keyword, Semantic --- only enabled modes are shown) and an expandable advanced options panel. Results display as entry cards with relevance scores and highlights. Press ``Enter`` to submit a query; searches always include the current advanced options and filters.

      .. figure:: /_static/screenshots/ariel_search.png
         :alt: ARIEL Search View
         :align: center
         :width: 90%

         Search view with Keyword mode selected.

   .. tab-item:: Browse

      Chronological, paginated listing of logbook entries (newest first) -- use Previous/Next to page back through older entries. Each entry shows its timestamp, author, and a text preview. Click an entry to view its full content.

      .. figure:: /_static/screenshots/ariel_browse.png
         :alt: ARIEL Browse View
         :align: center
         :width: 90%

         Browse view showing paginated entries sorted newest-first.

   .. tab-item:: New Entry

      Form for creating new logbook entries directly from the interface. Fields include subject, details, author, logbook, shift, and tags. When the configured logbook adapter is read-only (the common case for the standalone interface), entries are saved locally with ``source_system: "ARIEL Web"`` and a generated ``entry_id`` of the form ``ariel-<12-hex>``. When a write-capable facility adapter is configured, the entry is published to that logbook and takes the facility's ``source_system`` and ``entry_id``. Created entries are searchable immediately.

      .. figure:: /_static/screenshots/ariel_create.png
         :alt: ARIEL New Entry View
         :align: center
         :width: 90%

         New entry form for creating logbook entries from the web interface.

   .. tab-item:: Status

      Dashboard showing service health, database connection, entry count, embedding tables, enabled modules, and last ingestion timestamp. The dashboard polls ``/api/status`` on load, making it useful for verifying that the service is configured correctly after deployment.

      .. figure:: /_static/screenshots/ariel_status.png
         :alt: ARIEL Status View
         :align: center
         :width: 90%

         Status dashboard showing service health and configuration.

The four ARIEL views above were captured with OSPREY |captured_ariel| from the
``control-assistant`` tutorial's seeded logbook.

Display Preferences
===================

The button at the top right of the header --- the one drawn as three sliders
--- opens a small card holding everything about how the interface looks, plus
the way in to its settings:

**Appearance**
   Light or Dark. It flips the shade without changing which theme you are in.

**View**
   Expert or Simple. Expert is the full interface. Simple clears away the
   extras --- the four-tab strip becomes a single "Browse all entries" link,
   the search mode tabs and advanced options disappear, and results lose their
   relevance scores --- leaving a search box and a list of entries.

**Theme**
   One button per theme family: Main, DESY, High Contrast, and Retro. Picking
   a family keeps whichever appearance you are already in, so a switch from
   Main to High Contrast in dark stays dark.

**Settings**
   Opens the settings drawer, where you can read and edit ARIEL's
   configuration block.

Appearance, View, and Theme take effect straight away and leave the card open,
so you can compare two looks without re-opening the menu. Settings closes it,
because it takes you to a different surface. Click anywhere outside the card,
or press ``Escape``, to dismiss it.

**Your picks follow you.** The theme and view you choose here are remembered by
your browser and shared with every OSPREY interface served from the same
address --- pick a dark High Contrast look in ARIEL and the OSPREY web terminal
comes up that way the next time you load it, and the other way round too. There
is no separate preference to keep in step.

ARIEL running *inside* the web terminal, as a panel, behaves differently on
purpose. The terminal passes its own theme and view to the panel in the page
address, and an address always outranks the remembered preference, so an
embedded panel matches the terminal it sits in no matter what was set
elsewhere. The embedded panel has no header of its own either --- the
terminal's tile bar is the one header --- so the sliders button does not
appear there.

The ``GET /api/capabilities`` endpoint the interface calls at startup, and the
parameter descriptors it returns, are documented in
:doc:`/reference/contracts/ariel`.


Running the Web Interface
=========================

**CLI mode** (recommended for development):

.. code-block:: bash

   osprey ariel web                      # http://localhost:8085
   osprey ariel web --port 8080          # Custom port
   osprey ariel web --host 0.0.0.0       # Bind to all interfaces
   osprey ariel web --reload             # Auto-reload on code changes

.. note::

   The web UI runs in-process via ``osprey ariel web`` and is also exposed
   as a panel under ``osprey web``. There is no shipped container service
   template for it --- ``osprey up`` only brings up dependencies
   such as PostgreSQL.

**Programmatic usage:**

.. code-block:: python

   from osprey.interfaces.ariel.app import create_app

   app = create_app(config_path="config.yml")

   # Use with uvicorn
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8085)

.. admonition:: Collaboration Welcome
   :class: outreach

   The web interface is a great place to contribute --- whether that is a new view, improved accessibility, mobile-responsive layouts, or better error handling. If you build something useful, we encourage you to open a pull request so it becomes part of Osprey.


See Also
========

:doc:`search-modes`
    Search module architecture

:doc:`/reference/contracts/ariel`
    MCP tools, the capabilities API, and the database schema

:doc:`/reference/cli`
    CLI reference for ``osprey ariel web`` and all other ARIEL commands

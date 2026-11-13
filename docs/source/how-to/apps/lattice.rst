.. _how-to-lattice:

=======
Lattice
=======

The lattice app shows the optics, tunes, chromaticity and apertures of a
facility's lattice decks, computed by pyAT, elegant or the facility's own
code. Each analysis is computed once per deployment: every terminal, the agent
and the file downloads read the same result, cached by the deck's content, the
engine and its settings, the analysis settings and the energy. A restart
recomputes nothing; a changed deck or setting computes again. The app reads no
channel and writes none.

It ships with OSPREY and is off by default. Turn it on with
``osprey init --preset demo --set 'apps.lattice={}'``: init copies the app
into ``apps/lattice/`` and lists it under ``apps:`` in ``profile.yml``. The
retired *Lattice* web panel's ``web_panels`` keys are not read. To open the
web terminal on the app, set ``web.default_panel: lattice``.

From the panel
==============

Pick the model, the energy (when the model lists any) and the analysis. The
first terminal or agent to ask for a result starts it. A ready analysis shows
its charts, its scalars, and *CSV* and *SDDS* downloads of its table; the line
under the toolbar names the engine, the deck and the settings behind them. A
failed analysis shows its reason and stays failed until *Retry* or a restart.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. image:: /_static/screenshots/lattice_panel_light.png
         :alt: The lattice panel in the light theme: the analysis switch, the beta functions and dispersion along the ring, and the tunes, chromaticities and file downloads beside them
         :width: 100%

   .. grid-item::

      .. image:: /_static/screenshots/lattice_panel_dark.png
         :alt: The lattice panel in the dark theme
         :width: 100%

The *Optics* analysis of the example facility's storage ring, computed by
pyAT. Captured with OSPREY |captured_lattice_panel|.

Configuration
=============

The app's ``config`` block in ``profile.yml``. With no ``models`` the app
offers the deployment's simulator decks, computed by pyAT.

.. code-block:: yaml

   apps:
     lattice:
       config:
         deadline_s: 300          # one analysis's bound in seconds (default 300)
         max_workers: 2           # analyses computed at once (default 2)
         precompute: false        # true: compute every offered analysis at startup
         models:
           - name: booster
             label: Booster
             deck: decks/booster.lte      # absolute, or relative to the deployment
             engine: pyat                 # pyat (default), elegant, or one of engines:
             engine_settings: {bend_pass: ExactSectorBendPass, dp_step: 1.0e-5}
             energies_gev: [0.425, 3.0, 6.0]     # the panel's energy picker; first is default
             analyses: {lma: {enabled: false}}   # this model's overrides
         analyses:
           da: {settings: {nturns: 128, n_angles: 9, n_bisect: 8}, deadline_s: 1200}
         engines:
           aps-elegant:
             command: [python, -m, aps_lattice.engine]
             analyses: [optics, da, lma]  # omit: every analysis
             version: "3"                 # change it to drop this engine's cached results

``analyses`` sets, for every model or for one, which analyses run, their
settings and deadline; an analysis stopped at its deadline fails with that
reason. An engine reads the settings it knows:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Name
     - Shows
     - Settings (defaults)
   * - ``optics``
     - Tunes, chromaticities, Twiss functions, element layout
     - none
   * - ``chromaticity``
     - Tunes against momentum offset, chromatic orders to the third
     - ``dp_min_pct`` (-3), ``dp_max_pct`` (3), ``n_steps`` (25)
   * - ``da``
     - Dynamic aperture
     - ``nturns`` (512), ``n_angles`` (19), ``amp_max_mm`` (30), ``n_bisect`` (15)
   * - ``footprint``
     - Tune footprint against amplitude
     - ``n_amp`` (10), ``x_max_mm`` (3), ``y_max_mm`` (1), ``n_half`` (256)
   * - ``lma``
     - Local momentum aperture
     - ``nturns`` (512), ``n_refpts`` (15), ``dp_max_pct`` (5), ``n_bisect`` (15)

The chromatic orders are the derivatives of the tune at zero momentum offset
:math:`\delta`, from a cubic fitted to the curve, whichever engine solved its
points: :math:`\nu = \nu_0 + \xi_1\delta + \xi_2\delta^2/2 + \xi_3\delta^3/6`,
with ``chromaticity_x`` :math:`=\xi_1`, ``chromaticity2_x`` :math:`=\xi_2` and
``chromaticity3_x`` :math:`=\xi_3`. An offset with no stable optics is a gap
in the curve.

Engines
=======

``pyat`` computes every analysis. Without ``energies_gev`` it uses the deck's
energy. Its ``engine_settings``: ``bend_pass`` (the pass method of every
dipole), ``pass_methods`` (pass methods by element class), ``dp_step`` and
``xy_step`` (the finite-difference steps of the optics and chromaticity;
pyAT's defaults when omitted), ``use`` (the beamline of an elegant or MAD-X
deck) and ``six_d`` (``true`` keeps cavities and radiation on).

``elegant`` computes ``optics``, ``chromaticity``, ``da`` and ``lma`` by
running ``elegant`` and reading its output with ``sdds2stream``. Neither
program is in the OSPREY image: mount the facility's install, and the decks,
into the container (below). Its ``engine_settings``: ``elegant`` and
``sdds2stream`` (program paths), ``use_beamline``, ``p_central_mev`` (required
when the model lists no ``energies_gev``), ``setup`` (command-file text placed
after ``run_setup``), and one mapping per namelist (``run_setup``,
``twiss_output``, ``run_control``, ``find_aperture``, ``momentum_aperture``,
``closed_orbit``) whose entries add to or replace the ones the engine writes.

elegant runs in the deck's own directory, so a file the deck names by a
relative path (an ``#INCLUDE``, a kick map, a ``load_parameters`` file) is
found as it is by hand; every file it writes goes to a scratch directory, so
the decks may be mounted read-only. For input kept elsewhere, set
``run_setup: {search_path: "/lattices/kickmaps /lattices/errors"}`` (entries
separated by a space or a comma; none may contain a space).

For ``chromaticity`` the engine tracks one particle around the closed orbit at
each momentum offset and takes the tune from its turn-by-turn position, one
``elegant`` process per offset. ``tune_turns`` (1024) is the turns tracked and
``parallel`` (4) the processes run at once. It does not read the tunes off
``twiss_output`` at each offset: elegant's matrix there is truncated, which
leaves the linear chromaticity right and the higher orders wrong
(``dnux/dp2`` and ``dnux/dp3`` included).

.. code-block:: yaml

   apps:
     lattice:
       services:
         lattice:
           mounts:
             - {source: /nfs/physics/lattices, target: /lattices}
             - {source: /usr/local/oag, target: /opt/oag}
           env: [RPN_DEFNS]
       config:
         models:
           - name: sr
             deck: /lattices/sr/ring.lte
             engine: elegant
             energies_gev: [6.0]
             engine_settings:
               elegant: /opt/oag/bin/elegant
               sdds2stream: /opt/oag/bin/sdds2stream

``mounts`` is described under :doc:`build-your-own-app`.

**A facility's own engine** is any command named under ``engines:``. The
backend runs it as ``<command> <job.json> <out.json>``, the two files in a
scratch directory, and stops it at the analysis's deadline. The job:

.. code-block:: json

   {"schema": "lattice.job/1", "analysis": "da", "model": "booster",
    "deck": "/app/site/decks/booster.lte", "deck_sha256": "9f2c…", "energy_gev": 6.0,
    "settings": {"nturns": 128, "n_angles": 9, "amp_max_mm": 30.0, "n_bisect": 8},
    "engine": "aps-elegant", "engine_settings": {}}

The engine writes the result and exits 0, or writes the reason to stderr and
exits non-zero:

.. code-block:: json

   {"scalars": {"area_mm2": 391.1},
    "units": {"area_mm2": "mm^2"},
    "tables": {"aperture": {"columns": {"x_mm": [18.9, 12.1], "y_mm": [0.0, 9.4]},
                            "units": {"x_mm": "mm", "y_mm": "mm"}}}}

The panel plots the columns below; a result lacking one fails with its name.
It lists every scalar by name; further columns and tables are kept.

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - Analysis
     - Table
     - Required columns
   * - ``optics``
     - ``twiss``
     - ``s``, ``beta_x``, ``beta_y``, ``eta_x``
   * - ``chromaticity``
     - ``tune_vs_dp``
     - ``dp_pct``, ``tune_x``, ``tune_y``
   * - ``da``
     - ``aperture``
     - ``x_mm``, ``y_mm``
   * - ``footprint``
     - ``tunes``
     - ``x_mm``, ``y_mm``, ``tune_x``, ``tune_y``
   * - ``lma``
     - ``aperture``
     - ``s``, ``dp_plus_pct``, ``dp_minus_pct``

From the agent
==============

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Tool
     - Does
   * - ``lattice_state(model, energy_gev)``
     - The models, and for one model each analysis's settings, deadline, status
       and, when ready, scalars. Computes nothing.
   * - ``lattice_result(analysis, model, energy_gev, wait_s, retry, max_rows)``
     - One analysis's scalars and tables. Starts the analysis if nobody has,
       and waits up to 45 s; ``running`` means call again. Tables longer than
       ``max_rows`` (200) are thinned evenly.
   * - ``lattice_export(analysis, format, model, energy_gev, table, max_chars)``
     - A ready result's table as the panel's CSV or ASCII SDDS file, every
       row, with the scalars and provenance in the header; cut at
       ``max_chars`` (60 000), which ``truncated`` reports.

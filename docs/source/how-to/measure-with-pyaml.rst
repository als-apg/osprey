.. _measure-with-pyaml:

============================
Measure with the pyAML Tools
============================

A deployment whose build writes a pyAML view gives the agent two tools on the
``python`` server: ``pyaml_measure`` measures a response matrix or the
chromaticity on the control target, and ``model_response_matrix`` solves a
served model's orbit response on its deck. This page says when they appear,
what a measurement asks for, and what happens to the machine while it runs.
Every parameter and every result field is in
:ref:`executor-measurement-tools`.

When the tools are there
========================

The build writes ``data/pyaml/<model>/configuration.yaml`` for each served
model that names a deck and has a measurement file under
``data/facility/measurement/`` (see :doc:`describe-your-facility`). A render
with at least one such view registers both tools, the ``pyaml-tools`` skill and
the prompt section that teaches them; a render with none registers none of the
three.

The measurement file decides which kinds a model takes:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Kind
     - Measures
     - Steps
   * - ``orm``
     - the orbit response of the corrector setpoints
     - ``corrector_delta``
   * - ``dispersion``
     - the orbit response to the RF frequency
     - ``frequency_delta``
   * - ``trm``
     - the tune response of the quadrupoles
     - ``quad_delta``
   * - ``crm``
     - the chromaticity response of the sextupoles
     - ``sextu_delta``
   * - ``chromaticity_monitor``
     - the chromaticity
     - the RF, over ``n_step`` points

A kind the file does not list is refused, naming the group or instrument it
needs. A model solved in a single pass takes ``orm`` alone.

What a measurement asks
=======================

Every ``pyaml_measure`` call asks for approval. The prompt shows the model, the
kind, the modulation, the pyAML view's path and digest, and every setpoint the
call steps. A call that would step more setpoints than a prompt lists (500) is
refused without asking. When a guarded run on the target was interrupted, the
prompt also lists the setpoints it left displaced, with the values they are
restored to; approving the call restores them before the measurement steps
anything.

The call runs only on the view the build recorded and the prompt showed. A view
edited after the build, or one that differs from what the prompt showed, is
refused before any setpoint moves.

What happens to the machine
===========================

Each step is the measurement file's value, shrunk until no setpoint's largest
single write exceeds half its ``max_step`` in ``data/facility/limits.yaml``. A
strength, ``max_step`` or write that is not a finite number refuses the
measurement.

Every write of one measurement runs inside one guarded run on the control
target:

- the run holds the target's lock, and a second guarded run on the same
  target is refused while it does;
- each setpoint is journaled under ``var/guarded_run/<target>/`` before its
  first write;
- a run that stops part-way writes every displaced setpoint back before the
  call returns;
- a run that is killed leaves its journal, and the next approved guarded call
  on the target, from any session or container of the deployment, restores it
  first.

With approval off, the next guarded run restores a killed run's journal without
asking. A headless dispatch run that meets a pending journal is refused: only
a call a person approved restores one. A pyAML write from ``execute`` is
refused; measurements go through ``pyaml_measure``.

The result
==========

A completed measurement answers with its rows and columns named by address and
device id, the steps it used, and the measurement saved to the artifact store.
The saved file is the one the run left in its own execution folder; a run that
names any other file saves nothing and says so in ``data_file_refused``.

``crm`` by group is sized from the view's ``crm.json``, which the build writes
only when the sextupole group's design response can drive a correction; a view
without it refuses ``crm`` by group.

The model's answer
==================

``model_response_matrix`` solves the orbit response of a served model on its
own deck, without touching the machine and without asking for approval. Its
rows and columns carry the same addresses and device ids, so a measured matrix
and the model's line up entry by entry.

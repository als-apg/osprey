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

The tool refuses a kind the file does not list, naming the group or
instrument the model lacks, or the kinds the file lists when it lacks none;
such a call is never put to an approver. A model solved in a single pass takes ``orm`` alone.

What a measurement asks
=======================

Every ``pyaml_measure`` call for a kind the model's file lists asks for
approval. The prompt shows the model, the
kind, the modulation, the pyAML view's path and digest, and every setpoint the
call steps. A call that would step more setpoints than a prompt lists (500) is
refused without asking.

The call runs only on the view the build recorded and the prompt showed. A view
edited after the build, or one that differs from what the prompt showed, is
refused before any setpoint moves.

What happens to the machine
===========================

Each step is the measurement file's value, shrunk until no setpoint's largest
single write exceeds half its ``max_step`` in ``data/facility/limits.yaml``. A
strength, ``max_step`` or write that is not a finite number refuses the
measurement.

Every write of one measurement runs under the control target's run lock:

- a second read-write run on the target is refused while it runs;
- a measurement that stops part-way, fails or is interrupted leaves each
  setpoint where it stopped, and its result's ``moved`` lists each one with its
  value before the call; report them to the operator;
- a measurement that reaches the execution timeout is killed like any other
  run, with no early stop; its failure lists only the channels it attempted;
- a pyAML write from ``execute`` is refused; measurements go through
  ``pyaml_measure``.

:ref:`reference-guarded-run` has the lock and what a stopped run leaves.

The result
==========

A completed measurement answers with its rows and columns named by address and
device id, the steps it used, and the measurement saved to the artifact store.
The saved file is the one the run left in its own execution folder; a run that
names any other file saves nothing and says so in ``data_file_refused``.

``crm`` by group is sized from the view's ``crm.json``, which the build writes
whenever every member of the sextupole group has a modelled column; a group
with a member the model gives no column has no ``crm.json``, and its view
refuses ``crm`` by group.

The model's answer
==================

``model_response_matrix`` solves the orbit response of a served model on its
own deck, without touching the machine and without asking for approval. Its
rows and columns carry the same addresses and device ids, so a measured matrix
and the model's line up entry by entry.

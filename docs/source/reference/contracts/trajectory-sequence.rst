.. _reference-trajectory-sequence:

====================================
Trajectory Sequence Module Contract
====================================

A facility fires its injection its own way: a pulse on one channel, a counter
on another, a kicker to enable first. The trajectory service does not guess.
It loads one Python file the facility writes, the **sequence module**, and
calls it at three fixed points of a shot and a setup. This page is what that
module must hold, what it may do, and what the service refuses. How a shot
uses it is in :ref:`reference-trajectory-service`; the commands it may fire
are the :ref:`command slots <trajectory-command-slots>` of the model.

Where the module lives
======================

The module is the file ``sequence.py`` in the directory
``services.trajectory.sequence_dir`` names (see
:doc:`/reference/configuration/config`). The path is relative to the project
root and mounted read-only into the service's container. On a
virtual-accelerator target the build points it at the module OSPREY ships for
the virtual accelerator; on any other target it is unset until the facility
names its own.

The service loads the module once, when it starts, and checks it before it
serves anything:

- A directory that is unset, missing, or holds no ``sequence.py`` leaves the
  service without a module. It starts, and every shot and every setup is
  refused ``no_sequence_module``.
- A module that does not import, or does not pass the checks on this page,
  stops the service at startup, naming the file and what is wrong.

Changing the module takes a service restart.

Entry points
============

An entry point is a coroutine function (``async def``) taking one argument,
the **handle**. A module defines up to three:

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Entry point
     - Required
     - When the service calls it
   * - ``inject``
     - Yes
     - Once per shot, after every BPM is armed and reports armed. It fires
       one injection and nothing else.
   * - ``bpm_mode``
     - No
     - At a setup, first. It puts the BPMs into turn-by-turn acquisition. A
       facility whose BPMs need no change leaves it out, and setup skips it.
   * - ``bunch_clean``
     - No
     - At a setup, after ``bpm_mode``, only when the setup asks for it. A
       setup asking for it of a module that defines none is refused
       ``no_bunch_clean``.

An entry point that is defined but is not a coroutine function stops the
service at startup. The service alone arms the BPMs: no entry point arms, and
a shot calls ``inject`` only between its own arm and its own wait for the
armed flags to clear.

``SEQUENCE_METADATA``
=====================

The module declares, in one module-level mapping, every channel it writes and
the longest it ever pauses:

.. code-block:: python

   SEQUENCE_METADATA = {
       "slots": {
           "acquisition_mode": {"tag": "<signal on each BPM>"},
           "cleaning_enable": "<one address>",
       },
       "writes": {
           "bpm_mode": ["acquisition_mode"],
           "bunch_clean": ["cleaning_enable"],
       },
       "max_pause_s": 2.0,
   }

``max_pause_s`` (required)
   The longest the module ever waits between two handle calls, in seconds: a
   positive number no larger than 10. It sets the service's stop budget, so a
   Stop is answered within it plus one connector call.

``slots`` (optional)
   The settings the module may write, by a name of its own choosing. A slot
   is one address (a string), or one address per BPM, named by the facility
   signal those channels carry (``{"tag": "<signal>"}``). A slot is a
   setting, never a command: a slot naming an arm command or the injection
   request, or tagging ``bpm_arm_command``, stops the service at startup.

``writes`` (optional)
   Which slots each entry point may write. Every name is a declared slot, and
   every key is an entry point the module defines. ``inject`` has no entry: it
   writes the injection request and nothing else.

Any other key in the mapping stops the service at startup.

The handle
==========

Each entry point is called with a handle built for that one call, inside an
action that holds the target's guarded-run lock and names its owner (see
:ref:`reference-guarded-run`). Everything the module reads or writes goes
through it:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Call
     - What it does
   * - ``await handle.write("injection_request", value)``
     - Fires the injection request, from ``inject`` only, as a command (see
       `Commands and limits`_).
   * - ``await handle.write(slot, value)``
     - Writes a declared slot, from an entry point whose ``writes`` lists it,
       through the connector's checked setting path, with the confirm each
       channel's limits record asks for. A per-BPM slot gets *value* at every
       BPM.
   * - ``await handle.read(name)``
     - Reads the injection request, a declared slot (a per-BPM slot as a
       list in BPM order) or any channel address. ``"injection_request"``
       reads the channel the model wires, so a module that counts requests up
       never names its address.
   * - ``await handle.wait_for(name, condition, timeout=s)``
     - Reads one channel until *condition* (a predicate, or a value to
       equal) holds; ``TimeoutError`` after *timeout* seconds.
   * - ``await handle.sleep(s)``
     - Pauses, from 0 up to ``max_pause_s``.

Any other write --- an arm command, a bare address, an undeclared slot, a slot
this entry point does not list --- is refused ``sequence_refused`` before
anything is read or written. A ``sleep`` beyond ``max_pause_s`` and a
``wait_for`` on a per-BPM slot are refused the same way.

Before a slot's address is written, its current value is read. Every write
that was not refused is listed in the action's ``moved``, with that value as
``before``, so it can be put back (see :ref:`thread-the-first-turn`).

Every handle call is a stop checkpoint. Once a Stop or a take-over sets the
action's stop event, the next call raises, and a wait or a pause already
running ends within one poll. The module never sees the stop event itself and
has nothing to clean up: the service lists what moved and leaves the machine
where it stopped.

Commands and limits
===================

The injection request, like the BPMs' arm commands, is a **command**: a write
that acts and holds no setting. A command goes out with no confirmation
read-back, and is checked against the limits validator in the validator's own
mode:

- under ``optional``, a command with no limits record is written with no
  limits;
- under ``exclusive``, a command with no limits record is refused;
- a command that does have a record is held to its bounds in either mode.

A deployment that states limits checking off for the target fires commands
unchecked. One whose limits file could not be read refuses every command,
and so does one whose limits posture is neither on nor stated off: the shot is
refused ``limits_unavailable`` before anything is armed.

A module slot is a setting, not a command: it is written through the checked
setting path, held to its limits record and confirmed as that record asks.

The virtual accelerator's module
================================

The module OSPREY ships for the virtual accelerator is the smallest complete
one. The simulated machine fires one injection whenever its request is
written higher than the value it last saw, its BPMs always record turn by turn, and
it has no bunch to clean:

.. code-block:: python

   SEQUENCE_METADATA = {
       "slots": {},
       "max_pause_s": 0.5,
   }


   async def inject(handle):
       current = await handle.read("injection_request")
       await handle.write("injection_request", int(current) + 1)


   async def bpm_mode(handle):
       """Leave the BPMs as they are: they always record turn by turn."""

It declares no slots and names no address: it reaches the request only
through the handle.

.. seealso::

   :ref:`reference-trajectory-service`
      The routes a shot and a setup are taken through, and their refusals.

   :ref:`trajectory-command-slots`
      The arm commands and the injection request, and the rules the build
      holds them to.

   :doc:`/how-to/describe-your-facility`
      The measurement file that wires the injection request and the BPM
      signals.

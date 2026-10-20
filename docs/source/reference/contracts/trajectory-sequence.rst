.. _reference-trajectory-sequence:

====================================
Trajectory Sequence Module Contract
====================================

A facility fires its injection its own way: a pulse on one channel, a counter
on another, a kicker to enable first. The trajectory service does not guess.
It loads one Python file the facility writes, the **sequence module**, and
calls it at one point: the ``inject`` step of the chain. This page is what that
module must hold, what it may do, and what the service refuses. How a chain
uses it is in :ref:`reference-trajectory-service`.

Everything else the service writes around an injection --- the BPM arm
commands, the operator settings, command steps such as cleaning --- is
declared in the measurement file's ``trajectory`` block (see
:doc:`/how-to/describe-your-facility`), never in the module.

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
  service without a module. It starts, and every chain that injects is
  refused ``no_sequence_module``.
- A module that does not import, or does not pass the checks on this page,
  stops the service at startup, naming the file and what is wrong.

Changing the module takes a service restart.

``inject``
==========

``inject`` is the module's one entry point: a coroutine function
(``async def``) taking one argument, the **handle**. The service calls it once
per injection of a chain that ticks ``inject``, after its own arm and before
its own wait for the armed flags to clear. It fires one injection and nothing
else. A module without it, or whose ``inject`` is not a coroutine function,
stops the service at startup.

``SEQUENCE_METADATA``
=====================

The module declares, in one module-level mapping, the longest it ever pauses:

.. code-block:: python

   SEQUENCE_METADATA = {"max_pause_s": 2.0}

``max_pause_s`` (required)
   The longest the module ever waits between two handle calls, in seconds: a
   positive number no larger than 10. It sets the service's stop budget, so a
   Stop is answered within it plus one connector call, and it counts toward
   each injection's worst case.

Any other key in the mapping stops the service at startup.

The handle
==========

``inject`` is called with a handle built for that one injection, inside an
action that holds the target's guarded-run lock and names its owner (see
:ref:`reference-guarded-run`). Everything the module reads or writes goes
through it:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Call
     - What it does
   * - ``await handle.write("injection_request", value)``
     - Fires the injection request as a command (see `Commands and limits`_).
   * - ``await handle.read(name)``
     - Reads the injection request or any channel address.
       ``"injection_request"`` reads the channel the model wires, so a module
       that counts requests up never names its address.
   * - ``await handle.wait_for(name, condition, timeout=s)``
     - Reads one channel until *condition* (a predicate, or a value to
       equal) holds; ``TimeoutError`` after *timeout* seconds.
   * - ``await handle.sleep(s)``
     - Pauses, from 0 up to ``max_pause_s``.

Any other write --- an arm command, a setting, a bare address --- is refused
``sequence_refused`` before anything is read or written, and so is a
``sleep`` beyond ``max_pause_s``. The module never arms and never moves a
setting.

Every handle call is a stop checkpoint. Once a Stop or a take-over sets the
action's stop event, the next call raises, and a wait or a pause already
running ends within one poll. The module never sees the stop event itself and
has nothing to clean up: the service lists what it fired and leaves the
machine where it stopped.

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

The virtual accelerator's module
================================

The module OSPREY ships for the virtual accelerator is the smallest complete
one. The simulated machine fires one injection whenever its request is
written higher than the value it last saw:

.. code-block:: python

   SEQUENCE_METADATA = {
       "max_pause_s": 0.5,
   }


   async def inject(handle):
       current = await handle.read("injection_request")
       await handle.write("injection_request", int(current) + 1)

It names no address: it reaches the request only through the handle.

.. seealso::

   :ref:`reference-trajectory-service`
      The routes a chain is run through, and their refusals.

   :ref:`trajectory-command-slots`
      The command channels, and the rules the build holds them to.

   :doc:`/how-to/describe-your-facility`
      The measurement file's ``trajectory`` block: the devices, settings and
      chain steps around the injection.

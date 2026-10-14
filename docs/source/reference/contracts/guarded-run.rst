.. _reference-guarded-run:

====================
Guarded Run Contract
====================

A guarded run is the lock a read-write ``execute`` or ``execute_file`` run and
a ``pyaml_measure`` run hold on their control target, so two runs never drive
one machine at once. No other write path takes part.
:ref:`Guarded Runs <architecture-safety-chain-guarded-run>` places it among
the write paths and says what it does not cover. This page is the exact
behaviour: where the lock lives, who holds it, what a second run is told, and
what a stopped run leaves.

Every write made under the lock is an ordinary write. It goes through the
connector like any other, so the kill switch, the limits check and the
control-target check of :doc:`/architecture/safety-chain` all apply to it. The
lock adds no check of its own to those writes; it replaces none of them.

One run per target
==================

A guarded run holds one lock per control target, ``live``, ``va`` or
``standin``. The lock is a file,
``var/guarded_run/<target>/run.lock`` in the deployment repo, and the
directory is deployment-wide: the deploy provisions it on the host and binds
it at the same place in every container that runs the agent, so a run
started from a web terminal, a dispatched job or a local ``osprey chat``
contends for the same lock.

The lock is taken without waiting and held until the run ends. A second run
on the same target, of either kind, is refused before it starts, with the
process id of the holder and the time it took the lock. A read-only run is
refused before the lock is taken.

Who holds it
============

- A read-write ``execute`` or ``execute_file`` run takes the lock before its
  code starts and releases it after its record is saved.
- ``pyaml_measure`` runs its script through the same sandbox, so it takes the
  lock the same way.
- ``pyaml_cs_osprey.run_tool``, which runs every pyAML tool of a measurement,
  runs under the lock the process already holds, and takes it when the process
  holds none.

The process holds it
====================

The lock belongs to the process, not to the thread that took it: a thread or
task the run starts runs under it. Asking for the lock again from inside the
process is refused like any other second run. A process holds the lock of one
target at a time; asking for another target's lock while it holds one is
refused before anything is opened:
``this process holds the run lock on <held>; it cannot take one on <asked>``.

Joining a run never extends its hold: the lock is released when the code that
took it ends, whatever joined it; the executor takes it for the whole script.

A forked child never holds the lock, at any moment, from any thread: a fork
waits for a lock entry or release in progress to finish, and the child closes
its copies of the lock. A child started with ``subprocess`` or ``exec`` never
inherits it.

The lock file
=============

``run.lock`` is a regular file with one name, opened inside the target's
directory without following a link at ``guarded_run``, at the target's
directory or at the file. A link, a hard link, a FIFO or anything else that is
not such a file is refused with its path named: the run does not start, and
nothing is read or written through it.

While a run holds the lock, the file records its holder, ``{"pid", "started"}``;
the record is emptied on release. A created lock file has mode ``0o664``, and
each created directory ``0o2775``, setgid, so every identity of the deployment
sharing its group can take the lock. The directory holds no file but
``run.lock``.

The approval prompt never opens this directory: it binds the call to its target
(``approved_target``) and asks. Only the run refuses a held target, before its
code starts and with nothing moved:
``another guarded run is live under <directory> (pid <pid>, since <time>)``.

What the lock does not do
=========================

A killed or failed tool leaves setpoints where they stopped; the result lists
them with their old values.

- The lock records and restores nothing.
- A run that is killed, fails or is interrupted leaves every setpoint where it
  stopped, as the MATLAB Middle Layer and pyAML do.
- ``pyaml_measure`` lists in its result every setpoint the call moved, with its
  value before the call (``moved``), whether the run completed, stopped or
  failed.
- A run killed outright cannot list values, and its failure names the channels
  it attempted. That includes a run that reaches the execution timeout:
  nothing stops a tool early to leave time for a report, so it is killed like
  any other run.
- The agent's own Python lists nothing; its writes are in the run's write
  ledger.
- The putting back of a stepped knob inside a measurement is pyAML's own.

.. seealso::

   :doc:`/architecture/safety-chain`
      The checks every write passes, under the lock or not.

   :doc:`connectors`
      The checked write every write goes through.

   :doc:`/how-to/measure-with-pyaml`
      ``pyaml_measure`` and what its result lists.

.. _reference-guarded-run:

====================
Guarded Run Contract
====================

A guarded run is the lock a read-write ``execute`` or ``execute_file`` run and
a ``pyaml_measure`` run hold on their control target, so two runs never drive
one machine at once. No other write path takes part.
:ref:`Guarded Runs <architecture-safety-chain-guarded-run>` places it among
the write paths and says what it does not cover. This page is the exact
behaviour: where the lock lives, who holds it, whose run its record names,
what a second run is told, and what a stopped run leaves.

Every write made under the lock is an ordinary write. It goes through the
connector like any other, so the kill switch, the limits check and the
control-target check of :doc:`/architecture/safety-chain` all apply to it. The
lock adds no check of its own to those writes; it replaces none of them.

One run per target
==================

.. list-table:: Where the lock lives
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Value
   * - Lock file
     - ``var/guarded_run/<target>/run.lock`` in the deployment repo
   * - Targets
     - ``live``, ``va``, ``standin``: one lock per target
   * - Directory mode
     - ``0o2775`` for each guarded-run directory a run creates: setgid and
       group-writable. ``var/`` gets the process's default mode, and an
       existing directory keeps the mode it has.
   * - File mode
     - ``0o664`` for a created lock file
   * - Directory contents
     - ``run.lock`` only

The directory is deployment-wide: the deploy provisions it on the host and
binds it at the same place in every container that runs the agent, so a run
started from a web terminal, a dispatched job or a local ``osprey chat``
contends for the same lock. The setgid directories let every identity of the
deployment sharing their group take the lock.

The lock is taken without waiting and held until the run ends. A second run
on the same target, of either kind, is refused before it starts (see
`Refusals`_). A read-only run is refused before the lock is taken.

Who holds it
============

.. list-table:: Who takes the lock
   :header-rows: 1
   :widths: 35 65

   * - Caller
     - How it holds the lock
   * - A read-write ``execute`` or ``execute_file`` run
     - Takes it before its code starts and releases it after its record is
       saved.
   * - ``pyaml_measure``
     - Runs its script through the same sandbox, so it takes the lock the
       same way.
   * - ``pyaml_cs_osprey.run_tool``, which runs every pyAML tool of a
       measurement
     - Runs under the lock the process already holds, and takes it when the
       process holds none.

Whose run it is
===============

A run takes the lock with a record of whose run it is: the target it drives,
the account it acts for (or nobody) and the target it was admitted on. A
record whose target or admitted target is not the target the lock resolves to
is refused before anything is opened, so a run is never admitted on one target
and driven on another. Joining the process's hold with no record runs under
the held one; joining with a different record is refused.

- A read-write ``execute`` or ``execute_file`` run fills the record once, as
  its sandbox starts: the stamped target, the account the write path names
  (nobody when it names none) and that same target as the admitted one, which
  the executor checked before it started the sandbox. Nothing about the
  sandbox's writes changes.

The process holds it
====================

The lock belongs to the process, not to the thread that took it: a thread or
task the run starts runs under it. Asking for the lock again from inside the
process is refused like any other second run. A process holds the lock of one
target at a time; asking for another target's lock while it holds one is
refused before anything is opened.

Joining a run never extends its hold: the lock is released when the code that
took it ends, whatever joined it; the executor takes it for the whole script.

A forked child never holds the lock, at any moment, from any thread: a fork
waits for a lock entry or release in progress to finish, and the child closes
its copies of the lock. A child started with ``subprocess`` or ``exec`` never
inherits it.

The lock file
=============

While a run holds the lock, ``run.lock`` holds its holder record, a JSON
object with these keys. The record is emptied on release.

.. list-table:: Holder record
   :header-rows: 1
   :widths: 15 55 30

   * - Key
     - Value
     - ``null`` when
   * - ``pid``
     - The holder's process id
     - Never
   * - ``started``
     - The time the holder took the lock
     - Never
   * - ``owner``
     - The account the run acts for
     - The run acts for nobody
   * - ``target``
     - The control target
     - Never
   * - ``identity``
     - The name the process writes under on the control system, from
       ``OSPREY_CONTROL_IDENTITY``
     - The process sets none

A refusal of a held target names each key, and a key reads ``unknown`` when
the holder's record does not name it.

``run.lock`` is a regular file with one name, opened inside the target's
directory without following a link at ``guarded_run``, at the target's
directory or at the file. A link, a hard link, a FIFO or anything else that is
not such a file is refused with its path named: the run does not start, and
nothing is read or written through it.

The approval prompt never opens this directory: it binds the call to its target
(``approved_target``) and asks. Only the run refuses a held target, before its
code starts and with nothing moved.

Refusals
========

Each refusal is raised before the run's code starts. In the texts, ``<…>``
stands for a value the refusal fills in. A refusal of the operating system
itself (an ``OSError`` the lock file's open raises) is passed on as it is.

.. list-table:: Refusals
   :header-rows: 1
   :widths: 30 70

   * - Case
     - Text
   * - Second run on a held target
     - ``another guarded run is live under <directory> (pid <pid>, since <time>, owner <owner>, target <target>, identity <identity>)``
   * - Another target while the process holds one
     - ``this process holds the run lock on <held>; it cannot take one on <asked>``
   * - Read-only run
     - ``OSPREY refused the write: readonly execution mode: a guarded run needs execution_mode='readwrite'``
   * - Record names another target
     - ``a run on <target> cannot hold a context for target <record target> admitted on <admitted target>``
   * - Join with a different record
     - ``this process holds the run lock on <target> under another context; a join runs under the held one``
   * - Link at the lock file
     - ``<path> is a link; the guarded-run directory holds files only``
   * - Hard link at the lock file
     - ``<path> has <count> hard links; the guarded-run directory holds files only``
   * - FIFO or other non-regular file at the lock file
     - ``<path> is not a regular file; the guarded-run directory holds files only``
   * - Link at ``guarded_run`` or the target's directory
     - ``<path> is a link; a guarded-run directory is never a link``
   * - Something other than a directory at ``guarded_run`` or the target's directory
     - ``<path> is not a directory``
   * - Lock file replaced while it was opened
     - ``<path> was replaced while it was opened``
   * - Target that is not a control target
     - ``Unknown control target <name> for a guarded run. Valid targets are live, va, standin.``
   * - No deployment repo root
     - ``guarded runs need var/guarded_run: no deployment repo root found (resolved <root>, which holds no profile.yml or config.yml).``
   * - Directory cannot be created
     - ``guarded runs need var/guarded_run: cannot create <directory>: <error>``
   * - Directory not writable
     - ``guarded runs need var/guarded_run: <directory> is not writable.``

What the lock does not do
=========================

A killed or failed tool leaves setpoints where they stopped; the result lists
them with their old values.

- The lock records and restores nothing.
- A run that is killed, fails or is interrupted leaves every setpoint where it
  stopped, as the MATLAB Middle Layer and pyAML do.
- ``pyaml_measure`` lists in its result every setpoint whose write may have
  gone out, a write whose outcome is unknown included, with its value before
  the call (``moved``), whether the run completed, stopped or failed.
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

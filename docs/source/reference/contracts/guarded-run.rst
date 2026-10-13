.. _reference-guarded-run:

====================
Guarded Run Contract
====================

A guarded run is the lock a read-write ``execute`` or ``execute_file`` run and
a ``pyaml_measure`` run hold on their control target, and the journal a
``pyaml_measure`` run keeps of each setpoint it displaces, so a run that dies
half way is put back by the next. No other write path takes part.
:ref:`Guarded Runs <architecture-safety-chain-guarded-run>` places it among
the write paths and says what it does not cover. This page is the exact
behaviour: where the lock and the journal live, what the journal holds, when
a restore writes and when it refuses, and the lines a run prints.

Every write of a guarded run is an ordinary write. It goes through the
connector like any other, so the kill switch, the limits check and the
control-target check of :doc:`/architecture/safety-chain` all apply to it. The
guarded run adds a lock and a journal around those writes; it replaces none of
them.

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

A process has one guarded run. The lock it holds, the journal it has open and
the call it was approved for belong to the process, not to the thread that
took them: a thread or task the run starts runs under it, and its writes go
into the same journal. Asking for the lock again from inside the process is
refused like any other second run. A process holds the lock of one target at
a time; asking for another target's lock while it holds one is refused before
anything is opened:
``this process holds the run lock on <held>; it cannot take one on <asked>``.

The journal
===========

Beside the lock sits the journal, ``run.journal``, which only a
``pyaml_measure`` run writes. Before it first writes an address, it reads the setpoint that address holds and appends it to
the journal, flushed to disk before the write goes out. The journal keeps the
first value per address only, the value from before the run, however often the
run writes the address afterwards. Its header line records the control target,
the target's generation, who is acting, the process id and the start time. The lock and
the journal are regular files with one name each, opened inside the target's
directory without following a link at ``guarded_run``, at the target's
directory or at the file. A link, a hard link, a FIFO or anything else that is
not such a file is refused with its path named: the run does not start, and
nothing is read or written through it. The approval prompt and the Python
executor read the journal through the same open, so the journal a prompt lists
is the file the run restores.

A write through the guarded path is refused outside a journaled run and
touches no channel, so no code path writes a device under the guard without
the journal holding its way back.

A run that ends cleanly clears its journal and writes nothing back: leaving
the machine as it found it is then the run's own work. A journal with records
in it therefore means a guarded run left setpoints displaced: it was killed,
or its restore could not put every address back, or could not run.

Restoring a killed run
======================

A guarded run puts back what it displaced in one place, the exit path of its
outermost journaled run. When an exception escapes the run, an error in its
code or an interrupt, that exit path writes each journaled address back to its
recorded value before the exception goes on. While it does, it ignores
``SIGINT`` on the main thread, so a second interrupt does not cut the restore
short. A write already in flight on another thread finishes before the
restore starts, and none is accepted after the run has ended.

Releasing the lock ends a run still open under it, as when the code holding
the lock returns while a thread it started is still inside the run: the exit
path puts back what the run displaced and reports the run as aborted. Whatever
escapes a run opened inside another, on the same thread or on another one,
stops the whole run: every later guarded write is refused with
``the guarded run was stopped: <first line of the error>``, and the outer
run's exit puts back what the run displaced, reports it as aborted and raises
that error, even when its own code ended cleanly.

A run killed outright reaches no exit path. The next guarded run on the target
finds its journal. Holding the lock proves the run that wrote it is over, so
before its own work starts it writes each journaled address back to its
recorded value and prints one line,
``restored <n> addresses from a dead run (pid <pid>)``, followed by any
address it could not restore and why.

The restore forces nothing:

- Each address is written back through the same checked write as any other,
  so limits, write gates and the control-target check apply.
- An address reading exactly its journaled value is not written. Exactly
  means equal (``readings_equal``); the comparison that confirms a write
  landed (``values_match``) allows for the rounding of a float32 store and
  does not decide this, so an address displaced by less than that is still
  written back.
- Where a channel has a ``max_step`` limit and the way back is longer, the
  restore walks back in equal steps no larger than ``max_step``.
- An address the connector refuses, or whose write is not confirmed, is
  reported with its reason, and the restore goes on with the others. The
  journal then keeps its header line and exactly the addresses left
  displaced, for the next run to restore. A run whose own start finds an
  address it cannot put back does not start: a person decides what to do with
  a machine that could not be put back.
- Where the guarded tool's approval prompt asks, the prompt lists the pending
  journal, and the run restores it before the call's code runs only while the
  file is byte for byte the one the prompt listed; it is then restored
  whichever generation of the target wrote it. Any other journal is left
  untouched and the call is refused, so the next prompt lists the journal as
  it is.
- Without an approved journal, a journal written for another control target,
  or for an earlier generation of this one, is not replayed. The new run is
  refused and the file is left.

When the restore cannot run at all, nothing is written and the journal is left
byte for byte as it was, for the next guarded run on the target. One untagged
line on stderr says so:
``restore not attempted: <reason>; the journal <path> stays for the next guarded run on <target>``.

A write whose address reads NaN or an infinity before it, alone or anywhere in
an array, has no value a restore could write back. A guarded run refuses it
before it is journaled or written:
``reads nan before the write, so a guarded run could not put it back``.

Every outermost journaled run ends with exactly one report line on stderr,
``OSPREY_GUARDED_RUN_RESTORE`` followed by the report as JSON: the addresses
restored, the addresses already at their value, those refused with the value
each was left at, those that failed, and whether the run was aborted. A clean
run prints it too, with ``"aborted": false``. A journaled run opened inside
another prints none, and where the restore could not run the untagged line
takes its place. A restore of a killed run's journal prints one as well.

The line is for people. The report is the run's account of what it put back;
the journal is what the next run restores from. The Python executor takes a
run's reports from its execution record, never from the printed line, and
files them in the run's execution folder and the audit ledger beside what it
reads in the target's journal itself after the run.

.. seealso::

   :doc:`/architecture/safety-chain`
      The checks every write passes, guarded or not.

   :doc:`connectors`
      The checked write each restore goes through.

   :doc:`/how-to/measure-with-pyaml`
      ``pyaml_measure``, the one run that journals.

   :doc:`/how-to/describe-your-facility`
      ``limits.yaml``, where ``max_step`` is authored.

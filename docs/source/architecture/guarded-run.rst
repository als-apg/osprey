.. _architecture-guarded-run:

===========
Guarded Run
===========

A measurement moves many setpoints and has to leave them as it found them: an
orbit response measurement steps every corrector and puts each one back. A
*guarded run* is the runtime's mechanism for that kind of work. It gives a
multi-write run two properties a sequence of single writes does not have: only
one such run touches a machine at a time, and a run that dies half way leaves
a record the next run puts the machine back from.

Every write of a guarded run is an ordinary write. It goes through the
connector like any other, so the kill switch, the limits check and the
control-target check of :doc:`safety-chain` all apply to it. The guarded run
adds a lock and a journal around those writes; it replaces none of them.

One run per target
==================

A guarded run holds one lock per control target, ``live``, ``va`` or
``standin``. The lock is a file,
``var/guarded_run/<target>/run.lock`` in the deployment repo, and the
directory is deployment-wide: the deploy provisions it on the host and binds
it at the same place in every container that runs the agent, so a run
started from a web terminal, a dispatched job or a local ``osprey chat``
contends for the same lock.

The lock is taken without waiting. A second run on the same target is refused
before it starts, with the process id of the holder and the time it took the
lock. A read-only run is refused before the lock is taken.

The journal
===========

Beside the lock sits the journal, ``run.journal``. Before a guarded run first
writes an address, it reads the setpoint that address holds and appends it to
the journal, flushed to disk before the write goes out. The journal keeps the
first value per address only, the value from before the run, however often the
run writes the address afterwards. Its header line records the control target,
the target's generation, who is acting, the process id and the start time.

A write through the guarded path is refused outside a journaled run and
touches no channel, so no code path writes a device under the guard without
the journal holding its way back.

A run that ends, cleanly or with an error it survives, clears its journal. A
journal with records in it therefore means one thing: the run that wrote it
was killed.

Restoring a killed run
======================

The next guarded run on the target finds that journal. Holding the lock proves
the run that wrote it is dead, so before its own work starts it writes each
journaled address back to its recorded value and prints one line,
``restored <n> addresses from a dead run (pid <pid>)``.

The restore forces nothing:

- Each address is written back through the same checked write as any other,
  so limits, write gates and the control-target check apply.
- An address already at its journaled value is not written.
- Where a channel has a ``max_step`` limit and the way back is longer, the
  restore walks back in equal steps no larger than ``max_step``.
- An address the connector refuses, or whose write is not confirmed, is
  reported with its reason. The journal is then left byte for byte as it was
  and the new run does not start: a person decides what to do with a machine
  that could not be put back.
- A journal written for another control target, or for an earlier generation
  of this one, is not replayed. The new run is refused and the file is left.

.. seealso::

   :doc:`safety-chain`
      The checks every write passes, guarded or not.

   :doc:`/how-to/describe-your-facility`
      ``limits.yaml``, where ``max_step`` is authored.

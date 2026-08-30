.. _reference-connectors:

===================
Connector Contracts
===================

Every connector -- the ones that ship with OSPREY and any you register yourself --
has to behave the same way at its edges, because the agent, the plans and the
safety layers on top of it are written against that behaviour rather than against any
one control system. This page states those contracts: how large array values are
returned, how a write reports whether it actually took effect and which gates it
must pass first, and what an archiver's historical data has to look like. For the
day-to-day task of picking and configuring a connector, see
:doc:`/how-to/control-systems/use-connectors`.

Reading Large Values
--------------------

Array-valued channels -- camera frames, waveforms, orbit vectors -- are often
far too big to hand to the agent as raw numbers. ``channel_read`` applies a size
rule to every read, whatever protocol answered it: values within an element
budget come back inline as JSON lists, and anything larger is saved to the
artifact gallery and reported as a summary plus a handle to it.

.. code-block:: yaml

   control_system:
     read_inline_max_elements: 2000        # per-value element budget
     channel_read_artifact_retention: 20   # readings kept per channel

Only arrays are measured -- strings and single scalars are always inline. One
``channel_read`` call also has an aggregate budget of four times
``read_inline_max_elements``, spent in request order. Every withheld value
names which limit it hit as ``artifact_reason``: ``per_value_threshold`` (the
value alone is over budget; it will never come back inline) or
``aggregate_budget`` (earlier channels in the same request spent the call's
budget; a smaller batch returns it inline).

The summary that replaces the value reports shape, dtype and element count,
plus min/max/mean for numeric data, along with the artifact's id and its
``data_file`` path -- the authoritative copy of the values. 1-D arrays are
saved as JSON with an interactive chart in the gallery (x axis is sample
index); 2-D and 3-D data with a color axis get an auto-scaled PNG preview
beside a raw ``.npy``; shapes with no honest rendering are saved as ``.npy``
alone. The agent reaches the numbers by loading ``data_file`` inside
``execute`` (``numpy.load`` / ``json.load``). If the artifact store cannot
write, the read still reports success with ``artifact_error`` in place of the
handle -- the machine did answer.

``channel_read_artifact_retention`` (default 20) keeps only the newest N
**unpinned** artifacts per channel; pinned readings are never pruned and never
occupy a slot. ``0`` keeps everything -- remembering that an unattended
polling loop will then grow the gallery without bound.

Write Confirmation
------------------

A write is **confirmed** when the channel it wrote now holds the value that was
sent. The connector establishes that itself: it re-reads the channel once the
control system has accepted the put, and reports what it found as a single word
on :class:`~osprey.connectors.control_system.ChannelWriteResult`, which every
``write_channel()`` call returns.

.. code-block:: python

   from osprey.connectors.control_system import WriteOutcome

   result = await connector.write_channel("BEAM:CURRENT", 100.0)

   if result.outcome is WriteOutcome.CONFIRMED:
       print("the channel now holds the value that was sent")

Every write ends in exactly one of six outcomes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Outcome
     - What it means
   * - ``refused``
     - Nothing was written. A gate stopped the write before it left OSPREY --
       write posture, a limit, a channel marked not writable -- and
       ``refusal_reason`` names which one.
   * - ``failed``
     - The value was sent and the control system did not take it.
   * - ``confirmed``
     - The re-read holds the value that was sent.
   * - ``mismatch``
     - The re-read holds a different value. A setpoint the machine clamped or
       rounded is reported here, not smoothed over.
   * - ``unconfirmed``
     - The value was sent, but the re-read itself failed, so what the channel
       holds is unknown.
   * - ``unrequested``
     - Confirmation is switched off for this channel; nothing was checked.

Beside the outcome, the result carries ``observed_value`` (what the re-read
returned), ``alarm_status`` and ``alarm_severity`` (the channel's alarm state,
which is reported with the write and is never itself a reason to fail one), and
``notes`` -- a short display line such as ``observed 2.4, sent 2.5`` that is
there to be shown, not parsed.

**In Python, anything short of a confirmation raises.**
``osprey.runtime.write_channel`` raises ``ChannelWriteBlockedError`` on
``refused`` and ``ChannelWriteFailedError`` on ``failed``, ``mismatch`` and
``unconfirmed``; the mismatch message names both the value sent and the value
observed. The ``channel_write`` MCP tool reports the outcome for every channel
instead of raising, so the agent can describe a batch in which some channels
took the value and others did not.

Switching confirmation off
~~~~~~~~~~~~~~~~~~~~~~~~~~

``confirm`` is a per-channel boolean in the limits database, sitting beside
``writable``:

.. code-block:: json

   {
     "defaults": {
       "writable": true,
       "confirm": true
     },
     "SHUTTER:OPEN_CMD": {
       "writable": true,
       "confirm": false
     }
   }

A channel's own entry wins; failing that the ``defaults`` block applies; failing
that, confirmation is on. Switch it off for the channels a re-read cannot answer
for -- a command channel that resets itself, a register that cannot be read back
-- so those writes report ``unrequested`` rather than ``unconfirmed``. A call
site may pass ``confirm=True`` or ``confirm=False`` to decide for one call;
passing nothing means "no opinion" and leaves the choice to the configuration.
``write_multiple_channels()`` confirms each channel exactly as a single write
would, which is why ``osprey.runtime.write_channels`` behaves like
``osprey.runtime.write_channel``.

How values are compared
~~~~~~~~~~~~~~~~~~~~~~~

One rule decides every outcome, on every connector, and there is nothing to
configure. Two numbers agree when they are within a relative 1e-6 of each other
-- enough to absorb the rounding of a float on its way through a control system,
not enough to hide a clamped setpoint. A string written to an enum channel is
compared against the label the connector resolved for the reading, since the
channel itself reads back an index. A single-element array compares as the value
it holds, because control systems disagree about whether such a channel reads
back boxed. A scalar never matches a longer vector, and two vectors must be the
same length and agree element by element. Everything else compares for equality.
If the comparison cannot be made at all, the outcome is ``mismatch``: a write is
confirmed only when the values are known to agree.

.. _write-safety-config:

Write Safety Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Write operations are disabled by default, and write permission is set per
connector type:

.. code-block:: yaml

   control_system:
     writes_enabled: false                  # what a type inherits when it says
                                            # nothing about itself
     connector:
       virtual_accelerator:
         writes_enabled: true               # ... and this type's own answer
       epics:
         writes_enabled: false              # pinned, so the inherited key
                                            # cannot arm it later

A connector type that carries no ``writes_enabled`` of its own inherits
``control_system.writes_enabled``; one that carries it uses its own value and
never falls back. Both default to blocked when omitted, and **only a literal**
``true`` **arms writes** — the quoted string ``'true'`` and the number ``1`` do
not. A custom connector's block is keyed by the same dotted module path that
selects it, so ``mypackage.TangoConnector`` names one block and is never split
on its dots.

Write posture is a **launch-time deployment posture, not a live kill-switch**:
read from config and process-cached, so flipping it in ``config.yml`` does not
affect a running process. The enforced kill-switch lives at the harness layer
(a renderer ``permissions.deny`` on the write tool, then regenerate and
relaunch); in-flight control of an active plan is the RunEngine's own
``abort`` / ``pause``. One live control does exist alongside it, and it only
ever narrows: an operator can take one control target away from one session from
:ref:`the control-target chip <web-terminal-session-posture>`, and the
connector re-reads that on every put.

That rendered deny list is written once, before any session has chosen a
target, so it exists only where **no** target may write. A deployment armed on
one target and not another renders no deny at all, and the refusal arrives per
call instead, from the safety hook and from the connector, naming the target
that refused it. Tools a project lists under ``control_system.write_tools``
take that per-call path in every render: they are never in the deny list, and
the writes-check hook is what refuses them.

The connector applies **per-write mechanical safety** — the write-posture gate,
limits validation, the fail-closed validation path — on every put. That
is a separate, complementary layer from the **per-intent human authorization**
at the tool boundary (the approval hook, the launch token for plans), which
gates the *intent* once rather than every put. The approval layer cannot
substitute for the connector's mechanical refusal.

.. _limits-checking-config:

Limits Checking
~~~~~~~~~~~~~~~

.. code-block:: yaml

   control_system:
     limits_checking:
       enabled: true                     # Enable limits validation
       database_path: ./limits_db.json   # Path to the channel limits JSON
       allow_unlisted_channels: false    # Block writes to channels not in the database
     connector:
       virtual_accelerator:
         limits_checking:                # This connector type's own posture,
           enabled: true                 # replacing the pair above for it alone
           allow_unlisted_channels: true

When enabled, every ``write_channel()`` call is validated against the limits
database before the write reaches hardware. The database format is the
per-channel configuration above.

The posture is per connector type. ``control_system.limits_checking`` is what a
type inherits when it says nothing about itself, and
``control_system.connector.<type>.limits_checking`` answers for that type
instead — so one deployment can refuse unlisted channels on its live machine
while letting them through on a simulator. Three rules govern the per-type
block:

- **It replaces, it does not merge.** A per-type block must state both
  ``enabled`` and ``allow_unlisted_channels``; neither is inherited. A block
  stating one alone is refused by ``osprey build`` and ``osprey validate``,
  naming the missing setting — and a half-written block that reaches a running
  deployment anyway, by a hand-edited ``config.yml`` or an older render, blocks
  every write until it is completed.
- **The database stays deployment-wide.** ``database_path`` is not a per-type
  setting: the deployment mounts one limits file, and every target is checked
  against it.
- **Only explicit values decide.** ``allow_unlisted_channels`` is true, false,
  or unstated. With limits checking enabled, an ``allow_unlisted_channels``
  that no key states refuses unlisted channels — permission needs an explicit
  ``true``. A deployment that states no limits posture at all runs no limits
  checking, and nothing on that path refuses an unlisted channel.

A refusal about an unlisted channel — and the target switch's
``limits_posture`` refusal — names the key that answered, the per-type one where
a block spoke and the deployment-wide one where none did, so an operator edits
the line that decides rather than one it overrides. The ``channel_limits`` tool
reports the same pair for the target the session is on, including ``null`` where
nothing states an answer, alongside ``allow_unlisted_key`` naming the key it
read.

.. seealso::

   :class:`~osprey.connectors.control_system.ChannelValue`
       Channel read result data model

   :class:`~osprey.connectors.control_system.ChannelWriteResult`
       Complete write operation result

   :class:`~osprey.connectors.control_system.WriteOutcome`
       The six words a write can end in


Archiver Connectors
-------------------

An archiver connector answers the same questions about the past that a
control-system connector answers about the present.

.. versionchanged:: Unreleased

   ``get_data`` returns long-format data (below) instead of a shared-index wide
   ``DataFrame``. Out-of-tree connectors written against the old contract must be updated.

Subclass :class:`~osprey.connectors.archiver.base.ArchiverConnector` and implement
``connect``, ``disconnect``, ``get_data``, ``get_metadata``, ``check_availability``.

``get_data`` is the entire contract. It returns a **long-format** ``pandas.DataFrame``
with exactly three columns, sorted by ``channel`` then ``timestamp``:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Column
     - Dtype
     - Contents
   * - ``timestamp``
     - ``datetime64[ns, UTC]``
     - When the sample -- or, under a ``processing`` mode, the bin's aggregate --
       occurred.
   * - ``channel``
     - ``str``
     - The channel/PV name the row belongs to.
   * - ``value``
     - not dtype-constrained
     - ``float64`` when every requested channel's samples are numeric; pandas'
       natural mixed dtype (typically ``object``) once any channel is non-numeric.

An empty result is an empty frame with these same three columns.

**Nothing is manufactured.** Each channel contributes only its own real
samples -- never forward-filled, never reindexed onto a shared grid, never
padded with a row nothing was recorded at. A channel with no data in the range
contributes no rows; it never appears as an all-NaN column. If a custom
connector finds itself building a shared ``DatetimeIndex`` and reindexing
per-channel series onto it, that is the bug.

**Per-channel aggregation.** ``get_data`` takes ``processing: str = "raw"`` --
one of ``raw``, ``mean``, ``min``, ``max``, ``median``, ``std``, ``count`` --
applied independently to each channel's own samples:

- ``raw`` decimates each ``precision_ms`` bin to its **last real sample**,
  keeping that sample's own true timestamp (the Archiver Appliance's
  ``lastSample_N`` semantics).
- Every other mode aggregates the real samples in each bin. A bin with no
  samples is dropped, not emitted as ``NaN``.
- ``precision_ms <= 0`` means full resolution, valid only with
  ``processing="raw"`` -- anything else must raise ``ValueError``.
- Aggregating a non-numeric channel with anything but ``raw`` must raise
  ``ValueError`` naming the channel -- never coerce, drop, or emit ``NaN``.
- A bin width your backend cannot express must raise ``ValueError``, never
  round to one it can (the Archiver Appliance connector rejects a positive
  ``precision_ms`` that is not a multiple of 1000).

The shared helpers in ``osprey.connectors.archiver._timerange`` (``to_utc``,
``require_datetime``, ``resolve_processing``, ``long_frame``, ``decimate_raw``,
``aggregate_series``, ``reject_non_numeric``) implement all of the above --
every in-tree connector builds on them rather than reimplementing binning.

**The dtype rule for** ``value``\ **.** Enum/status channels carry string
values, and ``get_data`` never coerces them: a channel's own dtype flows
through, and only mixing non-numeric with numeric channels in one query promotes
the shared ``value`` column. Forcing ``value`` to ``float64`` "for consistency"
silently corrupts every enum/status channel. (This is deliberately not the live-read
contract, where an enum reads as its index with the label in ``enum_label`` --
an archiver reports what its backend recorded, and these backends recorded the
string.)

Query windows must be normalized to UTC before touching the wire: a naive
``start_date``/``end_date`` is facility-local and must be converted -- not
relabeled -- to your backend's UTC wire format; ``to_utc()`` does this.

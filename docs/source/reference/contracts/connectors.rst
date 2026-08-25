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
artifact gallery and reported as a summary plus a handle to it. A Channel Access
waveform takes exactly the same path as a pvAccess camera frame.

.. code-block:: yaml

   control_system:
     read_inline_max_elements: 2000        # per-value element budget
     channel_read_artifact_retention: 20   # readings kept per channel

``read_inline_max_elements`` (default 2000) is the per-value budget. The default
is the same inline budget OSPREY already plots against, which keeps ordinary
waveforms and orbit vectors inline while camera frames take the artifact path.
Only arrays are measured -- strings and single scalars are always inline,
however long they are.

One ``channel_read`` call also has an aggregate budget of four times that
number, spent in request order, so a batch of individually small arrays cannot
add up to a flood. Every withheld value says which of the two limits it hit, as
``artifact_reason``:

``per_value_threshold``
    The value on its own is over ``read_inline_max_elements``. It will never
    come back inline.

``aggregate_budget``
    The value is small enough by itself, but earlier channels in the same
    request had already spent the call's budget. Reading this channel in a
    smaller batch returns it inline.

The summary that replaces the value reports shape, dtype and element count, plus
min/max/mean for numeric data, along with the artifact's id and its
``data_file`` path. What gets saved depends on the shape:

- **1-D** -- an interactive chart in the artifact gallery, with the values as
  JSON in ``data_file``. The x axis is the sample index, not wall-clock time:
  one read of an array carries a single timestamp.
- **2-D and 3-D with a color axis** -- a PNG preview as the gallery image, with the raw array
  beside it as a ``.npy`` file (``data_file``). The preview is auto-scaled:
  brightness is relative to that frame's own minimum and maximum, so it carries
  no absolute units. ``data_file`` is the authoritative copy of the values.
- **Layouts with no honest rendering** -- four dimensions and up, or a 3-D stack
  with no color axis -- are still saved as a loadable ``.npy``, just without a
  preview image.

The agent reaches the numbers by loading ``data_file`` inside ``execute``:
``numpy.load(data_file)`` returns the array with its original shape and dtype,
and ``json.load`` opens a 1-D series file. If the artifact store cannot write at
all, the read is still reported as successful, with an ``artifact_error`` field
in place of the handle -- the machine did answer, and reporting the read as
failed would be a false alarm.

``channel_read_artifact_retention`` (default 20) bounds how far these readings
pile up: only the newest N **unpinned** artifacts per channel are kept, and
older ones are pruned as new readings are saved. Pinning a reading exempts it --
a pinned entry is never pruned and never occupies a slot in the window either.
Set the key to ``0`` to keep everything, remembering that an unattended polling
loop will then grow the gallery without bound.

Write Verification
------------------

All ``write_channel()`` calls return :class:`~osprey.connectors.control_system.base.ChannelWriteResult`:

.. code-block:: python

   connector = await ConnectorFactory.create_control_system_connector()

   result = await connector.write_channel("BEAM:CURRENT", 100.0)

   if result.verification and result.verification.verified:
       print(f"Write confirmed ({result.verification.level})")
   else:
       print(f"Verification failed: {result.verification.notes}")

   # Override verification level
   result = await connector.write_channel(
       "MOTOR:POSITION", 50.0,
       verification_level="readback",
       tolerance=0.1
   )

**Verification levels:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Level
     - Speed
     - Confidence
     - When to Use
   * - ``none``
     - Instant
     - Low
     - Development, non-critical writes
   * - ``callback``
     - Fast (~1-10ms)
     - Medium
     - Most production writes (default)
   * - ``readback``
     - Slow (~50-100ms)
     - High
     - Critical setpoints, safety-critical operations

**Configuration (global default):**

.. code-block:: yaml

   control_system:
     write_verification:
       default_level: "callback"
       default_tolerance_percent: 0.1   # interpreted as percent

**Per-channel configuration** (in limits database):

.. code-block:: json

   {
     "defaults": {
       "writable": true,
       "verification": { "level": "callback" }
     },
     "MOTOR:POSITION": {
       "min_value": -100.0,
       "max_value": 100.0,
       "max_step": 2.0,
       "writable": true,
       "verification": {
         "level": "readback",
         "tolerance_absolute": 0.1
       }
     }
   }

``tolerance_absolute`` takes priority over ``tolerance_percent`` (percentage of value).
Each channel inherits any field it does not set from the ``defaults`` block, and a
channel's own value always overrides it. ``writable`` defaults to ``true``; a channel's
verification falls back to the ``defaults`` block's verification. Set ``"writable": false``
-- on a channel, or in ``defaults`` to lock everything down by default -- to block writes.

**Where the level comes from.** Passing no ``verification_level`` means "no opinion",
not "no verification". The connector then resolves the level for that channel through
four layers, using the first one that supplies a value:

1. the channel's own ``verification`` entry in the limits database,
2. the limits database ``defaults.verification`` block,
3. ``control_system.write_verification`` in ``config.yml``,
4. the built-in fallback: ``callback``, with no tolerance.

The tolerance resolves through the same four layers, and it keeps doing so when the
level was passed explicitly, but a layer only supplies a tolerance when the level it
resolves to is ``readback``. Asking for ``readback`` on a channel whose limits entry
declares ``level: readback`` with ``tolerance_percent: 1.0`` verifies at 1%, not at
the built-in default -- pass ``tolerance`` as well to override that too.

Batch writes resolve the same way. ``write_multiple_channels()`` carries one level for
the whole batch, so when the caller omits it the keyword is left off each per-channel
write and every channel resolves its own. This is why ``osprey.runtime.write_channels``
verifies exactly like ``osprey.runtime.write_channel``.

.. _write-safety-config:

Write Safety Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Write operations are disabled by default and must be explicitly enabled at two levels:

**Global write permission** (in ``config.yml``):

.. code-block:: yaml

   control_system:
     writes_enabled: true          # Master switch for all write operations

If ``writes_enabled`` is omitted, it defaults to ``false`` and all writes are blocked.

``writes_enabled`` is a **launch-time deployment posture, not a live kill-switch.**
It is read from config and process-cached, so flipping it in ``config.yml`` does not
take effect in a running process. The enforced kill-switch lives at the harness layer
(a renderer ``permissions.deny`` on the write tool, then regenerate and relaunch the
agent); in-flight control of an active plan is the RunEngine's own ``abort`` / ``pause``.

The connector applies **per-write mechanical safety** — the ``writes_enabled`` gate,
limits validation, and the fail-closed validation path — on every Channel Access put.
This is a separate, complementary layer from the **per-intent human authorization**
enforced at the tool boundary (the PreToolUse approval hook, and the launch token for
plans), which gates the *intent* to write once per intent rather than once per put.
The approval layer cannot substitute for the connector's mechanical refusal.

.. _limits-checking-config:

Limits Checking
~~~~~~~~~~~~~~~

Automatic safety-limit validation for write operations:

.. code-block:: yaml

   control_system:
     limits_checking:
       enabled: true                     # Enable limits validation
       database_path: ./limits_db.json   # Path to the channel limits JSON
       allow_unlisted_channels: false    # Block writes to channels not in the database

When enabled, every ``write_channel()`` call is validated against the limits database
before the write is sent to hardware. See per-channel configuration above for the
database format.

.. seealso::

   :class:`~osprey.connectors.control_system.base.ChannelValue`
       Channel read result data model

   :class:`~osprey.connectors.control_system.base.ChannelWriteResult`
       Complete write operation result

   :class:`~osprey.connectors.control_system.base.WriteVerification`
       Verification result data model


Archiver Connectors
-------------------

An archiver connector answers the same questions about the past that a control-system
connector answers about the present, and it has its own contract for the shape and
typing of what it returns.

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

An empty result is an empty frame with these same three columns (``value`` defaults
to ``float64``, since there is no data to infer a dtype from).

**Nothing is manufactured.** Channels are never placed on a shared index. Each
channel contributes only its own real samples -- never forward-filled, never
reindexed onto a regular grid, never padded with a row for a bin or timestamp
nothing was actually recorded at. A channel with no data in the requested range
simply contributes no rows; it never appears as an all-NaN column. Connector
correctness bugs trace back to violating this rule, so hold to it strictly: if a
custom connector finds itself building a shared ``DatetimeIndex`` and reindexing
per-channel series onto it, that is the bug.

**Per-channel aggregation.** ``get_data`` takes a trailing ``processing: str =
"raw"`` keyword -- one of ``raw``, ``mean``, ``min``, ``max``, ``median``, ``std``,
``count`` -- applied independently to each channel's own real samples, never across
channels and never onto a shared grid:

- ``raw`` decimates each ``precision_ms`` bin down to its **last real sample**,
  keeping that sample's own true timestamp -- never a timestamp invented at the
  bin's edge to hold it. This matches the EPICS Archiver Appliance's long-standing
  ``lastSample_N`` semantics, and every in-tree backend applies it the same way.
- Every other mode aggregates the real samples that landed in each ``precision_ms``
  bin. A bin with no samples is dropped, not emitted as ``NaN`` -- so a sparse
  channel returns *fewer* rows than it has samples, never more, and no bin-width
  floor is ever needed to avoid upsampling.
- ``precision_ms <= 0`` means full resolution: every real sample, undecimated. It is
  only valid with ``processing="raw"`` -- an aggregate has no bin to aggregate over,
  and requesting one must raise ``ValueError`` rather than silently falling back to
  raw.
- Aggregating a non-numeric channel with anything but ``raw`` must raise
  ``ValueError`` naming the channel -- never coerce it, drop it, or silently emit
  ``NaN``. Backends that bin client-side get this from ``aggregate_series``; a
  backend that pushes the aggregation to its server must call
  ``reject_non_numeric`` on what comes back, since it never reaches
  ``aggregate_series``.
- A bin width your backend cannot express must raise ``ValueError``, never round
  to one it can. The EPICS Archiver Appliance's operator syntax takes whole
  seconds, so that connector rejects any positive ``precision_ms`` that is not a
  multiple of 1000 rather than serving a different resolution than was asked
  for.

The shared helpers in ``osprey.connectors.archiver._timerange`` (``to_utc``,
``require_datetime``, ``resolve_processing``, ``long_frame``, ``decimate_raw``,
``aggregate_series``, ``reject_non_numeric``)
implement all of the above and are the easiest way to get it right -- every in-tree
connector (EPICS, MongoDB, DOOCS, mock) builds on them rather than reimplementing
binning.

**Why the ``value`` dtype rule matters.** Enum/status channels -- machine mode,
interlock state, RF state, anything archived as EPICS ``mbbi`` or DOOCS
``DBR_STRING`` -- carry string values, not numbers. ``get_data`` never coerces them:
a channel's own dtype flows straight through, and only combining a non-numeric
channel with a numeric one in the same query promotes the shared ``value`` column to
a mixed dtype. A custom connector must resist forcing ``value`` to ``float64`` "for
consistency" -- doing so silently corrupts every enum/status channel it touches.

This is deliberately not the live-read contract, where an enum reads as its index
and the label travels in ``enum_label``. An archiver reports what its backend
recorded, and what these backends recorded is the string; converting one form into
the other would require a label list the archive does not carry.

Query windows must also be normalized to UTC before touching the wire: a naive
(timezone-less) ``start_date``/``end_date`` is facility-local, matching how the rest
of the framework reads operator wall-clock times, and must be converted -- not
relabeled -- to your backend's UTC wire format. ``to_utc()`` in
``osprey.connectors.archiver._timerange`` does this.

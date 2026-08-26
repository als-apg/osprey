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

Write Verification
------------------

All ``write_channel()`` calls return :class:`~osprey.connectors.control_system.base.ChannelWriteResult`:

.. code-block:: python

   result = await connector.write_channel("BEAM:CURRENT", 100.0)

   if result.verification and result.verification.verified:
       print(f"Write confirmed ({result.verification.level})")

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

**Global default** (in ``config.yml``) and **per-channel configuration** (in
the limits database):

.. code-block:: yaml

   control_system:
     write_verification:
       default_level: "callback"
       default_tolerance_percent: 0.1   # interpreted as percent

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

Each channel inherits any field it does not set from the ``defaults`` block,
and its own value always overrides it. ``writable`` defaults to ``true``; set
``"writable": false`` -- on a channel, or in ``defaults`` to lock everything
down -- to block writes. ``tolerance_absolute`` takes priority over
``tolerance_percent``.

**Where the level comes from.** Passing no ``verification_level`` means "no
opinion", not "no verification". The connector resolves the level through four
layers, first value wins: the channel's own limits entry, the limits
``defaults.verification`` block, ``control_system.write_verification``, then
the built-in fallback ``callback`` with no tolerance. The tolerance resolves
through the same four layers -- also when the level was passed explicitly --
but a layer only supplies a tolerance when the level it resolves to is
``readback``. ``write_multiple_channels()`` carries one optional level for the
whole batch; when omitted, every channel resolves its own, which is why
``osprey.runtime.write_channels`` verifies exactly like
``osprey.runtime.write_channel``.

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
never falls back. Both default to blocked when omitted, and **only a literal
``true`` arms writes** — the quoted string ``'true'`` and the number ``1`` do
not. A custom connector's block is keyed by the same dotted module path that
selects it, so ``mypackage.TangoConnector`` names one block and is never split
on its dots.

Write posture is a **launch-time deployment posture, not a live kill-switch**:
read from config and process-cached, so flipping it in ``config.yml`` does not
affect a running process. The enforced kill-switch lives at the harness layer
(a renderer ``permissions.deny`` on the write tool, then regenerate and
relaunch); in-flight control of an active plan is the RunEngine's own
``abort`` / ``pause``.

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

When enabled, every ``write_channel()`` call is validated against the limits
database before the write reaches hardware. The database format is the
per-channel configuration above.

.. seealso::

   :class:`~osprey.connectors.control_system.base.ChannelValue`
       Channel read result data model

   :class:`~osprey.connectors.control_system.base.ChannelWriteResult`
       Complete write operation result

   :class:`~osprey.connectors.control_system.base.WriteVerification`
       Verification result data model


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

**The ``value`` dtype rule.** Enum/status channels carry string values, and
``get_data`` never coerces them: a channel's own dtype flows through, and only
mixing non-numeric with numeric channels in one query promotes the shared
``value`` column. Forcing ``value`` to ``float64`` "for consistency" silently
corrupts every enum/status channel. (This is deliberately not the live-read
contract, where an enum reads as its index with the label in ``enum_label`` --
an archiver reports what its backend recorded, and these backends recorded the
string.)

Query windows must be normalized to UTC before touching the wire: a naive
``start_date``/``end_date`` is facility-local and must be converted -- not
relabeled -- to your backend's UTC wire format; ``to_utc()`` does this.

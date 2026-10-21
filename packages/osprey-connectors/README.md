# osprey-connectors

Lean control-system and archiver connectors for OSPREY: EPICS, DOOCS, and
mock/virtual-accelerator connectors for reading and writing control-system
channels, plus matching archiver connectors (EPICS Archiver Appliance,
DOOCS local history, MongoDB, mock) for historical data. The package also
carries the small set of support modules those connectors depend on —
configuration loading, logging setup, the error/exception taxonomy, and the
data-driven simulation core used by the mock connectors — without pulling in
the rest of the OSPREY framework (agents, capabilities, web UI, etc.).

## Install

```
pip install osprey-connectors
```

## Import map

These modules used to live under `osprey.*`. The old paths have been
removed; import from `osprey_connectors.*` instead. The table maps each old
path to its replacement.

| Old path (`osprey.*`) | New path (`osprey_connectors.*`) |
| --- | --- |
| `osprey.connectors` | `osprey_connectors` |
| `osprey.connectors.factory` | `osprey_connectors.factory` |
| `osprey.connectors.types` | `osprey_connectors.types` |
| `osprey.connectors.honesty` | `osprey_connectors.honesty` |
| `osprey.connectors.control_system` | `osprey_connectors.control_system` |
| `osprey.connectors.control_system.base` | `osprey_connectors.control_system.base` |
| `osprey.connectors.control_system.epics_connector` | `osprey_connectors.control_system.epics_connector` |
| `osprey.connectors.control_system.doocs_connector` | `osprey_connectors.control_system.doocs_connector` |
| `osprey.connectors.control_system.va_in_process_connector` | `osprey_connectors.control_system.va_in_process_connector` |
| `osprey.connectors.control_system.va_connector` | `osprey_connectors.control_system.va_connector` |
| `osprey.connectors.control_system.tango_connector` | `osprey_connectors.control_system.tango_connector` |
| `osprey.connectors.control_system.limits_validator` | `osprey_connectors.control_system.limits_validator` |
| `osprey.connectors.archiver` | `osprey_connectors.archiver` |
| `osprey.connectors.archiver.base` | `osprey_connectors.archiver.base` |
| `osprey.connectors.archiver.epics_archiver_connector` | `osprey_connectors.archiver.epics_archiver_connector` |
| `osprey.connectors.archiver.doocs_archiver_connector` | `osprey_connectors.archiver.doocs_archiver_connector` |
| `osprey.connectors.archiver.mongodb_archiver_connector` | `osprey_connectors.archiver.mongodb_archiver_connector` |
| `osprey.connectors.archiver.mock_archiver_connector` | `osprey_connectors.archiver.mock_archiver_connector` |
| `osprey.connectors.archiver.mya_archiver_connector` | `osprey_connectors.archiver.mya_archiver_connector` |
| `osprey.connectors.archiver._timerange` | `osprey_connectors.archiver._timerange` |
| `osprey.errors` | `osprey_connectors.errors` |
| `osprey.utils.config` | `osprey_connectors.config` |
| `osprey.utils.dotenv` | `osprey_connectors.dotenv` |
| `osprey.utils.identity` | `osprey_connectors.identity` |
| `osprey.utils.workspace` | `osprey_connectors.workspace` |
| `osprey.utils.logger` | `osprey_connectors.logger` |
| `osprey.utils.log_filter` | `osprey_connectors.log_filter` |
| `osprey.utils.relative_time` | `osprey_connectors.relative_time` |

## Optional runtime dependencies

Only the connectors you actually use need their backing libraries installed;
none of the following are declared as hard dependencies of this package, so
importing `osprey_connectors` never requires them:

- **EPICS Channel Access** — `EPICSConnector` speaks Channel Access through
  `aioca`, a declared dependency, which loads the per-architecture `libca`
  that `epicscorelibs` ships. Nothing else is needed to talk to real IOCs.
- **`pymongo`** — required by `MongoDBArchiverConnector`. Installed with this
  package; the connector imports it inside `connect()` so registration stays
  cheap, not because it is optional.
- **`doocs4py`** — required by `DOOCSConnector` and `DOOCSArchiverConnector`.
  Install separately (it is not on PyPI in all environments) if you connect
  to a DOOCS control system.

## Stability

`osprey-connectors` follows semantic versioning starting at `0.1.0`.

The exception taxonomy in `osprey_connectors.errors` is public API: the class
names `ChannelWriteBlockedError`, `ChannelWriteFailedError`,
`ChannelLimitsViolationError` and `ChannelNotFoundError`, and their `reason`
codes, are load-bearing for callers that branch on them.

- `ChannelWriteBlockedError` — the write was refused and no value was written.
  `reason` is one of `WRITES_DISABLED`, `LIMITS`, `VALIDATION_ERROR`,
  `CONTROL_SYSTEM_REFUSED`, `RAW_CLIENT_WRITE`, or `CHANNEL_NOT_FOUND`. The
  first three are refused before any `caput`; `CONTROL_SYSTEM_REFUSED` means
  the control system was asked and denied the write itself (EPICS access
  security, for instance); `RAW_CLIENT_WRITE` means user code called a client
  library's put directly instead of `osprey.runtime.write_channel`; and
  `CHANNEL_NOT_FOUND` means the address is not a channel of the facility.
  Nothing was sent in any of these cases except `CONTROL_SYSTEM_REFUSED`.
- `ChannelWriteFailedError` — the write was attempted but did not verifiably
  succeed. `reason` is one of `FAILED` (the control system did not take the
  value), `MISMATCH` (the channel holds a different value), or `UNCONFIRMED`
  (sent, but not acknowledged in time or the confirming re-read itself
  failed). The codes are the `WriteOutcome` words that
  raise; the exception also carries `outcome`, `value_written` and
  `observed_value`.
- `ChannelLimitsViolationError` — a channel write violated configured safety
  limits (range, read-only, step size, or unlisted channel).
- `ChannelNotFoundError` — a read named an address that is not a channel of the
  facility. Only a connector that holds the facility's whole address set raises
  it; `reason` is `CHANNEL_NOT_FOUND`, the refusal reason a write to such an
  address carries.

Removing or renaming any of these classes or reason codes is a major version
bump. Adding new classes or reason codes is a minor version bump.

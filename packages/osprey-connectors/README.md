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

`osprey-framework` still imports every one of these from their historical
`osprey.*` paths. Those old paths are compatibility shims — the real code
now lives under `osprey_connectors.*`, and new code should import from there
directly.

| Old path (`osprey.*`) | New path (`osprey_connectors.*`) |
| --- | --- |
| `osprey.connectors` | `osprey_connectors` |
| `osprey.connectors.factory` | `osprey_connectors.factory` |
| `osprey.connectors.types` | `osprey_connectors.types` |
| `osprey.connectors.channel_taxonomy` | `osprey_connectors.channel_taxonomy` |
| `osprey.connectors.control_system` | `osprey_connectors.control_system` |
| `osprey.connectors.control_system.base` | `osprey_connectors.control_system.base` |
| `osprey.connectors.control_system.epics_connector` | `osprey_connectors.control_system.epics_connector` |
| `osprey.connectors.control_system.doocs_connector` | `osprey_connectors.control_system.doocs_connector` |
| `osprey.connectors.control_system.mock_connector` | `osprey_connectors.control_system.mock_connector` |
| `osprey.connectors.control_system.va_connector` | `osprey_connectors.control_system.va_connector` |
| `osprey.connectors.control_system.limits_validator` | `osprey_connectors.control_system.limits_validator` |
| `osprey.connectors.archiver` | `osprey_connectors.archiver` |
| `osprey.connectors.archiver.base` | `osprey_connectors.archiver.base` |
| `osprey.connectors.archiver.epics_archiver_connector` | `osprey_connectors.archiver.epics_archiver_connector` |
| `osprey.connectors.archiver.doocs_archiver_connector` | `osprey_connectors.archiver.doocs_archiver_connector` |
| `osprey.connectors.archiver.mongodb_archiver_connector` | `osprey_connectors.archiver.mongodb_archiver_connector` |
| `osprey.connectors.archiver.mock_archiver_connector` | `osprey_connectors.archiver.mock_archiver_connector` |
| `osprey.errors` | `osprey_connectors.errors` |
| `osprey.utils.config` | `osprey_connectors.config` |
| `osprey.utils.logger` | `osprey_connectors.logger` |
| `osprey.utils.relative_time` | `osprey_connectors.relative_time` |
| `osprey.simulation` | `osprey_connectors.simulation` |
| `osprey.simulation.engine` | `osprey_connectors.simulation.engine` |
| `osprey.simulation.expressions` | `osprey_connectors.simulation.expressions` |
| `osprey.simulation.machine` | `osprey_connectors.simulation.machine` |
| `osprey.simulation.series` | `osprey_connectors.simulation.series` |

## Optional runtime dependencies

Only the connectors you actually use need their backing libraries installed;
none of the following are declared as hard dependencies of this package, so
importing `osprey_connectors` never requires them:

- **EPICS Channel Access client libraries** — `pyepics` itself is a declared
  dependency, but it needs a working `libca` at runtime to talk to real IOCs.
  Install `epicscorelibs` (or otherwise make a per-architecture `libca`
  available) if you use `EPICSConnector` or `EPICSArchiverConnector` against a
  live control system; `PYEPICS_LIBCA` can also point at one explicitly.
- **`pymongo`** — required by `MongoDBArchiverConnector`. Installed with this
  package; the connector imports it inside `connect()` so registration stays
  cheap, not because it is optional.
- **`doocs4py`** — required by `DOOCSConnector` and `DOOCSArchiverConnector`.
  Install separately (it is not on PyPI in all environments) if you connect
  to a DOOCS control system.

## Stability

`osprey-connectors` follows semantic versioning starting at `0.1.0`.

The exception taxonomy in `osprey_connectors.errors` is public API: the class
names `ChannelWriteBlockedError`, `ChannelWriteFailedError`, and
`ChannelLimitsViolationError`, and their `reason` codes, are load-bearing for
callers that branch on them.

- `ChannelWriteBlockedError` — the write was never attempted (refused before
  any `caput`). `reason` is one of `WRITES_DISABLED`, `LIMITS`,
  `VALIDATION_ERROR`.
- `ChannelWriteFailedError` — the write was attempted but did not verifiably
  succeed. `reason` is one of `CAPUT_FAILED`, `READBACK_UNVERIFIED`.
- `ChannelLimitsViolationError` — a channel write violated configured safety
  limits (range, read-only, step size, or unlisted channel).

Removing or renaming any of these classes or reason codes is a major version
bump. Adding new classes or reason codes is a minor version bump.

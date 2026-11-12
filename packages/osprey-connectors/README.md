# osprey-connectors

Lean connectors to a control system and the read-only services beside it: a
control-system connector reads and writes the channels of one machine; an
archiver reads their history; a channel directory, an alarm-state feed, an
alarm log and a put log answer what those services know. Every connector of a
kind answers the same API, so the code above them does not change when the
machine behind them does. The package also carries what those connectors need —
configuration loading, logging, the error taxonomy, the write-safety records and
the simulation core behind the virtual accelerator — without the rest of the
OSPREY framework (agents, capabilities, web UI, etc.).

Documentation: <https://als-apg.github.io/osprey/latest/how-to/control-systems/use-connectors.html>

## Install

```
pip install osprey-connectors
```

[Use the Connectors Without the Framework](https://als-apg.github.io/osprey/latest/how-to/control-systems/use-connectors-standalone.html)
reads a control system and its archiver from a script of your own, and says
what needs a deployment.

## Connectors

A connector is built by `ConnectorFactory` from its registered type name, after
`register_builtin_connectors()` has registered the built-in ones:

| Kind | Type name | Talks to |
| --- | --- | --- |
| control system | `epics` | EPICS IOCs over Channel Access, and pvAccess for the channels listed under `pva_channels` |
| control system | `virtual_accelerator` | OSPREY's simulated machine, in process or served over Channel Access (needs a facility built by `osprey build`) |
| control system | `live_standin` | The facility-shaped soft IOC a deployment runs for itself, over Channel Access |
| control system | `doocs` | DOOCS servers (needs `doocs4py`) |
| control system | `tango` | Tango device servers (needs `pytango`) |
| archiver | `epics_archiver` | The EPICS Archiver Appliance |
| archiver | `mongodb_archiver` | A deployment's own MongoDB time-series store |
| archiver | `doocs_archiver` | DOOCS local history (needs `doocs4py`) |
| archiver | `mya_archiver` | JLab's MYA, over the myquery HTTP service (needs `jlab-archiver-client`) |
| archiver | `mock_archiver` | The simulated machine's history (needs a facility built by `osprey build`) |
| directory | `channelfinder` | A ChannelFinder directory service |
| alarm_state | `phoebus_alarm_server` | A Phoebus alarm server's Kafka topic (needs `aiokafka`) |
| alarm_log | `phoebus_alarm_logger` | A Phoebus alarm logger |
| put_log | `caputlog` | A caPutLog put-history service |

A type name that holds a dot, such as `my_site.connectors.MyConnector`, is
imported as a module path and class name, so a connector of your own needs no
registration.

How to configure each one is in the
[Use Connectors](https://als-apg.github.io/osprey/latest/how-to/control-systems/use-connectors.html)
guide, and what every connector must do — reading large values, confirming a
write, the safety gates a write passes — in the
[connector contracts](https://als-apg.github.io/osprey/latest/reference/contracts/connectors.html).

## Public API

Every public module lists its public names in `__all__`, and a module whose
name starts with an underscore is internal and can change in any release. The
[API reference](https://als-apg.github.io/osprey/latest/reference/connectors-api/index.html)
documents every one of those names, grouped by area.

Most modules are their own API, each imported by its own path:

| Area | Modules |
| --- | --- |
| Building connectors | `factory` (the factory and the built-in registry), `types` (the type names), `connection` (an outbound connection block's address, bound, login and CA), `call_timeout` (a connector call's bound) |
| Configuration | `config`, `workspace` (where a process finds its config and agent data), `dotenv` (the deployment's env chain), `yaml_loader` |
| Errors and logging | `errors`, `logger`, `log_filter` |
| Writing safely | `posture_env` (the read-only run), `posture_store` (the write switch), `write_authorization` (a program's bounded write authorization), `refusals` (why a write was refused and who can lift it), `session_hold` (a session whose writes are held), `control_context`, `write_target` (which machine a write tool writes to), `settle` (waiting for a readback to settle) |
| The deployment's machines | `simulator` (its virtual accelerator), `standin` (its stand-in soft IOC), `archive_targets` (which machines its archive serves), `honesty` (the one connector pairing a deployment may not have) |
| Supervising connector processes | `process` (starting and stopping child processes), `recovery` (what to do about a failed connector call) |
| Read-only services | `service_answers` (what every service backend answers), `service_targets` (which source answers each kind on each target) |
| Utilities | `identity` (who an audit record names), `relative_time`, `safety_file` (replacing a file whole) |

So are the `simulation` package and its modules (`simulation.state`,
`simulation.view`, `simulation.archive`, ...): its root imports neither numpy
nor lume, and the heavier modules are imported by their own names.

Seven subpackages gather their implementation modules behind one namespace,
and their names are imported from the subpackage itself:

| Namespace | What it holds |
| --- | --- |
| `osprey_connectors.control_system` | The control-system connector contract (`ControlSystemConnector`, `ChannelValue`, `ChannelWriteResult`, `WriteOutcome`, `values_match`, ...) and `LimitsValidator` |
| `osprey_connectors.archiver` | The archiver connector contract (`ArchiverConnector`, `ArchiverMetadata`), the shared time-range helpers, the field name a channel is stored under, the MongoDB client arguments |
| `osprey_connectors.ipc` | What a connector-host supervisor needs: launching and stopping a child, the proxy and the pool, endpoint derivation and verification, and the errors a frame raises (the wire codec itself is internal) |
| `osprey_connectors.directory` | The channel directory contract (`ChannelDirectory`, `DirectoryQuery`, `DirectoryPage`, ...) |
| `osprey_connectors.alarm_state` | The alarm-state feed contract (`AlarmState`, `AlarmStateRecord`, `AlarmConfigRecord`) |
| `osprey_connectors.alarm_log` | The alarm log contract (`AlarmLog`, `AlarmHistoryRow`) |
| `osprey_connectors.put_log` | The put log contract (`PutLog`, `PutRow`) |

`osprey_connectors.control_system.write_door`, the door a connector opens around
its own write, is a public module of its own.

The read-only run — `POSTURE_ENV_VAR`, `SANDBOX_MODE` and `is_readonly_run()` —
lives in `osprey_connectors.posture_env`, a leaf that imports nothing from the
package. `osprey_connectors.control_system` exported `is_readonly_run` before
2026.10.0 and no longer does.

The concrete connectors are internal (`control_system._epics_connector`,
`archiver._mongodb_archiver_connector`, ...): a caller builds one through
`ConnectorFactory` from its registered type name.

`osprey-framework` imports these subpackages only through their namespaces, and
takes from every module only names its `__all__` lists. `lint-imports`
(import-linter, configured in the repository's root `pyproject.toml`) holds the
first, and ruff's `PLC2701` and `tests/connectors/test_public_api.py` the
second. import-linter also holds this package to never importing `osprey`, and
holds every top-level module of it in layers: the connector host above the
factory, the factory above the archivers, the archivers above the
control-system connectors, those above the simulation core, and the posture
store, config loading and the bottom layer (`posture_env`, `types`,
`identity`, ...) below them all.

## Import map

These modules used to live under `osprey.*`. The old paths have been
removed; the table maps each old path to the public spelling that replaces it.

| Old path (`osprey.*`) | Import from (`osprey_connectors.*`) |
| --- | --- |
| `osprey.connectors` | `osprey_connectors` |
| `osprey.connectors.factory` | `osprey_connectors.factory` |
| `osprey.connectors.types` | `osprey_connectors.types` |
| `osprey.connectors.honesty` | `osprey_connectors.honesty` |
| `osprey.connectors.control_system`, `.control_system.base`, `.control_system.limits_validator` | `osprey_connectors.control_system` |
| `osprey.connectors.control_system.*_connector` | built through `osprey_connectors.factory.ConnectorFactory` |
| `osprey.connectors.archiver`, `.archiver.base` | `osprey_connectors.archiver` |
| `osprey.connectors.archiver.*_archiver_connector` | built through `osprey_connectors.factory.ConnectorFactory` |
| `osprey.errors` | `osprey_connectors.errors` |
| `osprey.utils.config` | `osprey_connectors.config` |
| `osprey.utils.dotenv` (reading the env chain) | `osprey_connectors.dotenv` |
| `osprey.utils.identity` | `osprey_connectors.identity` |
| `osprey.utils.workspace` (config and agent-data resolution) | `osprey_connectors.workspace` |
| `osprey.utils.logger` | `osprey_connectors.logger` |
| `osprey.utils.log_filter` | `osprey_connectors.log_filter` |
| `osprey.utils.relative_time` | `osprey_connectors.relative_time` |

Not everything those modules held moved here. The deployment repo's layout
(layer directories, the image staging area, the audit and archive zones)
belongs to the framework, in a new `osprey.utils.workspace` that holds only
those names. The executor's vocabulary and the
facility timestamp formatter are the framework's `osprey.utils.execution_method`
and `osprey.utils.facility`, and writing `.env` files is
`osprey.deployment.dotenv`.

## Dependencies

Installing the package installs what every built-in connector except three
needs: `aioca` for Channel Access and `p4p` for pvAccess (both on the one EPICS
base `epicscorelibs` ships, so nothing else is needed to talk to real IOCs),
`pymongo` for `mongodb_archiver`, numpy and pandas, and `lume-base` for the
simulated machine's models. `pymongo` is imported inside `connect()`, which
keeps registration cheap; it is not optional.

Three connectors need a client library this package does not declare, because
only a site that runs that control system or archiver wants it. Each is imported
when the connector connects, never when the package is imported or the connector
registered, so a machine without it finds out only when it uses that connector:

- **`doocs4py`** — `doocs` and `doocs_archiver`. The DOOCS environment supplies
  it.
- **`pytango`** — `tango`.
- **`jlab-archiver-client`** — `mya_archiver`.

## Stability

`osprey-connectors` versions by calendar, `YYYY.M.P` (with `bN` / `rcN` for
pre-releases), on its own release tags: `osprey-connectors-vYYYY.M.P`, cut when
the package changes rather than with every framework release. Its numbers are
not the framework's, and no framework release requires a matching one. (The
`0.1.0` release on PyPI predates this scheme; its tag is retired.)

Each release stays backward-compatible with every supported `osprey-framework`
release, and CI enforces that in its `connectors-compat` and
`framework-at-connectors-floor` lanes. The framework therefore declares only a
floor on this package, never a pin.

The exception taxonomy in `osprey_connectors.errors` is public API: the class
names `ChannelError`, `ChannelWriteBlockedError`, `ChannelWriteFailedError`,
`ChannelLimitsViolationError` and `ChannelNotFoundError`, and their `reason`
codes, are load-bearing for callers that branch on them.

- `ChannelError` — the base of every failure about one channel; the classes
  below and `ChannelUnreachableError` are one. `channel_address` names the
  channel, in process and across the connector host.
- `ChannelWriteBlockedError` — the write was refused and no value was written.
  `reason` is a `RefusalReason`, one of `WRITES_DISABLED`, `LIMITS`,
  `VALIDATION_ERROR`, `CONTROL_SYSTEM_REFUSED`, `RAW_CLIENT_WRITE`, or
  `CHANNEL_NOT_FOUND`; each member names who refused (`refuser`) and what to
  do about it (`remedy`). The first three are refused before any `caput`;
  `CONTROL_SYSTEM_REFUSED` means
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

Adding classes or reason codes is compatible. Removing or renaming one breaks
the compatibility guarantee above, so it waits until no supported framework
release relies on it.

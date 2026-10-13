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

## Public API

Every public module lists its public names in `__all__`. Most modules are their
own API: `config`, `workspace`, `dotenv`, `yaml_loader`, `connection`,
`types`, `factory`, `errors`, `logger`, `log_filter`, `identity`,
`relative_time`, `channel_taxonomy`, `control_context`, `posture_store`,
`standin` and `honesty`, each imported by its own path.

Four subpackages gather implementation modules behind one namespace, and their
names are imported from the package itself:

| Namespace | What it holds |
| --- | --- |
| `osprey_connectors.control_system` | The control-system connector contract (`ControlSystemConnector`, `ChannelValue`, `ChannelWriteResult`, `WriteOutcome`, `values_match`, ...), call timeouts, `LimitsValidator` |
| `osprey_connectors.archiver` | The archiver connector contract (`ArchiverConnector`, `ArchiverMetadata`), the shared time-range helpers, the MongoDB client arguments |
| `osprey_connectors.simulation` | `SimulationEngine`, the machine model, the deploy-time archive seed |
| `osprey_connectors.ipc` | What a connector-host supervisor needs: frames, launch, proxy, endpoint verification |

`osprey_connectors.control_system.write_door`, the door a connector opens around
its own write, is a public module of its own.

A module whose name starts with an underscore is internal and can change in any
release. The concrete connectors are internal (`control_system._epics_connector`,
`archiver._mongodb_archiver_connector`, ...): a caller builds one through
`ConnectorFactory` from its registered type name.

`osprey-framework` imports these packages only through their namespaces.
`lint-imports` (import-linter, configured in the repository's root
`pyproject.toml`) holds it to that, holds this package to never importing
`osprey`, and holds the package's layers: the connector host above the factory,
the factory above the control-system and archiver connectors, and those above
the simulation engine.

## Historical paths

When these modules moved here from `osprey-framework`, each kept answering
under its old `osprey.*` path. These still do, as the same module object:

| Path (`osprey.*`) | Is (`osprey_connectors.*`) |
| --- | --- |
| `osprey.connectors.factory` | `osprey_connectors.factory` |
| `osprey.connectors.types` | `osprey_connectors.types` |
| `osprey.connectors.channel_taxonomy` | `osprey_connectors.channel_taxonomy` |
| `osprey.connectors.honesty` | `osprey_connectors.honesty` |
| `osprey.connectors.control_system.base`, `.limits_validator` | `osprey_connectors.control_system` |
| `osprey.connectors.archiver.base`, `._timerange` | `osprey_connectors.archiver` |
| `osprey.errors` | `osprey_connectors.errors` |
| `osprey.utils.config`, `.workspace`, `.dotenv`, `.logger`, `.log_filter`, `.identity`, `.relative_time` | the module of the same name |

`osprey.connectors`, `osprey.connectors.control_system` and
`osprey.connectors.archiver` re-export names from the matching namespace. The
old paths of the concrete connectors (`osprey.connectors.control_system.*_connector`,
`osprey.connectors.archiver.*_archiver_connector`) and of the simulation
internals (`osprey.simulation.engine`, `.machine`, `.series`, `.expressions`,
`.procedural`, `.archiver_seed`) are gone.

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
names `ChannelWriteBlockedError`, `ChannelWriteFailedError`, and
`ChannelLimitsViolationError`, and their `reason` codes, are load-bearing for
callers that branch on them.

- `ChannelWriteBlockedError` — the write was refused and no value was written.
  `reason` is one of `WRITES_DISABLED`, `LIMITS`, `VALIDATION_ERROR`, or
  `CONTROL_SYSTEM_REFUSED`. The first three are refused before any `caput`; the
  last means the control system was asked and denied the write itself (EPICS
  access security, for instance).
- `ChannelWriteFailedError` — the write was attempted but did not verifiably
  succeed. `reason` is one of `FAILED` (the control system did not take the
  value), `MISMATCH` (the channel holds a different value), or `UNCONFIRMED`
  (sent, but not acknowledged in time or the confirming re-read itself
  failed). The codes are the `WriteOutcome` words that
  raise; the exception also carries `outcome`, `value_written` and
  `observed_value`.
- `ChannelLimitsViolationError` — a channel write violated configured safety
  limits (range, read-only, step size, or unlisted channel).

Removing or renaming any of these classes or reason codes is a major version
bump. Adding new classes or reason codes is a minor version bump.

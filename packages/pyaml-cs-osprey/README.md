# pyaml-cs-osprey

A pyAML (`accelerator-middle-layer`) control-system backend for OSPREY. It
implements pyAML's `ControlSystem`, `DeviceAccess` and `DeviceAccessList` on
top of `osprey.runtime`, so every pyAML read and write passes through the
OSPREY `ControlSystemConnector`: channel limits, write posture, control-target
pinning, write verification and audit all apply. pyAML's native backends,
which open their own control-system clients, are not used.

The package ships with `osprey-framework`, which depends on it; it is not meant
to be installed on its own.

## Selecting the backend

In a pyAML configuration:

```yaml
controls:
  - type: pyaml_cs_osprey.controlsystem
    name: live
```

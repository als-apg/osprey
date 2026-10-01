The EPICS connector now uses pvapy (`pvaccess`) for both Channel Access and
PVAccess, and `pyepics` and `p4p` are no longer dependencies of
`osprey-connectors`. pvapy's wheels carry their own EPICS libraries, including
for linux/aarch64, so a bare-metal arm64 install no longer compiles the EPICS
client stack. Every Channel Access read now goes to the IOC, so there is no
monitor cache to go stale and no `fresh_reads` option to bypass one; write
confirmation still waits for the IOC's put-callback. Python 3.14 is not
supported until pvapy publishes wheels for it: OSPREY now requires Python 3.11,
3.12 or 3.13.
What else changes for callers:

- `raw_metadata`: `nt_id` is replaced by `nt_type` (`NTEnum`, `NTNDArray` or
  `None`); a new `provider` key names the protocol the read went over (`ca` or
  `pva`); a Channel Access `status` is now the normative alarm status rather
  than the CA status code; and Channel Access reads no longer report `type` or
  `count`.
- A timeout on an unreachable channel surfaces as `ConnectionError`, since
  pvapy reports a channel that never connects and one that times out the same
  way.
- Numeric writes are exact and typed to the channel: a fractional value
  written to an integer channel is refused rather than rounded.
- A write to an unreachable channel returns `FAILED` with "nothing was sent".
- Only one confirming write per channel is in flight at a time; a retry waits
  for the earlier one within its own deadline.
- A char waveform takes text on write and reads back as unsigned bytes.
- pvapy reads the `EPICS_CA_*` / `EPICS_PVA_*` environment once per process, at
  its first channel. A notebook kernel therefore no longer builds its EPICS
  connector in-process: `osprey.runtime` serves Channel Access targets from a
  connector-host child per target, so a kernel still follows a control-target
  switch on its next cell without a restart (the old target's child is stopped
  on the switch, and every child when the kernel exits). A write whose child is
  lost mid-call raises `ChannelWriteFailedError` (`UNCONFIRMED`). Any other
  process that reconnects the EPICS connector to a different endpoint is
  refused with `ControlTargetUnreachableError`, naming both endpoints, instead
  of silently talking to the previous gateway.
- The `max_step` read made by the runtime's limits check runs on the
  connector's worker threads, so a notebook cell no longer hangs on macOS.

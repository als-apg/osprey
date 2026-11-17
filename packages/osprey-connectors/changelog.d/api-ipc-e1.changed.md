Where a failure on a connector-host call came from is its class.
`osprey_connectors.errors.ChildLinkError` is the base of what the parent's
link to a child raises on its own: `ipc.ChildUnresponsiveError` and
`ipc.ConnectorHostSlowCallError` (both still `TimeoutError`s, never
`ConnectionError`s) and the new `ipc.ChildLostError`, a
`ConnectorUnavailableError` the proxy raises in place of a bare
`ConnectionError` whenever its pipe ends or it refuses new calls. Any other
exception a call raises is the one the child sent. `ipc.raised_by_child` is
gone: write `not isinstance(exc, ChildLinkError)`, or
`isinstance(exc, ChildLostError)` for a pipe that ended.
`ConnectorHostSlowCallError` is now exported from `osprey_connectors.ipc` and
its `recovery` may be passed positionally.

A reply of the wrong type now fails the call with `ConnectorInternalError`
(its `class_tag` names the reply's type) instead of `FrameDecodeError`, as an
unrebuildable reply already did.

The pool's key is `ipc.HostKey` (`target`, `execution_mode`, `owner`,
`authorization`): `ConnectorHostPool.pids()` is `dict[HostKey, int]`, and
`PooledConnector.key` replaces its `target`, `execution_mode`, `owner` and
`authorization` properties (read `handle.key.target` and so on).
`execution_mode` is `osprey_connectors.execution_mode.ExecutionMode | None`
(the word `"readonly"`); `ipc.READONLY` is gone, write `"readonly"`.
`ConnectorHostStartError.phases` holds `ipc.StartMark` records (`phase`,
`after_s`) instead of `(name, seconds)` tuples.

The IPC frames are unchanged on the wire. Inside the package an error frame
carries an `ExceptionFrameBody` (`ErrorFrame.error`), which `decode_exception`
rebuilds; `reconstruct_exception` and `exception_fields` are gone, and
`ExceptionSpec.fields` is `attributes`. A connector's own subclass of
`ChannelLimitsViolationError` or `ChannelNotFoundError` now sends its own text
in the frame's `message`, which the parent never reads. A child handed a
first frame that is not `init` answers with a `ConnectorInternalError`
instead of a `ConnectionError`.

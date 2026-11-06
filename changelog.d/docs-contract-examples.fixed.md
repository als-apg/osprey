The contract reference pages' examples now match the code, and a test loads
each one with the code it describes. The ARIEL service-factory example no
longer wraps the shared service in `async with`, which closed it for every
later caller in the process; it calls `close_ariel_service()` at shutdown
instead. The health report example's counts now agree with its rows, and the
page's `jq` recipes run over it. The hierarchical Channel Finder example has a
tree that `osprey channel-finder validate` accepts. The Python executor
parameter tables list the hook-filled `approved_target` and
`approved_view_sha256`.

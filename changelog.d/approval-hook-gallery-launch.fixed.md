The approval hook no longer starts an artifact gallery server from its own
short-lived process when it saves the pre-execution review notebook. The
notebook still lands in the shared store, where a running gallery shows it,
and the hook process exits cleanly instead of racing the server thread's
teardown.

The EPICS connector's channel-existence probe now uses the `timeout` the
deployment configured for it, like its reads and writes already did, instead of
a fixed 2 seconds. A facility on a slow or distant gateway can raise the one
key and have every probe follow.

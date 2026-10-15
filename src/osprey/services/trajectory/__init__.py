"""The trajectory threading service: one target, one connector, one gated app.

The service drives the injection and monitor-arming commands of one model's
``trajectory`` measurement on one control target. :mod:`.startup` decides
whether it may start at all and what it commands.
"""

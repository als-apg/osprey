`control_system.connector.<type>.fresh_reads: true` makes every Channel Access
read of an EPICS-family connector ask the IOC (`use_monitor=False`) instead of
answering from pyepics' monitor cache. An IOC that computes its readbacks on get
posts no monitor update, so a cached read of it never changes. The option is per
block, so a stand-in simulator can have it while the live machine does not.

`ChannelDirectory.list_servers(deadline)` lists every server the directory knows,
one `osprey_connectors.directory.ServerRow(name, host, channel_count, inactive_count)`
each. The list is complete or `Unavailable`, never partial. It is not abstract:
a backend that does not list keeps the default, which answers `Unavailable`.

An EPICS write to a channel that is not yet connected now waits for the
connection as long as the call's own timeout, as a read does, instead of
pyepics' 5 s default, so a busy IOC that answers the search late no longer
fails the write before anything is sent.

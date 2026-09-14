A file the web terminal's file panel or the artifact gallery would have missed
when the operating system's change notifications went quiet is now picked up
within a couple of seconds. Each watcher re-reads the directories it tracks on
an interval and announces what the notifications did not; set that interval with
`web.file_watch_reconcile_interval_s`.

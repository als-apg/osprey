`validate_plan` no longer reports the Bluesky bridge unreachable while the
bridge is still running the dry run it asked for. The request now waits for
the dry run, its grace and the upload of a pass, so a plan whose dry run takes
longer than fifteen seconds is reported as it was judged instead of failing
on the client while the bridge passes and uploads it. A `dry_run_timeout`
past 300 seconds is refused, by the tool and by the bridge. The plan-authoring
skill now says where a plan's channels come from: the queue worker's devices,
not every address the channel finder knows.

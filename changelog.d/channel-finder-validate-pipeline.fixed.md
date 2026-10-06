`osprey channel-finder validate --pipeline <paradigm>` without `--database`
now validates the file at `channel_finder.pipelines.<paradigm>.database.path`
instead of the auto-detected paradigm's database, and refuses, naming that
key, when it is unset.

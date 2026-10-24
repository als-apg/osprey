**Breaking change:** every write surface now decides by the one write verdict.
A host process refuses a write for an owner other than its login. A read-only
run on a machine with only a write gateway refuses to connect. Scheduled and
triggered dispatch jobs with no owner, and Bluesky items with no owner, no
longer write. A live lane writes only when its live connector block is
configured. A facility server guarded whole by `writes_check` refuses its reads
too while `live` is off.

**Breaking change:** every write surface now decides by the one write verdict.
A host process refuses a write for an owner other than its login. A live lane
writes only when its live connector block is configured. A facility server
guarded whole by `writes_check` refuses its reads too while `live` is off.

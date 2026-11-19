A read-only service backend may `GET` one percent-encoded segment under an
allowed `/*` prefix, which is how a channel name travels as a path segment.
The segment is refused unless it decodes to UTF-8 holding no `/`, `\` or
control character and is neither `.` nor `..`.

The web terminal opens one event stream per page instead of three. Every
module that listened on the file-events stream held its own connection, which
left a hub tab one short of the browser's six-per-host HTTP/1.1 limit; a second
tab or one more streaming panel then queued every later request indefinitely.
The subscribers now share a single connection.

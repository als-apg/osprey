Each user's bar arrangement is now stored on the server. The web terminal reads
it as the page loads, so the order of the header and status bar items, the
options you set on them (a UTC clock, say), and whether the status bar is shown
come back on the next visit. With nothing stored, the deployment's own
arrangement is shown. An arrangement this build cannot fully read is shown as
far as it goes and is never saved back over. One that cannot be read at all is
replaced by the deployment's own arrangement.

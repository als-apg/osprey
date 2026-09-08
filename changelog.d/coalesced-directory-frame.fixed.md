A file created or deleted inside a watched directory now shows up in the web
terminal's file panel and the artifacts gallery even when the operating system
reports the change as a single event about the directory rather than about the
file. Such a frame used to be forwarded with nothing in it (file panel) or
dropped outright (artifacts), so on macOS a burst of writes could leave both
listings stale until the next reload. The directory is now re-listed one level
deep and what actually differs is what gets announced.

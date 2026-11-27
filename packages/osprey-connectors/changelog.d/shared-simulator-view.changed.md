`SimulatorView.open`, `of_render`, `of_project` and `find` answer one view per
directory for the life of the process and read it again only when a view file
changed. A health run or a directory probe no longer parses the facility file
on every call, which held a server's event loop for seconds on a large
facility. `SimulatorView(path)` still reads afresh.

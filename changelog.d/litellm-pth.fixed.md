`osprey build` no longer recovers from a failed project-venv install by linking
the project at the build host's installed packages. An install that fails now
fails the build and reports the resolver's own output, so a built project runs
against the dependency set it recorded and nothing else.

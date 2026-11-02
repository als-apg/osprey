Web-terminal sidecar panels are described in one place: a registry entry now
carries the panel's label, sidebar icon and tooltip sentence, health route and
whether the panel proxy forwards the operator secret to it. The browser builds
one sidebar row per registered sidecar with no edit to its bundle, and the
JupyterLab panel keeps its routes and label.

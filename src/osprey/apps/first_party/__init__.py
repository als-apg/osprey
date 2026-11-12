"""OSPREY's own applications, one package directory each.

Each application is a package here, ``osprey/apps/first_party/<package>/``,
laid out exactly as a deployment's ``apps/<name>/``::

    <package>/
        __init__.py
        app.yml          # the manifest; its name is the entry-point name
        panel/           # the panel, when the application has one
        skills/<skill>/SKILL.md
        tools.py         # the tool server module, and any other Python modules

and declared by one entry point in OSPREY's own ``pyproject.toml``::

    [project.entry-points."osprey.apps"]
    <name> = "osprey.apps.first_party.<package>"

The wheel ships every file under the package. A deployment runs one only when its
profile's ``apps:`` block lists it (:mod:`osprey.apps.origin`). This module stays
free of side effects: locating an application imports this package, never the
application.
"""

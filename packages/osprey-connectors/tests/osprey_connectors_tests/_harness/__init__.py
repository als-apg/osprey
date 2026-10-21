"""The guards and helpers both test suites run on.

Everything here is framework-free: it imports ``osprey_connectors`` and the
standard library, never ``osprey`` and never the framework's ``tests`` package,
so the connectors suite can run on it with only that distribution installed.
The framework's suite imports the same modules, which is why a guard is written
once, here.

``_guards`` holds the conftest fixtures and hooks, imported by name into both
conftests; the other modules are the helpers they and the tests reach for.
Importing this package runs nothing: ``_guards`` acts on the environment at
import, so it is never imported from here.
"""

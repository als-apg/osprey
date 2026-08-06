"""Fixture registration for this directory.

Importing :func:`pubsub_emulator` here makes it available to every test in the
directory without each module importing it (which would shadow the fixture's own
name with the test's parameter). Modules outside this directory — the Google Chat
bridge e2e lane — import it from ``gchat_pubsub_emulator`` directly.
"""

from .gchat_pubsub_emulator import pubsub_emulator  # noqa: F401 - fixture registration

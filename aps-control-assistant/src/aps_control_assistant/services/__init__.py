"""
Service layer for control assistant.

Contains business logic independent of Osprey framework:
- channel_finder: Natural language channel finding with multiple pipeline modes

Note: Control system and archiver access is handled by Osprey connectors
      (configured in config.yml). See osprey.connectors for details.
"""


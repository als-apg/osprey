"""Stand-ins for heavy imports, put ahead of the source roots on connector-host children's path."""

from pathlib import Path

CHILD_STUBS = Path(__file__).resolve().parent

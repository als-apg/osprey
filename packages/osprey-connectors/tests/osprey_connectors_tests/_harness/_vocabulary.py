"""The refusal-phrase ban list both test suites match.

Framework-free, so the connectors suite holds its refusal phrases to it on a
host with only this distribution; ``tests/_vocabulary.py`` re-exports it for the
framework's own guards.
"""

from __future__ import annotations

import re

#: What no refusal phrase names: a config key, a config file, a build command or an
#: environment variable. Matched against the text an operator or agent reads.
REFUSAL_BAN_LIST = re.compile(
    r"writes_enabled|control_system\.|profile\.yml|config\.yml|osprey build|\bOSPREY_[A-Z0-9_]+"
    r"|\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"
)

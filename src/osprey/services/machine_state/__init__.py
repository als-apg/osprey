"""Machine State Service — bulk-read control system channels into a snapshot.

Provides:
- MachineStateReader: loads channel definitions and reads them in bulk
- MachineStateSnapshot: the result of a bulk read
- ChannelDefinition: parsed from the channels JSON file
- ChannelResult: one channel's read outcome
"""

from .models import ChannelDefinition, ChannelResult, MachineStateSnapshot
from .reader import MachineStateReader

__all__ = [
    "MachineStateReader",
    "MachineStateSnapshot",
    "ChannelDefinition",
    "ChannelResult",
]

"""The virtual accelerator's sequence module: each increase of the request is one shot.

The simulated machine fires one injection whenever its injection request is
written above the value it last saw, so ``inject`` reads the request and writes
it one higher. The simulated BPMs always record turn by turn, so ``bpm_mode``
has nothing to set, and there is no bunch to clean. The module names no
address: it declares no slots and reaches the request only through the handle.
"""

SEQUENCE_METADATA = {
    "slots": {},
    "max_pause_s": 0.5,
}


async def inject(handle):
    """Request one injection: write the request one above its current value."""
    current = await handle.read("injection_request")
    await handle.write("injection_request", int(current) + 1)


async def bpm_mode(handle):
    """Leave the BPMs as they are: the simulated BPMs always record turn by turn."""

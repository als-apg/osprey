"""The virtual accelerator's sequence module: each increase of the request is one shot.

The simulated machine fires one injection whenever its injection request is
written above the value it last saw, so ``inject`` reads the request and writes
it one higher. The module names no address: it reaches the request only
through the handle.
"""

SEQUENCE_METADATA = {
    "max_pause_s": 0.5,
}


async def inject(handle):
    """Request one injection: write the request one above its current value."""
    current = await handle.read("injection_request")
    await handle.write("injection_request", int(current) + 1)

"""A sequence module as the contract has it: ``inject`` and the longest pause.

``inject`` writes the injection request alone; everything else the service
writes around an injection is the measurement file's trajectory declaration.
"""

SEQUENCE_METADATA = {"max_pause_s": 0.5}


async def inject(handle):
    await handle.write("injection_request", 1)

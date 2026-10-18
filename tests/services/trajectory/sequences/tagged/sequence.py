"""A sequence module whose one slot is one address per BPM, by a signal the BPMs carry.

``turn_by_turn_x`` tags ``bpm_tbt_x``, a signal the test facility's BPM channels
carry, so the service's tag map holds an address for it on every BPM.
"""

SEQUENCE_METADATA = {
    "slots": {"turn_by_turn_x": {"tag": "bpm_tbt_x"}},
    "writes": {"bpm_mode": ["turn_by_turn_x"]},
    "max_pause_s": 0.5,
}


async def inject(handle):
    await handle.write("injection_request", 1)


async def bpm_mode(handle):
    await handle.write("turn_by_turn_x", 0)

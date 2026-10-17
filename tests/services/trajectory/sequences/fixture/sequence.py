"""A sequence module with every entry point and a module slot of each form.

``acquisition_mode`` is one address per BPM, by the signal its channels carry;
``kicker_enable`` is one address. ``bpm_mode`` writes the first, ``bunch_clean``
the second, and ``inject`` writes the injection request alone.
"""

KICKER_ENABLE = "SR:BUNCH:KICKER:ENABLE"

SEQUENCE_METADATA = {
    "slots": {
        "acquisition_mode": {"tag": "bpm_acquisition_mode"},
        "kicker_enable": KICKER_ENABLE,
    },
    "writes": {
        "bpm_mode": ["acquisition_mode"],
        "bunch_clean": ["kicker_enable"],
    },
    "max_pause_s": 0.5,
}


async def inject(handle):
    await handle.write("injection_request", 1)


async def bunch_clean(handle):
    await handle.write("kicker_enable", 1)
    await handle.sleep(0.1)
    await handle.write("kicker_enable", 0)


async def bpm_mode(handle):
    await handle.write("acquisition_mode", 1)

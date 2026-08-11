"""The archiver recorder: the live half of a deployment's stored history.

A deployed OSPREY project's archive has two authors. The base seeder writes a
deterministic synthesized history once per deploy, so the archive reaches back
before the machine was switched on. This service writes what happened *after*
that: it reads the running control system over Channel Access on a fixed
cadence and stores each reading in the same collection, on the same timeline,
in the same ``{date, <PV>: value}`` documents. There is one store and one
timeline — a reader cannot tell seeded history from recorded history except by
the one thing that should distinguish them, which is when it happened.

It records **only** while ``control_system.type`` is ``virtual_accelerator``,
and re-reads that setting from the mounted config on an interval rather than
latching it at startup, so the documented post-build flip takes effect without
restarting the service. On any other control system it idles: a real machine's
readings do not belong in an archive whose earlier half is synthesized, and a
mock control system has nothing worth recording.

Storage stays bounded without ever losing the recorded record, via the two
expiry tiers :func:`~osprey.services.archiver_recorder.store.expiry_for`
assigns. Every sample is kept for the hot span; the samples that land on the
tail grid are kept for the full retention window. Recorded reality therefore
thins out as it ages instead of vanishing, and it thins to exactly the cadence
the seeded tail already uses, so there is no visible seam at the point where
the dense copy expires.

The service ships as a compose template only — it runs the virtual
accelerator's image with a different command, so it adds no image build to any
deploy. Its knobs come from the profile's ``va_archiver:`` block through the
rendered ``config.yml``; nothing here carries a cadence or retention constant.
"""

from .config import (
    RecorderConfigError,
    RecorderSettings,
    load_settings,
    read_control_system_type,
    resolve_channel_addresses,
)
from .recorder import RECORDING_CONTROL_SYSTEM, Recorder
from .store import ArchiveWriter, expiry_for

__all__ = [
    "RECORDING_CONTROL_SYSTEM",
    "ArchiveWriter",
    "Recorder",
    "RecorderConfigError",
    "RecorderSettings",
    "expiry_for",
    "load_settings",
    "read_control_system_type",
    "resolve_channel_addresses",
]

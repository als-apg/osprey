"""Attachment format registry, re-exported from :mod:`osprey.imaging.formats`.

The registry lives in ``osprey.imaging`` so the isolated render worker can import
it without pulling in the ARIEL service; ARIEL code imports it from here.
"""

from osprey.imaging.formats import *  # noqa: F403

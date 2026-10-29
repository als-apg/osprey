"""Shared helpers for the Nextcloud Talk adapter's suites."""

import httpx

from osprey.bridges.nextcloud_talk.client import LAST_GIVEN_HEADER


def ocs(data, *, status=200, last_given=None):
    """An OCS v2 envelope response carrying ``data``.

    ``last_given`` sets the chat cursor header Talk sends with a page of messages.
    """
    headers = {} if last_given is None else {LAST_GIVEN_HEADER: str(last_given)}
    return httpx.Response(
        status,
        json={"ocs": {"meta": {"status": "ok", "statuscode": status}, "data": data}},
        headers=headers,
    )

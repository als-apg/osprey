"""The public endpoint Teams posts to, and the only writer to the bridge's queue.

A Teams bot receives messages one way: an HTTPS POST to an endpoint reachable
from the open internet. This module is that endpoint. It proves each activity
genuine through :mod:`validation` and hands the untouched request body to a
Service Bus queue, which the bridge consumes from wherever it runs -- so the
bridge itself keeps the property every Osprey bridge has, that it opens no
listening socket and only makes outbound calls.

The body is enqueued verbatim, wrapped in no envelope of our own. The bridge
parses the activity Microsoft sent rather than a re-serialisation of it, which
keeps intact every field this relay has no opinion about.

Three status codes carry the whole contract, because they are the three Teams
reacts to differently: 200 accepts the activity, 401 refuses it for good, and
503 says the relay could not judge it and the delivery should come again. A
relay that is itself mis-configured -- an unknown ``TEAMS_CLOUD`` -- raises
through as a 500, since that is a statement about the relay and not about the
request, and no amount of re-delivery would help.

Nothing here prints: Azure Functions collects the worker's ``logging`` output
and prints go nowhere an operator can read them.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import azure.functions as func

# The relay is a directory of plain modules, not a package: the Functions host
# imports ``function_app`` as a top-level module from the app root, so its
# sibling is reached by plain name. Putting that directory on the path as well
# keeps the same import working when the file is loaded by path -- which is how
# the tests read the relay, since it ships as a service template and is never
# installed.
_RELAY_DIR = str(Path(__file__).resolve().parent)
if _RELAY_DIR not in sys.path:
    sys.path.insert(0, _RELAY_DIR)

from validation import (  # noqa: E402 -- the path entry above has to exist first
    KeySourceUnavailable,
    ValidationError,
    validate_activity,
)

logger = logging.getLogger(__name__)

QUEUE_NAME_SETTING = "%TEAMS_SERVICEBUS_QUEUE%"
"""The app setting naming the queue, in the percent syntax bindings resolve.

The queue is named by deployment rather than by this file so that one published
relay can serve whichever queue its Function App is configured with.
"""

CONNECTION_SETTING = "SERVICEBUS_CONNECTION"
"""The app setting holding the Service Bus connection string.

It is named, never read here: the binding reads it in the host process, so the
send key never passes through Python and cannot reach a log line.
"""

DEFAULT_CLOUD = "commercial"
"""The cloud assumed when ``TEAMS_CLOUD`` is unset.

Commercial is where a bot lives unless someone deliberately put it elsewhere,
so the setting is required only of the deployments that are elsewhere. A value
that is set but unknown is not defaulted away -- it raises, and the relay
answers 500, because silently validating GCC High traffic against commercial
keys would refuse every genuine request with a 401 that looks like an attack.
"""

app = func.FunctionApp()


def _bearer_token(header: str | None) -> str | None:
    """Return the bare token carried by an ``Authorization`` header.

    ``None`` means there is nothing worth checking -- no header, another
    scheme, or a ``Bearer`` with nothing after it -- and the caller refuses the
    request without reaching for the signing keys. The scheme is compared
    case-insensitively because HTTP authentication schemes are, and a relay
    that refused ``bearer`` would be rejecting requests the standard calls
    well-formed.
    """
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


@app.function_name(name="teams_messages")
@app.route(route="messages", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
@app.service_bus_queue_output(
    arg_name="queue",
    queue_name=QUEUE_NAME_SETTING,
    connection=CONNECTION_SETTING,
)
def messages(req: func.HttpRequest, queue: func.Out[str]) -> func.HttpResponse:
    """Validate one Teams activity and put its body on the queue.

    The trigger is anonymous because the Bot Framework JWT is the
    authentication: a function key would be a second secret to distribute that
    Teams has no way to send, and the token proves more than a key could --
    which bot the activity is for, which cloud signed it, and that it was
    issued within the clock window.

    The refusals are ordered cheapest first. A request with no usable bearer
    value is answered before the key source is touched, so unauthenticated
    traffic to a public URL costs a header read and nothing else.
    """
    token = _bearer_token(req.headers.get("Authorization"))
    if token is None:
        logger.warning("refused a request carrying no Bearer authorization header")
        return func.HttpResponse(status_code=401)

    try:
        body = req.get_body().decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("refused a request whose body is not UTF-8 text")
        return func.HttpResponse(status_code=401)

    app_id = os.environ.get("TEAMS_APP_ID", "")
    if not app_id:
        logger.warning("TEAMS_APP_ID is unset, so every activity fails its audience check")
    cloud = os.environ.get("TEAMS_CLOUD") or DEFAULT_CLOUD

    try:
        activity = validate_activity(body, token, cloud=cloud, app_id=app_id)
    except ValidationError as exc:
        # The reason stays in the log rather than the response: the caller is
        # unauthenticated, and telling it which check it failed is how a forger
        # narrows down the next attempt.
        logger.warning("refused an activity: %s", exc)
        return func.HttpResponse(status_code=401)
    except KeySourceUnavailable as exc:
        logger.error("cannot judge an activity, asking Teams to re-deliver: %s", exc)
        return func.HttpResponse(status_code=503)

    queue.set(body)
    logger.info("enqueued a %r activity", activity.get("type"))
    return func.HttpResponse(status_code=200)

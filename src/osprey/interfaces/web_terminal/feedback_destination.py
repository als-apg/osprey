"""Where this deployment's documentation and feedback controls point.

One module owns the whole answer: the shipped defaults, the coercion of every
``web.docs_url`` / ``web.feedback.*`` value a facility may write, and the
resolution of those values into the destination the rail's Documentation link
and the Feedback dialog's channels actually use.

It is a **leaf**: it imports nothing from :mod:`osprey.interfaces.web_terminal.app`
and nothing from the route modules, so both may import it. That matters,
because before this module existed the app owned the constants and
``routes/panels.py`` re-typed them as literals in its ``getattr`` fallbacks —
with a comment blaming an ``app -> routes -> panels`` import cycle. The cycle
is real for a top-level ``from ...app import`` (the app pulls the router in at
import time), but it was never a reason to duplicate a value: a sibling leaf
beside :mod:`~osprey.interfaces.web_terminal.feedback_composer` and
:mod:`~osprey.interfaces.web_terminal.feedback_store` is reachable from both
sides with no cycle at all.

Duplicating them was worse than untidy. The copies lived on the *fallback*
path — ``getattr(app.state, "feedback_email", "<literal>")`` — which only runs
when app state is missing. Drift between the two spellings was therefore
invisible in every ordinary deployment and would surface exactly once, in the
degraded case nobody is watching.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Where the Documentation control in the rail's utility cluster points when
#: ``web.docs_url`` is absent. A facility hosting its own copy of the docs
#: overrides the key; the default is the published site.
DEFAULT_DOCS_URL = "https://als-apg.github.io/osprey"

#: ``owner/repo`` used to build the prefilled new-issue URL when
#: ``web.feedback.github_repo`` is absent.
DEFAULT_FEEDBACK_GITHUB_REPO = "als-apg/osprey"

#: Tracker kinds ``web.feedback.trackers`` accepts, each with the radio caption
#: used when an entry names no ``label`` of its own. The client's URL builders
#: (``static/js/feedback-prefill.js``) are keyed by the same two words.
FEEDBACK_TRACKER_LABELS: dict[str, str] = {"github": "GitHub", "gitlab": "GitLab"}

#: Recipient of the prefilled ``mailto:`` draft when ``web.feedback.email``
#: is absent — the OSPREY maintainers.
DEFAULT_FEEDBACK_EMAIL = "thellert@lbl.gov"

#: Ceiling (bytes) on the on-disk feedback store before the oldest saved
#: contexts are pruned; 256 MB unless ``web.feedback.max_store_bytes`` says
#: otherwise. Submission headers are never pruned, only their contexts.
DEFAULT_FEEDBACK_MAX_STORE_BYTES = 256 * 1024 * 1024


def coerce_config_str(key: str, value: object, default: str) -> str:
    """Return a configured string value, falling back to *default*.

    An **absent** key means "use the default": the caller reads it with
    *default* already in hand, so a facility with no ``config.yml`` still gets
    working Documentation and Feedback controls.

    An **explicitly blank** value (``docs_url: ""``) means "this deployment has
    no such target", and is returned as ``""``. That posture is what the rail
    anchor, the status-bar link and the dialog's channel guard are built on: an
    air-gapped control room blanks ``web.docs_url`` and gets no documentation
    link rather than one that opens a dead tab, and blanking
    ``web.feedback.github_repo`` retires the GitHub channel instead of aiming it
    at the upstream maintainers' tracker. Folding blank back into the default
    would make that whole posture unreachable while the UI kept claiming it.

    A YAML key written with no value at all (``docs_url:``, i.e. ``None``) reads
    as absent, not as blank — "I have not decided" rather than "there is none".

    A value of some other type (a nested mapping from a mis-indented
    ``config.yml``, say) is reported and discarded rather than repr'd into an
    ``href`` or a ``mailto:``, which would render a control that silently goes
    nowhere.

    Args:
        key: Dotted config key, used only for the warning message.
        value: Whatever the config reader returned.
        default: The shipped default for *key*.

    Returns:
        The stripped configured string (possibly ``""``), or *default*.
    """
    if isinstance(value, str):
        return value.strip()
    if value is not None:
        logger.warning("%s is %r, not a string; using %r instead", key, value, default)
    return default


def coerce_feedback_trackers(value: object) -> list[dict[str, str]]:
    """Return ``web.feedback.trackers`` as a list of normalised tracker entries.

    Each usable entry becomes ``{"kind", "label", "repo"}`` (GitHub, ``repo`` an
    ``owner/name``) or ``{"kind", "label", "url"}`` (GitLab, ``url`` the
    project's base URL — gitlab.com or self-hosted, trailing slash dropped).
    A missing ``label`` takes the kind's own name.

    Lenient per entry, strict per field: one malformed line — an unknown
    ``kind``, a GitHub entry without an ``owner/name`` repo, a GitLab entry
    whose ``url`` is not ``http(s)://`` — is reported and dropped while the rest
    of the list stands, because one typo must not retire every tracker the
    facility configured. A value that is not a list at all is reported and
    reads as no list.

    Args:
        value: Whatever the config reader returned for the key.

    Returns:
        The usable entries, in the order written.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        logger.warning("web.feedback.trackers is %r, not a list; ignoring it", value)
        return []
    trackers: list[dict[str, str]] = []
    for index, entry in enumerate(value):
        tracker = _coerce_feedback_tracker(entry)
        if tracker is None:
            logger.warning("web.feedback.trackers[%d] is %r; dropping it", index, entry)
            continue
        trackers.append(tracker)
    return trackers


def _coerce_feedback_tracker(entry: object) -> dict[str, str] | None:
    """One entry of :func:`coerce_feedback_trackers`, or ``None`` when unusable."""
    if not isinstance(entry, dict):
        return None
    kind = entry.get("kind")
    if not isinstance(kind, str) or kind.strip() not in FEEDBACK_TRACKER_LABELS:
        return None
    kind = kind.strip()
    label = entry.get("label")
    label = label.strip() if isinstance(label, str) and label.strip() else ""
    tracker = {"kind": kind, "label": label or FEEDBACK_TRACKER_LABELS[kind]}
    if kind == "github":
        repo = entry.get("repo")
        repo = repo.strip() if isinstance(repo, str) else ""
        if repo.count("/") != 1 or any(ch.isspace() for ch in repo) or not all(repo.split("/")):
            return None
        tracker["repo"] = repo
    else:
        url = entry.get("url")
        url = url.strip().rstrip("/") if isinstance(url, str) else ""
        if not url.startswith(("http://", "https://")) or any(ch.isspace() for ch in url):
            return None
        tracker["url"] = url
    return tracker


def resolve_feedback_trackers(
    trackers: list[dict[str, str]], github_repo: str
) -> list[dict[str, str]]:
    """The tracker list the dialog offers: the configured list plus the sugar.

    ``web.feedback.github_repo`` keeps its meaning as a single GitHub tracker,
    appended after the facility-authored list (blank retires it — that is the
    posture :func:`coerce_config_str` preserves). Two entries naming the same
    target collapse to the first, so a facility that lists the upstream repo
    under its own label does not get it rendered twice by the sugar.

    Args:
        trackers: Output of :func:`coerce_feedback_trackers`.
        github_repo: Resolved ``web.feedback.github_repo`` (``""`` when blank).

    Returns:
        The de-duplicated list, in render order.
    """
    candidates = list(trackers)
    if github_repo:
        candidates.append(
            {"kind": "github", "label": FEEDBACK_TRACKER_LABELS["github"], "repo": github_repo}
        )
    seen: set[tuple[str, str]] = set()
    resolved: list[dict[str, str]] = []
    for tracker in candidates:
        key = (tracker["kind"], tracker.get("repo") or tracker.get("url") or "")
        if key in seen:
            continue
        seen.add(key)
        resolved.append(dict(tracker))
    return resolved


def coerce_store_ceiling(value: object, default: int = DEFAULT_FEEDBACK_MAX_STORE_BYTES) -> int:
    """Return ``web.feedback.max_store_bytes`` as a positive byte count.

    Guarded rather than trusted: the pruner deletes stored contexts until the
    store fits under this number, so a ``0``, a negative, or a ``True`` that
    ``int()`` would happily turn into ``1`` would empty the store on the next
    submission while looking like ordinary pruning. A human-written ``256MB``
    is rejected the same way — this key is a plain byte count.

    Args:
        value: Whatever the config reader returned.
        default: The shipped ceiling to fall back to.

    Returns:
        A positive integer byte ceiling.
    """
    if value is None:
        return default
    if not isinstance(value, bool):
        try:
            ceiling = int(value)
        except (TypeError, ValueError, OverflowError):
            # OverflowError is not hypothetical: YAML parses `.inf` and any
            # overflowing exponent (1.0e+400) to float("inf"), which int()
            # refuses. This helper is called outside the lifespan's try, so an
            # escaping exception would abort startup outright.
            ceiling = 0
        if ceiling > 0:
            return ceiling
    logger.warning(
        "web.feedback.max_store_bytes is %r, not a positive byte count; using %d",
        value,
        default,
    )
    return default

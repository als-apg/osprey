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
from dataclasses import dataclass, field
from urllib.parse import urlencode

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


def coerce_feedback_owner(value: object) -> tuple[str, str | None, dict[str, str] | None]:
    """Read ``web.feedback.owner`` into ``(name, email, tracker)``.

    The block exists because a facility redirecting feedback has to move both
    the address and the tracker, and moving one is precisely the mistake it
    prevents. It is one block so that the two cannot be edited apart.

    ``email`` is returned as ``None`` when the block names none, which the
    caller feeds to :func:`coerce_config_str` as an *absent* value — that is
    what makes the leaf ``web.feedback.email`` outrank it without either key
    knowing about the other.

    The tracker is spelled ``{kind, target}`` rather than the internal
    ``{kind, repo}`` / ``{kind, url}`` pair: a facility writing this block is
    naming one destination and should not have to know which field name its
    forge happens to use. ``label`` is optional and defaults to the kind's own
    caption, exactly as in ``web.feedback.trackers``.

    Every field degrades on its own. A malformed tracker must not take the
    owner's address down with it — half a redirect is the failure mode, and
    silently reverting a facility's address to the upstream maintainers is the
    worst half to lose.

    Args:
        value: Whatever the config reader returned for ``web.feedback.owner``.

    Returns:
        The owner's name (``""`` when unnamed), the owner's address (``None``
        when unnamed) and the owner's tracker as one normalised entry
        (``None`` when unnamed or unusable).
    """
    if value is None:
        return "", None, None
    if not isinstance(value, dict):
        logger.warning("web.feedback.owner is %r, not a mapping; ignoring it", value)
        return "", None, None

    raw_name = value.get("name")
    name = raw_name.strip() if isinstance(raw_name, str) else ""

    raw_email = value.get("email")
    email = raw_email if isinstance(raw_email, str) or raw_email is None else None
    if raw_email is not None and email is None:
        logger.warning("web.feedback.owner.email is %r, not a string; ignoring it", raw_email)

    tracker = None
    raw_tracker = value.get("tracker")
    if raw_tracker is not None:
        tracker = _coerce_owner_tracker(raw_tracker)
        if tracker is None:
            logger.warning("web.feedback.owner.tracker is %r; ignoring it", raw_tracker)
    return name, email, tracker


def _coerce_owner_tracker(entry: object) -> dict[str, str] | None:
    """``{kind, target}`` as one normalised tracker entry, or ``None``.

    Translates the owner block's forge-agnostic ``target`` into the field name
    :func:`_coerce_feedback_tracker` expects for that kind, then validates
    through it — so the owner tracker and a ``web.feedback.trackers`` entry are
    held to exactly the same rules and cannot drift apart.
    """
    if not isinstance(entry, dict):
        return None
    kind = entry.get("kind")
    if not isinstance(kind, str):
        return None
    target_field = "repo" if kind.strip() == "github" else "url"
    return _coerce_feedback_tracker(
        {"kind": kind, "label": entry.get("label"), target_field: entry.get("target")}
    )


def resolve_feedback_trackers(
    trackers: list[dict[str, str]], owner_tracker: dict[str, str] | None
) -> list[dict[str, str]]:
    """The tracker list the dialog offers: the configured list, then the owner's.

    The deployment owner's own tracker is appended after the facility-authored
    ``web.feedback.trackers`` list, so a facility that also curates channels
    keeps its order and the owner's is the last resort. ``None`` retires it —
    that is the posture a blank ``web.feedback.github_repo`` reaches. Two
    entries naming the same target collapse to the first, so a facility that
    lists its own tracker explicitly does not get it rendered twice.

    Args:
        trackers: Output of :func:`coerce_feedback_trackers`.
        owner_tracker: The owner's tracker as one normalised entry, or ``None``
            when the deployment has no owner tracker at all.

    Returns:
        The de-duplicated list, in render order.
    """
    candidates = list(trackers)
    if owner_tracker:
        candidates.append(owner_tracker)
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


@dataclass(frozen=True)
class FeedbackDestination:
    """Everything the Documentation and Feedback controls need, resolved.

    One struct rather than five loose values, because the fields are not
    independent: ``trackers`` is partly *derived* from ``github_repo`` (the
    sugar expansion), so a caller that resolved them separately could hold a
    repo the tracker list does not mention. Returning them together makes that
    unrepresentable.
    """

    docs_url: str
    """``web.docs_url`` — ``""`` when the deployment retired the link."""

    email: str
    """``web.feedback.email`` — ``""`` when the deployment retired the channel."""

    github_repo: str
    """``web.feedback.github_repo`` — the sugar, already folded into *trackers*."""

    trackers: list[dict[str, str]] = field(default_factory=list)
    """The outbound channels the dialog offers, in render order."""

    max_store_bytes: int = DEFAULT_FEEDBACK_MAX_STORE_BYTES
    """Ceiling on the on-disk record store. Server-side only."""

    owner_name: str = ""
    """``web.feedback.owner.name`` — how this deployment's owner is written in a
    report. ``""`` when no owner block names one, including the unconfigured
    deployment: the OSPREY project is not a facility and does not caption
    itself as one."""


def resolve_feedback_destination(
    *,
    docs_url: object = None,
    email: object = None,
    github_repo: object = None,
    trackers: object = None,
    max_store_bytes: object = None,
    owner: object = None,
) -> FeedbackDestination:
    """Resolve the raw config values into one coherent destination.

    Deliberately **pure**: it reads no config of its own and takes whatever the
    reader returned, so the ``get_config_value`` calls stay in the lifespan
    where the config-key manifest can see them, and this function stays
    callable from a route that has no config at all.

    That second caller is the point. ``GET /api/panels`` falls back to shipped
    defaults whenever app state is missing, and it used to reach them by
    re-typing the literals — including a hand-rolled second copy of the
    ``github_repo`` -> one-GitHub-tracker sugar. Both callers now run this one
    function, so the lifespan's answer and the fallback's answer cannot
    disagree even in principle.

    Every argument is ``None`` by default and ``None`` means *absent*, so a
    bare call is the unconfigured deployment: the one whose owner is the
    OSPREY project itself.

    Each field is coerced separately. A single unusable value must not drag the
    others back to project defaults — silently redirecting a facility's
    feedback address to the upstream maintainers because its store ceiling was
    written ``256MB`` is exactly the failure a fail-open path must not produce.

    ``web.feedback.owner`` names the destination as one block. The two leaf
    keys still outrank it wherever they are spelled, which is what keeps every
    already-deployed profile meaning exactly what it meant before this key
    existed. The precedence falls out of :func:`coerce_config_str` rather than
    being written as a branch: the owner's value is passed as the *default* for
    the leaf key, so "leaf if spelled, else owner, else the project" is one
    expression and cannot be half-applied. A blank leaf key still retires its
    channel, because blank is a posture and outranks an owner that named one.

    Args:
        docs_url: Raw ``web.docs_url``.
        email: Raw ``web.feedback.email``.
        github_repo: Raw ``web.feedback.github_repo``.
        trackers: Raw ``web.feedback.trackers``.
        max_store_bytes: Raw ``web.feedback.max_store_bytes``.
        owner: Raw ``web.feedback.owner``.

    Returns:
        A freshly built :class:`FeedbackDestination`; nothing is shared between
        calls, so no caller can mutate the next one's tracker list.
    """
    owner_name, owner_email, owner_tracker = coerce_feedback_owner(owner)

    # The leaf repo, when spelled, replaces the owner's tracker outright — it
    # is the older way of saying the same thing, so honouring both would
    # render the destination twice.
    if github_repo is None and owner_tracker is not None:
        resolved_repo = owner_tracker.get("repo", "")
    else:
        resolved_repo = coerce_config_str(
            "web.feedback.github_repo", github_repo, DEFAULT_FEEDBACK_GITHUB_REPO
        )
        owner_tracker = (
            {"kind": "github", "label": FEEDBACK_TRACKER_LABELS["github"], "repo": resolved_repo}
            if resolved_repo
            else None
        )

    return FeedbackDestination(
        docs_url=coerce_config_str("web.docs_url", docs_url, DEFAULT_DOCS_URL),
        email=coerce_config_str(
            "web.feedback.email",
            email,
            coerce_config_str("web.feedback.owner.email", owner_email, DEFAULT_FEEDBACK_EMAIL),
        ),
        github_repo=resolved_repo,
        trackers=resolve_feedback_trackers(coerce_feedback_trackers(trackers), owner_tracker),
        max_store_bytes=coerce_store_ceiling(max_store_bytes),
        owner_name=owner_name,
    )


#: The OSPREY project's own issue tracker. A framework **constant**, not
#: facility config: it names the upstream project, which does not vary per
#: facility. A deployment configures where ITS reports go — never where the
#: framework's do — which is why the config surface has one owner block rather
#: than two configurable destinations.
UPSTREAM_ISSUE_REPO = DEFAULT_FEEDBACK_GITHUB_REPO

#: What a forwarded issue's title starts with. A prefix rather than a whole
#: title: the maintainer has diagnosed the bug and is the one who can name it,
#: and a canned title would survive into a tracker unedited.
UPSTREAM_ISSUE_TITLE_PREFIX = "[forwarded] "

#: How much of the preset content hash a report prints. Long enough to pin a
#: build, short enough to read in a metadata line; the same shortening
#: ``osprey build``'s drift advisory uses.
PRESET_HASH_CHARS = 19


def osprey_version() -> str:
    """The running OSPREY version, or ``"unknown"`` when it cannot be read.

    Never raises. Every caller is composing a feedback report or a deployment
    identity, and losing a bug report over a version lookup would be absurd.
    """
    try:
        from osprey import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 — a version lookup must not lose the report
        logger.debug("could not read the OSPREY version", exc_info=True)
        return "unknown"


@dataclass(frozen=True)
class DeploymentIdentity:
    """Which OSPREY this is — what a forwarded report must not make anyone re-derive.

    A facility maintainer who decides a report is a framework bug forwards it
    upstream, and upstream then needs to know what was actually running. Asking
    for that afterwards costs a round trip through two people and usually loses
    the report, so it rides along from the start.

    Every field degrades to ``""`` and an empty field is simply not printed: an
    identity is a convenience for the reader, and a deployment whose provenance
    cannot be read must still be able to file feedback.
    """

    osprey_version: str = ""
    """``osprey.__version__`` as the server sees it."""

    preset: str = ""
    """``provenance.preset`` — which shipped preset this profile was built from."""

    preset_hash: str = ""
    """``provenance.preset_hash`` — the content fingerprint of that preset, so a
    locally edited profile is distinguishable from the shipped one."""

    channel_finder_mode: str = ""
    """``channel_finder.pipeline_mode`` — the single biggest behavioural fork
    between two otherwise identical deployments."""

    def build_lines(self) -> dict[str, str]:
        """The BUILD facts as report-metadata lines, empty fields dropped.

        Deliberately excludes the version: both report builders already have
        their own source for it (the server reads the package, the browser
        reads ``/health``), and a second copy here would render it twice.

        The preset and its hash render as one line rather than two. They are
        one fact — "this build" — and a hash with no preset beside it tells a
        reader nothing they can act on.

        Labels are canonical Title Case. The two report builders write their
        blocks in different cases (the server's are lowercase, the browser's
        Title Case), which predates this and is not worth a format change to
        an already-shipped record; the server lowercases these on the way in,
        so both still render one set of facts from one definition.
        """
        lines: dict[str, str] = {}
        if self.preset:
            digest = self.preset_hash[:PRESET_HASH_CHARS]
            lines["Preset"] = f"{self.preset} ({digest}…)" if digest else self.preset
        if self.channel_finder_mode:
            lines["Channel finder"] = self.channel_finder_mode
        return lines

    def as_metadata(self) -> dict[str, str]:
        """The whole identity — the version, then the build facts."""
        version = {"OSPREY version": self.osprey_version} if self.osprey_version else {}
        return {**version, **self.build_lines()}


def resolve_deployment_identity(
    *,
    osprey_version: object = None,
    preset: object = None,
    preset_hash: object = None,
    channel_finder_mode: object = None,
) -> DeploymentIdentity:
    """Coerce the raw identity facts, dropping anything unusable.

    Pure, like :func:`resolve_feedback_destination`, and for the same reason:
    the config reads stay where the config-key manifest can see them.

    Nothing here warns. Unlike a feedback address, a missing identity field
    costs a maintainer a question rather than sending a report to the wrong
    place, and a deployment built before provenance was stamped is not
    misconfigured — it simply cannot answer.

    Args:
        osprey_version: The running version.
        preset: Raw ``provenance.preset``.
        preset_hash: Raw ``provenance.preset_hash``.
        channel_finder_mode: Raw ``channel_finder.pipeline_mode``.

    Returns:
        A :class:`DeploymentIdentity` with unusable fields left ``""``.
    """

    def _text(value: object) -> str:
        return value.strip() if isinstance(value, str) else ""

    return DeploymentIdentity(
        osprey_version=_text(osprey_version),
        preset=_text(preset),
        preset_hash=_text(preset_hash),
        channel_finder_mode=_text(channel_finder_mode),
    )


def upstream_escalation_url(identity: DeploymentIdentity, destination: FeedbackDestination) -> str:
    """A prefilled upstream issue for a maintainer forwarding a framework bug.

    This is the load-bearing half of the one-destination model. Users file to
    whoever owns the deployment, because a user cannot know whether a bug is
    OSPREY's code or the facility's configuration — working that out is usually
    the point of the report. The distinction is still made, just by the person
    who can actually make it. That only holds if forwarding is nearly free:
    a maintainer who has to open a tracker, re-describe the deployment and
    re-explain the bug will answer their user and upstream will never hear it.

    So the URL is prefilled with the deployment's identity and carries **no**
    user content. That is not a size compromise, it is the right split:

    * The maintainer is forwarding a bug they have now diagnosed, and their
      diagnosis is the valuable part. Prefilling the user's raw words would
      invite forwarding them unread.
    * A session id refers to a transcript on the *facility's* deployment and
      is unreadable upstream, so carrying it would only mislead.
    * Nothing varies per report, so the URL is resolved once at startup rather
      than composed per submission, and never has to be fitted to a length cap.

    Returns ``""`` when this deployment's reports already reach the OSPREY
    project — an unconfigured deployment IS owned by the project, and a link
    inviting a maintainer to forward a report to themselves is noise.

    Args:
        identity: What is running here.
        destination: The resolved destination, consulted only to detect that
            the owner is already the upstream project.

    Returns:
        A ``https://github.com/.../issues/new`` URL, or ``""``.
    """
    if any(
        tracker.get("kind") == "github" and tracker.get("repo") == UPSTREAM_ISSUE_REPO
        for tracker in destination.trackers
    ):
        return ""

    body_lines = ["Forwarded by a deployment maintainer.", ""]
    body_lines += [f"- **{key}:** {value}" for key, value in identity.as_metadata().items()]
    if destination.owner_name:
        body_lines.append(f"- **Deployment:** {destination.owner_name}")
    body_lines += ["", "Describe the framework bug and paste what the user reported."]

    query = urlencode({"title": UPSTREAM_ISSUE_TITLE_PREFIX, "body": "\n".join(body_lines)})
    return f"https://github.com/{UPSTREAM_ISSUE_REPO}/issues/new?{query}"

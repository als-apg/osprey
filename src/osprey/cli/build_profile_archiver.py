"""The ``va_archiver:`` block — the store a simulated facility's history lives in.

A deployment that serves simulated channels still needs somewhere to keep what
those channels did. This block declares that store: a MongoDB collection the
build seeds with a deterministic base series, a recorder samples the running
machine into, and the ``mongodb_archiver`` connector reads back. Declaring the
block is what makes the history *stored* rather than synthesized at read time.

Every number the store's shape depends on is a key here, because the facility
that owns the deployment owns those numbers: how far back history reaches, how
densely, how often the recorder writes, how the collection is compressed. None
of them are constants in the seeder or the recorder — they travel profile →
rendered ``config.yml`` → the services that read it, and changing one is a
profile edit and a rebuild, never a code change.

The block also knows the connector's connection keys, and emits them: a
facility that has said where its archive lives should not have to say it again
in ``config:`` under a second spelling that can disagree. Those keys are
therefore *rejected by name* in ``config:`` while this block is present, the
same way the ``deploy:`` block refuses a duplicated ``image_source``.

Shapes, parsing, rules and the config it contributes are all here rather than
split across the loader modules — the block is closed and small, and the build
step that injects the two archiver services wants exactly one import.

The build-time half of the honesty rule lives here too
(:func:`va_mock_archiver_errors`): declining to declare a store is a legal
choice for most deployments, but not for one serving a simulated machine, and
the profile is the earliest place that pairing can be caught.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, fields
from typing import Any

from osprey.connectors.honesty import VA_MOCK_ARCHIVER_WHY, pairing_in_profile
from osprey.connectors.types import MONGODB_ARCHIVER, VIRTUAL_ACCELERATOR
from osprey.errors import BuildProfileError

from .build_profile_schema import _ENV_VAR_RE

#: Block compressors ``mongod`` accepts for a WiredTiger collection. The value
#: reaches the container as a command flag and the collection inherits it at
#: implicit creation, so a name outside this set fails at container start —
#: hours after the profile that named it was written.
COMPRESSORS: tuple[str, ...] = ("zstd", "snappy", "zlib", "none")

#: Where the agent reaches a store this project deploys itself. The service
#: publishes its port on the deployment's bind address, and the agent runs on
#: the host, so loopback is the connection for every self-contained project.
#: An *attached* project overrides it with :attr:`VAArchiverConfig.host`.
DEFAULT_ARCHIVER_HOST = "localhost"

#: Rendered-config subtree the ``mongodb_archiver`` connector reads. Sibling to
#: ``archiver.type`` rather than nested under it — the archiver factory looks up
#: ``archiver.<type>`` for the selected type's settings.
CONNECTION_CONFIG_PREFIX = "archiver.mongodb_archiver"

#: Rendered-config block the seeder and the recorder read the archive's shape
#: from. Named after the profile block so the two are recognizably the same
#: knobs: the profile is where a facility sets them, this is where the services
#: read them, and nothing in between holds a second copy.
KNOBS_CONFIG_PREFIX = "va_archiver"

#: Health category the derived freshness check lands in. Deliberately not one of
#: the core category names (:data:`osprey.health.config.CORE_CATEGORY_NAMES`),
#: which reject a ``checks:`` list outright.
FRESHNESS_CATEGORY = "archiver"

#: Check and probe name. Both are the probe's own defaults, stated rather than
#: relied on so the rendered config reads as a complete declaration.
FRESHNESS_CHECK_NAME = "archiver_freshness"

#: Where the derived check lands in the rendered config.
FRESHNESS_CONFIG_KEY = f"health.categories.{FRESHNESS_CATEGORY}.checks"

#: How many recorder intervals may pass before the archive is called stale.
#:
#: The writer is the recorder, and its hot cadence is what governs: every sample
#: it takes is written, and ``recorder_tail_cadence_sec`` only decides which of
#: those *survive* ageing out of the hot span (see the recorder's store module —
#: a sample landing on the tail grid gets the longer expiry, it is not a
#: separate, slower write). So the newest sample is always one hot interval old
#: at worst, never one tail interval.
#:
#: At worst in steady state, that is — which is why the threshold is a multiple
#: rather than the cadence itself. A threshold equal to the cadence would sit
#: exactly on the sample age and flap on ordinary scheduling jitter: a container
#: under load, a CA read that takes most of its interval, a single missed tick.
#: Three intervals tolerates two consecutive misses before saying anything,
#: which is the point where "the recorder is behind" stops being noise.
FRESHNESS_SAFETY_MULTIPLE = 3

#: Floor for that threshold, in seconds. A facility recording every second would
#: otherwise derive a 3 s threshold, which any container restart trips; nothing
#: is learned from an alarm that fires faster than the deployment can recover.
FRESHNESS_FLOOR_SEC = 60

#: The knobs that describe the archive itself rather than the container serving
#: it. The container-shaping ones (port, user, compressor) are rendered into
#: ``services.mongodb`` by the injector, where the compose template reads them.
_KNOB_CONFIG_KEYS: tuple[str, ...] = (
    "retention_days",
    "hot_span_hours",
    "hot_cadence_sec",
    "tail_cadence_sec",
    "recorder_cadence_sec",
    "recorder_tail_cadence_sec",
    "recorder_poll_sec",
)

#: Mongo database names may not contain these, and a name that does is refused
#: at connect time with a pymongo error nobody reads as "fix your profile".
_ILLEGAL_DB_CHARS = set('/\\. "$*<>:|?')

#: Keys the block recognizes, derived from the dataclass rather than listed by
#: hand so the check cannot drift from what the parser honors.
_KNOWN_VA_ARCHIVER_KEYS: frozenset[str] = frozenset()  # populated below


@dataclass
class VAArchiverConfig:
    """The parsed ``va_archiver:`` block — one store, one timeline.

    The defaults describe the store a control-assistant deployment wants
    without being told: two days of dense history behind a month of coarse
    history, sampled every ten seconds, compressed. They are chosen to fit the
    documented storage budget at the shipped channel count; raising
    :attr:`retention_days` raises the disk footprint roughly in proportion.
    """

    retention_days: int = 30
    """How far back the archive reaches, in days. The coarse tail is seeded over
    this span and every recorder-written tail sample expires this long after it
    was taken, so the number is both "how much history a fresh deployment has"
    and "how much a running one keeps"."""

    hot_span_hours: int = 48
    """How much of the most recent history is kept at the dense cadence. Sized
    to cover an operator's plausible lookback ("what happened on the last
    shift") without paying dense storage for a month."""

    hot_cadence_sec: int = 10
    """Seconds between samples inside the hot span."""

    tail_cadence_sec: int = 60
    """Seconds between samples outside it. Must be a whole multiple of
    :attr:`hot_cadence_sec`: the tail grid is the sparse subset of the hot grid,
    and a cadence that does not divide it would put the two tiers on timestamps
    that never coincide."""

    recorder_cadence_sec: int = 10
    """How often the recorder samples the live machine. Its samples carry the
    hot-span expiry, so this is the cadence recent recorded history has."""

    recorder_tail_cadence_sec: int = 60
    """How often one of those samples is additionally kept for the full
    retention span. Recorded reality survives at the tail cadence as the dense
    copy ages out, which is what keeps steady-state storage bounded while
    leaving no hole where the seeded history ended."""

    recorder_poll_sec: int = 30
    """How often the recorder re-reads the deployment's config to decide whether
    it should be recording at all. It records only for a simulated control
    system, and that setting is a documented post-build flip — polling is what
    lets the flip take effect without restarting the service."""

    port_host: int = 27017
    """Host port the store publishes on. Only meaningful for a project that
    deploys its own store; an attached project is reaching someone else's
    published port, which is the same key seen from the other side."""

    database: str = "osprey_archiver"
    """Database holding the collection."""

    collection: str = "pv_history"
    """Collection the samples are written to and read from."""

    compression: str = "zstd"
    """WiredTiger block compressor for that collection (see :data:`COMPRESSORS`).
    Time-series documents compress well, and the choice is the difference
    between a base seed that fits the storage budget and one that does not."""

    username: str = "osprey"
    """Database user the deployment creates and the agent connects as."""

    auth_database: str = "admin"
    """Database that user authenticates against — ``admin`` for the root user a
    deployed store mints. Rendered as the connector's ``auth`` key."""

    password_env: str = "MONGO_ROOT_PASSWORD"
    """Name of the variable holding that user's password. The value is minted
    into the deployment's ``.env`` and read at connect time; it is never a
    profile field, and naming a different variable here does not create one."""

    timeout_sec: int = 5
    """Seconds the connector waits to reach the store. Deliberately short: the
    common failure is a project that was built but never deployed, and the
    honest answer there is a fast, explanatory error rather than a minute of
    silence on the first archiver call."""

    freshness_channel: str | None = None
    """Canary channel the ``archiver_freshness`` health check reads, if any.

    Naming one is what turns the archive from something the deployment *has*
    into something it is *checked on*: the check queries this channel's newest
    sample and grades its age, which is the difference between "the store is
    reachable" and "the store is still being written". Which channel is
    representative is a fact only the facility knows — a stored-beam current, a
    vacuum gauge, whatever stops moving first when the machine does — so there
    is no default, and a profile that names none derives no check rather than
    having the framework guess one.
    """

    host: str | None = None
    """Where the store actually is, when it is not this project's own.

    ``None`` — the default — means this project deploys the store and the agent
    reaches it on loopback. An *attached* project (``deploy_services: false``)
    deploys nothing, so it must say which host's store it is reading; that is
    the one configuration where this key is required rather than optional.
    """


_KNOWN_VA_ARCHIVER_KEYS = frozenset(f.name for f in fields(VAArchiverConfig))

# Integer keys and the floor each one has to clear. All of them are counts of
# something real (seconds, hours, days, a port), so zero and negative are
# meaningless rather than merely unusual.
_POSITIVE_INT_KEYS: tuple[str, ...] = (
    "retention_days",
    "hot_span_hours",
    "hot_cadence_sec",
    "tail_cadence_sec",
    "recorder_cadence_sec",
    "recorder_tail_cadence_sec",
    "recorder_poll_sec",
    "timeout_sec",
)

_STRING_KEYS: tuple[str, ...] = (
    "database",
    "collection",
    "compression",
    "username",
    "auth_database",
    "password_env",
)


def parse_va_archiver_block(raw: dict[str, Any]) -> VAArchiverConfig | None:
    """Parse a profile's ``va_archiver:`` block into its typed shape.

    Structural problems are reported together rather than one per run: the
    block is a dozen numbers being transcribed at once, and learning about them
    one rebuild at a time is the slowest way to fill in a table.

    Args:
        raw: The resolved raw profile dict.

    Returns:
        The parsed config, or ``None`` when the profile declares no
        ``va_archiver:`` block — a stored archive is opt-in, and a deployment
        without one reads whatever archiver its ``config:`` selects.

    Raises:
        BuildProfileError: If the block is not a mapping, names a key the block
            does not define, or gives one the wrong type. Range and consistency
            rules are :func:`va_archiver_errors`', so they can be reported
            alongside the rest of the profile's.
    """
    block = raw.get("va_archiver")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise BuildProfileError(
            f"Profile 'va_archiver' must be a mapping (got {type(block).__name__})"
        )

    problems: list[str] = []
    _reject_unknown_keys(block, problems)

    values: dict[str, Any] = {}
    for key in (*_POSITIVE_INT_KEYS, "port_host"):
        if key in block:
            values[key] = _parse_int(block[key], key, problems)
    for key in _STRING_KEYS:
        if key in block:
            values[key] = _parse_str(block[key], key, problems)
    for optional in ("host", "freshness_channel"):
        if block.get(optional) is not None:
            values[optional] = _parse_str(block[optional], optional, problems)

    if problems:
        raise BuildProfileError(
            "Profile 'va_archiver' block is invalid:\n  - " + "\n  - ".join(problems)
        )

    return VAArchiverConfig(**values)


def va_archiver_errors(
    va_archiver: VAArchiverConfig | None,
    config: Any,
    deploy_services: bool,
) -> list[str]:
    """Check the block's ranges and its agreement with the rest of the profile.

    Split from parsing so these land in ``BuildProfile.validate()``'s single
    report — a facility fixing a profile should meet every problem it has at
    once — and because two of the rules are not the block's to see alone: where
    the store lives depends on whether this project deploys one, and a
    duplicated connection key is a fact about ``config:``.

    Args:
        va_archiver: The parsed block, or ``None`` when the profile has none.
        config: The profile's ``config:`` block.
        deploy_services: Whether this project deploys its own service stack.

    Returns:
        The problems found, empty when the block is absent or clean.
    """
    if va_archiver is None:
        return []

    errors: list[str] = []
    cfg = va_archiver

    for key in _POSITIVE_INT_KEYS:
        value = getattr(cfg, key)
        if value < 1:
            errors.append(f"va_archiver.{key} must be >= 1 (got {value})")

    if not (1 <= cfg.port_host <= 65535):
        errors.append(f"va_archiver.port_host must be in 1..65535 (got {cfg.port_host})")

    # The hot tier is the recent end of the same timeline the tail covers, so a
    # hot span reaching past retention would describe dense history older than
    # the oldest history there is.
    if cfg.hot_span_hours > cfg.retention_days * 24:
        errors.append(
            f"va_archiver.hot_span_hours ({cfg.hot_span_hours}) exceeds retention_days "
            f"({cfg.retention_days} days = {cfg.retention_days * 24} hours) — the dense "
            f"tier is the recent part of the retained window, not a longer one"
        )

    # Same grid, two densities: the seeder writes the tail as the sparse subset
    # of the hot cadence, and the recorder promotes every Nth of its samples the
    # same way. A cadence that does not divide leaves the tiers on timestamps
    # that never line up, which shows up as a seam rather than as an error.
    for dense, sparse in (
        ("hot_cadence_sec", "tail_cadence_sec"),
        ("recorder_cadence_sec", "recorder_tail_cadence_sec"),
    ):
        fine, coarse = getattr(cfg, dense), getattr(cfg, sparse)
        if fine < 1 or coarse < 1:
            continue  # already reported above
        if coarse < fine:
            errors.append(
                f"va_archiver.{sparse} ({coarse}) must be >= {dense} ({fine}) — "
                f"the sparse tier cannot sample faster than the dense one"
            )
        elif coarse % fine:
            errors.append(
                f"va_archiver.{sparse} ({coarse}) must be a whole multiple of "
                f"{dense} ({fine}), so both tiers land on the same timestamps"
            )

    if cfg.compression not in COMPRESSORS:
        errors.append(
            f"va_archiver.compression must be one of "
            f"{', '.join(repr(name) for name in COMPRESSORS)} (got {cfg.compression!r})"
        )

    for key in ("database", "collection", "username", "auth_database"):
        if not getattr(cfg, key).strip():
            errors.append(f"va_archiver.{key} must be a non-empty string")
    illegal = sorted(_ILLEGAL_DB_CHARS & set(cfg.database))
    if illegal:
        errors.append(
            f"va_archiver.database {cfg.database!r} contains character(s) MongoDB "
            f"forbids in a database name: {' '.join(illegal)}"
        )

    if not _ENV_VAR_RE.match(cfg.password_env):
        errors.append(
            f"va_archiver.password_env must be an environment variable NAME "
            f"(uppercase letters, digits, underscores), got {cfg.password_env!r}. The "
            f"password itself belongs in the deployment's .env, never in the profile."
        )

    if cfg.host is not None and not cfg.host.strip():
        errors.append("va_archiver.host must be a non-empty string when present")

    # An attached project scaffolds no services, so nothing here deploys the
    # store. Without a host it would silently connect to loopback and find
    # nothing listening — a deploy-time mystery for a build-time omission.
    if not deploy_services and cfg.host is None:
        errors.append(
            "va_archiver.host is required when deploy_services is false — an "
            "attached project deploys no store of its own, so it has to name the "
            "host whose archive it reads (with port_host set to that host's "
            "published port)."
        )

    if cfg.freshness_channel is not None and not cfg.freshness_channel.strip():
        errors.append(
            "va_archiver.freshness_channel must be a non-empty channel name when present "
            "(omit the key entirely to derive no freshness check)"
        )

    errors.extend(_duplicate_derived_key_errors(config))
    errors.extend(_duplicate_freshness_check_errors(config, cfg))
    return errors


def va_mock_archiver_errors(config: Any) -> list[str]:
    """Refuse a profile that gives a simulated machine an invented past.

    The build is the earliest of the honesty rule's three sites and the only one
    that can name the *profile* keys to change, which is why the message here
    talks about ``va_archiver:`` and ``config:`` rather than about ``config.yml``.

    Reads the profile's resolved ``config:`` block — after presets, ``extends``
    parents and CLI layers have merged and the ``connector:`` shorthand has
    folded in — so what is checked is the pairing the rendered project will
    actually carry, not the one any single layer happened to author. Both the
    dotted and the nested spelling are read, and a disagreement between them
    fails closed; see :func:`~osprey.connectors.honesty.pairing_in_profile`.

    Args:
        config: The profile's ``config:`` block.

    Returns:
        The single problem as a one-element list, or empty when the profile
        pairs honestly. Shaped for :meth:`BuildProfile.validate`'s accumulated
        report — a facility fixing a profile meets all of its problems at once.
    """
    pairing = pairing_in_profile(config)
    if not pairing.is_invented_history:
        return []
    return [
        f"config sets control_system.type to {VIRTUAL_ACCELERATOR!r} while "
        f"archiver.type is {pairing.archiver_phrase} — {VA_MOCK_ARCHIVER_WHY} "
        f"Give the deployment a real archive instead: add a `va_archiver:` block "
        f"(the build then deploys a store and a recorder beside the virtual "
        f"accelerator, seeds the history, and derives the connector's keys) and set "
        f"`config: {{archiver.type: {MONGODB_ARCHIVER}}}`. If the archive lives in a "
        f"store you run yourself, set archiver.type to that connector and give it "
        f"its connection keys — anything but the mock is accepted here."
    ]


def va_archiver_config_overrides(va_archiver: VAArchiverConfig | None) -> dict[str, Any]:
    """The ``config:`` entries the block contributes to the rendered project.

    Two groups, both derived from the one block so the profile never states the
    archive twice:

    - the connector's eight connection keys, because the block already says
      where the archive is and the connector needs that fact under its own
      spelling. It is why a ``config:`` block spelling them anyway is refused
      rather than quietly overwritten;
    - the archive's shape — retention, tier spans, cadences — which the seeder
      and the recorder read at run time. They live in the rendered config rather
      than as constants in those services precisely so that changing one is a
      profile edit and a rebuild.

    Applied on the ordinary config-override path rather than by the archiver's
    build-step injector, because an *attached* project scaffolds no services and
    so never reaches that injector — yet it is exactly the project that most
    needs to be told where its archive is.

    The archiver *type* is deliberately not among them. Declaring where an
    archive lives is not the same decision as selecting it as the deployment's
    archiver, and a block that flipped the connector out from under a facility's
    ``archiver.type`` would be making that decision for them.

    Args:
        va_archiver: The parsed block, or ``None``.

    Returns:
        Dotted config keys to apply after the profile's own, empty when the
        profile declares no block.
    """
    if va_archiver is None:
        return {}
    cfg = va_archiver
    overrides: dict[str, Any] = {
        **freshness_check_override(cfg),
        f"{CONNECTION_CONFIG_PREFIX}.host": cfg.host or DEFAULT_ARCHIVER_HOST,
        f"{CONNECTION_CONFIG_PREFIX}.port": cfg.port_host,
        f"{CONNECTION_CONFIG_PREFIX}.name": cfg.database,
        f"{CONNECTION_CONFIG_PREFIX}.collection": cfg.collection,
        f"{CONNECTION_CONFIG_PREFIX}.auth": cfg.auth_database,
        f"{CONNECTION_CONFIG_PREFIX}.username": cfg.username,
        f"{CONNECTION_CONFIG_PREFIX}.password_env": cfg.password_env,
        f"{CONNECTION_CONFIG_PREFIX}.timeout": cfg.timeout_sec,
    }
    overrides.update(
        {f"{KNOBS_CONFIG_PREFIX}.{knob}": getattr(cfg, knob) for knob in _KNOB_CONFIG_KEYS}
    )
    return overrides


def freshness_max_age_s(va_archiver: VAArchiverConfig) -> int:
    """The staleness threshold this archive's own writer implies.

    Derived rather than configured, because the number that matters is not a
    preference — it is a fact about the recorder this same block configures. A
    facility that halves ``recorder_cadence_sec`` has halved how long a healthy
    archive can go without a sample, and the check should follow that edit
    without anyone remembering to re-tune a second number.

    See :data:`FRESHNESS_SAFETY_MULTIPLE` for why it is a multiple of the hot
    cadence and not the cadence itself, and why the hot cadence rather than the
    tail one is the writer that governs.
    """
    return max(
        FRESHNESS_SAFETY_MULTIPLE * va_archiver.recorder_cadence_sec,
        FRESHNESS_FLOOR_SEC,
    )


def freshness_check_override(va_archiver: VAArchiverConfig | None) -> dict[str, Any]:
    """The ``health:`` entry the block contributes, when it names a canary.

    Scoped to the block's presence on purpose, and that scope is what makes it
    right on both sides of the control-system flip. A deployment carrying the
    block has a store and a recorder, so there is something to be fresh about:
    on a virtual accelerator the recorder is writing and the check reports
    ``ok``; on a mock control system the recorder idles by design, the newest
    sample is the seeded tail, and the check reports staleness — as a
    ``warning``, because that is the only severity the probe gives staleness.
    Both are honest answers to "is this archive still being written". A
    storeless project has no block, derives no check, and claims nothing.

    Returns:
        The single dotted config key, or empty when there is no block or it
        names no canary channel.
    """
    if va_archiver is None or not va_archiver.freshness_channel:
        return {}
    return {
        FRESHNESS_CONFIG_KEY: [
            {
                "name": FRESHNESS_CHECK_NAME,
                "type": FRESHNESS_CHECK_NAME,
                "channel": va_archiver.freshness_channel,
                "max_age_s": freshness_max_age_s(va_archiver),
            }
        ]
    }


def config_derived_key_spellings(config: Any) -> list[str]:
    """How a ``config:`` block spells the keys this block derives, if it does.

    Several spellings reach the same rendered leaves and all are legal YAML: the
    dotted key, a mapping under the dotted prefix, and — for the connector's
    keys — a fully nested ``archiver:`` subtree. A duplicate hiding in any of
    them would defeat the point of looking, so all are checked.

    Returns:
        The offending ``config:`` keys as written, in a stable order; empty when
        the block spells none of them.
    """
    if not isinstance(config, dict):
        return []

    found: list[str] = []
    for prefix in (CONNECTION_CONFIG_PREFIX, KNOBS_CONFIG_PREFIX):
        found.extend(key for key in config if isinstance(key, str) and key.startswith(f"{prefix}."))
        subtree = config.get(prefix)
        if isinstance(subtree, dict):
            found.extend(f"{prefix}: {leaf}" for leaf in subtree)

    archiver = config.get("archiver")
    if isinstance(archiver, dict):
        nested = archiver.get("mongodb_archiver")
        if isinstance(nested, dict):
            found.extend(f"archiver: mongodb_archiver: {leaf}" for leaf in nested)

    return sorted(found)


def _duplicate_derived_key_errors(config: Any) -> list[str]:
    """Refuse a profile that states the archive's coordinates in both homes.

    The build derives these keys from this block, so a ``config:`` entry saying
    the same thing is not an override — it is a second home for one fact, free
    to disagree with the first. Disagreement is the dangerous case: a stale
    ``host`` or ``collection`` there points the agent at an archive nothing is
    writing, and it reads as empty rather than as broken.
    """
    spellings = config_derived_key_spellings(config)
    if not spellings:
        return []
    return [
        f"'{spelling}' is set in the profile's `config:` block while a "
        f"`va_archiver:` block is present — one fact, two homes, free to "
        f"disagree. The va_archiver block is the home: the build writes both "
        f"`{CONNECTION_CONFIG_PREFIX}.*` and `{KNOBS_CONFIG_PREFIX}.*` from it. "
        f"Delete the `config:` entry and set the matching va_archiver key instead."
        for spelling in spellings
    ]


def _expand_dotted(config: Any) -> dict[str, Any]:
    """The nested tree a ``config:`` block addresses, dotted keys folded in.

    Built the way the renderer builds it — every dotted key is a path to a leaf,
    and a nested mapping addresses the same leaves — so a lookup here answers
    "does this config reach that key", whichever of the legal spellings was
    used. Order-independent by construction: it records only *whether* a path is
    addressed, never which spelling would win, because which one wins is
    decided at render time by key order and is exactly what must not be relied
    on.
    """
    tree: dict[str, Any] = {}
    if not isinstance(config, dict):
        return tree
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        node = tree
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        leaf = parts[-1]
        if isinstance(value, dict):
            existing = node.get(leaf)
            merged = existing if isinstance(existing, dict) else {}
            node[leaf] = {**merged, **_expand_dotted(value)}
        else:
            node.setdefault(leaf, value)
    return tree


def _addresses_freshness_category(config: Any) -> bool:
    """Whether a ``config:`` block reaches the health category the block derives."""
    node: Any = _expand_dotted(config)
    for part in ("health", "categories", FRESHNESS_CATEGORY):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _duplicate_freshness_check_errors(config: Any, cfg: VAArchiverConfig) -> list[str]:
    """Refuse a profile that declares the freshness check the block derives.

    Same rule as the connection keys above, for the same reason: the build
    writes this category from ``freshness_channel`` and the recorder's cadence,
    so a ``config:`` block writing it too is a second home for one fact. This
    one is worse than most if left alone — the two would be merged by whichever
    key the renderer happens to reach last, so the deployment's staleness
    threshold would depend on the order two unrelated lines were typed in.

    Refused rather than merged or silently overridden: a facility that wants a
    different threshold has a knob for the number it is really changing
    (``recorder_cadence_sec``), and one that wants a different *check* should
    say so by dropping ``freshness_channel`` and declaring the category itself.
    """
    if not cfg.freshness_channel or not _addresses_freshness_category(config):
        return []
    return [
        f"the profile's `config:` block declares `health.categories."
        f"{FRESHNESS_CATEGORY}` while `va_archiver.freshness_channel` is set — the "
        f"build derives that category from the block (the canary channel, and a "
        f"staleness threshold from `recorder_cadence_sec`), so the two would be one "
        f"fact in two homes, merged in whichever order the keys happen to be read. "
        f"Drop the `config:` entry to keep the derived check, or drop "
        f"`va_archiver.freshness_channel` to own the category yourself."
    ]


def _reject_unknown_keys(block: dict[str, Any], problems: list[str]) -> None:
    """Reject anything the block does not define, naming the closest spelling."""
    unknown = sorted(set(block) - _KNOWN_VA_ARCHIVER_KEYS)
    if not unknown:
        return
    known = sorted(_KNOWN_VA_ARCHIVER_KEYS)
    for key in unknown:
        close = difflib.get_close_matches(str(key), known, n=1)
        suggestion = f" (did you mean '{close[0]}'?)" if close else ""
        problems.append(
            f"unknown key '{key}'{suggestion} — valid va_archiver keys are: {', '.join(known)}."
        )


def _parse_int(value: Any, key: str, problems: list[str]) -> int:
    """Read one integer knob, refusing the bool YAML would otherwise let pass."""
    if isinstance(value, bool) or not isinstance(value, int):
        problems.append(f"'{key}' must be an integer (got {value!r}).")
        return 1
    return value


def _parse_str(value: Any, key: str, problems: list[str]) -> str:
    """Read one string knob."""
    if not isinstance(value, str):
        problems.append(f"'{key}' must be a string (got {type(value).__name__}).")
        return ""
    return value

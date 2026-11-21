"""The words the channel directory service serves for its sources: one producer of each.

The connectors' answers name only the kind and the type of the service that
answered; what the panel and the agent read as a source's label, the label of
an alarm configuration and the reason a raising service gives are made here.

The control system's own words for a channel and for the server that serves it
are made here too (:class:`Vocabulary`): an EPICS machine names them PV and
IOC, a Tango one attribute and device server, a DOOCS one property and server,
and a control system that states no words of its own gets the neutral ones,
channel and server. They are resolved from the connector type a control target
selects (:func:`target_vocabulary`), so they follow the target an answer comes
from. A facility connector class, named by a dotted ``control_system.type``,
states its own words in an optional class attribute,
:data:`VOCABULARY_ATTRIBUTE`.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache

from osprey_connectors.types import (
    CONTROL_TARGETS,
    DOOCS,
    EPICS,
    LIVE_STANDIN,
    TANGO,
    VIRTUAL_ACCELERATOR,
    resolve_target,
)

__all__ = [
    "CONFIG_LABELS",
    "EPICS_WORDS",
    "LOGBOOK_SERVICE",
    "NEUTRAL_WORDS",
    "VOCABULARY_ATTRIBUTE",
    "Vocabulary",
    "Words",
    "connector_vocabulary",
    "service_failed",
    "source_label",
    "target_vocabulary",
    "target_words",
    "words_of",
]

logger = logging.getLogger("osprey.services.channel_directory.vocabulary")

#: The service type of the logbook source.
LOGBOOK_SERVICE = "ariel"


@dataclass(frozen=True)
class Vocabulary:
    """A control system's words for a channel and for the server that serves it.

    The nouns are spelled as they read inside a sentence (``PV``, ``attribute``,
    ``device server``); a reader that starts a heading with one capitalises its
    first letter. ``directory`` and ``servers_view`` are the titles of the
    panel's directory view and of its server fleet view.
    """

    channel: str
    channels: str
    server: str
    servers: str
    directory: str
    servers_view: str

    def as_json(self) -> dict[str, str]:
        """The words as the panel and the agent are served them."""
        return asdict(self)


#: Every control target's words, by target name, as a service serves them.
Words = Mapping[str, Vocabulary]

#: The words of an EPICS machine: Channel Access and PV Access serve PVs, IOCs serve them.
EPICS_WORDS = Vocabulary(
    channel="PV",
    channels="PVs",
    server="IOC",
    servers="IOCs",
    directory="All PVs",
    servers_view="IOCs",
)

#: The words of a control system that states none.
NEUTRAL_WORDS = Vocabulary(
    channel="channel",
    channels="channels",
    server="server",
    servers="servers",
    directory="Channels",
    servers_view="Servers",
)

#: A Tango database lists the attributes of devices, served by device-server instances.
_TANGO_WORDS = Vocabulary(
    channel="attribute",
    channels="attributes",
    server="device server",
    servers="device servers",
    directory="Attributes",
    servers_view="Device servers",
)

#: A DOOCS address names a property, served by a server the ENS lists.
_DOOCS_WORDS = Vocabulary(
    channel="property",
    channels="properties",
    server="server",
    servers="servers",
    directory="Properties",
    servers_view="Servers",
)

#: The words of each built-in connector type. The virtual accelerator and the
#: live stand-in are EPICS-shaped soft IOCs: they serve Channel Access, and the
#: simulator's directory lists its servers as IOCs carrying IOC health PVs.
_BUILTIN_WORDS: dict[str, Vocabulary] = {
    EPICS: EPICS_WORDS,
    VIRTUAL_ACCELERATOR: EPICS_WORDS,
    LIVE_STANDIN: EPICS_WORDS,
    TANGO: _TANGO_WORDS,
    DOOCS: _DOOCS_WORDS,
}

#: The optional class attribute a facility connector states its words in: a
#: mapping with ``channel``, ``channels``, ``server`` and ``servers``, and
#: optionally ``directory`` and ``servers_view``, each a non-empty string. A
#: view title left out is its plural with the first letter capitalised.
VOCABULARY_ATTRIBUTE = "channel_vocabulary"

_REQUIRED = ("channel", "channels", "server", "servers")
_TITLES = (("directory", "channels"), ("servers_view", "servers"))

#: The label of an alarm configuration, by the kind of service that answered it.
CONFIG_LABELS = {
    "alarm_state": "live configuration",
    "alarm_log": "last logged configuration",
}


def _heading(word: str) -> str:
    return word[:1].upper() + word[1:]


def _stated_words(stated: object) -> Vocabulary | None:
    """The words a connector's :data:`VOCABULARY_ATTRIBUTE` states; ``None`` when malformed."""
    if not isinstance(stated, Mapping):
        return None
    words: dict[str, str] = {}
    for key in _REQUIRED:
        value = stated.get(key)
        if not isinstance(value, str) or not value:
            return None
        words[key] = value
    for key, plural in _TITLES:
        value = stated.get(key, _heading(words[plural]))
        if not isinstance(value, str) or not value:
            return None
        words[key] = value
    return Vocabulary(**words)


def _connector_class_words(connector_type: str) -> object:
    """The :data:`VOCABULARY_ATTRIBUTE` of the class a dotted type names, ``None`` if absent."""
    module_path, class_name = connector_type.rsplit(".", 1)
    connector_class = getattr(importlib.import_module(module_path), class_name)
    return getattr(connector_class, VOCABULARY_ATTRIBUTE, None)


@lru_cache(maxsize=64)
def connector_vocabulary(connector_type: str) -> Vocabulary:
    """The words of the control system *connector_type* names, resolved once per type.

    A built-in type has its own words. A dotted type names a facility's
    connector class, whose :data:`VOCABULARY_ATTRIBUTE` gives the words when it
    is present and well formed. Every other type, a class that does not import
    included, gets :data:`NEUTRAL_WORDS`.
    """
    builtin = _BUILTIN_WORDS.get(connector_type)
    if builtin is not None:
        return builtin
    if "." not in connector_type:
        return NEUTRAL_WORDS
    try:
        stated = _connector_class_words(connector_type)
    except Exception:
        logger.debug("cannot read the words of connector %r", connector_type, exc_info=True)
        return NEUTRAL_WORDS
    if stated is None:
        return NEUTRAL_WORDS
    words = _stated_words(stated)
    if words is None:
        logger.warning(
            "connector %r states a malformed %s; the neutral words are served",
            connector_type,
            VOCABULARY_ATTRIBUTE,
        )
        return NEUTRAL_WORDS
    return words


def target_vocabulary(control_system: object, target: str) -> Vocabulary:
    """The words of the control system *target* selects in a ``control_system:`` section.

    A target the section cannot resolve (``live`` on a deployment that names no
    real machine) gets :data:`NEUTRAL_WORDS`.
    """
    try:
        connector_type = resolve_target(control_system, target)
    except ValueError:
        return NEUTRAL_WORDS
    return connector_vocabulary(connector_type)


def target_words(config: Mapping[str, object]) -> dict[str, Vocabulary]:
    """Every control target's words in *config*, by target name."""
    control_system = config.get("control_system")
    return {target: target_vocabulary(control_system, target) for target in CONTROL_TARGETS}


def words_of(words: Words | None, target: str) -> Vocabulary:
    """*target*'s words in *words*; :data:`NEUTRAL_WORDS` when none are given for it."""
    return (words or {}).get(target, NEUTRAL_WORDS)


def source_label(kind: str, service: str, words: Vocabulary = NEUTRAL_WORDS) -> str:
    """The served label of a source: the directory's title in *words*, else the service's type."""
    return words.directory if kind == "directory" else service


def service_failed(failure: BaseException) -> str:
    """The served reason of a service that raised: its exception's class, never its text.

    A backend's exception can carry its URL, a login included; the caller logs
    the whole exception, and only the class name is served.
    """
    return f"the service failed: {type(failure).__name__}"

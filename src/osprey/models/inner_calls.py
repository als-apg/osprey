"""Which enabled features make their own model call, and whether each has its key.

The agent's model is reached by Claude Code, which can sign in with a personal
login when no API key is set. A handful of OSPREY features do not go through the
agent: they send a completion themselves, through :mod:`osprey.models`, and that
path authenticates with the provider's API key and nothing else. With no key the
agent works and these features do not.

:data:`FEATURES` is the one list of them. Each entry states, for a project
config, which provider (if any) the feature calls; :func:`inner_model_calls`
turns the list into one row per feature and provider with the key's variable and
whether it is set. ``osprey build``, ``osprey validate``, ``osprey web`` and the
session namer all read that function and keep no list of their own, so a feature
added here is reported by each of them.

A feature joins by adding one :class:`Feature` to :data:`FEATURES` in the change
that adds its ``get_chat_completion`` call.

The rule for "no key" is :func:`osprey.models.api_keys.api_key_missing`, which
this module and :func:`osprey.models.completion.get_chat_completion` both import,
so a feature is reported here exactly when its call would be refused there.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from osprey.models.api_keys import _placeholder_variable, api_key_missing
from osprey_connectors.config import is_unresolved_placeholder, resolve_env_vars

__all__ = [
    "FEATURES",
    "Feature",
    "SESSION_NAMING",
    "InnerModelCall",
    "inner_model_calls",
    "missing_key_warnings",
]


def _section(config: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    """The mapping at *path* in *config*; empty when any step is not a mapping."""
    node: Any = config
    for key in path:
        node = node.get(key) if isinstance(node, Mapping) else None
    return node if isinstance(node, Mapping) else {}


def _agent_provider(config: Mapping[str, Any]) -> list[str]:
    """``claude_code.provider``: the provider a feature with none of its own calls."""
    provider = _section(config, "claude_code").get("provider")
    return [provider] if isinstance(provider, str) and provider else []


def _in_context_channel_finder(config: Mapping[str, Any]) -> list[str]:
    # mcp_server/channel_finder_in_context/server_context.py: only the
    # in_context paradigm asks a model; its subagent_provider wins over the
    # agent's provider.
    if _section(config, "channel_finder").get("pipeline_mode") != "in_context":
        return []
    own = _section(config, "channel_finder", "pipelines", "in_context").get("subagent_provider")
    return [own] if isinstance(own, str) and own else _agent_provider(config)


def _logbook_compose(config: Mapping[str, Any]) -> list[str]:
    # interfaces/artifacts/logbook.py, _composition_provider.
    own = _section(config, "logbook", "composition").get("provider")
    return [own] if isinstance(own, str) and own else _agent_provider(config)


def _ariel_module(name: str) -> Callable[[Mapping[str, Any]], list[str]]:
    # services/ariel_search/enhancement/<name>: runs only when enabled, on the
    # module's own provider (ariel.embedding.provider never stands in).
    def providers(config: Mapping[str, Any]) -> list[str]:
        module = _section(config, "ariel", "enhancement_modules", name)
        provider = module.get("provider")
        if module.get("enabled") is not True or not isinstance(provider, str) or not provider:
            return []
        return [provider]

    return providers


def _health_model_chat(config: Mapping[str, Any]) -> list[str]:
    # health/core/model_chat.py: one completion per (provider, model_id) under
    # `models:`.
    from osprey.health.core.model_chat import unique_model_pairs

    return sorted({provider for provider, _ in unique_model_pairs(config)})


@dataclass(frozen=True, slots=True)
class Feature:
    """One feature that sends its own completion through :mod:`osprey.models`.

    Attributes:
        name: What an operator calls the feature.
        without_key: What the operator loses while the key is missing, as the
            tail of a sentence.
        providers: The providers the feature calls under a project config;
            empty when the config does not turn the feature on.
    """

    name: str
    without_key: str
    providers: Callable[[Mapping[str, Any]], list[str]]


#: services/session_naming.py: on in every web terminal, on the agent's provider.
#: Named because the session namer looks its own row up before it asks.
SESSION_NAMING = Feature(
    "session naming", "sessions keep their first-prompt title", _agent_provider
)

#: Every feature that makes its own model call. The one list: see the module
#: docstring for how a feature joins it.
FEATURES: tuple[Feature, ...] = (
    SESSION_NAMING,
    Feature(
        "logbook compose",
        "the compose panel cannot draft an entry",
        _logbook_compose,
    ),
    Feature(
        "in-context channel finder",
        "channel questions return an error",
        _in_context_channel_finder,
    ),
    Feature(
        "ARIEL semantic processor",
        "new logbook entries get no keywords or summary",
        _ariel_module("semantic_processor"),
    ),
    Feature(
        "ARIEL image captions",
        "pictures get no caption",
        _ariel_module("image_caption"),
    ),
    Feature(
        "health model chat",
        "`osprey health --full` reports the model as failing",
        _health_model_chat,
    ),
)


@dataclass(frozen=True, slots=True)
class InnerModelCall:
    """One enabled feature's own model call, and the state of the key it needs.

    Attributes:
        feature: The feature's name.
        provider: The provider the call goes to.
        key_env_var: The variable that provider's API key arrives in.
        key_present: Whether the key is set for this deployment.
        without_key: What the operator loses while it is not.
    """

    feature: str
    provider: str
    key_env_var: str
    key_present: bool
    without_key: str

    @property
    def warning(self) -> str:
        """The one sentence every command prints for a call whose key is missing."""
        return f"{self.feature} needs {self.key_env_var} (not set): {self.without_key}"


def inner_model_calls(
    config: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
) -> list[InnerModelCall]:
    """Each enabled feature that makes its own model call, with its key's state.

    A provider that needs no key (ollama, vllm, …) is never reported, nor is a
    name the provider registry does not hold: the first has nothing to miss, and
    the second is reported by the call itself.

    Args:
        config: The project's ``config.yml`` as a mapping, ``${VAR}`` references
            expanded or not.
        environ: The environment the deployment's processes read their keys
            from, used to expand a reference *config* still carries. ``None``
            reads ``os.environ``.

    Returns:
        One row per feature and provider, in :data:`FEATURES` order.
    """
    from osprey.models.provider_registry import get_provider_registry

    registry = get_provider_registry()
    catalog = _section(config, "api", "providers")
    calls: list[InnerModelCall] = []
    for feature in FEATURES:
        for provider in feature.providers(config):
            declared = registry.api_key_env_var(provider)
            if declared is None:
                continue
            entry = catalog.get(provider)
            raw_key = entry.get("api_key") if isinstance(entry, Mapping) else None
            key = resolve_env_vars(raw_key, environ=environ)
            # The variable the entry itself references is the one the deployment
            # reads; an entry with no reference left falls to the provider's own.
            env_var = _placeholder_variable(key) if is_unresolved_placeholder(key) else declared
            calls.append(
                InnerModelCall(
                    feature=feature.name,
                    provider=provider,
                    key_env_var=env_var,
                    key_present=not api_key_missing(key),
                    without_key=feature.without_key,
                )
            )
    return calls


def missing_key_warnings(
    config: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
) -> list[str]:
    """One sentence per enabled feature whose own model call has no key."""
    return [
        call.warning for call in inner_model_calls(config, environ=environ) if not call.key_present
    ]

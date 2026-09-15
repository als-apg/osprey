"""Shared base for thin LiteLLM-delegating provider adapters.

Several providers do nothing but forward :meth:`execute_completion` and
:meth:`check_health` to the shared ``litellm_adapter`` helpers, differing only
in their class-attribute metadata (name, routing flags, default models, API-key
help). Those providers -- anthropic, als-apg, amsc-i2, cborg, google, openai,
stanford -- subclass :class:`LiteLLMDelegatingProvider` and supply only class
attributes; the two delegating methods live here, once.

**Patch-target preservation.** Each provider's test suite patches
``osprey.models.providers.<name>.execute_litellm_completion`` (and
``check_litellm_health``) -- the helper name bound in that concrete provider's
own module. The method bodies live here, so to keep those patches effective
the helpers are looked up from the concrete subclass's module namespace
at call time (via ``sys.modules[type(self).__module__]``) rather than captured
in this module's globals. Every subclass module therefore still imports the two
helpers at module level, which is what the patch targets rebind. If a subclass
module omits those imports (or an intermediate subclass lives in another
module), the lookup falls back to the real helpers imported here instead of
failing with an AttributeError.
"""

import sys
from typing import Any

from .base import BaseProvider
from .litellm_adapter import check_litellm_health, execute_litellm_completion


class LiteLLMDelegatingProvider(BaseProvider):
    """Data-driven base for providers that only delegate to ``litellm_adapter``.

    Subclasses override the :class:`BaseProvider` metadata attributes and
    inherit the ``execute_completion`` / ``check_health`` bodies unchanged.
    """

    # The endpoint is resolved through BaseProvider.resolve_base_url, which
    # refuses a provider that requires one and has no source for it. The refusal
    # belongs here because a direct adapter call never passes through
    # osprey.models.completion: without it such a call reached litellm with no
    # api_base, and a gateway-fronted provider then failed with an
    # authentication error against a host nobody configured.

    def execute_completion(
        self,
        message: str,
        model_id: str,
        api_key: str | None,
        base_url: str | None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        thinking: dict | None = None,
        system_prompt: str | None = None,
        output_format: Any | None = None,
        **kwargs,
    ) -> str | Any:
        """Execute a chat completion by delegating to the litellm helper."""
        execute = getattr(
            sys.modules[type(self).__module__],
            "execute_litellm_completion",
            execute_litellm_completion,
        )
        return execute(
            provider=self.name,
            message=message,
            model_id=model_id,
            api_key=api_key,
            base_url=self.resolve_base_url(base_url),
            max_tokens=max_tokens,
            temperature=temperature,
            output_format=output_format,
            **kwargs,
        )

    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 5.0,
        model_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check provider health by delegating to the litellm health helper."""
        check = getattr(
            sys.modules[type(self).__module__],
            "check_litellm_health",
            check_litellm_health,
        )
        result: tuple[bool, str] = check(
            provider=self.name,
            api_key=api_key,
            base_url=self.resolve_base_url(base_url),
            timeout=timeout,
            model_id=model_id or self.health_check_model_id,
        )
        return result

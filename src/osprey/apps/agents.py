"""An application's subagent: the frontmatter of ``agents/<name>.md``.

The file is what Claude Code reads from ``.claude/agents/``; the build copies it
there as written, adding only the ``model:`` line the deployment decides. Its
frontmatter is an allowlist: a key OSPREY has not decided on is refused, and
each key an agent file may carry elsewhere but an application may not is
refused with what to do instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    field_validator,
)

from osprey.apps.errors import AppManifestError, Violation
from osprey.apps.manifest import NO_HOOKS_MESSAGE, NonEmpty, Slug

#: The directory of an application holding its agent files, one ``<name>.md`` each.
AGENTS_DIR = "agents"

#: The placeholder in ``delegation`` that the delegated tools replace.
DELEGATED_TOOLS = "{delegated_tools}"

#: Keys an agent file may carry elsewhere, refused in an application: rule, message.
REFUSED_KEYS: dict[str, tuple[str, str]] = {
    "model": (
        "agent_key_refused",
        "the build sets the model from `claude_code.agent_models`; remove model:",
    ),
    "hooks": ("no_hooks", NO_HOOKS_MESSAGE),
    "permissionMode": (
        "agent_key_refused",
        "the login's tier and the write switch decide what an agent may do; remove permissionMode:",
    ),
    "mcpServers": (
        "agent_key_refused",
        "an app's tools are its `tools:` server; remove mcpServers:",
    ),
}


def delegated_tools_phrase(tools: Sequence[str]) -> str:
    """The tools in the brief's house form: `` `a` ``, `` `a` or `b` ``, `` `a`, `b`, or `c` ``."""
    quoted = [f"`{tool}`" for tool in tools]
    if len(quoted) <= 2:
        return " or ".join(quoted)
    return ", ".join(quoted[:-1]) + ", or " + quoted[-1]


class AppAgent(BaseModel):
    """The frontmatter of one application agent file."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    name: Slug
    #: What the agent is for; Claude Code routes by it.
    description: NonEmpty
    summary: NonEmpty | None = None
    #: The tools the agent may call: an application's agent never inherits every tool.
    tools: list[NonEmpty] = Field(min_length=1)
    disallowed_tools: list[NonEmpty] | None = Field(default=None, alias="disallowedTools")
    max_turns: PositiveInt | None = Field(default=None, alias="maxTurns")
    #: The note the main agent's brief carries for this agent.
    delegation: NonEmpty
    #: Tools of the application the main agent is told to leave to this agent.
    delegates: list[NonEmpty] = Field(default_factory=list)

    @field_validator("tools", "disallowed_tools", mode="before")
    @classmethod
    def _comma_line(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    def brief_note(self) -> str:
        """``delegation`` with the delegated tools in place of ``{delegated_tools}``."""
        return self.delegation.replace(DELEGATED_TOOLS, delegated_tools_phrase(self.delegates))


def parse_agent(text: str, *, app: str, name: str, source: str | None = None) -> AppAgent:
    """Read one application agent file.

    Args:
        text: The file's contents.
        app: The application carrying the agent.
        name: The name the manifest lists, which the file's ``name`` must equal.
        source: The file a refusal names when it is not the application's ``app.yml``.

    Raises:
        AppManifestError: Carrying every violation of the file, each on field
            ``agents.<name>.<key>``.
    """
    from osprey.cli.validate_claude_artifacts import _parse_frontmatter, _unreadable_frontmatter

    at = f"agents.{name}"
    frontmatter = _parse_frontmatter(text)
    if frontmatter is None or _unreadable_frontmatter(text):
        raise AppManifestError.from_violations(
            [
                Violation(
                    app,
                    at,
                    "agent_frontmatter",
                    f"{AGENTS_DIR}/{name}.md opens with no ---\\n...\\n--- frontmatter mapping",
                    source=source,
                )
            ]
        )
    found = [
        Violation(app, f"{at}.{key}", rule, message, source=source)
        for key, (rule, message) in REFUSED_KEYS.items()
        if key in frontmatter
    ]
    allowed = {k: v for k, v in frontmatter.items() if k not in REFUSED_KEYS}
    if "name" not in allowed:
        found.append(
            Violation(app, f"{at}.name", "missing", f"name is required: {name!r}", source=source)
        )
    elif allowed["name"] != name:
        found.append(
            Violation(
                app,
                f"{at}.name",
                "agent_name",
                f"name is {allowed['name']!r}; it is the file name, {name!r}",
                source=source,
            )
        )
    agent: AppAgent | None = None
    try:
        agent = AppAgent.model_validate({**allowed, "name": name})
    except ValidationError as exc:
        for error in exc.errors(include_url=False):
            field = ".".join(str(part) for part in error["loc"])
            found.append(
                Violation(app, f"{at}.{field}", error["type"], error["msg"], source=source)
            )
    if found or agent is None:
        raise AppManifestError.from_violations(found)
    return agent

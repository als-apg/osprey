"""The application manifest: the schema of ``apps/<name>/app.yml``.

Every level refuses unknown keys. A container key that belongs to compose (its
binds, volumes, network, ...) is refused with a message naming where it lives
instead: the application's compose fragment, or for ``network`` the profile's
``apps.<name>.services.<key>.network``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    StringConstraints,
    Tag,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from osprey.apps.errors import AppManifestError, Violation

if TYPE_CHECKING:
    from osprey.apps.origin import AppOrigin

APP_NAME_PATTERN = r"^[a-z0-9-]+$"

#: Methods an agent route may declare.
HTTP_METHODS = ("GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE")

#: Methods that only read. An agent route with any other method is a write route.
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: The one template a ``{package, version, file}`` vendor entry resolves through.
PACKAGE_URL_TEMPLATE = "https://cdn.jsdelivr.net/npm/{package}@{version}/{file}"

#: Container keys that belong to the compose service, not the manifest.
_FRAGMENT_KEYS = ("binds", "volumes", "env_file", "depends_on", "ports")

_PANEL_PREFIX = "/panel"

Slug = Annotated[str, StringConstraints(pattern=APP_NAME_PATTERN)]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
EnvName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
ModuleName = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ExactVersion = Annotated[
    str,
    StringConstraints(pattern=r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$"),
]


def _backend_path(value: str) -> str:
    if not value.startswith("/"):
        raise ValueError(f"{value!r} must start with '/' (a path from the backend's root)")
    if value == _PANEL_PREFIX or value.startswith(_PANEL_PREFIX + "/"):
        raise ValueError(
            f"{value!r} spells the proxy prefix; give the path from the backend's root, "
            "the /panel/<id> prefix is added by OSPREY"
        )
    return value


BackendPath = Annotated[str, AfterValidator(_backend_path)]


def launch_token_env(app_name: str) -> str:
    """Return the environment variable that carries an application's launch token.

    The name is the application name upper-cased with ``-`` turned into ``_``,
    followed by ``_LAUNCH_TOKEN``. This is the only place the name is derived.

    Raises:
        ValueError: ``app_name`` is not an application name.
    """
    if not re.fullmatch(APP_NAME_PATTERN, app_name):
        raise ValueError(f"{app_name!r} is not an application name ([a-z0-9-]+)")
    return app_name.upper().replace("-", "_") + "_LAUNCH_TOKEN"


def is_write_method(method: str) -> bool:
    """Return whether an agent route with this HTTP method is a write route."""
    return method.upper() not in READ_METHODS


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Container(_Strict):
    """One container of the application; its key is the compose service name."""

    runtime: Literal["osprey"] | None = None
    image: NonEmpty | None = None
    command: list[str] | None = None
    fragment: str | None = None
    port: Annotated[StrictInt, Field(ge=1, le=65535)] | None = None
    health: BackendPath | None = None
    drives_target: StrictBool = False
    control_identity: StrictBool = False
    state: NonEmpty | None = None
    env: list[EnvName] = Field(default_factory=list)

    @field_validator("env", mode="before")
    @classmethod
    def _env_is_a_list(cls, value: Any) -> Any:
        if isinstance(value, dict):
            raise PydanticCustomError(
                "env_names",
                "container env is a list of host variable names (the service env axis), "
                "not a mapping; set values on the host",
            )
        return value

    @model_validator(mode="after")
    def _runtime_or_image(self) -> Container:
        if (self.runtime is None) == (self.image is None):
            raise PydanticCustomError(
                "runtime_or_image", "a container sets exactly one of runtime: osprey or image:"
            )
        return self


class AgentRoute(_Strict):
    """An HTTP route of the panel's backend the agent may call."""

    method: Literal["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"]
    path: BackendPath

    @model_validator(mode="before")
    @classmethod
    def _from_pair(cls, value: Any) -> Any:
        if isinstance(value, list | tuple):
            if len(value) != 2:
                raise PydanticCustomError(
                    "agent_route_pair", "an agent route is a [METHOD, <path>] pair"
                )
            return {"method": value[0], "path": value[1]}
        return value

    @property
    def is_write(self) -> bool:
        """Whether the route writes: any method but GET, HEAD and OPTIONS."""
        return is_write_method(self.method)


class RosterGrant(_Strict):
    single_user_alias: StrictBool = False


class Panel(_Strict):
    """The application's web panel, served by one of its containers."""

    id: Slug
    label: NonEmpty
    container: NonEmpty
    entry: BackendPath = "/"
    launch_token: StrictBool = False
    read_timeout_s: Annotated[StrictInt, Field(gt=0)] | None = None
    agent_routes: list[AgentRoute] = Field(default_factory=list)
    roster_grant: RosterGrant | None = None


class StateField(_Strict):
    """Copy the fetched state's ``key`` into the stamp field ``field``.

    A dotted ``key`` (``step.digest``) is a path through nested mappings of the
    state. A ``key`` the state does not hold, or a path that crosses a value
    that is not a mapping, leaves the field unbound.
    """

    field: NonEmpty
    key: NonEmpty


class InputDigest(_Strict):
    """Bind ``field`` to the digest of the call's inputs named in ``over``.

    Every digest is required: ``require.all_of`` names it, and a call missing an
    ``over`` key is refused (``approval_missing``) before the prompt. A tool that
    takes two input shapes is two tools, each with its own digest.
    """

    field: NonEmpty
    over: list[NonEmpty] = Field(min_length=1)


class Bind(_Strict):
    state_fields: list[StateField] = Field(default_factory=list)
    input_digests: list[InputDigest] = Field(default_factory=list)


class Require(_Strict):
    """Stamp fields a call needs: all of ``all_of``, exactly one of each ``one_of`` group.

    A ``one_of`` group holds readback (``bind.state_fields``) names only, never a
    digest (:func:`osprey.apps.rules.validate` refuses one, ``one_of_readbacks``).
    """

    all_of: list[NonEmpty] = Field(default_factory=list)
    one_of: list[Annotated[list[NonEmpty], Field(min_length=1)]] = Field(default_factory=list)


class Approval(_Strict):
    """How one write tool is shown to the approver and bound to what was approved."""

    tool: NonEmpty
    prompt: NonEmpty
    state_path: BackendPath | None = None
    display_lines_key: NonEmpty | None = None
    bind: Bind = Field(default_factory=Bind)
    require: Require = Field(default_factory=Require)


class ToolErrors(_Strict):
    unreachable: NonEmpty | None = None
    refused: NonEmpty | None = None


class Tools(_Strict):
    """The application's MCP server, a Python module run beside the agent."""

    module: ModuleName
    shown_when: Literal["always", "panel"] = "always"
    deny_in_jobs: StrictBool = False
    read: list[NonEmpty] = Field(default_factory=list)
    approval: list[Approval] = Field(default_factory=list)
    errors: ToolErrors | None = None


class UrlVendorEntry(_Strict):
    """A file fetched from an explicit URL into the application directory."""

    url: Annotated[str, StringConstraints(pattern=r"^https://\S+$")]
    into: NonEmpty
    sha256: Sha256

    @property
    def source_url(self) -> str:
        return self.url


class PackageVendorEntry(_Strict):
    """One file of an npm package at an exact version, fetched through the CDN template."""

    package: Annotated[
        str, StringConstraints(pattern=r"^(@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*$")
    ]
    version: ExactVersion
    file: Annotated[str, StringConstraints(pattern=r"^[^/\s]\S*$")]
    into: NonEmpty
    sha256: Sha256

    @property
    def source_url(self) -> str:
        return PACKAGE_URL_TEMPLATE.format(
            package=self.package, version=self.version, file=self.file
        )


def _vendor_source(value: Any) -> str | None:
    if isinstance(value, dict):
        has_url, has_package = "url" in value, "package" in value
        if has_url != has_package:
            return "url" if has_url else "package"
        return None
    if isinstance(value, UrlVendorEntry):
        return "url"
    if isinstance(value, PackageVendorEntry):
        return "package"
    return None


VendorEntry = Annotated[
    Annotated[UrlVendorEntry, Tag("url")] | Annotated[PackageVendorEntry, Tag("package")],
    Discriminator(
        _vendor_source,
        custom_error_type="vendor_source",
        custom_error_message="a vendor entry names exactly one source: url or package",
    ),
]


class AppManifest(_Strict):
    """An application's ``app.yml``."""

    name: Slug
    summary: NonEmpty | None = None
    #: The application's containers, by compose service name; none for an application
    #: of a tool server, skills or both. A backend-less application keeps no state and
    #: opens no deployment data store: its tool server runs beside the agent, and
    #: keeping to that is the tool server author's duty, since no check can see it.
    containers: dict[NonEmpty, Container] = Field(default_factory=dict)
    panel: Panel | None = None
    tools: Tools | None = None
    #: Names the containers' spool and state directories; defaults to ``name``.
    audit_identity: NonEmpty = ""
    vendor: list[VendorEntry] = Field(default_factory=list)
    #: The agent skills the application carries, each ``skills/<name>/SKILL.md`` in its directory.
    skills: list[Slug] = Field(default_factory=list)
    _audit_identity_spelled: bool = PrivateAttr(default=False)

    @property
    def audit_identity_spelled(self) -> bool:
        """Whether ``app.yml`` spells ``audit_identity`` rather than taking the default."""
        return self._audit_identity_spelled

    #: Where the loader found the manifest: a fact of discovery, never a key of ``app.yml``.
    _origin: AppOrigin | None = PrivateAttr(default=None)

    @property
    def origin(self) -> AppOrigin | None:
        """Where the application's directory is; ``None`` for a manifest no loader read."""
        return self._origin

    def bind_origin(self, origin: AppOrigin) -> AppManifest:
        """Record where the loader found this manifest, and return it."""
        self._origin = origin
        return self

    @model_validator(mode="before")
    @classmethod
    def _defaults_from_name(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            return data
        name = data["name"]
        data = dict(data)
        panel = data.get("panel")
        if isinstance(panel, dict) and "id" not in panel:
            data["panel"] = {"id": name, **panel}
        return data

    @model_validator(mode="after")
    def _has_a_part(self) -> AppManifest:
        if not (self.containers or self.panel or self.tools or self.skills):
            raise PydanticCustomError(
                "app_has_part",
                "an application declares at least one of containers, panel, tools or skills",
                {"at": ""},
            )
        self._audit_identity_spelled = "audit_identity" in self.model_fields_set
        if not self._audit_identity_spelled:
            self.audit_identity = self.name
        return self

    @field_validator("containers", mode="before")
    @classmethod
    def _compose_keys_name_their_home(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, dict):
            return value
        app = info.data.get("name") or "<name>"
        for key, raw in value.items():
            if not isinstance(raw, dict):
                continue
            fragment = f"compose/{key}/docker-compose.yml.j2"
            for compose_key in raw:
                if compose_key == "network":
                    home = f"profile.yml under apps.{app}.services.{key}.network"
                elif compose_key in _FRAGMENT_KEYS or (
                    compose_key == "command" and "image" in raw and "runtime" not in raw
                ):
                    home = f"the application's fragment {fragment}"
                else:
                    continue
                raise PydanticCustomError(
                    "compose_key",
                    f"{compose_key} is a compose key, not an application key: set it in {home}",
                    {"at": f"containers.{key}.{compose_key}"},
                )
        return value

    @field_validator("containers", mode="after")
    @classmethod
    def _fragment_is_the_container_directory(
        cls, value: dict[str, Container]
    ) -> dict[str, Container]:
        for key, container in value.items():
            if container.fragment is None:
                continue
            at = {"at": f"containers.{key}.fragment"}
            if container.runtime is not None:
                raise PydanticCustomError(
                    "image_only", "fragment is for image: containers only", at
                )
            directory = f"compose/{key}/"
            if container.fragment.rstrip("/") + "/" != directory:
                raise PydanticCustomError(
                    "fragment_directory",
                    f"fragment names the directory {directory} holding docker-compose.yml.j2, "
                    f"not {container.fragment!r}",
                    at,
                )
            container.fragment = directory
        return value


def parse_manifest(data: Any, *, app: str, source: str | None = None) -> AppManifest:
    """Validate the parsed contents of one ``app.yml``.

    Args:
        data: The YAML document as loaded.
        app: The application's name, used in the refusal.
        source: The file a refusal names when it is not ``apps/<app>/app.yml``:
            an installed application's manifest
            (:attr:`~osprey.apps.origin.AppOrigin.manifest_source`).

    Raises:
        AppManifestError: The manifest is refused; it carries every violation.
    """
    if not isinstance(data, dict):
        raise AppManifestError.from_violations(
            [Violation(app, "", "mapping", "app.yml must be a mapping of keys", source=source)]
        )
    try:
        return AppManifest.model_validate(data)
    except ValidationError as exc:
        violations = []
        for error in exc.errors(include_url=False):
            ctx = error.get("ctx") or {}
            field = ctx.get("at") or ".".join(str(part) for part in error["loc"])
            violations.append(Violation(app, field, error["type"], error["msg"], source=source))
        raise AppManifestError.from_violations(violations) from None

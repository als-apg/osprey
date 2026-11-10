"""The deploy and CLI pages quote the names, shapes and messages the code holds.

Each block is found by page, language and a distinctive substring (never by line
number) through the snippet-coverage scanner, then compared with the code that
writes, reads or prints what it shows:

* ``how-to/deploy-project/project-image.rst``: the three ``.gguf`` names are
  ``qmd_service.MODEL_FILENAMES``, both directions; the proxy ``RUN`` opens with
  the bridge every shipped recipe spells, the project ``Dockerfile.j2`` among
  them, and its apt steps are a shipped recipe's.
* ``how-to/deploy-project/index.rst``: the docker and podman compose invocations
  are what ``compose_base_cmd`` builds for a rendered repo, flag by flag, with
  the repo and the service as the variable parts; the ``[network]`` ini block is
  the backend ``container_lifecycle`` requires and the setting its refusal
  prints.
* ``how-to/deploy-project/env-chain.rst``, ``how-to/deploy-a-facility.rst``,
  ``how-to/build-profiles.rst``: each ``cp``/``mv`` names the env files the
  build, the compose chain and the web-tier migration use.
* ``how-to/llm-providers/configure-providers.rst``: the policy JSON is read by
  ``read_managed_policy_env`` and agrees with the launch of a ``/v1`` gateway.
* ``reference/configuration/profile.rst``: the derived-key refusal is what
  ``derived_key_errors`` returns.
* ``how-to/agent-interfaces/cli-agent.rst``: the companion block is what
  ``osprey chat`` prints for the gallery and ARIEL at the default port base.
* ``reference/configuration/config.rst``: the image-name pattern is the order
  ``resolve_image_defaults`` composes.
"""

from __future__ import annotations

import ast
import configparser
import inspect
import json
import re
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest

from osprey.agent_runner.provider_env import (
    detect_managed_policy_conflicts,
    inject_provider_env,
    load_provider_spec,
    read_managed_policy_env,
)
from osprey.cli import chat_cmd, output, profile_cmd
from osprey.cli.derived_keys import derived_key_errors
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import (
    _OSPREY_IMAGE_SUFFIXES,
    LABELS_OVERRIDE_FILENAME,
    compose_base_cmd,
    compose_provider_env,
    labels_override_path,
    resolve_image_defaults,
)
from osprey.deployment.compose_merge import MERGED_COMPOSE_FILENAME
from osprey.deployment.container_lifecycle import (
    _REQUIRED_PODMAN_NETWORK_BACKEND,
    ENV_MERGED_RELPATH,
)
from osprey.deployment.qmd_service import MODEL_FILENAMES
from osprey.deployment.runtime_helper import ComposeProvider
from osprey.deployment.web_terminals.env_production import (
    LEGACY_USERS_ENV_FILENAME,
    USERS_ENV_FILENAME,
    migrate_users_env,
)
from osprey.interfaces.common_middleware import EXTERNAL_ORIGIN_ENV, WEB_PORT_ENV
from osprey.registry.web import FRAMEWORK_WEB_SERVERS, framework_web_port_default
from osprey_connectors.dotenv import ENV_CHAIN_FILENAMES, ENV_LOCAL_FILENAME
from tests._by_path import load_module
from tests.docs.test_env_production_retired import RETIRED_NAME

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs" / "source"
_SCANNER = _REPO_ROOT / "scripts" / "docs" / "snippet_coverage.py"
_TEMPLATES = _REPO_ROOT / "src" / "osprey" / "templates"

PROJECT_IMAGE_PAGE = "how-to/deploy-project/project-image.rst"
DEPLOY_INDEX_PAGE = "how-to/deploy-project/index.rst"
ENV_CHAIN_PAGE = "how-to/deploy-project/env-chain.rst"
DEPLOY_FACILITY_PAGE = "how-to/deploy-a-facility.rst"
BUILD_PROFILES_PAGE = "how-to/build-profiles.rst"
PROVIDERS_PAGE = "how-to/llm-providers/configure-providers.rst"
PROFILE_PAGE = "reference/configuration/profile.rst"
CLI_AGENT_PAGE = "how-to/agent-interfaces/cli-agent.rst"
CONFIG_PAGE = "reference/configuration/config.rst"


@lru_cache(maxsize=1)
def _scanner() -> ModuleType:
    """The snippet-coverage scanner, loaded by path; registered for its dataclasses."""
    return load_module("snippet_coverage", _SCANNER, register=True)


def _page_text(page: str) -> str:
    return (_DOCS / page).read_text(encoding="utf-8")


def _blocks(page: str, language: str, needle: str) -> list[str]:
    return [
        block.text
        for block in _scanner().scan_page(_page_text(page), page, _DOCS)
        if block.language == language and needle in block.text
    ]


def _block(page: str, language: str, needle: str) -> str:
    """The one block on *page* in *language* whose text holds *needle*."""
    found = _blocks(page, language, needle)
    assert len(found) == 1, f"{page}: {len(found)} {language} blocks hold {needle!r}"
    return found[0]


# -- how-to/deploy-project/project-image.rst -----------------------------------


def _model_names(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def test_the_model_file_names_are_the_ones_the_sidecar_checks() -> None:
    names = _model_names(_block(PROJECT_IMAGE_PAGE, "text", ".gguf"))
    assert names == list(MODEL_FILENAMES)


def test_a_planted_model_name_is_caught() -> None:
    names = _model_names(_block(PROJECT_IMAGE_PAGE, "text", ".gguf"))
    names[1] = "qwen3-reranker-0.6b-q8_0.gguf"  # the bare huggingface basename
    assert names != list(MODEL_FILENAMES)


def _logical_lines(text: str) -> list[str]:
    """Dockerfile lines with continuations joined and runs of whitespace collapsed."""
    joined = re.sub(r"\\\n", " ", text)
    return [" ".join(line.split()) for line in joined.splitlines() if line.strip()]


def _shell_steps(run: str) -> list[str]:
    """A ``RUN`` line's shell, split into its ``;``/``&&`` steps, mounts dropped."""
    shell = re.sub(r"^RUN\s+(--mount=\S+\s+)*", "", run)
    return [step.strip() for step in re.split(r";|&&", shell) if step.strip()]


def _shipped_runs() -> dict[str, list[str]]:
    """Every ``RUN`` logical line of every shipped Dockerfile, by template path."""
    recipes = [*_TEMPLATES.rglob("Dockerfile"), *_TEMPLATES.rglob("Dockerfile.j2")]
    return {
        str(path.relative_to(_TEMPLATES)): [
            line
            for line in _logical_lines(path.read_text(encoding="utf-8"))
            if line.startswith("RUN ")
        ]
        for path in recipes
    }


def _proxy_steps() -> list[str]:
    (run,) = _logical_lines(_block(PROJECT_IMAGE_PAGE, "dockerfile", "http_proxy"))
    return _shell_steps(run)


def _bridge_problems(bridge: str, runs: dict[str, list[str]]) -> list[str]:
    """Shipped bridges spelled otherwise than *bridge*, and recipes lacking it."""
    problems = []
    for path, lines in runs.items():
        exports = [step for line in lines for step in _shell_steps(line) if "http_proxy=" in step]
        problems += [f"{path}: {step}" for step in exports if step != bridge]
        if path == "project/Dockerfile.j2" and bridge not in exports:
            problems.append(f"{path}: no bridge line")
    return problems


def test_the_proxy_bridge_is_the_one_every_shipped_recipe_spells() -> None:
    bridge = _proxy_steps()[0]
    assert bridge.startswith("export http_proxy=")
    assert _bridge_problems(bridge, _shipped_runs()) == []


def test_a_planted_bridge_spelling_is_caught() -> None:
    bridge = _proxy_steps()[0].replace("${HTTP_PROXY:-}", "${HTTP_PROXY}")
    assert _bridge_problems(bridge, _shipped_runs()) != []


def _recipes_running(steps: list[str], runs: dict[str, list[str]]) -> list[str]:
    """Recipes with one ``RUN`` holding every one of *steps*, in that order."""
    found = []
    for path, lines in runs.items():
        for line in lines:
            shipped = iter(_shell_steps(line))
            if all(step in shipped for step in steps):
                found.append(path)
                break
    return found


def test_the_proxy_example_is_a_step_sequence_a_shipped_recipe_runs() -> None:
    assert _recipes_running(_proxy_steps(), _shipped_runs()) != []


def test_a_planted_apt_step_is_caught() -> None:
    steps = [*_proxy_steps()[:-1], "apt-get install -y ca-certificates"]
    assert _recipes_running(steps, _shipped_runs()) == []


# -- how-to/deploy-project/index.rst: compose invocations ----------------------

_SERVICE = "postgresql"


def _rendered_repo(root: Path) -> Path:
    """A repo with one rendered service, the labels override and both chain files."""
    compose = root / "build" / "services" / _SERVICE / "docker-compose.yml"
    compose.parent.mkdir(parents=True)
    compose.write_text(f"services:\n  {_SERVICE}:\n    image: postgres:16\n", encoding="utf-8")
    override = labels_override_path(root)
    override.write_text(
        f"services:\n  {_SERVICE}:\n    labels:\n      osprey.service: {_SERVICE}\n",
        encoding="utf-8",
    )
    for name in ENV_CHAIN_FILENAMES:
        (root / name).write_text("A=1\n", encoding="utf-8")
    return compose


def _page_argv(text: str) -> tuple[list[str], list[str]]:
    """The block's argv tokens (parentheticals dropped) and its comment lines."""
    comments = [line.strip() for line in text.splitlines() if line.strip().startswith("#")]
    argv = " ".join(
        re.sub(r"\(.*?\)", "", line) for line in text.splitlines() if line.strip()[:1] != "#"
    )
    return argv.split(), comments


def _generic(argv: list[str], root: Path) -> list[str]:
    """*argv* with the repo and the service spelled as the page's placeholders."""
    return [
        token.replace(str(root), "<repo>").replace(f"/{_SERVICE}/", "/<service>/") for token in argv
    ]


def _docker_block() -> str:
    return _block(DEPLOY_INDEX_PAGE, "text", "docker compose --project-directory")


def _podman_block() -> str:
    return _block(DEPLOY_INDEX_PAGE, "text", "podman compose")


def test_the_docker_invocation_is_the_one_compose_base_cmd_builds(tmp_path: Path) -> None:
    compose = _rendered_repo(tmp_path)
    built = compose_base_cmd(["docker", "compose"], [compose], tmp_path)
    page, _ = _page_argv(_docker_block())
    assert _generic(built, tmp_path) == page
    assert f"<repo>/build/{LABELS_OVERRIDE_FILENAME}" in page
    assert [f"<repo>/{name}" for name in ENV_CHAIN_FILENAMES] == [
        page[i + 1] for i, token in enumerate(page) if token == "--env-file"
    ]


def test_the_podman_invocation_is_the_one_compose_base_cmd_builds(tmp_path: Path) -> None:
    compose = _rendered_repo(tmp_path)
    built = compose_base_cmd(
        ["podman", "compose"], [compose], tmp_path, provider=ComposeProvider.PODMAN_COMPOSE
    )
    page, comments = _page_argv(_podman_block())
    assert _generic(built, tmp_path) == page
    assert f"<repo>/{MERGED_COMPOSE_FILENAME}" in page
    assert f"<repo>/{ENV_MERGED_RELPATH.as_posix()}" in page
    env = compose_provider_env(ComposeProvider.PODMAN_COMPOSE, tmp_path)
    (variable,) = env
    assert env[variable] == str(tmp_path.absolute())
    assert comments == [f"# plus {variable}=<repo> in the command's environment"]


@pytest.mark.parametrize(
    "plant",
    [
        lambda text: text.replace("--env-file <repo>/.env.shared ", ""),
        lambda text: text.replace("-f <repo>/build/osprey-labels.override.yml", ""),
        lambda text: text.replace("--project-directory <repo>", "--project-dir <repo>"),
    ],
    ids=["one-env-file", "no-labels-override", "wrong-flag"],
)
def test_a_planted_docker_flag_is_caught(plant, tmp_path: Path) -> None:
    compose = _rendered_repo(tmp_path)
    built = compose_base_cmd(["docker", "compose"], [compose], tmp_path)
    page, _ = _page_argv(plant(_docker_block()))
    assert _generic(built, tmp_path) != page


def test_a_planted_podman_file_name_is_caught(tmp_path: Path) -> None:
    compose = _rendered_repo(tmp_path)
    built = compose_base_cmd(
        ["podman", "compose"], [compose], tmp_path, provider=ComposeProvider.PODMAN_COMPOSE
    )
    page, _ = _page_argv(_podman_block().replace(".env.merged", ".env"))
    assert _generic(built, tmp_path) != page


# -- how-to/deploy-project/index.rst: network backend --------------------------


def _network_block() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read_string(_block(DEPLOY_INDEX_PAGE, "ini", "netavark"))
    return parser


def _refusal_lines(monkeypatch: pytest.MonkeyPatch, backend: str) -> list[str]:
    """The ``[network]`` lines the preflight prints for a podman host on *backend*."""
    monkeypatch.setattr(container_lifecycle, "get_runtime_command", lambda config: ["podman"])
    monkeypatch.setattr(container_lifecycle, "_podman_network_backend", lambda runtime: backend)
    with pytest.raises(RuntimeError) as excinfo:
        container_lifecycle._preflight_internal_network_backend({"deployed_services": ["bluesky"]})
    lines = [line.strip() for line in str(excinfo.value).splitlines()]
    start = lines.index("[network]")
    return lines[start : start + 2]


def test_the_network_backend_is_the_one_the_preflight_requires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _network_block()["network"]["network_backend"]
    assert json.loads(value) == _REQUIRED_PODMAN_NETWORK_BACKEND
    page = [line.strip() for line in _block(DEPLOY_INDEX_PAGE, "ini", "netavark").splitlines()]
    assert page == _refusal_lines(monkeypatch, "cni")


def test_the_required_backend_passes_the_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    value = json.loads(_network_block()["network"]["network_backend"])
    monkeypatch.setattr(container_lifecycle, "get_runtime_command", lambda config: ["podman"])
    monkeypatch.setattr(container_lifecycle, "_podman_network_backend", lambda runtime: value)
    container_lifecycle._preflight_internal_network_backend({"deployed_services": ["bluesky"]})


def test_a_planted_backend_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    planted = _block(DEPLOY_INDEX_PAGE, "ini", "netavark").replace("netavark", "cni")
    page = [line.strip() for line in planted.splitlines()]
    assert page != _refusal_lines(monkeypatch, "cni")


# -- env-chain, deploy-a-facility, build-profiles: env file names --------------

_ENV_FILE_PAGES = (ENV_CHAIN_PAGE, DEPLOY_FACILITY_PAGE, BUILD_PROFILES_PAGE)
_FILE_COMMAND = re.compile(r"^(cp|mv) (\.env\S*) (\.env\S*)$")


def _file_commands() -> list[tuple[str, str, str, str]]:
    """``(page, verb, source, target)`` for every ``cp``/``mv .env…`` line."""
    commands = []
    for page in _ENV_FILE_PAGES:
        for block in _scanner().scan_page(_page_text(page), page, _DOCS):
            if block.language != "bash":
                continue
            for line in block.text.splitlines():
                match = _FILE_COMMAND.match(line.strip())
                if match:
                    commands.append((page, *match.groups()))
    return commands


def _known_moves() -> set[tuple[str, str, str]]:
    """The copies and renames the code spells: seed the secrets file, migrate the web env."""
    return {
        ("cp", profile_cmd._PROFILE_ENV_EXAMPLE_FILENAME, profile_cmd._PROFILE_ENV_FILENAME),
        ("mv", LEGACY_USERS_ENV_FILENAME, USERS_ENV_FILENAME),
    }


def _unknown_moves(commands: list[tuple[str, str, str, str]]) -> list[tuple[str, ...]]:
    return [command for command in commands if tuple(command[1:]) not in _known_moves()]


def test_every_env_file_command_names_files_the_code_uses() -> None:
    commands = _file_commands()
    assert {page for page, *_ in commands} == set(_ENV_FILE_PAGES)
    assert _unknown_moves(commands) == []


def test_the_copy_target_is_the_local_chain_member() -> None:
    assert profile_cmd._PROFILE_ENV_FILENAME == ENV_LOCAL_FILENAME
    assert ENV_LOCAL_FILENAME in ENV_CHAIN_FILENAMES
    assert profile_cmd._PROFILE_ENV_EXAMPLE_FILENAME not in ENV_CHAIN_FILENAMES


def test_the_rename_is_the_migration_up_performs(tmp_path: Path) -> None:
    (_page, _verb, source, target) = next(
        command for command in _file_commands() if command[1] == "mv"
    )
    assert RETIRED_NAME.search(source) and not RETIRED_NAME.search(target)
    (tmp_path / source).write_text("A=1\n", encoding="utf-8")
    assert migrate_users_env(tmp_path) == tmp_path / target
    assert (tmp_path / target).read_text(encoding="utf-8") == "A=1\n"
    assert not (tmp_path / source).exists()


def test_a_planted_env_file_name_is_caught() -> None:
    planted = [
        (page, verb, source, ".env.local" if verb == "cp" else ".env.prod")
        for page, verb, source, _target in _file_commands()
    ]
    assert len(_unknown_moves(planted)) == len(planted)


# -- how-to/llm-providers/configure-providers.rst ------------------------------

_GATEWAY_CONFIG = """\
api:
  providers:
    gw:
      base_url: https://gateway.example.org/v1
      api_protocol: anthropic
      default_model: gw-large
      models: [gw-small, gw-large]
claude_code:
  provider: gw
"""


def _policy(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "managed-settings.json"
    path.write_text(text, encoding="utf-8")
    return path


def _gateway_launch(tmp_path: Path) -> dict[str, str]:
    """The launch environment of the page's native gateway (``…/v1``)."""
    (tmp_path / "config.yml").write_text(_GATEWAY_CONFIG, encoding="utf-8")
    spec = load_provider_spec(tmp_path, include_telemetry=False)
    assert spec is not None
    env = {"GW_API_KEY": "sk-gw"}
    inject_provider_env(env, spec)
    return env


def _policy_block() -> str:
    return _block(PROVIDERS_PAGE, "json", "ANTHROPIC_BASE_URL")


def test_the_policy_is_read_in_the_settings_shape_claude_code_uses(tmp_path: Path) -> None:
    text = _policy_block()
    assert list(json.loads(text)) == ["env"]
    found = read_managed_policy_env([_policy(tmp_path, text)])
    assert {var: value for var, (value, _source) in found.items()} == json.loads(text)["env"]


def test_the_policy_agrees_with_a_v1_gateway_launch(tmp_path: Path) -> None:
    policy = _policy(tmp_path, _policy_block())
    assert detect_managed_policy_conflicts(_gateway_launch(tmp_path), [policy]) == []


def test_a_planted_policy_endpoint_is_caught(tmp_path: Path) -> None:
    text = _policy_block().replace("gateway.example.org", "gateway.example.org/v1")
    policy = _policy(tmp_path, text)
    assert detect_managed_policy_conflicts(_gateway_launch(tmp_path), [policy]) != []


# -- reference/configuration/profile.rst ---------------------------------------


def _refusal_block() -> str:
    return " ".join(_block(PROFILE_PAGE, "text", "rendered by the build").split())


def test_the_derived_key_refusal_is_the_message_the_validator_returns() -> None:
    key = _refusal_block().removeprefix("config: ").split(" ", 1)[0]
    assert derived_key_errors({key: "x"}) == [_refusal_block()]


def test_a_planted_derived_key_message_is_caught() -> None:
    planted = _refusal_block().replace("Remove it from", "Delete it from")
    key = planted.removeprefix("config: ").split(" ", 1)[0]
    assert derived_key_errors({key: "x"}) != [planted]


# -- how-to/agent-interfaces/cli-agent.rst -------------------------------------

_COMPANIONS = ("artifact", "ariel")


def _section_title() -> str:
    """The title ``_run_agent`` hands ``output.section`` for the started servers."""
    tree = ast.parse(inspect.getsource(chat_cmd._run_agent))
    (title,) = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "section"
        and isinstance(node.args[0], ast.Constant)
    ]
    return title


class _Launcher:
    """A launcher stub: started at the key's default port for the page's two, else not."""

    def __init__(self, key: str) -> None:
        self._launched = key in _COMPANIONS
        self._config_reader = lambda: ("127.0.0.1", framework_web_port_default(key))


def _printed_companions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> list[str]:
    """What ``osprey chat`` prints when the gallery and ARIEL start, token elided.

    Every registered companion has a launcher; all but the page's two report
    not launched, so the loop takes its real skip path and warns about none.
    """
    from osprey.infrastructure import server_launcher

    monkeypatch.setenv(WEB_PORT_ENV, "10100")
    monkeypatch.delenv(EXTERNAL_ORIGIN_ENV, raising=False)
    monkeypatch.setattr(server_launcher, "ensure_web_server", lambda key: None)
    monkeypatch.setattr(
        server_launcher, "_launchers", {key: _Launcher(key) for key in FRAMEWORK_WEB_SERVERS}
    )
    original = install_reporter(PhaseReporter(color=False))
    try:
        started = chat_cmd._launch_companion_servers(tmp_path)
        output.section(_section_title(), started)
    finally:
        install_reporter(original)
    captured = capsys.readouterr()
    assert "did not start" not in captured.out + captured.err
    printed = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", captured.out)
    return [re.sub(r"\?token=\S+", "?token=…", line.rstrip()) for line in printed.splitlines()]


def _companion_block() -> list[str]:
    return [
        line.rstrip() for line in _block(CLI_AGENT_PAGE, "text", "Companion servers").splitlines()
    ]


def test_the_companion_block_is_what_chat_prints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    printed = _printed_companions(monkeypatch, tmp_path, capsys)
    assert printed == _companion_block()
    labels = [line.split("http")[0].strip() for line in _companion_block()[1:]]
    assert labels == [FRAMEWORK_WEB_SERVERS[key].name for key in _COMPANIONS]


def test_a_planted_companion_row_is_caught(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    printed = _printed_companions(monkeypatch, tmp_path, capsys)
    planted = [line.replace("Artifact gallery", "Artifacts       ") for line in _companion_block()]
    assert printed != planted
    assert printed != [line.replace(":10300", ":8300") for line in _companion_block()]


# -- reference/configuration/config.rst ----------------------------------------


def _filled(template: str, values: dict[str, str]) -> str:
    """*template* with each ``<placeholder>`` replaced by its value."""
    return re.sub(r"<([^>]+)>", lambda match: values[match.group(1)], template)


def _image_names(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.delenv("OSPREY_IMAGE_REGISTRY", raising=False)
    monkeypatch.delenv("OSPREY_IMAGE_TAG", raising=False)
    config = {"project_name": "demo", "images": {"registry": "reg.example", "tag": "T1"}}
    return resolve_image_defaults(config)


def _misnamed(template: str, images: dict[str, str]) -> list[str]:
    """Images the build composes that *template*, filled in, does not spell."""
    return [
        image
        for key, image in images.items()
        if image
        != _filled(
            template,
            {
                "registry": "reg.example",
                "project": "demo",
                "service suffix": _OSPREY_IMAGE_SUFFIXES[key],
                "tag": "T1",
            },
        )
    ]


def test_the_image_name_pattern_is_the_order_the_build_composes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _block(CONFIG_PAGE, "literal", "<registry>").strip()
    assert _misnamed(template, _image_names(monkeypatch)) == []


def test_a_planted_image_name_order_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    template = "<registry>/<project>:<tag><service suffix>"
    assert _misnamed(template, _image_names(monkeypatch)) != []

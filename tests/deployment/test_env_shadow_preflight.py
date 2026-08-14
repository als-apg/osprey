"""``osprey up`` warns when a shell export shadows the pinned ``.env`` value.

``<repo>/.env`` is a deployment's one secret store, and every compose invocation
is pointed at it with ``--env-file``. But that is the *lower-precedence* source
for compose's document interpolation: a variable also exported in the calling
shell wins, silently. The stack then starts on the exported value while the
volumes were initialized from the store — a database handed a password its data
directory does not answer to, and no error anywhere.

Nothing else on the deploy path says this. The mint deliberately skips a
variable the process env already sets (so the store is never overwritten with
the divergent value, which is right), and the build-side divergence warning
compares a project against its *profile*, which a deployment repo does not have.

What these tests pin:

* the comparison is scoped to the variables the rendered compose files actually
  interpolate, read from the file that will be passed as ``--env-file``;
* every interpolation form counts (a default does not stop an export from
  winning), while ``$$`` — compose's escaped literal dollar — does not;
* the result is a WARNING the deploy carries on past, never a refusal;
* the warning names variables and **never** values, on either side.

No container runtime is touched: the unit tests call the preflight directly, and
the one end-to-end test drives ``osprey up`` with the runtime stubbed out.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.container_lifecycle import (
    _compose_interpolated_vars,
    _preflight_env_shadowing,
)

_PINNED = "p1nnedvaluefromtherepostore"
_EXPORTED = "d1vergentexportvaluenotfromthisrepo"


@pytest.fixture(autouse=True)
def _no_inherited_exports(monkeypatch):
    """The test's own environment decides; nothing may leak in from the runner."""
    for var in ("ARIEL_DB_PASSWORD", "ZO_ROOT_USER_PASSWORD", "EVENT_DISPATCHER_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def _repo(tmp_path: Path, env_text: str | None, *compose_texts: str) -> Path:
    """A repo root with a ``.env`` and compose files under ``build/services/``.

    The zone layout a real render uses, so the relative ``-f`` spellings the
    preflight is handed here are the ones ``up`` hands it.
    """
    root = tmp_path / "repo"
    services = root / "build" / "services"
    services.mkdir(parents=True)
    if env_text is not None:
        (root / ".env").write_text(env_text, encoding="utf-8")
    for index, text in enumerate(compose_texts):
        (services / f"docker-compose.{index}.yml").write_text(text, encoding="utf-8")
    return root


def _files(count: int = 1) -> list[str]:
    """The repo-relative compose paths matching what :func:`_repo` wrote."""
    return [f"build/services/docker-compose.{index}.yml" for index in range(count)]


# ---------------------------------------------------------------------------
# What compose reads from the environment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fragment",
    [
        "${ARIEL_DB_PASSWORD}",
        "$ARIEL_DB_PASSWORD",
        "${ARIEL_DB_PASSWORD:-ariel}",
        "${ARIEL_DB_PASSWORD-ariel}",
        "${ARIEL_DB_PASSWORD:?minted by deploy}",
        "${ARIEL_DB_PASSWORD?minted by deploy}",
        "${ARIEL_DB_PASSWORD:+set}",
        "${ARIEL_DB_PASSWORD+set}",
        'POSTGRES_PASSWORD: "${ARIEL_DB_PASSWORD:-ariel}"',
    ],
)
def test_every_interpolation_form_counts_as_a_reference(fragment):
    """A modifier changes what an UNSET name yields, not whether it is read.

    ``${VAR:-default}`` is the form the shipped postgres and openobserve
    templates use for exactly the passwords this preflight exists to protect. If
    a default excused a variable from the comparison, the two divergences most
    likely to cost an operator a data directory would be the two it never
    reported.
    """
    assert _compose_interpolated_vars(fragment) == {"ARIEL_DB_PASSWORD"}


def test_an_escaped_dollar_is_not_a_reference():
    """``$$`` is compose's literal dollar — the name behind it is never resolved.

    Warning about it would be a false positive, and a preflight that cries wolf
    on a literal is one an operator learns to scroll past.
    """
    assert _compose_interpolated_vars("command: echo $$ARIEL_DB_PASSWORD") == set()


def test_scanning_continues_inside_the_braces():
    """A default that itself interpolates contributes both names."""
    assert _compose_interpolated_vars("${OSPREY_WORKER_IMAGE:-${PROJECT_IMAGE}}") == {
        "OSPREY_WORKER_IMAGE",
        "PROJECT_IMAGE",
    }


def test_a_bare_dollar_is_not_a_reference():
    """``$`` before a non-name character names nothing and must not crash."""
    assert _compose_interpolated_vars("price: $9.99 and a trailing $") == set()


# ---------------------------------------------------------------------------
# The divergence itself
# ---------------------------------------------------------------------------


def test_a_divergent_export_is_reported_by_name(tmp_path, caplog):
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD:-ariel}\n",
    )

    with caplog.at_level(logging.WARNING):
        shadowed = _preflight_env_shadowing(
            _files(), repo, environ={"ARIEL_DB_PASSWORD": _EXPORTED}
        )

    assert shadowed == ["ARIEL_DB_PASSWORD"]
    assert "ARIEL_DB_PASSWORD" in caplog.text


def test_the_entry_time_dotenv_override_does_not_hide_the_divergence(tmp_path, caplog, monkeypatch):
    """The default comparison sees the shell as it was BEFORE the CLI's ``.env`` load.

    ``load_project_dotenv()`` (``override=True``, at CLI entry) replaces a
    divergent export with the pinned value inside the osprey process — so
    comparing against the live ``os.environ`` would find agreement and never
    warn, exactly when the operator most needs to hear otherwise. The recorded
    shell overrides are what keep the comparison honest.
    """
    import osprey.utils.config as config

    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD:-ariel}\n",
    )
    # The post-entry-load state: the process env holds the pinned value, and
    # the shell's own differing value survives only in the recorded overrides.
    monkeypatch.setenv("ARIEL_DB_PASSWORD", _PINNED)
    monkeypatch.setattr(config, "_dotenv_shell_overrides", {"ARIEL_DB_PASSWORD": _EXPORTED})

    with caplog.at_level(logging.WARNING):
        shadowed = _preflight_env_shadowing(_files(), repo)

    assert shadowed == ["ARIEL_DB_PASSWORD"]
    assert "ARIEL_DB_PASSWORD" in caplog.text


def test_neither_value_ever_reaches_the_warning(tmp_path, caplog):
    """The variable is the actionable fact; both values are secrets.

    A warning is the one place a secret would land in a terminal, a CI log and a
    pasted bug report at once, so neither the store's value nor the exported one
    may appear — the same rule ``compose_unsafe_vars`` and the build-side
    divergence warning already hold to.
    """
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )

    with caplog.at_level(logging.WARNING):
        _preflight_env_shadowing(_files(), repo, environ={"ARIEL_DB_PASSWORD": _EXPORTED})

    assert _PINNED not in caplog.text, "the preflight printed the value pinned in .env"
    assert _EXPORTED not in caplog.text, "the preflight printed the exported value"


def test_the_warning_says_which_source_wins(tmp_path, caplog):
    """An operator who cannot tell which value is running cannot act on this."""
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )

    with caplog.at_level(logging.WARNING):
        _preflight_env_shadowing(_files(), repo, environ={"ARIEL_DB_PASSWORD": _EXPORTED})

    text = caplog.text
    assert "EXPORTED" in text, "the warning does not say the shell's value is the one that runs"
    assert "--env-file" in text, "the warning does not name the source that loses"
    assert str(repo / ".env") in text, "the warning does not name the store it compared against"


def test_an_agreeing_export_is_not_a_divergence(tmp_path, caplog):
    """Exporting the same value changes nothing, so there is nothing to say."""
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )

    with caplog.at_level(logging.WARNING):
        assert (
            _preflight_env_shadowing(_files(), repo, environ={"ARIEL_DB_PASSWORD": _PINNED}) == []
        )

    assert caplog.text == ""


def test_a_pinned_key_no_compose_file_reads_is_not_reported(tmp_path, caplog):
    """It cannot change what starts, so it is not this preflight's business.

    ``.env`` also feeds the dispatch worker's whole-file ``env_file:`` mount and
    the agent's own provider auth. Reporting every key in it that happens to
    differ from the shell would bury the ones that actually alter the stack
    compose brings up.
    """
    repo = _repo(
        tmp_path,
        f"ANTHROPIC_API_KEY={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )

    with caplog.at_level(logging.WARNING):
        assert (
            _preflight_env_shadowing(_files(), repo, environ={"ANTHROPIC_API_KEY": _EXPORTED}) == []
        )

    assert caplog.text == ""


def test_an_export_the_store_does_not_pin_is_not_a_divergence(tmp_path, caplog):
    """Nothing is being overridden — the export is the only value there is."""
    repo = _repo(
        tmp_path,
        "ANTHROPIC_API_KEY=x\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )

    with caplog.at_level(logging.WARNING):
        assert (
            _preflight_env_shadowing(_files(), repo, environ={"ARIEL_DB_PASSWORD": _EXPORTED}) == []
        )

    assert caplog.text == ""


def test_an_export_over_a_deliberately_empty_pin_is_reported(tmp_path, caplog):
    """``TOKEN=`` in the store means "no token" — a deliberate fail-closed setting.

    The deploy layer treats an empty value as a setting rather than a blank (it
    declines to mint over one), so an export that re-arms the service behind the
    operator's back is exactly the divergence worth naming.
    """
    repo = _repo(
        tmp_path,
        "EVENT_DISPATCHER_TOKEN=\n",
        "environment:\n  EVENT_DISPATCHER_TOKEN: ${EVENT_DISPATCHER_TOKEN}\n",
    )

    with caplog.at_level(logging.WARNING):
        shadowed = _preflight_env_shadowing(
            _files(), repo, environ={"EVENT_DISPATCHER_TOKEN": _EXPORTED}
        )

    assert shadowed == ["EVENT_DISPATCHER_TOKEN"]


def test_every_diverging_variable_is_named(tmp_path, caplog):
    """Sorted, and complete: fixing one export and hitting the next is a trap."""
    repo = _repo(
        tmp_path,
        f"ZO_ROOT_USER_PASSWORD={_PINNED}\nARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD:-ariel}\n",
        "environment:\n  ZO_ROOT_USER_PASSWORD: ${ZO_ROOT_USER_PASSWORD:-Complexpass}\n",
    )

    with caplog.at_level(logging.WARNING):
        shadowed = _preflight_env_shadowing(
            _files(2),
            repo,
            environ={"ARIEL_DB_PASSWORD": _EXPORTED, "ZO_ROOT_USER_PASSWORD": _EXPORTED},
        )

    assert shadowed == ["ARIEL_DB_PASSWORD", "ZO_ROOT_USER_PASSWORD"]


def test_an_absolute_compose_path_is_read_as_given(tmp_path, caplog):
    """``compose_base_cmd`` passes absolutes through, so this must read them."""
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )
    absolute = str(repo / "build" / "services" / "docker-compose.0.yml")

    with caplog.at_level(logging.WARNING):
        shadowed = _preflight_env_shadowing(
            [absolute], repo, environ={"ARIEL_DB_PASSWORD": _EXPORTED}
        )

    assert shadowed == ["ARIEL_DB_PASSWORD"]


# ---------------------------------------------------------------------------
# It never becomes the reason a deploy fails
# ---------------------------------------------------------------------------


def test_no_env_file_is_silent(tmp_path, caplog):
    """With no store, compose is given no ``--env-file`` and nothing is shadowed."""
    repo = _repo(tmp_path, None, "environment:\n  X: ${ARIEL_DB_PASSWORD}\n")

    with caplog.at_level(logging.WARNING):
        assert (
            _preflight_env_shadowing(_files(), repo, environ={"ARIEL_DB_PASSWORD": _EXPORTED}) == []
        )

    assert caplog.text == ""


def test_a_missing_compose_file_is_skipped_not_raised(tmp_path, caplog):
    """A compose file that is not there is the start's error to raise, not this one's.

    An advisory that aborted the deploy would turn a warning into a refusal by
    accident, and it would do so *before* the start reached the error that
    actually explains the missing render.
    """
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )

    with caplog.at_level(logging.WARNING):
        shadowed = _preflight_env_shadowing(
            [*_files(), "build/services/gone.yml", "build/services"],
            repo,
            environ={"ARIEL_DB_PASSWORD": _EXPORTED},
        )

    assert shadowed == ["ARIEL_DB_PASSWORD"]


def test_the_default_environ_is_the_live_process_environment(tmp_path, monkeypatch, caplog):
    """``environ=None`` is the deploy path's own call — it must read os.environ."""
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )
    monkeypatch.setenv("ARIEL_DB_PASSWORD", _EXPORTED)

    with caplog.at_level(logging.WARNING):
        assert _preflight_env_shadowing(_files(), repo) == ["ARIEL_DB_PASSWORD"]


# ---------------------------------------------------------------------------
# It is on the deploy path
# ---------------------------------------------------------------------------


def test_the_start_sequence_runs_the_preflight_before_the_image_build(tmp_path, monkeypatch):
    """Wired into ``_start_stack``, ahead of the minutes-long image build.

    ``_start_stack`` is the one sequence both start verbs funnel through, so a
    check placed here covers the legacy re-render path and the deployment repo's
    as-built path without either knowing about it. Ordering matters as much as
    presence: an operator who has to wait out an image build to be told their
    export is stale has already lost the time the warning was meant to save.
    """
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )
    monkeypatch.setenv("ARIEL_DB_PASSWORD", _EXPORTED)

    order: list[str] = []
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_preflight_host_ports", lambda config, files: None)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "log_endpoint_summary", lambda config, files: None)
    monkeypatch.setattr(
        container_lifecycle,
        "_preflight_env_shadowing",
        lambda files, root, *a, **k: order.append("shadow") or [],
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_build_project_image",
        lambda config, dev, env, ctx=None: order.append("image"),
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(list(cmd), 0),
    )

    container_lifecycle._start_stack(
        {"project_name": "proj", "deployed_services": ["postgresql"]},
        _files(),
        repo,
        detached=True,
        env_path=repo / ".env",
    )

    assert order == ["shadow", "image"]


def test_a_divergent_export_warns_but_still_starts_the_stack(tmp_path, monkeypatch, caplog):
    """A warning, never a refusal.

    Exporting over the store is a legitimate gesture — a one-off run against
    another host's credentials, a rotation in progress. A deploy that refused it
    would break the escape hatch in order to protect people from using it.
    """
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )
    monkeypatch.setenv("ARIEL_DB_PASSWORD", _EXPORTED)

    ran: list[list[str]] = []
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_preflight_host_ports", lambda config, files: None)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "log_endpoint_summary", lambda config, files: None)
    monkeypatch.setattr(
        container_lifecycle, "_build_project_image", lambda config, dev, env, ctx=None: None
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        lambda cmd, **k: ran.append(list(cmd)) or subprocess.CompletedProcess(list(cmd), 0),
    )

    with caplog.at_level(logging.WARNING):
        container_lifecycle._start_stack(
            {"project_name": "proj", "deployed_services": ["postgresql"]},
            _files(),
            repo,
            detached=True,
            env_path=repo / ".env",
        )

    assert any("up" in cmd for cmd in ran), "the deploy did not reach `compose up`"
    assert "ARIEL_DB_PASSWORD" in caplog.text
    assert _PINNED not in caplog.text and _EXPORTED not in caplog.text

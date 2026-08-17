"""``osprey up``'s env-chain preflights: shadowing, layering, and drift.

The chain under ``<repo>`` — ``.env.shared`` then ``.env``, later winning — is a
deployment's one secret store, and every compose invocation is pointed at it.
But the env file is not the last word on compose's document interpolation, and
*which* word it is inverts between the supported providers: Docker Compose lets
a shell export win, while podman-compose reads its env file first and drops the
export. Either way the shell and the chain can disagree silently, and the stack
starts on a value its volumes were never initialized with — a database handed a
password its data directory does not answer to, and no error anywhere.

Nothing else on the deploy path says this. The mint deliberately skips a
variable the process env already sets (so the store is never overwritten with
the divergent value, which is right), and the build-side divergence warning
compares a project against its *profile*, which a deployment repo does not have.

What these tests pin:

* the comparison is scoped to the variables the rendered compose files actually
  interpolate, read from the *merged chain* (``.env.shared`` then ``.env``,
  later winning) that compose will be pointed at;
* every interpolation form counts (a default does not stop an export from
  winning), while ``$$`` — compose's escaped literal dollar — does not;
* the result is a WARNING the deploy carries on past, never a refusal;
* the warning names variables and **never** values, on either side;
* which side of the divergence actually runs is stated per provider, because
  the two supported providers invert it.

The chain also brings two failure modes of its own, reported at deliberately
different volumes:

* what the local file overrides in the shared defaults is the chain working as
  designed — ONE info line, names only;
* a **stale pin** — the local file still holding the value the shared defaults
  carried before they changed — is the one case that is nobody's decision, and
  is the only thing here that warns;
* a chain whose membership no longer matches what the project was rendered
  against is a REFUSAL, naming ``osprey build``, because a file added or
  removed after the build contributes nothing to what the stack reads.

No container runtime is touched: the unit tests call the preflights directly,
and the end-to-end tests drive ``osprey up`` with the runtime stubbed out.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import (
    ENV_CHAIN_MARKER_FILENAME,
    ENV_CHAIN_MARKER_KEY,
)
from osprey.deployment.container_lifecycle import (
    ENV_CHAIN_STATE_RELPATH,
    _compose_interpolated_vars,
    _preflight_env_chain_drift,
    _preflight_env_shadowing,
    _report_chain_overrides,
)
from osprey.deployment.runtime_helper import ComposeProvider

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


def _shared(root: Path, text: str) -> Path:
    """Add a ``.env.shared`` to a repo built by :func:`_repo`."""
    path = root / ".env.shared"
    path.write_text(text, encoding="utf-8")
    return path


def _files(count: int = 1) -> list[str]:
    """The repo-relative compose paths matching what :func:`_repo` wrote."""
    return [f"build/services/docker-compose.{index}.yml" for index in range(count)]


def _worker_compose(*env_files: str) -> str:
    """A rendered worker service whose ``env_file:`` names the given chain files.

    The shape the dispatch-worker template renders: the membership decided at
    build time, spelled relative to the pinned compose project directory (the
    deployment repo root).
    """
    entries = "".join(f"      - {name}\n" for name in env_files)
    block = f"    env_file:\n{entries}" if entries else ""
    return f"services:\n  dispatch-worker-1:\n    image: proj:local\n{block}"


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


def test_an_export_over_an_empty_pin_is_reported(tmp_path, caplog):
    """``TOKEN=`` in the store and a different value exported over it diverge.

    What this preflight reports is the divergence itself, whatever either side
    holds — it reads the files as they are and does not model the mint (which
    treats a blank service token as a blank and fills it in). An export that
    silently decides what a service authenticates with is worth naming either
    way.
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


# ---------------------------------------------------------------------------
# The store being compared against is the whole chain, not just `.env`
# ---------------------------------------------------------------------------


def test_a_shared_only_key_is_part_of_the_store(tmp_path, caplog):
    """``.env.shared`` is a chain member, so a key only it sets is still pinned.

    Comparing against ``.env`` alone would go quiet for exactly the keys a
    multi-host deployment keeps in the shared file — the ones every host is
    meant to agree on, and therefore the ones an export diverging from them is
    most likely to be a mistake.
    """
    repo = _repo(tmp_path, None, "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n")
    _shared(repo, f"ARIEL_DB_PASSWORD={_PINNED}\n")

    with caplog.at_level(logging.WARNING):
        shadowed = _preflight_env_shadowing(
            _files(), repo, environ={"ARIEL_DB_PASSWORD": _EXPORTED}
        )

    assert shadowed == ["ARIEL_DB_PASSWORD"]


def test_the_comparison_is_against_the_merged_value_not_the_shared_one(tmp_path, caplog):
    """A local override wins the chain, so the export is compared against IT.

    An export that happens to match the shared default is still a divergence
    once this host has overridden it — the value that would run and the value
    exported are different, which is the whole question.
    """
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )
    _shared(repo, f"ARIEL_DB_PASSWORD={_EXPORTED}\n")

    with caplog.at_level(logging.WARNING):
        shadowed = _preflight_env_shadowing(
            _files(), repo, environ={"ARIEL_DB_PASSWORD": _EXPORTED}
        )

    assert shadowed == ["ARIEL_DB_PASSWORD"]


def test_an_export_agreeing_with_the_merged_value_is_silent(tmp_path, caplog):
    """Matching the value that would run is agreement, whichever file supplied it."""
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )
    _shared(repo, f"ARIEL_DB_PASSWORD={_EXPORTED}\n")

    with caplog.at_level(logging.WARNING):
        assert (
            _preflight_env_shadowing(_files(), repo, environ={"ARIEL_DB_PASSWORD": _PINNED}) == []
        )


def test_the_warning_names_every_chain_file_it_compared_against(tmp_path, caplog):
    """An operator who cannot tell which files were read cannot go fix one."""
    repo = _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )
    _shared(repo, "OTHER=x\n")

    with caplog.at_level(logging.WARNING):
        _preflight_env_shadowing(_files(), repo, environ={"ARIEL_DB_PASSWORD": _EXPORTED})

    assert str(repo / ".env.shared") in caplog.text
    assert str(repo / ".env") in caplog.text


# ---------------------------------------------------------------------------
# Which side wins is provider-scoped, and the two providers invert it
# ---------------------------------------------------------------------------


def _diverging_repo(tmp_path: Path) -> Path:
    return _repo(
        tmp_path,
        f"ARIEL_DB_PASSWORD={_PINNED}\n",
        "environment:\n  POSTGRES_PASSWORD: ${ARIEL_DB_PASSWORD}\n",
    )


def test_docker_is_told_the_exported_value_is_the_one_that_runs(tmp_path, caplog):
    """Docker Compose's env file is its LOWEST-precedence interpolation source."""
    repo = _diverging_repo(tmp_path)

    with caplog.at_level(logging.WARNING):
        _preflight_env_shadowing(
            _files(),
            repo,
            environ={"ARIEL_DB_PASSWORD": _EXPORTED},
            provider=ComposeProvider.DOCKER_V2,
        )

    assert "EXPORTED" in caplog.text
    assert "ENV-FILE" not in caplog.text, "docker was told the env file wins"


def test_podman_compose_is_told_its_export_reaches_nothing(tmp_path, caplog):
    """podman-compose reads its env file FIRST — the inverse of docker.

    Telling a podman-compose operator to unset the export "so the pinned value
    runs" would be advice for a precedence their provider does not have: the
    pinned value is already what runs, and the export they set is the thing
    doing nothing. Same divergence, opposite consequence, so opposite wording.
    """
    repo = _diverging_repo(tmp_path)

    with caplog.at_level(logging.WARNING):
        _preflight_env_shadowing(
            _files(),
            repo,
            environ={"ARIEL_DB_PASSWORD": _EXPORTED},
            provider=ComposeProvider.PODMAN_COMPOSE,
        )

    assert "ENV-FILE" in caplog.text
    assert "EXPORTED" not in caplog.text, "podman-compose was told the export wins"


def test_an_unprobed_provider_names_both_precedences(tmp_path, caplog):
    """Not knowing is reported as not knowing, never as a guess.

    Picking one provider's wording when the probe has not run would be right
    half the time and confidently backwards the other half — and the operator
    has no way to tell which they got.
    """
    repo = _diverging_repo(tmp_path)

    with caplog.at_level(logging.WARNING):
        _preflight_env_shadowing(_files(), repo, environ={"ARIEL_DB_PASSWORD": _EXPORTED})

    assert "EXPORTED" in caplog.text and "podman-compose" in caplog.text


@pytest.mark.parametrize(
    "provider", [ComposeProvider.DOCKER_V2, ComposeProvider.PODMAN_COMPOSE, None]
)
def test_no_provider_wording_ever_prints_a_value(tmp_path, caplog, provider):
    """The names-only rule holds on every branch, not just the one first written."""
    repo = _diverging_repo(tmp_path)

    with caplog.at_level(logging.WARNING):
        _preflight_env_shadowing(
            _files(), repo, environ={"ARIEL_DB_PASSWORD": _EXPORTED}, provider=provider
        )

    assert _PINNED not in caplog.text and _EXPORTED not in caplog.text


# ---------------------------------------------------------------------------
# The chain's own layering: one info line for the deliberate, a warning for stale
# ---------------------------------------------------------------------------


def test_a_deliberate_override_is_one_info_line_of_names(tmp_path, caplog):
    """Overriding shared defaults is what the local file is FOR.

    A deployment repo that overrides nothing would not need two files. Warning
    per override would make the loudest signal on the deploy path the one thing
    guaranteed to be intentional, and teach operators to scroll past the line
    that is not.
    """
    repo = _repo(tmp_path, f"A={_PINNED}\nB={_PINNED}\n")
    _shared(repo, f"A={_EXPORTED}\nB={_EXPORTED}\n")

    with caplog.at_level(logging.INFO):
        deliberate, stale = _report_chain_overrides(repo)

    assert (deliberate, stale) == (["A", "B"], [])
    info_lines = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_lines) == 1, "the deliberate overrides were not collapsed into one line"
    assert "A, B" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_the_info_line_carries_names_and_no_values(tmp_path, caplog):
    repo = _repo(tmp_path, f"A={_PINNED}\n")
    _shared(repo, f"A={_EXPORTED}\n")

    with caplog.at_level(logging.INFO):
        _report_chain_overrides(repo)

    assert "A" in caplog.text
    assert _PINNED not in caplog.text and _EXPORTED not in caplog.text


def test_a_key_the_two_files_agree_on_is_not_an_override(tmp_path, caplog):
    """Restating a shared value locally changes nothing, so there is nothing to say."""
    repo = _repo(tmp_path, f"A={_PINNED}\n")
    _shared(repo, f"A={_PINNED}\n")

    with caplog.at_level(logging.INFO):
        assert _report_chain_overrides(repo) == ([], [])

    assert caplog.text == ""


def test_a_local_only_key_is_not_an_override(tmp_path, caplog):
    """There is no shared value to override — the local file is the only source."""
    repo = _repo(tmp_path, f"A={_PINNED}\n")
    _shared(repo, "B=x\n")

    with caplog.at_level(logging.INFO):
        assert _report_chain_overrides(repo) == ([], [])


def test_a_repo_with_no_shared_file_reports_nothing(tmp_path, caplog):
    """The single-file shape is the common one and must stay silent."""
    repo = _repo(tmp_path, f"A={_PINNED}\n")

    with caplog.at_level(logging.INFO):
        assert _report_chain_overrides(repo) == ([], [])

    assert caplog.text == ""


def test_the_first_deploy_has_no_history_so_nothing_is_stale(tmp_path, caplog):
    """A stale pin is a claim about a CHANGE, which needs a previous state.

    With no record, an override is reported as what it is — an override — and
    calling it stale would be an accusation the evidence does not support.
    """
    repo = _repo(tmp_path, f"A={_PINNED}\n")
    _shared(repo, f"A={_EXPORTED}\n")

    with caplog.at_level(logging.INFO):
        deliberate, stale = _report_chain_overrides(repo)

    assert (deliberate, stale) == (["A"], [])


def test_a_shared_value_moving_past_an_unchanged_local_pin_warns(tmp_path, caplog):
    """THE stale pin: the local file holds the shared default of the day before.

    Nobody chose this value for this host; it was copied to match, and then the
    shared file moved on. The stack starts on the superseded value and comes up
    looking perfectly healthy, which is why this is the one layering case that
    warrants a warning.
    """
    repo = _repo(tmp_path, "A=old-shared-value\n")
    _shared(repo, "A=old-shared-value\nB=x\n")
    _report_chain_overrides(repo)  # the deploy that recorded the shared value

    _shared(repo, "A=new-shared-value\nB=x\n")

    with caplog.at_level(logging.INFO):
        deliberate, stale = _report_chain_overrides(repo)

    assert (deliberate, stale) == ([], ["A"])
    assert [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "A" in caplog.text


def test_a_stale_pin_is_kept_out_of_the_deliberate_line(tmp_path, caplog):
    """The split has to be a split: a stale pin reported at both volumes is noise."""
    repo = _repo(tmp_path, "A=old-shared-value\nB=chosen-for-this-host\n")
    _shared(repo, "A=old-shared-value\nB=shared-b\n")
    _report_chain_overrides(repo)

    _shared(repo, "A=new-shared-value\nB=shared-b\n")

    deliberate, stale = _report_chain_overrides(repo)

    assert deliberate == ["B"]
    assert stale == ["A"]


def test_a_local_value_that_matches_neither_is_deliberate(tmp_path):
    """A value that was never the shared default is a value somebody chose."""
    repo = _repo(tmp_path, "A=chosen-for-this-host\n")
    _shared(repo, "A=old-shared-value\n")
    _report_chain_overrides(repo)

    _shared(repo, "A=new-shared-value\n")

    assert _report_chain_overrides(repo) == (["A"], [])


def test_a_stale_pin_keeps_being_reported_until_it_is_resolved(tmp_path):
    """The record must not "catch up" and go quiet with the pin still stale.

    Advancing the recorded shared value unconditionally would make this a
    one-shot warning: the next deploy would find the shared file unchanged
    since the record, report nothing, and leave the stack running on the
    superseded value for as long as anybody cares to redeploy it.
    """
    repo = _repo(tmp_path, "A=old-shared-value\n")
    _shared(repo, "A=old-shared-value\n")
    _report_chain_overrides(repo)
    _shared(repo, "A=new-shared-value\n")

    assert _report_chain_overrides(repo) == ([], ["A"])
    assert _report_chain_overrides(repo) == ([], ["A"])
    assert _report_chain_overrides(repo) == ([], ["A"])


def test_adopting_the_shared_value_ends_the_report(tmp_path):
    """Deleting the pin is the remedy the warning names, and it must work."""
    repo = _repo(tmp_path, "A=old-shared-value\n")
    _shared(repo, "A=old-shared-value\n")
    _report_chain_overrides(repo)
    _shared(repo, "A=new-shared-value\n")
    _report_chain_overrides(repo)

    (repo / ".env").write_text("", encoding="utf-8")

    assert _report_chain_overrides(repo) == ([], [])


def test_choosing_a_local_value_downgrades_the_report_to_info(tmp_path):
    """The other remedy: keep a local value, but one this host actually wants."""
    repo = _repo(tmp_path, "A=old-shared-value\n")
    _shared(repo, "A=old-shared-value\n")
    _report_chain_overrides(repo)
    _shared(repo, "A=new-shared-value\n")
    _report_chain_overrides(repo)

    (repo / ".env").write_text("A=chosen-for-this-host\n", encoding="utf-8")

    assert _report_chain_overrides(repo) == (["A"], [])


# ---------------------------------------------------------------------------
# The record the stale-pin check remembers with
# ---------------------------------------------------------------------------


def test_the_record_holds_digests_and_never_a_value(tmp_path):
    """Every stale-pin question is an equality test, which a digest answers.

    Keeping a second plaintext copy of the shared secrets would buy nothing and
    add one more file that must never be pasted into a bug report.
    """
    repo = _repo(tmp_path, f"A={_PINNED}\n")
    _shared(repo, f"A={_EXPORTED}\n")

    _report_chain_overrides(repo)

    raw = (repo / ENV_CHAIN_STATE_RELPATH).read_text(encoding="utf-8")
    assert _PINNED not in raw and _EXPORTED not in raw
    assert "A" in json.loads(raw)["shared_value_digests"]


def test_the_record_lives_in_the_machine_zone_under_a_dot_env_name(tmp_path):
    """It is derived, never committed — and every ``.env*`` sweep must catch it."""
    assert ENV_CHAIN_STATE_RELPATH.parts[0] == "build"
    assert ENV_CHAIN_STATE_RELPATH.name.startswith(".env")


def test_an_unreadable_record_is_treated_as_no_history(tmp_path, caplog):
    """Garbage in the cache means "no evidence", never a crashed deploy."""
    repo = _repo(tmp_path, "A=old-shared-value\n")
    _shared(repo, "A=new-shared-value\n")
    state = repo / ENV_CHAIN_STATE_RELPATH
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("{not json at all", encoding="utf-8")

    assert _report_chain_overrides(repo) == (["A"], [])


def test_a_record_from_another_schema_is_discarded_not_migrated(tmp_path):
    repo = _repo(tmp_path, "A=old-shared-value\n")
    _shared(repo, "A=new-shared-value\n")
    state = repo / ENV_CHAIN_STATE_RELPATH
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"version": 99, "shared_value_digests": {"A": "x"}}), "utf-8")

    assert _report_chain_overrides(repo) == (["A"], [])


def test_an_unwritable_record_never_fails_the_deploy(tmp_path, monkeypatch, caplog):
    """This is an advisory's memory. It must not become the reason nothing starts."""
    repo = _repo(tmp_path, f"A={_PINNED}\n")
    _shared(repo, f"A={_EXPORTED}\n")

    def _refuse(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(container_lifecycle, "atomic_write", _refuse)

    with caplog.at_level(logging.INFO):
        assert _report_chain_overrides(repo) == (["A"], [])


# ---------------------------------------------------------------------------
# Chain-set drift: a refusal, because membership is fixed at build time
# ---------------------------------------------------------------------------


def test_a_chain_file_added_since_the_build_refuses(tmp_path, caplog):
    """The render names the files that existed then — a new one contributes nothing.

    Dropping a ``.env.shared`` into a repo built without one is the most natural
    thing an operator can do, and the most misleading: the deploy comes up
    healthy on values the new file never delivered, and nothing anywhere says
    the file was ignored.
    """
    repo = _repo(tmp_path, "A=x\n", _worker_compose("./.env"))
    _shared(repo, "B=y\n")

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        _preflight_env_chain_drift(_files(), repo)

    assert ".env.shared" in caplog.text
    assert "osprey build" in caplog.text


def test_a_chain_file_removed_since_the_build_refuses(tmp_path, caplog):
    """The render points at a path that is gone — compose would fail obscurely."""
    repo = _repo(tmp_path, "A=x\n", _worker_compose("./.env.shared", "./.env"))

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as excinfo:
        _preflight_env_chain_drift(_files(), repo)

    assert ".env.shared" in caplog.text
    assert "osprey build" in str(excinfo.value)


def test_a_chain_matching_its_render_is_silent(tmp_path, caplog):
    repo = _repo(tmp_path, "A=x\n", _worker_compose("./.env.shared", "./.env"))
    _shared(repo, "B=y\n")

    with caplog.at_level(logging.INFO):
        _preflight_env_chain_drift(_files(), repo)

    assert caplog.text == ""


def test_the_deploys_own_minted_env_is_not_drift(tmp_path, caplog):
    """`.env` appearing after the build is this command's own doing.

    `osprey init` writes only the shared half, so a project's first `osprey
    build` records a chain of `.env.shared` alone — and the `osprey up` that
    follows mints the service tokens into a brand-new `.env` a few steps before
    this check runs. Reading that as operator drift refuses the FIRST START of
    every new deployment, which is the ordinary `init` -> `build` -> `up` order.
    """
    repo = _repo(tmp_path, "A=x\n", _worker_compose("./.env.shared"))
    _shared(repo, "B=y\n")

    with caplog.at_level(logging.INFO):
        _preflight_env_chain_drift(_files(), repo)

    assert caplog.text == ""


def test_a_missing_env_is_not_this_checks_refusal_to_make(tmp_path, caplog):
    """A render naming `.env` with no `.env` on disk stays silent HERE.

    Not because it is fine, but because `osprey up` already refuses a missing
    `.env` on its own, with a message that tells the operator to create one.
    Refusing here too would answer that situation with `osprey build`, which
    re-renders around the missing file instead of fixing it.
    """
    repo = _repo(tmp_path, None, _worker_compose("./.env.shared", "./.env"))
    _shared(repo, "B=y\n")

    with caplog.at_level(logging.INFO):
        _preflight_env_chain_drift(_files(), repo)

    assert caplog.text == ""


def test_a_render_that_names_no_chain_file_is_not_evidence(tmp_path, caplog):
    """No record is not the same as a record of an empty chain.

    A project deploying no service with an ``env_file:`` block leaves exactly
    the same evidence as one rendered against an empty chain. Refusing on that
    ambiguity would block deploys this check has no finding against.
    """
    repo = _repo(tmp_path, "A=x\n", _worker_compose())
    _shared(repo, "B=y\n")

    with caplog.at_level(logging.INFO):
        _preflight_env_chain_drift(_files(), repo)

    assert caplog.text == ""


def test_a_non_chain_env_file_is_not_read_as_membership(tmp_path, caplog):
    """``.env.auth`` and friends are other subsystems' files, not chain members."""
    repo = _repo(tmp_path, "A=x\n", _worker_compose("./.env.auth"))
    _shared(repo, "B=y\n")

    with caplog.at_level(logging.INFO):
        _preflight_env_chain_drift(_files(), repo)

    assert caplog.text == ""


def test_the_mapping_form_of_an_env_file_entry_counts(tmp_path):
    """Compose accepts ``- path: ./.env`` as well as a bare string."""
    repo = _repo(
        tmp_path,
        "A=x\n",
        "services:\n  w:\n    env_file:\n      - path: ./.env\n        required: true\n",
    )
    _shared(repo, "B=y\n")

    with pytest.raises(RuntimeError):
        _preflight_env_chain_drift(_files(), repo)


def test_an_unparseable_compose_file_is_skipped_not_raised(tmp_path):
    """A broken render is the start's own error to raise, not this preflight's."""
    repo = _repo(tmp_path, "A=x\n", "{{{ not yaml", _worker_compose("./.env.shared", "./.env"))
    _shared(repo, "B=y\n")

    _preflight_env_chain_drift(_files(2), repo)


def test_membership_is_collected_across_every_compose_file(tmp_path):
    """Two services can each name one half of the chain; together they are the set."""
    repo = _repo(tmp_path, "A=x\n", _worker_compose("./.env.shared"), _worker_compose("./.env"))
    _shared(repo, "B=y\n")

    _preflight_env_chain_drift(_files(2), repo)


# ---------------------------------------------------------------------------
# Both new checks are on the deploy path
# ---------------------------------------------------------------------------


def _stub_runtime(monkeypatch, order: list[str]) -> list[list[str]]:
    """Stand in for everything ``_start_stack`` would touch on a real host."""
    ran: list[list[str]] = []
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_preflight_host_ports", lambda config, files: None)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "log_endpoint_summary", lambda config, files: None)
    monkeypatch.setattr(
        container_lifecycle,
        "_build_project_image",
        lambda config, dev, env, ctx=None: order.append("image"),
    )

    # `container_lifecycle.subprocess` IS the stdlib module, so this patch is
    # process-global: every subprocess.run in the deploy path lands here,
    # including the read-only volume probe `_start_stack` makes through
    # `osprey.deployment.reset.RuntimeProbe`. The double therefore has to accept
    # whatever any of them passes and hand back a real CompletedProcess — a
    # narrower signature fails the caller furthest from this test.
    def _fake_run(cmd, *args, **kwargs):
        ran.append(list(cmd))
        return subprocess.CompletedProcess(list(cmd), 0, stdout="", stderr="")

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return ran


def test_drift_refuses_the_deploy_before_the_image_build(tmp_path, monkeypatch):
    """Refusing after a minutes-long build would waste the time it exists to save.

    And it must refuse rather than warn: unlike an export over the store, a
    chain that does not match its render produces an outcome nobody asked for.
    """
    repo = _repo(tmp_path, "A=x\n", _worker_compose("./.env"))
    _shared(repo, "B=y\n")
    order: list[str] = []
    ran = _stub_runtime(monkeypatch, order)

    with pytest.raises(RuntimeError, match="osprey build"):
        container_lifecycle._start_stack(
            {"project_name": "proj", "deployed_services": ["postgresql"]},
            _files(),
            repo,
            detached=True,
            env_path=repo / ".env",
        )

    assert order == [], "the image build ran despite the refusal"
    # Compose invocations specifically, not "no subprocess at all": the refusal
    # path legitimately makes read-only engine probes (a labelled `volume ls`)
    # before it decides. What must not happen is the deploy proceeding.
    assert [cmd for cmd in ran if "compose" in cmd] == [], (
        "a compose command ran despite the refusal"
    )


def test_a_matching_chain_lets_the_deploy_through(tmp_path, monkeypatch, caplog):
    """The refusal must be about drift, not about having a chain at all."""
    repo = _repo(tmp_path, "A=x\n", _worker_compose("./.env.shared", "./.env"))
    _shared(repo, "B=y\n")
    ran = _stub_runtime(monkeypatch, [])

    container_lifecycle._start_stack(
        {"project_name": "proj", "deployed_services": ["postgresql"]},
        _files(),
        repo,
        detached=True,
        env_path=repo / ".env",
    )

    assert any("up" in cmd for cmd in ran), "the deploy did not reach `compose up`"


def test_the_deploy_path_reports_a_stale_pin(tmp_path, monkeypatch, caplog):
    """End to end: the layering warning reaches an operator running ``osprey up``."""
    repo = _repo(tmp_path, "A=old-shared-value\n", _worker_compose("./.env.shared", "./.env"))
    _shared(repo, "A=old-shared-value\n")
    _report_chain_overrides(repo)
    _shared(repo, "A=new-shared-value\n")
    _stub_runtime(monkeypatch, [])

    with caplog.at_level(logging.WARNING):
        container_lifecycle._start_stack(
            {"project_name": "proj", "deployed_services": ["postgresql"]},
            _files(),
            repo,
            detached=True,
            env_path=repo / ".env",
        )

    assert "A" in caplog.text
    assert "old-shared-value" not in caplog.text and "new-shared-value" not in caplog.text


# ---------------------------------------------------------------------------
# Membership comes from the render's own marker when it wrote one
# ---------------------------------------------------------------------------


def _marker(root: Path, *names: str) -> Path:
    """Write the membership marker a render leaves behind."""
    path = root / "build" / ENV_CHAIN_MARKER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({ENV_CHAIN_MARKER_KEY: list(names)}), encoding="utf-8")
    return path


def test_the_marker_is_authoritative_over_the_env_file_reading(tmp_path):
    """The render states its membership; the compose files only imply it.

    A service can be rendered with no ``env_file:`` and still have been built
    against a chain — interpolation delivery is the other half of membership —
    so where the two disagree, what the build recorded wins.
    """
    repo = _repo(tmp_path, "A=x\n", _worker_compose("./.env"))
    _shared(repo, "B=y\n")
    _marker(repo, ".env.shared", ".env")

    _preflight_env_chain_drift(_files(), repo)


def test_a_marker_recording_the_empty_chain_still_catches_a_new_file(tmp_path, caplog):
    """The gap the marker closes: an empty chain is a RECORD, not an absence.

    Inferring membership from ``env_file:`` cannot tell "rendered against no
    chain" from "no service reads one", so it has to stay silent. The marker
    says which, so the first ``.env.shared`` dropped into a project built
    without one is caught instead of quietly contributing nothing.

    The shared half, not ``.env``: the deploy mints its own ``.env`` before this
    check runs, so that file is compared away on both sides.
    """
    repo = _repo(tmp_path, "A=x\n", _worker_compose())
    _shared(repo, "B=y\n")
    _marker(repo)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="osprey build"):
        _preflight_env_chain_drift(_files(), repo)

    assert ".env.shared" in caplog.text


def test_a_marker_matching_the_disk_is_silent(tmp_path, caplog):
    repo = _repo(tmp_path, "A=x\n", _worker_compose())
    _shared(repo, "B=y\n")
    _marker(repo, ".env.shared", ".env")

    with caplog.at_level(logging.INFO):
        _preflight_env_chain_drift(_files(), repo)

    assert caplog.text == ""


def test_a_marker_naming_a_file_that_is_gone_refuses(tmp_path, caplog):
    repo = _repo(tmp_path, "A=x\n", _worker_compose())
    _marker(repo, ".env.shared", ".env")

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        _preflight_env_chain_drift(_files(), repo)

    assert ".env.shared" in caplog.text


def test_an_unreadable_marker_falls_back_to_the_env_file_reading(tmp_path):
    """A corrupt marker is "no record", so the weaker reading still applies."""
    repo = _repo(tmp_path, "A=x\n", _worker_compose("./.env"))
    _shared(repo, "B=y\n")
    marker = repo / "build" / ENV_CHAIN_MARKER_FILENAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{not json", encoding="utf-8")

    with pytest.raises(RuntimeError):
        _preflight_env_chain_drift(_files(), repo)

"""Unit tests for the Neo4j graph-store password in the token-mint map.

Mirrors ``test_postgres_password_mint.py``/``test_mongo_password_mint.py`` for
the ``graphdb`` deployed-service entry in ``_SERVICE_TOKEN_VARS``
(container_lifecycle.py): ``osprey up`` mints a strong ``GRAPHDB_PASSWORD``
into the project ``.env``, which the graphdb compose template will read as the
password half of the composite ``NEO4J_AUTH: "neo4j/${GRAPHDB_PASSWORD:-…}"``
and which the connection resolver hands the driver separately
(``driver(uri, auth=(user, password))``).

Two things make this credential different from the other store passwords, and
both are what these tests pin:

* the container variable is a **composite** ``user/password``, so a ``/`` in
  the value moves the split and the container comes up as a different user (or
  refuses to start) — the shared URI-safe rule already rejects ``/``, but this
  var states it on its own ground so a later loosening of that shared helper
  cannot silently take the composite with it;
* Neo4j enforces an **8-character minimum** on the initial password and
  crash-loops below it, so a short operator-supplied value has to be refused at
  the deploy boundary rather than at container start.

The external-store path is the third case: with an explicit
``services.graphdb.uri`` and ``graphdb`` absent from ``deployed_services``,
nothing mints this var — the operator sets it themselves and it is validated
when present, never fabricated when absent (the ``ARIEL_DSN`` validate-only
pattern).
"""

from __future__ import annotations

import subprocess

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.service_tokens import (
    _VAR_FORBIDDEN_VALUES,
    _generate_token,
    _validate_var,
)

#: The bare password half of the compose template's published
#: ``NEO4J_AUTH: "neo4j/${GRAPHDB_PASSWORD:-ospreygraph}"`` fallback. Spelled
#: here as a literal rather than read out of the registry so a drifting
#: registry entry fails a test instead of quietly re-pinning the test to
#: whatever the code now says.
PUBLISHED_DEFAULT = "ospreygraph"


def _deploy_fixture(monkeypatch, tmp_path, services):
    """Patch deploy_up's collaborators for a project deploying ``services``.

    The stubbed config carries a ``services.graphdb`` block whenever graphdb is
    deployed, because that block is *required* rather than defaulted: the deploy
    refuses a block-less ``graphdb`` in ``deployed_services`` before it reaches
    any mint (``preflight_graphdb_config``). A fixture without it would describe
    a deployment osprey does not accept, and would test the refusal rather than
    the mint.

    The graph-store staging step is stubbed out for the same reason the compose
    invocation is: it dials the store it just started, and a mint test has no
    store to answer — left in, every test here would sit out the staging step's
    full connection budget before failing on something unrelated to minting.
    """
    captured: dict = {}

    config: dict = {"deployed_services": list(services)}
    if "graphdb" in services:
        config["services"] = {"graphdb": {"port_host": 7687, "http_port_host": 7474}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (config, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_stage_graphdb_store",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_run(cmd, env=None, check=False, **kwargs):
        captured["cmd"] = cmd
        # run_captured re-wraps this result to carry its spool path, so the
        # stand-in has to be a real completed process, and it passes redirection
        # kwargs this ignores.
        return subprocess.CompletedProcess(list(cmd), 0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return captured


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """A project with only 'graphdb' deployed — the minting path."""
    return _deploy_fixture(monkeypatch, tmp_path, ["graphdb"])


@pytest.fixture
def external_store_argv(monkeypatch, tmp_path):
    """A project with NO graphdb deployed — the external-store path.

    ``postgresql`` stands in for "some other service is deployed" so the deploy
    still exercises a normal mint loop; what matters is only that ``graphdb``
    is absent from ``deployed_services``, which is what makes GRAPHDB_PASSWORD
    validate-only rather than minted.
    """
    return _deploy_fixture(monkeypatch, tmp_path, ["postgresql"])


@pytest.fixture(autouse=True)
def _clean_token_env(monkeypatch):
    """Keep every test here off whatever the host exports for these vars.

    Autouse rather than threaded through each signature, because the mint path
    is not the only one that reads them: a shell export outranks ``.env`` in the
    deploy's effective environment, so an ambient ``GRAPHDB_PASSWORD`` also
    decides what the staleness comparison in :class:`TestStalenessAndAdoption`
    puts beside the container's credential. Naming the fixture per test left
    exactly the classes that do not mint it exposed to the host. The tests that
    care about an ambient value set it themselves, which still wins — this only
    clears what the host brought.
    """
    monkeypatch.delenv("GRAPHDB_PASSWORD", raising=False)
    monkeypatch.delenv("ARIEL_DB_PASSWORD", raising=False)
    monkeypatch.delenv("ARIEL_DSN", raising=False)


def _parse_env(tmp_path):
    from osprey.utils.dotenv import parse_dotenv_file

    path = tmp_path / ".env"
    return parse_dotenv_file(path) if path.is_file() else {}


def test_graphdb_deploy_mints_graphdb_password(captured_argv, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    assert env.get("GRAPHDB_PASSWORD")
    # token_hex(32) → 64 hex chars, 256 bits of entropy.
    assert len(env["GRAPHDB_PASSWORD"]) == 64
    # A graphdb-only deploy must not mint unrelated service tokens.
    assert "ARIEL_DB_PASSWORD" not in env
    assert "MONGO_ROOT_PASSWORD" not in env
    assert "ZO_ROOT_USER_PASSWORD" not in env


def test_graphdb_password_generator_survives_the_neo4j_auth_composite_every_time():
    """Every mint must be splittable out of ``neo4j/<password>`` unchanged.

    ``NEO4J_AUTH`` carries user and password in one string separated by ``/``,
    so a mint containing ``/`` would hand the container a different user and a
    truncated password. The hex alphabet is what makes that impossible; assert
    the whole registered rule (alphabet, no ``/``, ≥ 8 chars) rather than the
    generator's shape alone, so the two halves cannot drift apart.
    """
    for _ in range(50):
        value = _generate_token("GRAPHDB_PASSWORD")
        assert value.isalnum()
        assert "/" not in value
        assert len(value) >= 8
        assert _validate_var("GRAPHDB_PASSWORD", value)


def test_minted_value_is_never_the_published_default():
    """The mint can never produce the value the forbidden map refuses."""
    for _ in range(50):
        assert _generate_token("GRAPHDB_PASSWORD") != PUBLISHED_DEFAULT


def test_graphdb_mint_is_idempotent(captured_argv, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    first = _parse_env(tmp_path)
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    second = _parse_env(tmp_path)

    assert first["GRAPHDB_PASSWORD"] == second["GRAPHDB_PASSWORD"]
    assert (tmp_path / ".env").read_text().count("GRAPHDB_PASSWORD=") == 1


def test_existing_graphdb_password_is_preserved(captured_argv, tmp_path):
    """The volume that adopted the value outlives any later mint."""
    (tmp_path / ".env").write_text("GRAPHDB_PASSWORD=preexistingvalue\n", encoding="utf-8")
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert _parse_env(tmp_path)["GRAPHDB_PASSWORD"] == "preexistingvalue"


@pytest.mark.parametrize(
    "bad",
    [
        "neo4j/secret",  # the composite's own separator — moves the user/password split
        "a/b/c",
        "p@ssword",
        "user:pw",
        "q?xxxxxxxx",
        "frag#mentxx",
        "pass wordxx",
    ],
)
def test_operator_password_with_reserved_char_is_rejected(captured_argv, tmp_path, bad):
    """Refuse at the deploy boundary rather than let NEO4J_AUTH mis-split."""
    (tmp_path / ".env").write_text(f"GRAPHDB_PASSWORD={bad}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="GRAPHDB_PASSWORD"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)


@pytest.mark.parametrize("short", ["a", "short", "sevench"])
def test_operator_password_under_the_neo4j_minimum_is_rejected(captured_argv, tmp_path, short):
    """Neo4j refuses an initial password under 8 characters and crash-loops.

    Nothing about a short value is malformed as a URI or as a composite half,
    so the shared URI-safe rule admits it — the length floor is the only thing
    that turns that container-start crash-loop into a deploy-time error.
    """
    assert len(short) < 8
    (tmp_path / ".env").write_text(f"GRAPHDB_PASSWORD={short}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="GRAPHDB_PASSWORD"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)


def test_password_at_the_minimum_length_is_accepted(captured_argv, tmp_path):
    """Exactly 8 characters is Neo4j's floor, not one over it."""
    (tmp_path / ".env").write_text("GRAPHDB_PASSWORD=eightchr\n", encoding="utf-8")
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert _parse_env(tmp_path)["GRAPHDB_PASSWORD"] == "eightchr"


def test_published_template_default_is_refused(captured_argv, tmp_path):
    """The template's ``:-`` fallback is a shared password, not a secret.

    It is well-formed by every format rule this var registers — that is exactly
    why the forbidden-values map exists — so nothing but an explicit refusal
    catches an operator who pinned it as their real password.
    """
    (tmp_path / ".env").write_text(f"GRAPHDB_PASSWORD={PUBLISHED_DEFAULT}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="GRAPHDB_PASSWORD"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)


def test_rejection_never_echoes_the_password(captured_argv, tmp_path):
    (tmp_path / ".env").write_text("GRAPHDB_PASSWORD=neo4j/secret\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert "neo4j/secret" not in str(excinfo.value)


def test_forbidden_rejection_never_echoes_the_default(captured_argv, tmp_path):
    (tmp_path / ".env").write_text(f"GRAPHDB_PASSWORD={PUBLISHED_DEFAULT}\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert PUBLISHED_DEFAULT not in str(excinfo.value)


class TestExternalStore:
    """graphdb not deployed: validated when present, never minted when absent."""

    def test_absent_password_is_never_fabricated(self, external_store_argv, tmp_path):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
        env = _parse_env(tmp_path)
        assert "GRAPHDB_PASSWORD" not in env
        # The deploy still ran its own service's mint, so this is a scoped
        # absence rather than a deploy that did nothing.
        assert env.get("ARIEL_DB_PASSWORD")

    def test_operator_supplied_password_is_preserved(self, external_store_argv, tmp_path):
        (tmp_path / ".env").write_text("GRAPHDB_PASSWORD=externalstorepw\n", encoding="utf-8")
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
        assert _parse_env(tmp_path)["GRAPHDB_PASSWORD"] == "externalstorepw"

    @pytest.mark.parametrize("bad", ["neo4j/secret", "p@ssword", "short"])
    def test_malformed_operator_password_is_still_rejected(
        self, external_store_argv, tmp_path, bad
    ):
        """A store this deploy does not run still parses the value the same way.

        Without the validate-only registration the var would be invisible to
        the deploy boundary whenever graphdb is absent from
        ``deployed_services`` — the exact configuration in which the operator,
        not the mint, wrote the value.
        """
        (tmp_path / ".env").write_text(f"GRAPHDB_PASSWORD={bad}\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="GRAPHDB_PASSWORD"):
            container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)


class TestRegistryAgreement:
    """The registry entries that make the mint and the validation one contract."""

    def test_graphdb_declares_exactly_the_password_var(self):
        from osprey.deployment.container_lifecycle import _SERVICE_TOKEN_VARS

        assert _SERVICE_TOKEN_VARS["graphdb"] == ("GRAPHDB_PASSWORD",)

    def test_the_password_is_registered_validate_only(self):
        """Without this the external-store path validates nothing at all."""
        from osprey.deployment.container_lifecycle import _VALIDATE_ONLY_VARS

        assert "GRAPHDB_PASSWORD" in _VALIDATE_ONLY_VARS

    def test_the_forbidden_value_is_the_bare_password_not_the_composite(self):
        """The template interpolates ``neo4j/${GRAPHDB_PASSWORD:-ospreygraph}``.

        The var holds only the password half, so registering the composite
        would mean the map never matches anything an operator can actually
        write into ``.env`` — a refusal that silently never fires.
        """
        assert _VAR_FORBIDDEN_VALUES["GRAPHDB_PASSWORD"] == frozenset({PUBLISHED_DEFAULT})

    def test_the_validator_failure_names_both_constraints(self):
        """One message has to send the operator to the right one of two fixes."""
        from osprey.deployment.service_tokens import _VAR_VALIDATOR_DESCRIPTIONS

        description = _VAR_VALIDATOR_DESCRIPTIONS["GRAPHDB_PASSWORD"]
        assert "NEO4J_AUTH" in description
        assert "8" in description

    def test_the_minted_var_is_registered_as_volume_initialized(self):
        """A missing entry makes the stale-volume preflight silently never fire.

        Neo4j reads ``NEO4J_AUTH`` only while initializing an empty ``/data``,
        and that volume carries the server's own auth store — so a freshly
        minted password beside a surviving volume is a credential mismatch that
        surfaces only as an authentication failure minutes into the deploy. The
        registry entry is what makes ``_preflight_stale_store_volumes`` look for
        that volume at all.
        """
        from osprey.deployment.container_lifecycle import _VOLUME_INITIALIZED_VARS

        assert _VOLUME_INITIALIZED_VARS["GRAPHDB_PASSWORD"].service == "graphdb"

    def test_the_registry_names_the_volume_the_template_declares(self):
        """A drifted volume name makes the preflight look for something absent.

        The failure mode is silent and one-directional: the probe finds no such
        volume, concludes this is an ordinary first deploy, and lets the
        credential mismatch through to the store — the exact bug the preflight
        exists to stop.
        """
        from osprey.deployment.container_lifecycle import _VOLUME_INITIALIZED_VARS

        store = _VOLUME_INITIALIZED_VARS["GRAPHDB_PASSWORD"]
        declared = _declared_top_level_volumes(_graphdb_template_text())
        assert store.volume in declared
        # The data volume, specifically. `graphdb_plugins` also survives a
        # recreate but holds only a downloaded jar, so registering it would make
        # the preflight refuse deploys over a volume that costs nothing to drop.
        assert store.volume != "graphdb_plugins"

    def test_the_registry_names_the_container_the_template_uses(self):
        """A wrong suffix makes ``--reuse-stores`` report a live volume as lost.

        The harvest reads the original credential off this exact container; a
        name that does not resolve looks identical to a container that is gone,
        and the operator discards data that could have been kept.
        """
        from osprey.deployment.container_lifecycle import _VOLUME_INITIALIZED_VARS

        store = _VOLUME_INITIALIZED_VARS["GRAPHDB_PASSWORD"]
        template = _graphdb_template_text()
        assert f"container_name: {{{{ osprey_labels.project_name }}}}{store.container}" in template

    def test_the_registry_cred_env_and_prefix_match_the_template_composite(self):
        """The one entry in this map whose container value is not the credential.

        ``cred_prefix`` is pinned against the template's ACTUAL ``NEO4J_AUTH``
        spelling rather than against the literal ``neo4j/``, so respelling the
        composite in the template — a different user, or dropping the composite
        for a bare password — breaks this test instead of silently leaving the
        harvest stripping a prefix that is no longer there.
        """
        from osprey.deployment.container_lifecycle import _VOLUME_INITIALIZED_VARS

        store = _VOLUME_INITIALIZED_VARS["GRAPHDB_PASSWORD"]
        key, prefix, default = _password_composite_in_template(_graphdb_template_text())
        assert store.cred_env == key
        assert store.cred_prefix == prefix
        # Belt and braces on the parse itself: a regex that matched some other
        # line would pin the entry to whatever it happened to find.
        assert (key, prefix) == ("NEO4J_AUTH", "neo4j/")
        # Same line, and the source of the forbidden value pinned above — so the
        # two registry entries provably describe one template assignment.
        assert default == PUBLISHED_DEFAULT

    def test_every_other_store_declares_no_prefix(self):
        """graphdb is the exception; the mechanism must be inert for the rest.

        ``cred_prefix`` defaults to empty and ``str.removeprefix('')`` is the
        identity, so postgres, mongo and openobserve harvest exactly as before.
        Pinned because a stray prefix on one of them would corrupt a credential
        that currently round-trips.
        """
        from osprey.deployment.container_lifecycle import _VOLUME_INITIALIZED_VARS

        prefixed = {
            var
            for var, store in _VOLUME_INITIALIZED_VARS.items()
            if store.cred_prefix  # non-empty
        }
        assert prefixed == {"GRAPHDB_PASSWORD"}


class _StubProbe:
    """Answers ``env_of_container`` and nothing else, for the harvest alone."""

    def __init__(self, envs: dict[str, dict[str, str]]):
        self.envs = envs
        self.asked: list[str] = []

    def env_of_container(self, name: str):
        self.asked.append(name)
        return self.envs.get(name)


class TestCredentialHarvest:
    """``_harvest_original_credential`` returns ``.env`` terms, not container terms.

    It is the single point both consumers read the volume's credential
    through — the staleness comparison and the ``--reuse-stores`` write — so a
    prefix left on here would be wrong in both places at once and in the same
    direction.
    """

    def test_the_user_prefix_is_stripped_off_the_composite(self):
        from osprey.deployment.container_lifecycle import (
            _VOLUME_INITIALIZED_VARS,
            _harvest_original_credential,
        )

        store = _VOLUME_INITIALIZED_VARS["GRAPHDB_PASSWORD"]
        probe = _StubProbe({"proj-graphdb": {"NEO4J_AUTH": "neo4j/s3cretvalue"}})

        assert _harvest_original_credential(probe, "proj", store) == "s3cretvalue"
        assert probe.asked == ["proj-graphdb"]

    def test_a_composite_that_is_only_the_prefix_reads_as_unreadable(self):
        """``neo4j/`` is an empty password, which is no credential at all.

        Reporting it as a value would let ``--reuse-stores`` "restore" an empty
        password into ``.env`` and call the volume adopted.
        """
        from osprey.deployment.container_lifecycle import (
            _VOLUME_INITIALIZED_VARS,
            _harvest_original_credential,
        )

        store = _VOLUME_INITIALIZED_VARS["GRAPHDB_PASSWORD"]
        probe = _StubProbe({"proj-graphdb": {"NEO4J_AUTH": "neo4j/"}})

        assert _harvest_original_credential(probe, "proj", store) is None

    def test_a_composite_without_the_prefix_is_returned_whole(self):
        """``removeprefix`` is a no-op on a value that does not carry it.

        Not a shape OSPREY's own template produces, but an operator-run
        container can carry any ``NEO4J_AUTH``, and truncating one that happens
        to start differently would be worse than passing it through.
        """
        from osprey.deployment.container_lifecycle import (
            _VOLUME_INITIALIZED_VARS,
            _harvest_original_credential,
        )

        store = _VOLUME_INITIALIZED_VARS["GRAPHDB_PASSWORD"]
        probe = _StubProbe({"proj-graphdb": {"NEO4J_AUTH": "bareword"}})

        assert _harvest_original_credential(probe, "proj", store) == "bareword"

    def test_an_unprefixed_store_harvests_exactly_what_the_container_holds(self):
        """The zero-change guarantee for the three stores that predate this."""
        from osprey.deployment.container_lifecycle import (
            _VOLUME_INITIALIZED_VARS,
            _harvest_original_credential,
        )

        store = _VOLUME_INITIALIZED_VARS["MONGO_ROOT_PASSWORD"]
        probe = _StubProbe(
            {"proj-archiver-mongodb": {"MONGO_INITDB_ROOT_PASSWORD": "neo4j/keepme"}}
        )

        # Even a value that *looks* like a composite is untouched: the strip is
        # per-store, not a global rule about slashes.
        assert _harvest_original_credential(probe, "proj", store) == "neo4j/keepme"


class TestStalenessAndAdoption:
    """The two consumers of the harvest, on the real ``RuntimeProbe``."""

    def test_a_matching_composite_is_not_stale(self, tmp_path):
        """The healthy steady state: container says ``neo4j/X``, ``.env`` says ``X``.

        Without the strip this is the case that breaks — every working graphdb
        deployment would be reported stale on every restart, because the
        comparison would put ``neo4j/X`` beside ``X``.
        """
        env_path = tmp_path / ".env"
        env_path.write_text("GRAPHDB_PASSWORD=matchingpw\n", encoding="utf-8")

        stale, originals = _stale_graphdb(
            volumes=[f"{_PROJECT}_graphdb_data"],
            container_env={f"{_PROJECT}-graphdb": {"NEO4J_AUTH": "neo4j/matchingpw"}},
            env_path=env_path,
        )

        assert stale == []

    def test_a_disagreeing_composite_is_stale_and_reports_the_bare_password(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("GRAPHDB_PASSWORD=whatEnvSays\n", encoding="utf-8")

        stale, originals = _stale_graphdb(
            volumes=[f"{_PROJECT}_graphdb_data"],
            container_env={f"{_PROJECT}-graphdb": {"NEO4J_AUTH": "neo4j/whatVolumeHas"}},
            env_path=env_path,
        )

        assert [var for var, _ in stale] == ["GRAPHDB_PASSWORD"]
        assert originals["GRAPHDB_PASSWORD"] == "whatVolumeHas"

    def test_adoption_writes_the_bare_password_not_the_composite(self, tmp_path):
        """What ``--reuse-stores`` puts back must be re-composable by the template.

        The template builds ``neo4j/${GRAPHDB_PASSWORD}``, so restoring the
        composite would render ``neo4j/neo4j/<password>`` and lock the operator
        out of the very store the flag exists to keep.
        """
        from osprey.deployment.container_lifecycle import _adopt_original_credentials

        env_path = tmp_path / ".env"
        env_path.write_text("GRAPHDB_PASSWORD=freshlyminted\n", encoding="utf-8")

        stale, originals = _stale_graphdb(
            volumes=[f"{_PROJECT}_graphdb_data"],
            container_env={f"{_PROJECT}-graphdb": {"NEO4J_AUTH": "neo4j/theOriginal"}},
            env_path=env_path,
        )
        _adopt_original_credentials(env_path, stale, originals)

        written = _parse_env(tmp_path)["GRAPHDB_PASSWORD"]
        assert written == "theOriginal"
        assert not written.startswith("neo4j/")


class TestExternalStoreForbiddenValue:
    """The validate-only loop must ask both questions, not just the format one.

    A value that reaches that loop was never minted — graphdb is absent from
    ``deployed_services``, so nothing in this deploy could have written it. It is
    operator-supplied by construction, and copying the template's published
    fallback is exactly the mistake an operator wiring an external store makes.
    No format rule can catch it: ``ospreygraph`` is well-formed by every
    constraint the var registers.
    """

    def test_the_published_default_is_refused_on_the_external_store_path(
        self, external_store_argv, tmp_path
    ):
        (tmp_path / ".env").write_text(f"GRAPHDB_PASSWORD={PUBLISHED_DEFAULT}\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="GRAPHDB_PASSWORD"):
            container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    def test_the_external_store_refusal_never_echoes_the_default(
        self, external_store_argv, tmp_path
    ):
        """Same no-echo rule the minted path already holds to."""
        (tmp_path / ".env").write_text(f"GRAPHDB_PASSWORD={PUBLISHED_DEFAULT}\n", encoding="utf-8")

        with pytest.raises(RuntimeError) as excinfo:
            container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

        assert PUBLISHED_DEFAULT not in str(excinfo.value)

    def test_the_refusal_is_the_forbidden_value_diagnosis_not_a_format_one(
        self, external_store_argv, tmp_path
    ):
        """Two failures, two fixes — the message has to name the right one.

        "your password is public" and "your password has a bad character" send
        an operator to different places, and only the first is true here.
        """
        (tmp_path / ".env").write_text(f"GRAPHDB_PASSWORD={PUBLISHED_DEFAULT}\n", encoding="utf-8")

        with pytest.raises(RuntimeError) as excinfo:
            container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

        message = str(excinfo.value)
        from osprey.deployment.service_tokens import _VAR_FORBIDDEN_DESCRIPTIONS

        assert _VAR_FORBIDDEN_DESCRIPTIONS["GRAPHDB_PASSWORD"] in message

    def test_a_forbidden_value_from_the_process_env_is_refused_too(
        self, external_store_argv, tmp_path, monkeypatch
    ):
        """A shell export outranks ``.env`` for compose, so it must be checked."""
        monkeypatch.setenv("GRAPHDB_PASSWORD", PUBLISHED_DEFAULT)

        with pytest.raises(RuntimeError, match="GRAPHDB_PASSWORD"):
            container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    def test_a_clean_external_password_still_passes(self, external_store_argv, tmp_path):
        """The new check must refuse the default and nothing else."""
        (tmp_path / ".env").write_text("GRAPHDB_PASSWORD=notthedefault\n", encoding="utf-8")

        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

        assert _parse_env(tmp_path)["GRAPHDB_PASSWORD"] == "notthedefault"

    def test_a_validate_only_var_with_no_forbidden_entry_is_unaffected(self, tmp_path):
        """ARIEL_DSN registers no forbidden values; the lookup must be a no-op."""
        env_path = tmp_path / ".env"
        dsn = "postgresql://ariel:ariel@ariel-postgres:5432/ariel"
        env_path.write_text(f"ARIEL_DSN={dsn}\n", encoding="utf-8")

        container_lifecycle._ensure_service_tokens(
            {"deployed_services": []}, expose_network=False, env_path=env_path
        )

        assert _parse_env(tmp_path)["ARIEL_DSN"] == dsn


# ---------------------------------------------------------------------------
# Helpers for the registry-agreement and harvest tests
# ---------------------------------------------------------------------------

#: Compose project name for the staleness/adoption fixtures. Arbitrary, but the
#: volume and container names are derived from it exactly as compose derives
#: them, so a registry entry that disagrees with the template fails here too.
_PROJECT = "graph-project"


def _graphdb_template_text() -> str:
    """The graphdb compose template as written, jinja expressions intact."""
    from pathlib import Path

    import osprey

    path = (
        Path(osprey.__file__).parent
        / "templates"
        / "services"
        / "graphdb"
        / "docker-compose.yml.j2"
    )
    return path.read_text(encoding="utf-8")


def _declared_top_level_volumes(template: str) -> set[str]:
    """Bare volume names from the template's top-level ``volumes:`` mapping.

    Scanned textually rather than with ``yaml.safe_load``: the service body
    carries jinja expressions (``${VAR:-{{ … }}}``) that are not valid YAML, so
    parsing the file whole would fail for reasons unrelated to what is asserted.
    Only the first indentation level inside the block is collected, since each
    volume here carries a nested ``labels:`` map that is not a volume name.
    """
    names: set[str] = set()
    in_block = False
    for line in template.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[:1].isspace():
            in_block = line.rstrip() == "volumes:"
            continue
        if in_block and len(line) - len(line.lstrip()) == 2 and line.rstrip().endswith(":"):
            names.add(line.strip().rstrip(":"))
    return names


def _password_composite_in_template(template: str) -> tuple[str, str, str]:
    """The template line that builds a container variable from GRAPHDB_PASSWORD.

    Returns ``(container_var, prefix, published_default)`` read off the template
    itself, so the registry's ``cred_env``/``cred_prefix`` are pinned to what
    the template actually spells rather than to a literal repeated in the test.
    """
    import re

    match = re.search(
        r'^\s*(?P<key>[A-Za-z0-9_]+):\s*"(?P<prefix>[^"$]*)'
        r'\$\{GRAPHDB_PASSWORD:-(?P<default>[^}]*)\}"\s*$',
        template,
        re.MULTILINE,
    )
    assert match is not None, (
        "the graphdb template no longer builds a container variable from "
        "${GRAPHDB_PASSWORD:-…}; the registry's cred_env/cred_prefix describe "
        "an assignment that is gone"
    )
    return match["key"], match["prefix"], match["default"]


class _FakeRuntime:
    """Answers the argv the volume probe and the credential harvest build.

    Argv-shaped rather than a mock of ``RuntimeProbe``: the thing under test is
    the credential the deploy reads back, so a stand-in that accepted anything
    would pass while the real command was wrong.
    """

    def __init__(self, volumes: list[str], container_env: dict[str, dict[str, str]]):
        self.volumes = volumes
        self.container_env = container_env

    def __call__(self, cmd, **kwargs):
        argv = list(cmd)[1:]  # drop the runtime binary
        if argv[:2] == ["volume", "ls"]:
            return subprocess.CompletedProcess(list(cmd), 0, stdout="\n".join(self.volumes))
        if argv[:2] == ["container", "inspect"]:
            env = self.container_env.get(argv[2])
            if env is None:
                return subprocess.CompletedProcess(list(cmd), 1, stdout="")
            body = "\n".join(f"{k}={v}" for k, v in env.items())
            return subprocess.CompletedProcess(list(cmd), 0, stdout=body)
        return subprocess.CompletedProcess(list(cmd), 0, stdout="")


def _stale_graphdb(volumes, container_env, env_path):
    """``_stale_store_volumes`` for a graphdb-only project, nothing minted.

    ``minted`` is empty on purpose: this exercises the *disagreement* ground,
    which is the one that reads the container's credential and compares it with
    ``.env``. The minted ground would qualify the store without ever consulting
    the harvest, and so would pass whether or not the prefix is stripped.
    """
    from osprey.deployment.reset import RuntimeProbe

    probe = RuntimeProbe("docker", run=_FakeRuntime(volumes, container_env))
    return container_lifecycle._stale_store_volumes(probe, _PROJECT, set(), {"graphdb"}, env_path)

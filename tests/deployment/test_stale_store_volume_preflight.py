"""A freshly minted store credential must never meet a volume that predates it.

The three stores in ``_VOLUME_INITIALIZED_VARS`` read their root credential
only while initializing an empty data volume. Mint a new value beside a volume
that already exists and the store keeps the password it was born with: the
stack starts, the deploy waits, and the store refuses the credential the
``.env`` now claims.

``osprey up`` used to only *guess* at this — it warned "if this deployment's
volume already exists" at the mint and started the stack anyway. Two things
make that worse than a plain refusal:

* the diagnosis arrives after the project image build and a health-probe
  timeout, and names only the first store to be asked, leaving the operator to
  rediscover the other two one restart at a time;
* ``compose up`` recreates the store container on its way past, and that
  container's environment held the *only* copy of the credential the volume was
  initialized with. Proceeding destroys the evidence that would let the volume
  be reopened at all.

So the check belongs before anything touches a container, and it is a check
rather than a warning: the runtime can be asked whether the volume exists.
"""

from __future__ import annotations

import subprocess

import pytest

from osprey.deployment import container_lifecycle

PROJECT = "my-control-assistant"


class FakeRuntime:
    """Answers the argv the volume probe and the credential harvest build.

    Deliberately argv-shaped rather than a mock of the probe: the thing under
    test is which questions the deploy asks the runtime, so a stand-in that
    accepted anything would pass while the real command was wrong.
    """

    def __init__(self, volumes: list[str], container_env: dict[str, dict[str, str]] | None = None):
        self.volumes = volumes
        self.container_env = container_env or {}
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        argv = list(cmd)[1:]  # drop the runtime binary
        if argv[:2] == ["volume", "ls"]:
            return _completed(cmd, "\n".join(self.volumes))
        if argv[:2] == ["container", "inspect"]:
            name = argv[2]
            env = self.container_env.get(name)
            if env is None:
                return _completed(cmd, "", returncode=1)
            return _completed(cmd, "\n".join(f"{k}={v}" for k, v in env.items()))
        return _completed(cmd, "")


def _completed(cmd, stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(list(cmd), returncode, stdout=stdout, stderr="")


@pytest.fixture
def deploy(monkeypatch, tmp_path):
    """A one-store project whose runtime answers are set per test."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MONGO_ROOT_PASSWORD", raising=False)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {"deployed_services": ["mongodb"], "project_name": PROJECT},
            ["docker-compose.yml"],
        ),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _run(fake: FakeRuntime):
        monkeypatch.setattr(container_lifecycle.subprocess, "run", fake)
        return lambda **kw: container_lifecycle.deploy_up(
            str(tmp_path / "config.yml"), detached=True, **kw
        )

    return _run


def _env(tmp_path):
    from osprey.utils.dotenv import parse_dotenv_file

    path = tmp_path / ".env"
    return parse_dotenv_file(path) if path.is_file() else {}


def test_refuses_when_a_minted_credential_meets_a_surviving_volume(deploy, tmp_path):
    """The whole point: stop before the image build, not after a probe timeout."""
    up = deploy(FakeRuntime(volumes=[f"{PROJECT}_archiver_mongodb_data"]))

    with pytest.raises(RuntimeError) as excinfo:
        up()

    message = str(excinfo.value)
    assert "archiver_mongodb_data" in message
    assert "mongodb" in message


def test_a_first_deploy_with_no_volumes_is_untouched(deploy, tmp_path):
    """The ordinary case the old code was afraid of breaking must stay silent."""
    up = deploy(FakeRuntime(volumes=[]))

    up()

    assert _env(tmp_path).get("MONGO_ROOT_PASSWORD")


def test_an_operator_supplied_credential_is_never_second_guessed(deploy, tmp_path):
    """Nothing was minted, so the value and the volume agree by construction."""
    (tmp_path / ".env").write_text("MONGO_ROOT_PASSWORD=preexistingvalue\n", encoding="utf-8")
    up = deploy(FakeRuntime(volumes=[f"{PROJECT}_archiver_mongodb_data"]))

    up()

    assert _env(tmp_path)["MONGO_ROOT_PASSWORD"] == "preexistingvalue"


class TestAlreadyBrokenDeployment:
    """A mismatch persists in ``.env`` long after the run that minted it.

    Keying only on "this run minted it" prevents the situation but cannot
    recognise one already in it: after a failed start the ``.env`` holds the new
    value, so the next start mints nothing and sails past. The second rule is
    provable rather than inferred — a running store container is on the
    credential its volume actually has, so a ``.env`` that disagrees with the
    container cannot authenticate, whoever wrote it.
    """

    def test_a_container_credential_that_disagrees_with_env_is_refused(self, deploy, tmp_path):
        (tmp_path / ".env").write_text("MONGO_ROOT_PASSWORD=thenewone\n", encoding="utf-8")
        up = deploy(
            FakeRuntime(
                volumes=[f"{PROJECT}_archiver_mongodb_data"],
                container_env={
                    f"{PROJECT}-archiver-mongodb": {"MONGO_INITDB_ROOT_PASSWORD": "theoriginal"}
                },
            )
        )

        with pytest.raises(RuntimeError) as excinfo:
            up()

        assert "archiver_mongodb_data" in str(excinfo.value)

    def test_a_healthy_redeploy_is_untouched(self, deploy, tmp_path):
        """The regression that would matter most: every ordinary restart.

        Container and ``.env`` agree, so the volume will accept the credential
        and there is nothing to report — even though the volume is old and
        nothing was minted.
        """
        (tmp_path / ".env").write_text("MONGO_ROOT_PASSWORD=agreed\n", encoding="utf-8")
        up = deploy(
            FakeRuntime(
                volumes=[f"{PROJECT}_archiver_mongodb_data"],
                container_env={
                    f"{PROJECT}-archiver-mongodb": {"MONGO_INITDB_ROOT_PASSWORD": "agreed"}
                },
            )
        )

        up()

        assert _env(tmp_path)["MONGO_ROOT_PASSWORD"] == "agreed"

    def test_a_stopped_store_with_no_container_is_not_second_guessed(self, deploy, tmp_path):
        """Nothing minted and nothing to compare against is not evidence of a fault.

        Refusing here would block every deploy whose stack is fully down, which
        is the normal way a deployment sits between sessions.
        """
        (tmp_path / ".env").write_text("MONGO_ROOT_PASSWORD=whatever\n", encoding="utf-8")
        up = deploy(FakeRuntime(volumes=[f"{PROJECT}_archiver_mongodb_data"], container_env={}))

        up()

        assert _env(tmp_path)["MONGO_ROOT_PASSWORD"] == "whatever"


class TestRecoverability:
    """Whether the volume can still be reopened decides what the operator can do.

    A store container holds, in its own environment, the credential its volume
    was initialized with. While that container survives the data is reachable;
    once it is recreated the volume is orphaned for good. The refusal has to
    tell those two apart, because they offer the operator different choices.
    """

    def test_a_surviving_container_is_reported_as_recoverable(self, deploy, tmp_path):
        up = deploy(
            FakeRuntime(
                volumes=[f"{PROJECT}_archiver_mongodb_data"],
                container_env={
                    f"{PROJECT}-archiver-mongodb": {"MONGO_INITDB_ROOT_PASSWORD": "theoriginal"}
                },
            )
        )

        with pytest.raises(RuntimeError) as excinfo:
            up()

        message = str(excinfo.value)
        assert "original recoverable from" in message
        assert "--reuse-stores" in message  # the way out this store leaves open
        # Never the value itself, in keeping with every other secret this path logs.
        assert "theoriginal" not in message

    def test_an_absent_container_is_reported_as_unrecoverable(self, deploy, tmp_path):
        """The shape the failed run leaves behind: volume alive, container gone."""
        up = deploy(FakeRuntime(volumes=[f"{PROJECT}_archiver_mongodb_data"], container_env={}))

        with pytest.raises(RuntimeError) as excinfo:
            up()

        message = str(excinfo.value)
        assert "the original is unrecoverable" in message
        assert "original recoverable from" not in message
        # Offering --reuse-stores here would send the operator down a path that
        # cannot work for this store.
        assert "--reuse-stores     keep the data" not in message


class TestReuseStores:
    """``--reuse-stores``: adopt the volumes instead of the freshly minted value."""

    def test_it_restores_the_original_credential_and_proceeds(self, deploy, tmp_path):
        up = deploy(
            FakeRuntime(
                volumes=[f"{PROJECT}_archiver_mongodb_data"],
                container_env={
                    f"{PROJECT}-archiver-mongodb": {"MONGO_INITDB_ROOT_PASSWORD": "theoriginal"}
                },
            )
        )

        up(reuse_stores=True)

        assert _env(tmp_path)["MONGO_ROOT_PASSWORD"] == "theoriginal"

    def test_it_refuses_when_the_original_cannot_be_read(self, deploy, tmp_path):
        """Reuse must not half-succeed: a store it cannot reopen is a hard stop.

        Proceeding would start the recoverable stores on adopted credentials
        and leave the unrecoverable one to fail at its health probe — the
        original bug, reintroduced one store narrower.
        """
        up = deploy(FakeRuntime(volumes=[f"{PROJECT}_archiver_mongodb_data"], container_env={}))

        with pytest.raises(RuntimeError) as excinfo:
            up(reuse_stores=True)

        assert "archiver_mongodb_data" in str(excinfo.value)

    def test_it_is_a_no_op_when_no_volume_survives(self, deploy, tmp_path):
        """Passing the flag on an ordinary first deploy must not change it."""
        up = deploy(FakeRuntime(volumes=[]))

        up(reuse_stores=True)

        assert len(_env(tmp_path)["MONGO_ROOT_PASSWORD"]) == 64  # the freshly minted value


class TestRestartChecksBeforeItStops:
    """``restart`` stops the stack first — which is what destroys the evidence.

    ``down`` removes the store containers, and a removed container takes the
    only host-side copy of its volume's credential with it. So a restart that
    checked at the same point ``up`` does would find every stale volume
    "unrecoverable", having itself made them so moments earlier, and would tell
    the operator to discard data that was recoverable when they typed the
    command.
    """

    @pytest.fixture
    def restart(self, monkeypatch, tmp_path):
        repo = tmp_path / "repo"
        (repo / "build").mkdir(parents=True)
        (repo / ".env").write_text("", encoding="utf-8")
        stopped: list[bool] = []

        monkeypatch.setattr(
            container_lifecycle,
            "_resolve_as_built_inputs",
            lambda root, *, dev_mode: (
                {"deployed_services": ["mongodb"], "project_name": PROJECT},
                ["docker-compose.yml"],
                False,
            ),
        )
        monkeypatch.setattr(
            container_lifecycle, "down_deployment", lambda root: stopped.append(True)
        )
        monkeypatch.setattr(
            container_lifecycle, "_start_as_built", lambda *a, **k: stopped.append(False)
        )
        monkeypatch.setattr(
            container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
        )
        monkeypatch.delenv("MONGO_ROOT_PASSWORD", raising=False)

        def _run(fake: FakeRuntime, **kw):
            monkeypatch.setattr(container_lifecycle.subprocess, "run", fake)
            return container_lifecycle.restart_deployment(repo, **kw), stopped, repo

        return _run

    def test_it_refuses_without_stopping_anything(self, restart):
        fake = FakeRuntime(
            volumes=[f"{PROJECT}_archiver_mongodb_data"],
            container_env={
                f"{PROJECT}-archiver-mongodb": {"MONGO_INITDB_ROOT_PASSWORD": "theoriginal"}
            },
        )

        with pytest.raises(RuntimeError) as excinfo:
            restart(fake)

        assert "original recoverable from" in str(excinfo.value)

    def test_reuse_adopts_the_credential_before_the_stop(self, restart):
        """Order is the whole point: harvest, write .env, and only then stop."""
        fake = FakeRuntime(
            volumes=[f"{PROJECT}_archiver_mongodb_data"],
            container_env={
                f"{PROJECT}-archiver-mongodb": {"MONGO_INITDB_ROOT_PASSWORD": "theoriginal"}
            },
        )

        _, stopped, repo = restart(fake, reuse_stores=True)

        from osprey.utils.dotenv import parse_dotenv_file

        assert parse_dotenv_file(repo / ".env")["MONGO_ROOT_PASSWORD"] == "theoriginal"
        assert stopped == [True, False]  # stopped, then started


def test_the_volume_probe_is_label_filtered_to_this_project(deploy, tmp_path):
    """A host-wide listing would let another project's volume block this deploy."""
    fake = FakeRuntime(volumes=[])
    up = deploy(fake)

    up()

    (volume_ls,) = [c for c in fake.calls if c[1:3] == ["volume", "ls"]]
    assert f"label=com.docker.compose.project={PROJECT}" in volume_ls

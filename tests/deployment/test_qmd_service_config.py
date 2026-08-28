"""Tests for the ``services.qmd`` config schema resolver."""

from pathlib import Path

import pytest

from osprey.deployment.errors import DeploymentPreconditionError
from osprey.deployment.host_ports import _SERVICE_REMEDY_KEYS
from osprey.deployment.qmd_service import (
    DEFAULT_BIND_ADDRESS,
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_PORT,
    MODEL_FILENAMES,
    MODELS_DIR_CONFIG_KEY,
    PORT_CONFIG_KEY,
    QMDServiceConfig,
    preflight_qmd_models_dir,
    resolve_bind_address,
    resolve_qmd_service_config,
)


class TestAbsentBlock:
    """A deployment without a qmd sidecar resolves to ``None``, not defaults."""

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"services": {}},
            {"services": {"postgresql": {"port_host": 5432}}},
            {"services": None},
            {"services": {"qmd": None}},
        ],
        ids=["none", "empty", "no-qmd", "other-service", "services-null", "qmd-null"],
    )
    def test_absent_resolves_to_none(self, config: dict | None) -> None:
        assert resolve_qmd_service_config(config) is None


class TestDefaults:
    """A present-but-empty block fills in every default."""

    def test_empty_block_gets_defaults(self) -> None:
        resolved = resolve_qmd_service_config({"services": {"qmd": {}}})
        assert resolved == QMDServiceConfig(
            port=DEFAULT_PORT,
            bind_address=DEFAULT_BIND_ADDRESS,
            interval_seconds=DEFAULT_INTERVAL_SECONDS,
        )

    def test_default_port_avoids_qmd_own_daemon_port(self) -> None:
        """8181 names the container-internal daemon; the published port differs."""
        assert DEFAULT_PORT != 8181

    def test_explicit_values_win(self) -> None:
        resolved = resolve_qmd_service_config(
            {
                "deployment": {"bind_address": "0.0.0.0"},
                "services": {"qmd": {"port": 9999, "interval": 300}},
            }
        )
        assert resolved is not None
        assert (resolved.port, resolved.interval_seconds) == (9999, 300)
        assert resolved.bind_address == "0.0.0.0"


class TestBindAddress:
    """``bind_address`` is project-wide, never a per-service key."""

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"deployment": {}},
            {"deployment": None},
            {"deployment": {"bind_address": None}},
            {"deployment": {"bind_address": "   "}},
        ],
        ids=["none", "empty", "no-address", "deployment-null", "address-null", "blank"],
    )
    def test_defaults_to_loopback(self, config: dict | None) -> None:
        assert resolve_bind_address(config) == "127.0.0.1"

    def test_reads_project_wide_key(self) -> None:
        assert resolve_bind_address({"deployment": {"bind_address": " 10.0.0.5 "}}) == "10.0.0.5"

    def test_per_service_bind_address_is_ignored(self) -> None:
        """A hand-written ``services.qmd.bind_address`` must not take effect.

        Honouring it would let the sidecar publish on an interface the rest of
        the stack does not, which is exactly the split the single project-wide
        key exists to prevent.
        """
        resolved = resolve_qmd_service_config(
            {"services": {"qmd": {"bind_address": "0.0.0.0"}}},
        )
        assert resolved is not None
        assert resolved.bind_address == "127.0.0.1"


class TestValidation:
    """A malformed scalar refuses rather than silently defaulting."""

    @pytest.mark.parametrize("bad", [0, -1, "8180", 8180.0, True, [8180]])
    def test_bad_port_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"services\.qmd\.port"):
            resolve_qmd_service_config({"services": {"qmd": {"port": bad}}})

    @pytest.mark.parametrize("bad", [0, -30, "30", 30.0, True])
    def test_bad_interval_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"services\.qmd\.interval"):
            resolve_qmd_service_config({"services": {"qmd": {"interval": bad}}})


class TestBaseUrl:
    """Clients connect over loopback regardless of the publish interface."""

    def test_uses_configured_port(self) -> None:
        assert QMDServiceConfig(port=8180).base_url == "http://127.0.0.1:8180"

    def test_wildcard_publish_still_dials_loopback(self) -> None:
        assert QMDServiceConfig(port=8180, bind_address="0.0.0.0").base_url == (
            "http://127.0.0.1:8180"
        )


class TestModelsDir:
    """The optional pre-staged-model directory, and its shape validation."""

    def test_absent_by_default(self) -> None:
        resolved = resolve_qmd_service_config({"services": {"qmd": {}}})
        assert resolved is not None
        assert resolved.models_dir is None

    def test_absolute_path_is_kept(self) -> None:
        resolved = resolve_qmd_service_config(
            {"services": {"qmd": {"models_dir": " /srv/qmd-models "}}}
        )
        assert resolved is not None
        assert resolved.models_dir == "/srv/qmd-models"

    @pytest.mark.parametrize("bad", ["", "   ", "models", "./models", "~/models", 7, True])
    def test_non_absolute_path_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"services\.qmd\.models_dir"):
            resolve_qmd_service_config({"services": {"qmd": {"models_dir": bad}}})


class TestModelsDirPreflight:
    """A set ``models_dir`` refuses the deploy unless all three models are staged."""

    @staticmethod
    def _stage(directory: Path, names: tuple[str, ...]) -> None:
        for name in names:
            (directory / name).write_bytes(b"gguf")

    @pytest.mark.parametrize(
        "config",
        [None, {}, {"services": {"qmd": {}}}, {"services": {"qmd": {"port": 9000}}}],
        ids=["none", "empty", "no-models-dir", "other-keys-only"],
    )
    def test_unset_key_is_a_no_op(self, config: dict | None) -> None:
        assert preflight_qmd_models_dir(config) is None

    def test_missing_directory_refuses(self, tmp_path: Path) -> None:
        absent = tmp_path / "nope"
        with pytest.raises(DeploymentPreconditionError) as excinfo:
            preflight_qmd_models_dir({"services": {"qmd": {"models_dir": str(absent)}}})
        assert str(absent) in excinfo.value.reason
        assert MODELS_DIR_CONFIG_KEY in excinfo.value.reason

    def test_fully_staged_directory_passes(self, tmp_path: Path) -> None:
        self._stage(tmp_path, MODEL_FILENAMES)
        assert (
            preflight_qmd_models_dir({"services": {"qmd": {"models_dir": str(tmp_path)}}}) is None
        )

    def test_partial_staging_names_only_what_is_missing(self, tmp_path: Path) -> None:
        self._stage(tmp_path, MODEL_FILENAMES[:1])
        with pytest.raises(DeploymentPreconditionError) as excinfo:
            preflight_qmd_models_dir({"services": {"qmd": {"models_dir": str(tmp_path)}}})
        assert MODEL_FILENAMES[0] not in excinfo.value.reason
        assert MODEL_FILENAMES[1] in excinfo.value.reason
        assert MODEL_FILENAMES[2] in excinfo.value.reason

    def test_wrong_filename_reads_as_absent(self, tmp_path: Path) -> None:
        """The download basename is not the cache name qmd recognises."""
        self._stage(tmp_path, MODEL_FILENAMES[1:])
        (tmp_path / "embeddinggemma-300M-Q8_0.gguf").write_bytes(b"gguf")
        with pytest.raises(DeploymentPreconditionError) as excinfo:
            preflight_qmd_models_dir({"services": {"qmd": {"models_dir": str(tmp_path)}}})
        assert MODEL_FILENAMES[0] in excinfo.value.reason

    def test_empty_file_reads_as_absent(self, tmp_path: Path) -> None:
        """An interrupted copy leaves a zero-byte file that must not pass."""
        self._stage(tmp_path, MODEL_FILENAMES[1:])
        (tmp_path / MODEL_FILENAMES[0]).touch()
        with pytest.raises(DeploymentPreconditionError) as excinfo:
            preflight_qmd_models_dir({"services": {"qmd": {"models_dir": str(tmp_path)}}})
        assert MODEL_FILENAMES[0] in excinfo.value.reason

    def test_a_path_that_is_a_file_refuses(self, tmp_path: Path) -> None:
        """``models_dir`` names a directory to mount, not an archive to unpack.

        A file there passes the schema — it is an absolute path — and would be
        bind-mounted over the container's model directory, hiding it behind a
        single file. Refused with the same message a missing directory gets.
        """
        staged = tmp_path / "models.tar"
        staged.write_bytes(b"not a directory")
        with pytest.raises(DeploymentPreconditionError) as excinfo:
            preflight_qmd_models_dir({"services": {"qmd": {"models_dir": str(staged)}}})
        assert str(staged) in excinfo.value.reason

    def test_every_refusal_lists_all_three_names_verbatim(self, tmp_path: Path) -> None:
        """The remedy is the staging instruction, so it must be complete."""
        for config_dir in (tmp_path / "absent", tmp_path):
            with pytest.raises(DeploymentPreconditionError) as excinfo:
                preflight_qmd_models_dir(
                    {"services": {"qmd": {"models_dir": str(config_dir)}}},
                )
            for name in MODEL_FILENAMES:
                assert name in excinfo.value.remedy


def test_the_deploy_runs_the_preflight_before_it_touches_the_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwired preflight is a preflight that never runs.

    Every other assertion here calls the function directly, which proves the
    check is correct and nothing about it being reached. This one goes in
    through ``_start_stack``, the one sequence both ``osprey up`` paths share,
    with a config whose model directory does not exist — and fails if the
    runtime is probed at all, because a refusal that arrives after the runtime
    has been touched is a refusal that arrives after work has started.
    """
    from osprey.deployment import container_lifecycle

    def _unreachable(config: object) -> tuple[bool, str]:
        raise AssertionError(
            "the container runtime was probed before the models-dir preflight refused"
        )

    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", _unreachable)

    with pytest.raises(DeploymentPreconditionError):
        container_lifecycle._start_stack(
            {
                "deployed_services": ["qmd"],
                "services": {"qmd": {"models_dir": str(tmp_path / "never-staged")}},
            },
            [],
            tmp_path,
        )


def test_model_filenames_match_the_dockerfile_pins() -> None:
    """The preflight checks for exactly the files the image build produces."""
    dockerfile = (
        Path(__file__).resolve().parents[2] / "src/osprey/templates/services/qmd/Dockerfile"
    )
    text = dockerfile.read_text()
    for name in MODEL_FILENAMES:
        assert name in text, f"{name} is not the cache name the Dockerfile writes"


def test_model_fetches_retry_a_dropped_stream() -> None:
    """Every model fetch carries ``--retry-all-errors``.

    The three GGUF downloads total ~2.1 GB, and the CDN drops a stream
    mid-transfer often enough to fail a whole image build (``curl: (92) HTTP/2
    stream was not closed cleanly``). curl retries neither that nor an HTTP
    error body unless asked: ``--retry`` covers transient transfer failures and
    ``--retry-connrefused`` only adds a refused connection, so without this flag
    the configured retry budget is never spent on the failure that actually
    happens. Pinned as text because the fetch only runs during a real image
    build — no unit test can reach it.
    """
    dockerfile = (
        Path(__file__).resolve().parents[2] / "src/osprey/templates/services/qmd/Dockerfile"
    )
    fetches = [line for line in dockerfile.read_text().splitlines() if "curl -fSL" in line]
    assert len(fetches) == len(MODEL_FILENAMES), "one fetch per pinned model"
    for line in fetches:
        assert "--retry-all-errors" in line, f"fetch retries only some errors: {line.strip()}"


def test_port_conflict_preflight_knows_the_sidecar() -> None:
    """The deploy-time port sweep can name the key that moves the qmd port."""
    assert _SERVICE_REMEDY_KEYS["qmd"] == PORT_CONFIG_KEY == "services.qmd.port"

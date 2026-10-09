"""Tests for the ``services.qmd`` config schema resolver."""

from pathlib import Path

import pytest

from osprey.deployment.errors import DeploymentPreconditionError
from osprey.deployment.host_ports import _SERVICE_REMEDY_KEYS
from osprey.deployment.qmd_service import (
    DEFAULT_BIND_ADDRESS,
    DEFAULT_FIRST_INDEX_GRACE_SECONDS,
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_PORT,
    INDEX_MANAGED,
    INDEX_PREBUILT,
    MAX_DECLARED_CORPORA,
    MODEL_FILENAMES,
    MODELS_DIR_CONFIG_KEY,
    PORT_CONFIG_KEY,
    DeclaredCorpus,
    QMDServiceConfig,
    corpus_service_name,
    corpus_url_env,
    preflight_qmd_corpora,
    preflight_qmd_models_dir,
    resolve_bind_address,
    resolve_qmd_corpus_config,
    resolve_qmd_service_config,
)
from osprey.port_layout import QMD_CORPUS_MAX, default_port


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

    def test_first_index_grace_defaults(self) -> None:
        """An unstated grace period keeps the number the template has always used."""
        resolved = resolve_qmd_service_config({"services": {"qmd": {}}})
        assert resolved is not None
        assert resolved.first_index_grace_seconds == DEFAULT_FIRST_INDEX_GRACE_SECONDS

    def test_first_index_grace_is_read(self) -> None:
        """A facility whose corpus takes longer than an hour to index says so."""
        resolved = resolve_qmd_service_config({"services": {"qmd": {"first_index_grace": 14400}}})
        assert resolved is not None
        assert resolved.first_index_grace_seconds == 14400


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

    @pytest.mark.parametrize("bad", [0, -30, "3600", 3600.0, True])
    def test_bad_first_index_grace_raises(self, bad: object) -> None:
        """A zero or malformed grace period would fail the container on first boot."""
        with pytest.raises(ValueError, match=r"services\.qmd\.first_index_grace"):
            resolve_qmd_service_config({"services": {"qmd": {"first_index_grace": bad}}})


class TestBaseUrl:
    """Clients dial the address the publish interface is actually reached on."""

    def test_uses_configured_port(self) -> None:
        assert QMDServiceConfig(port=8180).base_url == "http://127.0.0.1:8180"

    def test_wildcard_publish_still_dials_loopback(self) -> None:
        assert QMDServiceConfig(port=8180, bind_address="0.0.0.0").base_url == (
            "http://127.0.0.1:8180"
        )

    def test_pinned_interface_is_dialed_there_not_on_loopback(self) -> None:
        """A concrete bind publishes on that interface only, so dial it."""
        assert QMDServiceConfig(port=8180, bind_address="10.0.0.7").base_url == (
            "http://10.0.0.7:8180"
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

    def _unreachable(_config: object) -> tuple[bool, str]:
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


class TestCorpora:
    """One sidecar per corpus: okf and ariel at fixed offsets, declared ones after."""

    def test_okf_and_ariel_sit_at_fixed_offsets_whatever_the_render_configures(self) -> None:
        # A render that knows only one of the two still dials the port the
        # deployment published it on.
        resolved = resolve_qmd_service_config({"services": {"qmd": {"port": 9000}}})
        assert resolved.for_corpus("okf").port == 9000
        assert resolved.for_corpus("ariel").port == 9001

    def test_declared_corpora_follow_in_list_order(self) -> None:
        resolved = resolve_qmd_service_config(
            {
                "services": {
                    "qmd": {
                        "port": 9000,
                        "corpora": [
                            {"name": "papers", "index": "prebuilt", "index_dir": "/i/papers"},
                            {"name": "ascc", "source": "./data/ascc"},
                        ],
                    }
                }
            }
        )
        assert resolved.corpora == (
            DeclaredCorpus("papers", INDEX_PREBUILT, None, "/i/papers"),
            DeclaredCorpus("ascc", INDEX_MANAGED, "./data/ascc", None),
        )
        assert resolved.for_corpus("papers").port == 9002
        assert resolved.for_corpus("ascc").port == 9003
        assert resolved.for_corpus("ascc").base_url == "http://127.0.0.1:9003"

    def test_resolve_corpus_config_without_a_block_is_none(self) -> None:
        assert resolve_qmd_corpus_config({}, "okf") is None

    def test_an_unknown_corpus_is_refused_not_guessed(self) -> None:
        with pytest.raises(ValueError, match="no qmd corpus named 'papers'"):
            resolve_qmd_corpus_config({"services": {"qmd": {}}}, "papers")

    @pytest.mark.parametrize(
        ("corpora", "match"),
        [
            ("papers", "must be a list"),
            ([{"source": "./x"}], "name must be"),
            ([{"name": "Papers", "source": "./x"}], "name must be"),
            ([{"name": "okf", "source": "./x"}], "already taken"),
            ([{"name": "a", "source": "./x"}, {"name": "a", "source": "./y"}], "already taken"),
            ([{"name": "a"}], "needs a `source`"),
            ([{"name": "a", "index": "prebuilt"}], "needs the `index_dir`"),
            ([{"name": "a", "source": "./x", "index_dir": "/i"}], "corpus is managed"),
            ([{"name": "a", "index": "remote", "source": "./x"}], "index must be"),
            ([{"name": "a", "source": "./x", "catalogue": "t"}], "unknown key"),
            ([{"name": f"c{i}", "source": "./x"} for i in range(9)], "room for 8"),
        ],
    )
    def test_malformed_corpora_are_refused(self, corpora, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            resolve_qmd_service_config({"services": {"qmd": {"corpora": corpora}}})

    def test_the_family_fits_below_the_next_slot(self) -> None:
        # okf + ariel + the most declared corpora fill the qmd band exactly.
        assert MAX_DECLARED_CORPORA + 2 == QMD_CORPUS_MAX + 1
        assert default_port("qmd", QMD_CORPUS_MAX) < default_port("tiled")


class TestCorporaPreflight:
    """Declared corpora must have something on the host before the build."""

    def _config(self, **corpus) -> dict:
        return {"services": {"qmd": {"corpora": [{"name": "papers", **corpus}]}}}

    def test_no_declared_corpora_is_a_no_op(self, tmp_path: Path) -> None:
        preflight_qmd_corpora({"services": {"qmd": {}}}, tmp_path)
        preflight_qmd_corpora({}, tmp_path)

    def test_a_missing_managed_source_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DeploymentPreconditionError, match="not a directory"):
            preflight_qmd_corpora(self._config(source="data/papers"), tmp_path)

    def test_a_present_managed_source_passes(self, tmp_path: Path) -> None:
        (tmp_path / "data" / "papers").mkdir(parents=True)
        preflight_qmd_corpora(self._config(source="data/papers"), tmp_path)

    def test_a_prebuilt_dir_without_an_index_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "idx").mkdir()
        with pytest.raises(DeploymentPreconditionError, match="holds no qmd index"):
            preflight_qmd_corpora(
                self._config(index="prebuilt", index_dir=str(tmp_path / "idx")), tmp_path
            )

    def test_a_prebuilt_dir_with_an_index_passes(self, tmp_path: Path) -> None:
        (tmp_path / "idx" / ".qmd").mkdir(parents=True)
        (tmp_path / "idx" / ".qmd" / "index.sqlite").write_bytes(b"x")
        preflight_qmd_corpora(self._config(index="prebuilt", index_dir="idx"), tmp_path)


class TestInNetworkDial:
    """A client inside the compose network dials the sidecar by its service name.

    The ``services.qmd`` block only knows where the sidecar is PUBLISHED, which
    from inside a bridge-networked container is that container's own loopback.
    The render hands such a container the in-network URL under
    :func:`corpus_url_env`, and the corpus resolver honours it.
    """

    BLOCK = {"services": {"qmd": {"port": 9000, "corpora": [{"name": "papers", "source": "./p"}]}}}

    def test_service_name_and_env_name_derive_from_the_corpus(self) -> None:
        assert corpus_service_name("ariel") == "qmd-ariel"
        assert corpus_url_env("ariel") == "OSPREY_QMD_ARIEL_URL"
        assert corpus_url_env("site_docs") == "OSPREY_QMD_SITE_DOCS_URL"

    def test_without_an_override_the_host_address_is_dialled(self) -> None:
        resolved = resolve_qmd_corpus_config(self.BLOCK, "ariel", env={})
        assert resolved is not None
        assert resolved.dial_url is None
        assert resolved.base_url == "http://127.0.0.1:9001"

    def test_the_override_names_the_url_dialled(self) -> None:
        env = {"OSPREY_QMD_ARIEL_URL": "http://qmd-ariel:9001"}
        resolved = resolve_qmd_corpus_config(self.BLOCK, "ariel", env=env)
        assert resolved is not None
        assert resolved.base_url == "http://qmd-ariel:9001"
        # The published port is still the deployment's fact; only the dial moves.
        assert resolved.port == 9001

    def test_the_override_is_per_corpus(self) -> None:
        env = {"OSPREY_QMD_ARIEL_URL": "http://qmd-ariel:9001"}
        assert resolve_qmd_corpus_config(self.BLOCK, "okf", env=env).base_url == (
            "http://127.0.0.1:9000"
        )
        assert resolve_qmd_corpus_config(self.BLOCK, "papers", env=env).base_url == (
            "http://127.0.0.1:9002"
        )

    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_empty_override_is_no_override(self, value: str) -> None:
        resolved = resolve_qmd_corpus_config(
            self.BLOCK, "ariel", env={"OSPREY_QMD_ARIEL_URL": value}
        )
        assert resolved.base_url == "http://127.0.0.1:9001"

    def test_a_trailing_slash_is_dropped(self) -> None:
        env = {"OSPREY_QMD_ARIEL_URL": "http://qmd-ariel:9001/"}
        assert resolve_qmd_corpus_config(self.BLOCK, "ariel", env=env).base_url == (
            "http://qmd-ariel:9001"
        )

    def test_the_process_environment_is_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OSPREY_QMD_OKF_URL", "http://qmd-okf:9000")
        assert resolve_qmd_corpus_config(self.BLOCK, "okf").base_url == "http://qmd-okf:9000"

    def test_an_override_without_a_block_still_resolves_to_none(self) -> None:
        # No block means no sidecar this deployment knows; a stray variable does
        # not conjure one.
        assert (
            resolve_qmd_corpus_config({}, "okf", env={"OSPREY_QMD_OKF_URL": "http://x:1"}) is None
        )

    def test_another_corpus_never_inherits_a_dial_url(self) -> None:
        env = {"OSPREY_QMD_ARIEL_URL": "http://qmd-ariel:9001"}
        ariel = resolve_qmd_corpus_config(self.BLOCK, "ariel", env=env)
        assert ariel.dial_url == "http://qmd-ariel:9001"
        assert ariel.for_corpus("okf").dial_url is None

    def test_the_render_hands_each_sidecar_its_in_network_url(self, tmp_path: Path) -> None:
        from osprey.deployment.compose_generator import _resolve_qmd_render_context

        config = {
            **self.BLOCK,
            "facility_knowledge": {"bundle_path": "./okf"},
        }
        context = _resolve_qmd_render_context(config, str(tmp_path))
        assert [c["service"] for c in context["corpora"]] == [
            corpus_service_name("okf"),
            corpus_service_name("papers"),
        ]
        assert context["network_env"] == {
            "OSPREY_QMD_OKF_URL": "http://qmd-okf:9000",
            "OSPREY_QMD_PAPERS_URL": "http://qmd-papers:9002",
        }

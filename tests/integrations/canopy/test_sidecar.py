"""The CANOPY sidecar without CANOPY: what it renders, writes, starts and ends.

The two children here are a script that records what it was started with —
argv, environment, working directory — and then behaves as told: stay, leave
with a status, or leave with 75 (or the status the move file holds) when asked to. Everything the sidecar decides
is visible that way, and nothing needs CANOPY installed. The real servers are
``test_sidecar_smoke.py``'s.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import os
import socket
import stat
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

import osprey.integrations.canopy as canopy_package
from osprey.integrations.canopy import (
    BACKEND_PATH,
    CHANNELS_FILE,
    COLLECTIONS_SUBPATH,
    CONDITIONAL_DATA_CANOPY_VARS,
    CONDITIONAL_FRONT_CANOPY_VARS,
    EXPECTED_DATA_CANOPY_VARS,
    EXPECTED_FRONT_CANOPY_VARS,
    FACILITY_COLLECTIONS_SUBPATH,
    HOST_STYLESHEETS,
    HOST_STYLESHEETS_ENV,
    LIBRARY_SUBPATH,
    MODULES,
    MOLE_ENV_PREFIX,
    PERSONAL_SUBPATH,
    PUSH_SUBPATH,
)
from osprey.integrations.canopy import sidecar as sidecar_module
from osprey.integrations.canopy.sidecar import (
    CanopySidecar,
    mole_block,
    rendered_for,
    scrubbed_environment,
    write_time_budget,
)
from osprey.interfaces.web_terminal.supervised_child import PREVIOUS_EXIT_ENV
from osprey.runtime.budget import write_budget
from osprey_connectors import control_context, posture_store, types
from osprey_connectors.identity import AUDIT_IDENTITY_ENV

#: How long a stub child gets to show up, or to be gone.
WAIT_S = 20.0

STUB = textwrap.dedent(
    """
    import json, os, sys, time
    dump, role, status = sys.argv[1], sys.argv[2], sys.argv[4]
    record = {"argv": sys.argv, "env": dict(os.environ), "cwd": os.getcwd(), "pid": os.getpid()}
    path = os.path.join(dump, f"{role}-{os.getpid()}.json")
    with open(path + ".part", "w") as handle:
        json.dump(record, handle)
    os.rename(path + ".part", path)
    if status != "stay":
        sys.exit(int(status))
    move = os.path.join(dump, f"{role}.move")
    while True:
        if os.path.exists(move):
            with open(move) as handle:
                code = handle.read().strip() or "75"
            os.unlink(move)
            sys.exit(int(code))
        time.sleep(0.02)
    """
)

MONGODB_BLOCK = {
    "type": types.MONGODB_ARCHIVER,
    "settings": {
        "host": "store.example.org",
        "port": 27017,  # osprey:not-a-port
        "name": "history",
        "collection": "samples",
        "auth": {"username": "reader", "source": "admin", "password_env": "STORE_PASSWORD"},
    },
}


def write_record(state_dir: Path, *, target: str, generation: int) -> None:
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation, posture={}),
        path=state_dir / control_context.RECORD_FILENAME,
    )
    control_context.invalidate_cache()


def wait_for(predicate, what: str, timeout: float = WAIT_S):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    pytest.fail(f"{what} did not happen within {timeout} s")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def refuses(url: str) -> bool:
    """Whether nothing listens at *url*'s port any more."""
    port = int(url.rsplit(":", 1)[1])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(2)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def relevant(env: dict[str, str], role: str) -> set[str]:
    """The names the sidecar answers for in a child's environment."""
    keep = {name for name in env if name.startswith(("CANOPY_", "OSPREY_CANOPY_"))}
    if role == "data":
        keep |= {name for name in env if name in ("CONFIG_FILE", "OSPREY_CONFIG")}
    return {name for name in keep if not name.startswith(MOLE_ENV_PREFIX)}


class Deployment:
    """A config, a record directory and a payload in ``tmp_path``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config_path = root / "config.yml"
        self.state_dir = root / "control_target"
        self.state_dir.mkdir()
        self.payload = root / "canopy_payload"
        (self.payload / "viewer").mkdir(parents=True)
        (self.payload / "viewer" / "index.html").write_text("<html></html>", encoding="utf-8")
        (self.payload / "viewer" / "ir-map.json").write_text("{}", encoding="utf-8")
        self.dump = root / "dump"
        self.dump.mkdir()
        self.stub = root / "stub.py"
        self.stub.write_text(STUB, encoding="utf-8")
        self.write_config({})
        self.sidecars: list[CanopySidecar] = []

    def write_config(self, config: dict) -> None:
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def sidecar(self, user: str = "alice", *, data: str = "stay", front: str = "stay"):
        deployment = self

        class _Stubbed(CanopySidecar):
            def _front_command(self, fd):
                return [sys.executable, str(deployment.stub), str(deployment.dump), "front",
                        str(fd), front]  # fmt: skip

            def _data_command(self, fd):
                return [sys.executable, str(deployment.stub), str(deployment.dump), "data",
                        str(fd), data]  # fmt: skip

        built = _Stubbed(self.root / "agent-data" / user, "", None, payload_dir=self.payload)
        self.sidecars.append(built)
        return built

    def dumps(self, role: str) -> list[dict]:
        found = sorted(self.dump.glob(f"{role}-*.json"), key=lambda path: path.stat().st_mtime_ns)
        return [json.loads(path.read_text(encoding="utf-8")) for path in found]

    def started(self, role: str, count: int = 1) -> list[dict]:
        return wait_for(
            lambda: (lambda found: found if len(found) >= count else None)(self.dumps(role)),
            f"{count} {role} start(s)",
        )


def write_screen(library, screen_id: str) -> None:
    """A minimal screen file at the library id *screen_id*."""
    target = library / "screens" / f"{screen_id}.screen.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"kind": "screen"}), encoding="utf-8")


def home_config(screen_id) -> dict:
    return {"web": {"panels": {"canopy": {"home_screen": screen_id}}}}


def home_warnings(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "home_screen" in record.getMessage()
    ]


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("CANOPY_", "OSPREY_CANOPY_", "OSPREY_CONTROL_TARGET")):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    built = Deployment(tmp_path)
    monkeypatch.setenv("OSPREY_CONFIG", str(built.config_path))
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(built.state_dir))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.chdir(tmp_path)
    control_context.invalidate_cache()
    yield built
    for sidecar in built.sidecars:
        sidecar.stop()
    control_context.invalidate_cache()


# -- pure rendering ----------------------------------------------------------------


class TestScrubbedEnvironment:
    def test_inherited_canopy_names_and_terminal_credentials_are_gone(self):
        env = scrubbed_environment(
            {
                "PATH": "/bin",
                "CANOPY_API_KEYS": "k:read",
                "CANOPY_BASE_URL": "http://elsewhere",
                "OSPREY_CANOPY_WRITES": "1",
                "OSPREY_CANOPY_PAYLOAD": "/payload",
                "OSPREY_TERMINAL_SECRET": "s",
                "OSPREY_PANEL_TOKEN": "t",
                "STORE_PASSWORD": "p",
                AUDIT_IDENTITY_ENV: "alice",
            }
        )
        assert env == {"PATH": "/bin", "STORE_PASSWORD": "p", AUDIT_IDENTITY_ENV: "alice"}

    def test_a_forbidden_session_variable_is_passed_through(self):
        """DATA must see it and refuse; the sidecar never hides it."""
        names = (
            posture_store.CONTROL_OWNER_ENV_VAR,
            posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
            posture_store.LAUNCH_POSTURE_ENV_VAR,
            "OSPREY_CONTROL_TARGET",
        )
        env = scrubbed_environment(dict.fromkeys(names, "x"))
        assert set(env) == set(names)


class TestRenderedFor:
    def test_it_spells_the_record_as_data_compares_it(self):
        record = control_context.ControlContext(target="va", generation=3, posture={})
        assert rendered_for(record) == "va@3"
        assert rendered_for(None) == "none"

    def test_the_backend_reads_the_names_this_module_renders(self):
        # The backend imports CANOPY's models and this module runs without
        # CANOPY, so the backend is read from its source rather than its
        # namespace: it takes the names from the package, as this module does,
        # and defines the class the data process loads.
        spec = importlib.util.find_spec("osprey.integrations.canopy.backend")
        assert spec is not None and spec.origin
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))

        imported = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "osprey.integrations.canopy"
            for alias in node.names
        }
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        shared = {"RENDERED_FOR_ENV", "NO_RECORD", "REACH_SINCE_ENV", "STATEMENT_FILENAME"}
        assert shared <= imported
        assert all(
            getattr(sidecar_module, name) == getattr(canopy_package, name) for name in shared
        )
        module, _, class_name = BACKEND_PATH.partition(":")
        assert module == spec.name
        assert class_name in classes


def _writes(value):
    return {"control_system": {"writes": value}}


class TestWriteTimeBudget:
    """The sidecar's budget is :func:`osprey.runtime.write_budget`'s, whatever the config."""

    @staticmethod
    def config(**blocks):
        return {"control_system": {"connector": blocks}}

    @pytest.mark.parametrize(
        "blocks",
        [
            pytest.param({}, id="defaults"),
            pytest.param(
                {
                    "epics": {"timeout_s": 3, "step_read_timeout_s": 4},
                    "virtual_accelerator": {"timeout_s": 8},
                },
                id="maximum-over-blocks",
            ),  # fmt: skip
            pytest.param({"epics": {"timeout_s": 60, "step_read_timeout_s": 4}}, id="clamped"),
            pytest.param({"epics": {"step_read_timeout_s": 90}}, id="step-read-capped"),
            pytest.param({"epics": {"timeout": 30}, "tango": "nonsense"}, id="unreadable"),
        ],
    )
    def test_it_delegates_to_the_runtime_budget(self, blocks):
        config = self.config(**blocks)
        assert write_time_budget(config) == write_budget(config)
        assert write_time_budget(config).grace_s == write_budget(config).grace_s

    def test_the_defaults_give_a_fourteen_second_grace_and_a_fifteen_second_relay(self):
        budget = write_time_budget({})
        assert (budget.T, budget.S, budget.D) == (5.0, 2.0, 12.0)
        assert (budget.grace_s, budget.relay_timeout_s) == (14.0, 15.0)

    def test_a_step_read_beyond_the_ceiling_keeps_a_one_second_call_bound(self):
        """The 90 s step read: S is capped at 22 so T keeps 1 s, and D stays 24."""
        budget = write_time_budget(self.config(epics={"step_read_timeout_s": 90}))
        assert (budget.T, budget.S, budget.D) == (1.0, 22.0, 24.0)
        assert (budget.grace_s, budget.relay_timeout_s) == (26.0, 27.0)

    def test_the_budget_text_is_what_the_data_process_parses(self):
        assert sidecar_module.write_budget_text(write_time_budget({})) == "5.0,2.0,12.0"


class TestMoleBlock:
    @pytest.fixture
    def store_archive(self, monkeypatch):
        """The deployment's store, attributed to ``va`` by the table."""
        monkeypatch.setattr(sidecar_module, "archive_targets", lambda config: frozenset({"va"}))
        monkeypatch.delenv("OSPREY_ARCHIVER_MONGODB_HOST", raising=False)
        monkeypatch.delenv("OSPREY_ARCHIVER_MONGODB_PORT", raising=False)

    @pytest.mark.usefixtures("store_archive")
    def test_the_simulators_user_gets_the_simulators_archive(self):
        """alice is on ``va`` and the store holds the virtual accelerator's history."""
        block = mole_block({"archiver": MONGODB_BLOCK}, "va")
        prefix = "CANOPY_MOLE__ARCHIVERS__OSPREY__"
        assert block == {
            "CANOPY_MOLE__DEFAULT_ARCHIVER": "osprey",
            f"{prefix}TYPE": (
                "osprey.integrations.canopy.mongo_recorder:OspreyMongoRecorderAdapter"
            ),
            f"{prefix}URL": "mongodb://reader@store.example.org:27017/history?authSource=admin",
            f"{prefix}EXTRAS__COLLECTION": "samples",
            f"{prefix}API_KEY_ENV": "STORE_PASSWORD",
            f"{prefix}TIMEOUT_S": "10",
        }

    @pytest.mark.usefixtures("store_archive")
    @pytest.mark.parametrize("target", ["standin", "live", None])
    def test_a_user_on_another_target_gets_no_archive(self, target):
        """bob is on ``standin``: the simulator's past is not his to be shown."""
        assert mole_block({"archiver": MONGODB_BLOCK}, target) == {}

    @pytest.mark.usefixtures("store_archive")
    def test_the_in_network_address_wins_over_the_configs(self, monkeypatch):
        monkeypatch.setenv("OSPREY_ARCHIVER_MONGODB_HOST", "archiver-mongodb")
        monkeypatch.setenv("OSPREY_ARCHIVER_MONGODB_PORT", "27017")
        block = mole_block({"archiver": MONGODB_BLOCK}, "va")
        assert "@archiver-mongodb:27017/" in block["CANOPY_MOLE__ARCHIVERS__OSPREY__URL"]

    @pytest.mark.usefixtures("store_archive")
    def test_a_block_missing_its_login_renders_no_archive(self):
        settings = {**MONGODB_BLOCK["settings"], "auth": {"username": "reader"}}
        assert mole_block({"archiver": {**MONGODB_BLOCK, "settings": settings}}, "va") == {}

    def test_an_appliance_is_the_live_machines_and_the_stand_ins_archive(self):
        """The real attribution: the stand-in rehearses the real machine, so a
        user on either is shown the appliance; a user on the simulator is not."""
        config = {
            "archiver": {
                "type": types.EPICS_ARCHIVER,
                "settings": {"url": "https://archiver.example.org"},
            }
        }
        prefix = "CANOPY_MOLE__ARCHIVERS__OSPREY__"
        expected = {
            "CANOPY_MOLE__DEFAULT_ARCHIVER": "osprey",
            f"{prefix}TYPE": "epics_archiver_appliance",
            f"{prefix}URL": "https://archiver.example.org",
            f"{prefix}TIMEOUT_S": "10",
        }
        assert mole_block(config, "live") == expected
        assert mole_block(config, "standin") == expected
        assert mole_block(config, "va") == {}

    def test_a_virtual_accelerators_own_store_is_its_archive_beside_a_stand_in(self):
        """The real attribution: the store is ``va``'s whether or not a stand-in
        was built and recorded, and whatever machine the sessions start on —
        the recorder samples the simulator this deployment built."""
        config = {
            "control_system": {"type": types.LIVE_STANDIN},
            "archiver": MONGODB_BLOCK,
            "services": {"virtual_accelerator": {"port": 5064}, "live_standin": {"port": 5074}},
            "deployed_services": ["virtual_accelerator", "live_standin", "archiver_recorder"],
        }
        assert mole_block(config, "va")
        assert mole_block(config, "standin") == {}
        assert mole_block(config, "live") == {}


# -- the reachability a respawn carries --------------------------------------------

#: An epoch-seconds ``since`` a predecessor stated.
PREDECESSOR_SINCE = 1_760_700_060.25


def _leave_statement(sidecar, payload) -> Path:
    """Write DATA's statement file into *sidecar*'s key directory, as DATA does."""
    path = sidecar.key_dir / sidecar_module.STATEMENT_FILENAME
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


def _statement(binding_key="va@3", state="no", since=PREDECESSOR_SINCE):
    return {"binding_key": binding_key, "state": state, "since": since, "written_at": 1.0}


class TestCarriedReach:
    """``OSPREY_CANOPY_REACH_SINCE``: rendered on a respawn whose record identity is unchanged."""

    def test_the_first_spawn_carries_nothing(self, deployment):
        write_record(deployment.state_dir, target="va", generation=3)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar_module.REACH_SINCE_ENV not in sidecar.data_env()

    def test_an_unchanged_record_carries_the_no_and_its_since(self, deployment):
        write_record(deployment.state_dir, target="va", generation=3)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        path = _leave_statement(sidecar, _statement())
        env = sidecar.data_env()
        assert env[sidecar_module.REACH_SINCE_ENV] == f"no,{PREDECESSOR_SINCE!r}"
        assert not path.exists()
        assert float(env[sidecar_module.REACH_SINCE_ENV].partition(",")[2]) == PREDECESSOR_SINCE

    def test_a_carried_yes_has_no_since(self, deployment):
        write_record(deployment.state_dir, target="va", generation=3)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        _leave_statement(sidecar, _statement(state="yes", since=None))
        assert sidecar.data_env()[sidecar_module.REACH_SINCE_ENV] == "yes"

    def test_an_integer_since_is_said_as_seconds(self, deployment):
        write_record(deployment.state_dir, target="va", generation=3)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        _leave_statement(sidecar, _statement(since=1_760_700_060))
        assert sidecar.data_env()[sidecar_module.REACH_SINCE_ENV] == "no,1760700060.0"

    @pytest.mark.parametrize(
        ("target", "generation"),
        [("va", 4), ("live", 3)],
        ids=["new-generation", "other-target"],
    )
    def test_a_changed_record_identity_renders_nothing(self, deployment, target, generation):
        write_record(deployment.state_dir, target=target, generation=generation)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        path = _leave_statement(sidecar, _statement(binding_key="va@3"))
        env = sidecar.data_env()
        assert sidecar_module.REACH_SINCE_ENV not in env
        assert env[sidecar_module.RENDERED_FOR_ENV] == f"{target}@{generation}"
        assert not path.exists(), "a statement for another record is removed, never kept"

    def test_a_statement_is_handed_on_once(self, deployment):
        write_record(deployment.state_dir, target="va", generation=3)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        _leave_statement(sidecar, _statement())
        assert sidecar_module.REACH_SINCE_ENV in sidecar.data_env()
        assert sidecar_module.REACH_SINCE_ENV not in sidecar.data_env()

    @pytest.mark.parametrize(
        "payload",
        [
            "{not json",
            "[]",
            _statement(state="unknown", since=None),
            _statement(since=None),
            _statement(since="1760700060"),
            _statement(since=True),
            '{"binding_key": "va@3", "state": "no", "since": Infinity}',
            {"state": "no", "since": PREDECESSOR_SINCE},
        ],
        ids=[
            "malformed",
            "not-an-object",
            "unknown",
            "no-without-since",
            "string-since",
            "boolean-since",
            "infinite-since",
            "no-binding-key",
        ],
    )
    def test_a_statement_that_says_no_definite_state_renders_nothing(self, deployment, payload):
        write_record(deployment.state_dir, target="va", generation=3)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        path = _leave_statement(sidecar, payload)
        assert sidecar_module.REACH_SINCE_ENV not in sidecar.data_env()
        assert not path.exists()

    def test_the_carried_name_is_one_the_sidecar_answers_for(self, deployment):
        """A respawn's carry is scrubbed from what any child inherits, like every rendered name."""
        assert sidecar_module.REACH_SINCE_ENV.startswith(sidecar_module.STRIPPED_PREFIXES)
        write_record(deployment.state_dir, target="va", generation=3)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        _leave_statement(sidecar, _statement())
        assert sidecar_module.REACH_SINCE_ENV in relevant(sidecar.data_env(), "data")
        assert sidecar_module.REACH_SINCE_ENV not in sidecar.front_env()


# -- the rendered environments -----------------------------------------------------


class TestEnvironments:
    def test_the_rendered_names_are_the_exported_sets(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert relevant(sidecar.data_env(), "data") == (
            EXPECTED_DATA_CANOPY_VARS - {"CANOPY_THEME_FILE"}
        )
        assert relevant(sidecar.front_env(), "front") == (
            EXPECTED_FRONT_CANOPY_VARS - CONDITIONAL_FRONT_CANOPY_VARS
        )

    def test_with_a_theme_a_trace_and_a_library_every_exported_name_is_rendered(self, deployment):
        deployment.write_config(
            {
                "web": {
                    "panels": {"canopy": {"theme_file": "/app/theme.css", "home_screen": "sr/home"}}
                }
            }
        )
        write_screen(deployment.root / LIBRARY_SUBPATH, "sr/home")
        (deployment.root / FACILITY_COLLECTIONS_SUBPATH).mkdir(parents=True)
        (deployment.payload / "trace").mkdir()
        (deployment.payload / "trace" / "index.html").write_text("<html></html>")
        sidecar = deployment.sidecar()
        sidecar.spawn()
        data, front = sidecar.data_env(), sidecar.front_env()
        assert relevant(data, "data") == EXPECTED_DATA_CANOPY_VARS
        assert relevant(front, "front") == EXPECTED_FRONT_CANOPY_VARS
        assert data["CANOPY_THEME_FILE"] == front["CANOPY_THEME_FILE"] == "/app/theme.css"
        assert front["CANOPY_TRACE_DIST"] == str(deployment.payload / "trace")

    def test_a_relative_theme_file_resolves_against_the_project_root(self, deployment):
        deployment.write_config({"web": {"panels": {"canopy": {"theme_file": "themes/als.css"}}}})
        sidecar = deployment.sidecar()
        sidecar.spawn()
        expected = str(deployment.root / "themes" / "als.css")
        assert sidecar.data_env()["CANOPY_THEME_FILE"] == expected
        assert sidecar.front_env()["CANOPY_THEME_FILE"] == expected

    def test_a_relative_theme_file_under_a_build_render_resolves_above_build(
        self, deployment, monkeypatch
    ):
        render = deployment.root / "build"
        render.mkdir()
        config = render / "config.yml"
        config.write_text(
            yaml.safe_dump({"web": {"panels": {"canopy": {"theme_file": "als.css"}}}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(config))
        sidecar = deployment.sidecar()
        sidecar.spawn()
        expected = str(deployment.root / "als.css")
        assert sidecar.data_env()["CANOPY_THEME_FILE"] == expected
        assert sidecar.front_env()["CANOPY_THEME_FILE"] == expected

    def test_an_absolute_theme_file_is_passed_unchanged(self, deployment, tmp_path):
        absolute = str(tmp_path / "elsewhere" / "theme.css")
        deployment.write_config({"web": {"panels": {"canopy": {"theme_file": absolute}}}})
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.data_env()["CANOPY_THEME_FILE"] == absolute
        assert sidecar.front_env()["CANOPY_THEME_FILE"] == absolute

    def test_the_library_is_handed_over_when_it_exists(self, deployment):
        library = deployment.root / LIBRARY_SUBPATH
        library.mkdir(parents=True)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.front_env()["CANOPY_LIBRARY_DIR"] == str(library)

    def test_without_a_library_no_library_dir_is_rendered(self, deployment, caplog):
        caplog.set_level(logging.INFO, logger=sidecar_module.__name__)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        caplog.clear()
        assert "CANOPY_LIBRARY_DIR" not in sidecar.front_env()
        reasons = [record for record in caplog.records if "screen library" in record.getMessage()]
        assert len(reasons) == 1
        assert reasons[0].levelno == logging.INFO
        assert str(deployment.root / LIBRARY_SUBPATH) in reasons[0].getMessage()

    def test_a_library_that_is_a_file_is_not_a_library(self, deployment):
        library = deployment.root / LIBRARY_SUBPATH
        library.parent.mkdir(parents=True)
        library.write_text("", encoding="utf-8")
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert "CANOPY_LIBRARY_DIR" not in sidecar.front_env()

    def test_without_a_config_no_library_dir_is_rendered(self, deployment, monkeypatch, caplog):
        (deployment.root / LIBRARY_SUBPATH).mkdir(parents=True)
        monkeypatch.delenv("OSPREY_CONFIG")
        caplog.set_level(logging.INFO, logger=sidecar_module.__name__)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        caplog.clear()
        assert "CANOPY_LIBRARY_DIR" not in sidecar.front_env()
        reasons = [
            record
            for record in caplog.records
            if "screen library" in record.getMessage() and "OSPREY_CONFIG" in record.getMessage()
        ]
        assert len(reasons) == 1
        assert reasons[0].levelno == logging.INFO

    def test_the_library_under_a_build_render_resolves_above_build(self, deployment, monkeypatch):
        library = deployment.root / LIBRARY_SUBPATH
        library.mkdir(parents=True)
        render = deployment.root / "build"
        render.mkdir()
        (render / LIBRARY_SUBPATH).mkdir(parents=True)
        config = render / "config.yml"
        config.write_text(yaml.safe_dump({}), encoding="utf-8")
        monkeypatch.setenv("OSPREY_CONFIG", str(config))
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.front_env()["CANOPY_LIBRARY_DIR"] == str(library)

    def test_the_library_is_looked_for_at_each_render(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert "CANOPY_LIBRARY_DIR" not in sidecar.front_env()
        library = deployment.root / LIBRARY_SUBPATH
        library.mkdir(parents=True)
        assert sidecar.front_env()["CANOPY_LIBRARY_DIR"] == str(library)

    def test_the_home_screen_is_the_configured_screen_the_library_holds(self, deployment):
        write_screen(deployment.root / LIBRARY_SUBPATH, "facility/overview")
        deployment.write_config(home_config("facility/overview"))
        sidecar = deployment.sidecar()
        sidecar.spawn()
        front = sidecar.front_env()
        assert front["CANOPY_HOME_SCREEN"] == "facility/overview"
        assert front["CANOPY_LIBRARY_DIR"] == str(deployment.root / LIBRARY_SUBPATH)

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({}, id="unset"),
            pytest.param(home_config(""), id="empty"),
            pytest.param(home_config(None), id="null"),
        ],
    )
    def test_without_the_key_no_home_screen_is_rendered_and_nothing_is_warned(
        self, deployment, config, caplog
    ):
        write_screen(deployment.root / LIBRARY_SUBPATH, "facility/overview")
        deployment.write_config(config)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        caplog.set_level(logging.WARNING, logger=sidecar_module.__name__)
        caplog.clear()
        front = sidecar.front_env()
        assert "CANOPY_LIBRARY_DIR" in front
        assert "CANOPY_HOME_SCREEN" not in front
        assert not [r for r in caplog.records if "home_screen" in r.getMessage()]

    def test_a_configured_screen_the_library_lacks_is_warned_once_and_not_rendered(
        self, deployment, caplog
    ):
        library = deployment.root / LIBRARY_SUBPATH
        write_screen(library, "sr/orbit-feedback")
        deployment.write_config(home_config("facility/overview"))
        sidecar = deployment.sidecar()
        sidecar.spawn()
        caplog.set_level(logging.WARNING, logger=sidecar_module.__name__)
        caplog.clear()
        front = sidecar.front_env()
        assert "CANOPY_HOME_SCREEN" not in front
        warnings = home_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "web.panels.canopy.home_screen" in message
        assert str(library / "screens" / "facility" / "overview.screen.json") in message

    def test_a_configured_file_that_is_not_a_screen_is_not_rendered(self, deployment, caplog):
        library = deployment.root / LIBRARY_SUBPATH
        target = library / "screens" / "sr" / "home.screen.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}", encoding="utf-8")
        deployment.write_config(home_config("sr/home"))
        sidecar = deployment.sidecar()
        sidecar.spawn()
        caplog.set_level(logging.WARNING, logger=sidecar_module.__name__)
        caplog.clear()
        assert "CANOPY_HOME_SCREEN" not in sidecar.front_env()
        warnings = home_warnings(caplog)
        assert len(warnings) == 1
        assert str(target) in warnings[0].getMessage()

    @pytest.mark.parametrize(
        "screen_id",
        [
            pytest.param("library/sr/home", id="library-prefix"),
            pytest.param("sr/home.screen.json", id="screen-suffix"),
        ],
    )
    def test_a_misspelled_id_is_not_rendered_even_when_the_screen_exists(
        self, deployment, screen_id, caplog
    ):
        write_screen(deployment.root / LIBRARY_SUBPATH, "sr/home")
        deployment.write_config(home_config(screen_id))
        sidecar = deployment.sidecar()
        sidecar.spawn()
        caplog.set_level(logging.WARNING, logger=sidecar_module.__name__)
        caplog.clear()
        assert "CANOPY_HOME_SCREEN" not in sidecar.front_env()
        assert len(home_warnings(caplog)) == 1

    def test_a_configured_screen_without_a_library_is_warned_and_not_rendered(
        self, deployment, caplog
    ):
        deployment.write_config(home_config("facility/overview"))
        sidecar = deployment.sidecar()
        sidecar.spawn()
        caplog.set_level(logging.WARNING, logger=sidecar_module.__name__)
        caplog.clear()
        front = sidecar.front_env()
        assert "CANOPY_LIBRARY_DIR" not in front
        assert "CANOPY_HOME_SCREEN" not in front
        warnings = home_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "web.panels.canopy.home_screen" in message
        assert "facility/overview.screen.json" in message

    def test_without_a_config_no_home_screen_is_rendered(self, deployment, monkeypatch):
        write_screen(deployment.root / LIBRARY_SUBPATH, "facility/overview")
        monkeypatch.delenv("OSPREY_CONFIG")
        sidecar = deployment.sidecar()
        sidecar.spawn()
        front = sidecar.front_env()
        assert "CANOPY_LIBRARY_DIR" not in front
        assert "CANOPY_HOME_SCREEN" not in front

    def test_the_home_screen_is_looked_for_at_each_render(self, deployment):
        library = deployment.root / LIBRARY_SUBPATH
        library.mkdir(parents=True)
        deployment.write_config(home_config("facility/overview"))
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert "CANOPY_HOME_SCREEN" not in sidecar.front_env()
        write_screen(library, "facility/overview")
        assert sidecar.front_env()["CANOPY_HOME_SCREEN"] == "facility/overview"

    def test_the_personal_collections_are_always_handed_over(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        personal = deployment.root / "agent-data" / "alice" / "canopy" / "collections"
        assert sidecar.collections_dir == personal
        assert sidecar.front_env()["CANOPY_COLLECTIONS_DIR"] == str(personal)

    def test_the_personal_collections_are_rendered_without_a_config(self, deployment, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG")
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.front_env()["CANOPY_COLLECTIONS_DIR"] == str(sidecar.collections_dir)

    def test_the_facility_collections_are_handed_over_when_they_exist(self, deployment):
        store = deployment.root / FACILITY_COLLECTIONS_SUBPATH
        store.mkdir(parents=True)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.front_env()["CANOPY_FACILITY_COLLECTIONS_DIR"] == str(store)

    def test_without_facility_collections_no_facility_dir_is_rendered(self, deployment, caplog):
        caplog.set_level(logging.INFO, logger=sidecar_module.__name__)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        caplog.clear()
        assert "CANOPY_FACILITY_COLLECTIONS_DIR" not in sidecar.front_env()
        reasons = [
            record for record in caplog.records if "facility collections" in record.getMessage()
        ]
        assert len(reasons) == 1
        assert reasons[0].levelno == logging.INFO
        assert str(deployment.root / FACILITY_COLLECTIONS_SUBPATH) in reasons[0].getMessage()

    def test_facility_collections_that_are_a_file_are_not_a_store(self, deployment):
        store = deployment.root / FACILITY_COLLECTIONS_SUBPATH
        store.parent.mkdir(parents=True)
        store.write_text("", encoding="utf-8")
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert "CANOPY_FACILITY_COLLECTIONS_DIR" not in sidecar.front_env()

    def test_without_a_config_no_facility_collections_dir_is_rendered(
        self, deployment, monkeypatch, caplog
    ):
        (deployment.root / FACILITY_COLLECTIONS_SUBPATH).mkdir(parents=True)
        monkeypatch.delenv("OSPREY_CONFIG")
        caplog.set_level(logging.INFO, logger=sidecar_module.__name__)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        caplog.clear()
        assert "CANOPY_FACILITY_COLLECTIONS_DIR" not in sidecar.front_env()
        reasons = [
            record
            for record in caplog.records
            if "facility collections" in record.getMessage()
            and "OSPREY_CONFIG" in record.getMessage()
        ]
        assert len(reasons) == 1
        assert reasons[0].levelno == logging.INFO

    def test_the_facility_collections_under_a_build_render_resolve_above_build(
        self, deployment, monkeypatch
    ):
        store = deployment.root / FACILITY_COLLECTIONS_SUBPATH
        store.mkdir(parents=True)
        render = deployment.root / "build"
        render.mkdir()
        (render / FACILITY_COLLECTIONS_SUBPATH).mkdir(parents=True)
        config = render / "config.yml"
        config.write_text(yaml.safe_dump({}), encoding="utf-8")
        monkeypatch.setenv("OSPREY_CONFIG", str(config))
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.front_env()["CANOPY_FACILITY_COLLECTIONS_DIR"] == str(store)

    def test_the_facility_collections_are_looked_for_at_each_render(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert "CANOPY_FACILITY_COLLECTIONS_DIR" not in sidecar.front_env()
        store = deployment.root / FACILITY_COLLECTIONS_SUBPATH
        store.mkdir(parents=True)
        assert sidecar.front_env()["CANOPY_FACILITY_COLLECTIONS_DIR"] == str(store)

    def test_the_personal_and_facility_collections_are_distinct(self, deployment):
        (deployment.root / FACILITY_COLLECTIONS_SUBPATH).mkdir(parents=True)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        front = sidecar.front_env()
        assert front["CANOPY_COLLECTIONS_DIR"] != front["CANOPY_FACILITY_COLLECTIONS_DIR"]

    def test_the_waveform_default_query_is_rendered_verbatim(self, deployment):
        deployment.write_config(
            {
                "web": {
                    "panels": {
                        "canopy": {"waveform_default_query": "pv=SR:MAG:SEPTUM:01:CURRENT:WAVEFORM"}
                    }
                }
            }
        )
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert (
            sidecar.front_env()["CANOPY_WAVEFORM_DEFAULT_QUERY"]
            == "pv=SR:MAG:SEPTUM:01:CURRENT:WAVEFORM"
        )
        assert "CANOPY_WAVEFORM_DEFAULT_QUERY" in EXPECTED_FRONT_CANOPY_VARS
        assert "CANOPY_WAVEFORM_DEFAULT_QUERY" not in CONDITIONAL_FRONT_CANOPY_VARS
        assert "CANOPY_WAVEFORM_DEFAULT_QUERY" not in sidecar.data_env()

    def test_the_drafts_dir_is_the_personal_store_with_or_without_a_library(self, deployment):
        """The library is a second root; it never takes the personal store's place."""
        sidecar = deployment.sidecar()
        sidecar.spawn()
        personal = str(deployment.root / "agent-data" / "alice" / "canopy" / "drafts")
        assert sidecar.front_env()["CANOPY_DRAFTS_DIR"] == personal
        library = deployment.root / LIBRARY_SUBPATH
        library.mkdir(parents=True)
        front = sidecar.front_env()
        assert front["CANOPY_DRAFTS_DIR"] == personal
        assert front["CANOPY_DRAFTS_DIR"] != front["CANOPY_LIBRARY_DIR"]

    def test_authoring_is_on_with_or_without_a_library(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.front_env()["CANOPY_AUTHORING"] == "1"
        (deployment.root / LIBRARY_SUBPATH).mkdir(parents=True)
        assert sidecar.front_env()["CANOPY_AUTHORING"] == "1"

    def test_the_workspace_is_served_by_the_per_user_front(self, deployment, monkeypatch):
        """The opt-in is rendered over one inherited from the terminal, and FRONT's alone."""
        monkeypatch.setenv("CANOPY_WORKSPACE", "0")
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.front_env()["CANOPY_WORKSPACE"] == "1"
        assert "CANOPY_WORKSPACE" not in sidecar.data_env()

    def test_the_browser_runs_without_its_own_sandbox(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.front_env()["CANOPY_BROWSER_SANDBOX"] == "0"

    def test_the_per_user_subpaths_are_four_siblings(self):
        assert PERSONAL_SUBPATH == ("canopy", "drafts")
        assert PUSH_SUBPATH == ("canopy", "push")
        assert COLLECTIONS_SUBPATH == ("canopy", "collections")
        subpaths = {
            PERSONAL_SUBPATH,
            PUSH_SUBPATH,
            COLLECTIONS_SUBPATH,
            sidecar_module.VIEWS_SUBPATH,
        }
        assert len(subpaths) == 4
        assert {subpath[0] for subpath in subpaths} == {"canopy"}

    def test_the_library_subpath_is_relative(self):
        assert LIBRARY_SUBPATH == Path("data/canopy/library")
        assert not LIBRARY_SUBPATH.is_absolute()

    def test_the_facility_collections_subpath_is_relative_beside_the_library(self):
        assert FACILITY_COLLECTIONS_SUBPATH == Path("data/canopy/collections")
        assert not FACILITY_COLLECTIONS_SUBPATH.is_absolute()
        assert FACILITY_COLLECTIONS_SUBPATH.parent == LIBRARY_SUBPATH.parent

    def test_the_predecessors_exit_is_in_no_exported_set(self):
        """The supervisor's own variable is not one the sidecar renders or strips."""
        for exported in (
            EXPECTED_DATA_CANOPY_VARS,
            EXPECTED_FRONT_CANOPY_VARS,
            CONDITIONAL_DATA_CANOPY_VARS,
            CONDITIONAL_FRONT_CANOPY_VARS,
        ):
            assert PREVIOUS_EXIT_ENV not in exported

    def test_the_conditional_names_are_within_the_exported_sets(self):
        assert CONDITIONAL_DATA_CANOPY_VARS < EXPECTED_DATA_CANOPY_VARS
        assert CONDITIONAL_FRONT_CANOPY_VARS < EXPECTED_FRONT_CANOPY_VARS

    def test_the_home_screen_is_a_conditional_front_name(self):
        assert "CANOPY_HOME_SCREEN" in EXPECTED_FRONT_CANOPY_VARS
        assert "CANOPY_HOME_SCREEN" in CONDITIONAL_FRONT_CANOPY_VARS
        assert "CANOPY_HOME_SCREEN" not in EXPECTED_DATA_CANOPY_VARS

    def test_the_values_are_the_contracts(self, deployment):
        deployment.write_config(
            {
                "web": {
                    "panels": {
                        "canopy": {
                            "vole_bytes_max": 1024,
                            "trace_default_query": "pv=SR:DIAG:BPM:03:POSITION:X",
                            "correlation_default_query": "x=SR:MAG:HCM:01:CURRENT:SP&dt=1",
                            "waveform_default_query": "pv=SR:MAG:SEPTUM:01:CURRENT:WAVEFORM",
                        }
                    }
                }
            }
        )
        sidecar = deployment.sidecar()
        sidecar.spawn()
        key_dir = sidecar.key_dir
        data, front = sidecar.data_env(), sidecar.front_env()
        config = str(deployment.config_path)
        assert {name: data[name] for name in relevant(data, "data")} == {
            "CONFIG_FILE": config,
            "OSPREY_CONFIG": config,
            "CANOPY_EPICS__BACKEND": BACKEND_PATH,
            "CANOPY_MODULES": "hare,vole,formula-engine,mole,converter",
            "CANOPY_API_KEYS_FILE": str(key_dir / "data.keys"),
            "CANOPY_CLOSED_META": "1",
            "CANOPY_VOLE__GLOBAL_BYTES_MAX": "1024",
            "OSPREY_CANOPY_RENDERED_FOR": "none",
            "OSPREY_CANOPY_WRITE_BUDGET": "5.0,2.0,12.0",
        }
        assert {name: front[name] for name in relevant(front, "front")} == {
            "CANOPY_BASE_URL": sidecar.data_url,
            "CANOPY_SELF_URL": sidecar.url,
            "CANOPY_ACCESS_TOKEN_FILE": str(key_dir / "front.access"),
            "CANOPY_API_KEY_FILE": str(key_dir / "front.read"),
            "CANOPY_DRAFTS_DIR": str(
                deployment.root / "agent-data" / "alice" / "canopy" / "drafts"
            ),
            "CANOPY_PUSH_DIR": str(deployment.root / "agent-data" / "alice" / "canopy" / "push"),
            "CANOPY_COLLECTIONS_DIR": str(
                deployment.root / "agent-data" / "alice" / "canopy" / "collections"
            ),
            "CANOPY_VIEWS_DIR": str(deployment.root / "agent-data" / "alice" / "canopy" / "views"),
            "CANOPY_VIEWER_DIST": str(deployment.payload / "viewer"),
            "CANOPY_URL_PREFIX": "/panel/canopy",
            "CANOPY_PROXY_MARK_HEADERS": "X-Osprey-Terminal-Secret",
            "CANOPY_LIBRARY_IGNORED_NAMES": "claude.local.md dockerfile prefetched-models var",
            "CANOPY_AUTHORING": "1",
            "CANOPY_WORKSPACE": "1",
            "CANOPY_BROWSER_SANDBOX": "0",
            "CANOPY_RELAY_CONNECT_RETRY_S": "3",
            "CANOPY_TRACE_DEFAULT_QUERY": "pv=SR:DIAG:BPM:03:POSITION:X",
            "CANOPY_CORRELATION_DEFAULT_QUERY": "x=SR:MAG:HCM:01:CURRENT:SP&dt=1",
            "CANOPY_WAVEFORM_DEFAULT_QUERY": "pv=SR:MAG:SEPTUM:01:CURRENT:WAVEFORM",
            "CANOPY_CHANNELS_FILE": "/etc/osprey/channels.json",
            "CANOPY_HOST_STYLESHEETS": (
                "design-system/css/tokens.css static/fonts/fonts.css"
                " design-system/css/canopy-host.css"
            ),
            "CANOPY_PUT_RELAY": "1",
            "CANOPY_PUT_RELAY_TIMEOUT_S": "15.0",
            "CANOPY_WRITE_API_KEY_FILE": str(key_dir / "front.write"),
        }
        assert sidecar.url != sidecar.data_url

    def test_the_defaults_apply_where_the_config_says_nothing(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.data_env()["CANOPY_VOLE__GLOBAL_BYTES_MAX"] == "268435456"
        assert sidecar.front_env()["CANOPY_TRACE_DEFAULT_QUERY"] == ""
        assert sidecar.front_env()["CANOPY_CORRELATION_DEFAULT_QUERY"] == ""
        assert sidecar.front_env()["CANOPY_WAVEFORM_DEFAULT_QUERY"] == ""

    def test_the_channel_catalog_path_is_the_one_every_web_container_mounts(self, deployment):
        """FRONT is told the constant path whether or not the build staged a catalog.

        Without the mount the file is simply missing there, and FRONT answers
        ``/channels`` with 404; the render mounts the staged snapshot at exactly
        this path (``tests/deployment/web_terminals/test_channel_catalog_mount.py``).
        """
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert (
            sidecar.front_env()["CANOPY_CHANNELS_FILE"]
            == CHANNELS_FILE
            == ("/etc/osprey/channels.json")
        )

    def test_the_host_stylesheets_are_rendered_in_cascade_order(self, deployment):
        """FRONT is told OSPREY's three sheets unconditionally, tokens first.

        The embedded pages link them in this order after their own, so the
        host sheet's mapping resolves against tokens that are already defined.
        """
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert HOST_STYLESHEETS_ENV == "CANOPY_HOST_STYLESHEETS"
        assert (
            sidecar.front_env()[HOST_STYLESHEETS_ENV]
            == " ".join(HOST_STYLESHEETS)
            == (
                "design-system/css/tokens.css static/fonts/fonts.css"
                " design-system/css/canopy-host.css"
            )
        )

    def test_no_widget_directory_is_named(self, deployment):
        """The catalog is the viewer's ir-map.json; FRONT is given no widgets directory."""
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert "CANOPY_WIDGETS_DIR" not in sidecar.front_env()
        assert "CANOPY_WIDGETS_DIR" not in EXPECTED_FRONT_CANOPY_VARS

    def test_nothing_about_writes_is_rendered_at_any_tier(self, deployment):
        """DATA judges every write on the verdict it reads; no switch reaches either child."""
        sidecar = deployment.sidecar()
        sidecar.spawn()
        for config in ({}, _writes("allowed"), _writes("never")):
            deployment.write_config(config)
            data, front = sidecar.data_env(), sidecar.front_env()
            assert not [name for name in data if name.startswith("OSPREY_CANOPY_WRITES")]
            assert "CANOPY_CONFIRM_WRITES" not in front

    def test_seeded_canopy_variables_do_not_reach_a_child(self, deployment, monkeypatch):
        monkeypatch.setenv("CANOPY_API_KEYS", "leaked:read+write")
        monkeypatch.setenv("CANOPY_API_KEY", "leaked")
        monkeypatch.setenv("CANOPY_BASE_URL", "http://elsewhere:1")
        monkeypatch.setenv("CANOPY_MOLE__DEFAULT_ARCHIVER", "elsewhere")
        monkeypatch.setenv("OSPREY_CANOPY_WRITES", "1")
        monkeypatch.setenv("OSPREY_TERMINAL_SECRET", "terminal")
        sidecar = deployment.sidecar()
        sidecar.spawn()
        data, front = sidecar.data_env(), sidecar.front_env()
        for env in (data, front):
            assert "CANOPY_API_KEYS" not in env
            assert "CANOPY_API_KEY" not in env
            assert "CANOPY_MOLE__DEFAULT_ARCHIVER" not in env
            assert "OSPREY_TERMINAL_SECRET" not in env
        assert "CANOPY_BASE_URL" not in data
        assert front["CANOPY_BASE_URL"] == sidecar.data_url
        assert "OSPREY_CANOPY_WRITES" not in data
        assert "OSPREY_CANOPY_WRITES" not in front

    def test_a_forbidden_variable_is_passed_through_to_data(self, deployment, monkeypatch):
        monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "someone")
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.data_env()[posture_store.CONTROL_OWNER_ENV_VAR] == "someone"
        (started,) = deployment.started("data")
        assert started["env"][posture_store.CONTROL_OWNER_ENV_VAR] == "someone"

    def test_a_dotenv_in_the_working_directory_changes_nothing(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        before = (sidecar.data_env(), sidecar.front_env())
        (deployment.root / ".env").write_text(
            "CANOPY_MOLE__DEFAULT_ARCHIVER=elsewhere\nCANOPY_API_KEYS=leaked:read+write\n",
            encoding="utf-8",
        )
        assert (sidecar.data_env(), sidecar.front_env()) == before
        assert "CANOPY_MOLE__DEFAULT_ARCHIVER" not in os.environ
        assert "CANOPY_API_KEYS" not in os.environ

    def test_the_mole_block_follows_the_record(self, deployment, monkeypatch):
        """alice on the VA gets its archive; bob on the stand-in gets none."""
        monkeypatch.delenv("OSPREY_ARCHIVER_MONGODB_HOST", raising=False)
        monkeypatch.delenv("OSPREY_ARCHIVER_MONGODB_PORT", raising=False)
        deployment.write_config(
            {"services": {"virtual_accelerator": {"port": 5064}}, "archiver": MONGODB_BLOCK}
        )
        sidecar = deployment.sidecar()
        sidecar.spawn()

        def mole_names():
            return {name for name in sidecar.data_env() if name.startswith(MOLE_ENV_PREFIX)}

        assert mole_names() == set()
        write_record(deployment.state_dir, target="va", generation=1)
        assert "CANOPY_MOLE__ARCHIVERS__OSPREY__URL" in mole_names()
        assert not any(name.startswith(MOLE_ENV_PREFIX) for name in sidecar.front_env())
        write_record(deployment.state_dir, target="standin", generation=2)
        assert mole_names() == set()

    def test_data_is_rendered_from_the_record_and_config_as_they_are_now(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar.data_env()["OSPREY_CANOPY_RENDERED_FOR"] == "none"
        write_record(deployment.state_dir, target="va", generation=4)
        deployment.write_config({"web": {"panels": {"canopy": {"vole_bytes_max": 77}}}})
        env = sidecar.data_env()
        assert env["OSPREY_CANOPY_RENDERED_FOR"] == "va@4"
        assert env["CANOPY_VOLE__GLOBAL_BYTES_MAX"] == "77"

    def test_two_users_views_directories_differ(self, deployment):
        alice, bob = deployment.sidecar("alice"), deployment.sidecar("bob")
        alice.spawn()
        bob.spawn()
        views = [side.front_env()["CANOPY_VIEWS_DIR"] for side in (alice, bob)]
        assert views[0] != views[1]
        assert views[0].endswith("alice/canopy/views") and views[1].endswith("bob/canopy/views")

    def test_two_users_personal_stores_and_push_directories_differ(self, deployment):
        alice, bob = deployment.sidecar("alice"), deployment.sidecar("bob")
        alice.spawn()
        bob.spawn()
        for name, leaf in (("CANOPY_DRAFTS_DIR", "drafts"), ("CANOPY_PUSH_DIR", "push")):
            mine, theirs = (side.front_env()[name] for side in (alice, bob))
            assert mine != theirs
            assert mine.endswith(f"alice/canopy/{leaf}") and theirs.endswith(f"bob/canopy/{leaf}")

    def test_the_write_budget_is_fixed_at_spawn(self, deployment):
        config = {"control_system": {"connector": {"epics": {"timeout_s": 60}}}}
        deployment.write_config(config)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        rendered = sidecar.data_env()["OSPREY_CANOPY_WRITE_BUDGET"]
        deployment.write_config({})
        assert sidecar.budget == write_budget(config)
        assert sidecar.budget.grace_s == write_budget(config).grace_s
        assert sidecar.data_env()["OSPREY_CANOPY_WRITE_BUDGET"] == rendered


# -- key files and children --------------------------------------------------------


class TestKeyFiles:
    def test_the_directory_is_0700_and_its_files_0600(self, deployment):
        """The four secrets, and the sessions file naming the terminal's sessions."""
        sidecar = deployment.sidecar()
        sidecar.spawn()
        key_dir = sidecar.key_dir
        assert stat.S_IMODE(key_dir.stat().st_mode) == 0o700
        files = sorted(path.name for path in key_dir.iterdir())
        assert files == ["data.keys", "front.access", "front.read", "front.write", "sessions.json"]
        for name in files:
            assert stat.S_IMODE((key_dir / name).stat().st_mode) == 0o600

    def test_data_accepts_the_two_keys_front_sends_and_nothing_else(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        key_dir = sidecar.key_dir
        read_key = (key_dir / "front.read").read_text().strip()
        write_key = (key_dir / "front.write").read_text().strip()
        access = (key_dir / "front.access").read_text().strip()
        assert (key_dir / "data.keys").read_text() == (
            f"{read_key}:read@hare+formula-engine+mole+converter\n{write_key}:read+write@hare\n"
        )
        assert len({access, read_key, write_key}) == 3
        assert sidecar.auth_headers == {"X-Canopy-Access": access}
        assert sidecar.token == access

    def test_the_read_key_is_scoped_to_the_modules_front_calls_that_data_mounts(self):
        mounted = set(MODULES.split(","))
        assert (
            set(sidecar_module.FRONT_MODULES) == set(sidecar_module.FRONT_CALLED_MODULES) & mounted
        )
        assert sidecar_module.FRONT_MODULES == ("hare", "formula-engine", "mole", "converter")
        assert "vole" not in sidecar_module.FRONT_MODULES

    def test_no_secret_is_in_a_childs_environment_or_argv(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        key_dir = sidecar.key_dir
        secrets_ = [
            (key_dir / name).read_text().strip()
            for name in ("front.access", "front.read", "front.write")
        ]
        children = deployment.started("data") + deployment.started("front")
        assert len(children) == 2
        for child in children:
            for text in [*child["argv"], *child["env"].values()]:
                assert not any(secret in text for secret in secrets_)
        for env in (sidecar.data_env(), sidecar.front_env()):
            assert not any(secret in value for value in env.values() for secret in secrets_)

    def test_a_child_is_started_with_exactly_the_rendered_environment(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        (data,) = deployment.started("data")
        (front,) = deployment.started("front")
        assert relevant(data["env"], "data") == relevant(sidecar.data_env(), "data")
        assert relevant(front["env"], "front") == relevant(sidecar.front_env(), "front")
        assert front["env"]["CANOPY_BASE_URL"] == sidecar.data_url

    def test_both_children_run_in_the_key_directory(self, deployment):
        """So a ``.env`` beside the terminal is never the children's."""
        (deployment.root / ".env").write_text("CANOPY_MOLE__DEFAULT_ARCHIVER=elsewhere\n")
        sidecar = deployment.sidecar()
        sidecar.spawn()
        key_dir = os.path.realpath(sidecar.key_dir)
        for role in ("data", "front"):
            (child,) = deployment.started(role)
            assert os.path.realpath(child["cwd"]) == key_dir
            assert "CANOPY_MOLE__DEFAULT_ARCHIVER" not in child["env"]
        assert not (sidecar.key_dir / ".env").exists()

    def test_each_child_is_given_its_own_listening_socket(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        (data,) = deployment.started("data")
        (front,) = deployment.started("front")
        assert data["argv"][3] != front["argv"][3]
        assert (data["pid"], front["pid"]) == (sidecar.data_pid, sidecar.front_pid)

    def test_the_real_commands_are_the_contracts(self, deployment):
        sidecar = CanopySidecar(deployment.root, "", None, payload_dir=deployment.payload)
        assert sidecar._front_command(7) == [
            sys.executable, "-m", "canopy_mcp", "--http", "--fd", "7",
        ]  # fmt: skip
        assert sidecar._data_command(8) == [
            sys.executable, "-m", "uvicorn", "canopy.main:app", "--fd", "8",
        ]  # fmt: skip


class TestLifecycle:
    def test_stop_leaves_no_process_no_port_and_no_key_directory(self, deployment):
        fired = []
        sidecar = deployment.sidecar()
        sidecar.spawn()
        sidecar.on_exit = lambda: fired.append(True)
        (data,) = deployment.started("data")
        (front,) = deployment.started("front")
        key_dir, url, data_url = sidecar.key_dir, sidecar.url, sidecar.data_url
        sidecar.stop()
        sidecar.stop()
        assert not alive(data["pid"]) and not alive(front["pid"])
        assert refuses(url) and refuses(data_url)
        assert not key_dir.exists()
        assert sidecar.exit_status is None
        assert fired == []

    def test_a_moved_binding_replaces_data_and_nothing_else(self, deployment):
        write_record(deployment.state_dir, target="va", generation=1)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        (first,) = deployment.started("data")
        (front,) = deployment.started("front")
        url, data_url = sidecar.url, sidecar.data_url
        assert first["env"]["OSPREY_CANOPY_RENDERED_FOR"] == "va@1"

        write_record(deployment.state_dir, target="standin", generation=2)
        (deployment.dump / "data.move").touch()
        _, second = deployment.started("data", 2)
        assert second["pid"] != first["pid"]
        assert second["env"]["OSPREY_CANOPY_RENDERED_FOR"] == "standin@2"
        assert second["argv"][3] == first["argv"][3]
        assert sidecar.front_pid == front["pid"] and alive(front["pid"])
        assert (sidecar.url, sidecar.data_url) == (url, data_url)
        assert len(deployment.dumps("front")) == 1
        assert sidecar.exit_status is None

    def test_data_is_replaced_at_once_after_a_move_or_a_stale_transport(self, deployment):
        """75 and 76 are DATA's respawn-now statuses; FRONT has none."""
        sidecar = deployment.sidecar()
        sidecar.spawn()
        assert sidecar._data._respawn_now == {75, 76}
        assert sidecar._front._respawn_now == frozenset()

    def test_a_stale_transport_replaces_data_which_learns_why(self, deployment):
        """Exit 76: a new DATA, told its predecessor's status; FRONT untouched."""
        write_record(deployment.state_dir, target="va", generation=1)
        sidecar = deployment.sidecar()
        sidecar.spawn()
        (first,) = deployment.started("data")
        (front,) = deployment.started("front")
        assert PREVIOUS_EXIT_ENV not in first["env"]

        part = deployment.dump / "data.move.part"
        part.write_text("76", encoding="utf-8")
        os.replace(part, deployment.dump / "data.move")
        _, second = deployment.started("data", 2)
        assert second["pid"] != first["pid"]
        assert second["env"][PREVIOUS_EXIT_ENV] == "76"
        assert second["env"]["OSPREY_CANOPY_RENDERED_FOR"] == "va@1"
        assert PREVIOUS_EXIT_ENV not in relevant(second["env"], "data")
        assert sidecar.front_pid == front["pid"] and alive(front["pid"])
        assert len(deployment.dumps("front")) == 1
        assert sidecar.exit_status is None

    def test_a_refused_start_ends_the_sidecar_without_a_second_try(self, deployment):
        fired = []
        sidecar = deployment.sidecar(data="78")
        sidecar.spawn()
        # FRONT may be stopped before it has run a line, so its pid is taken
        # from the sidecar rather than from what the child wrote.
        front_pid, url, data_url = sidecar.front_pid, sidecar.url, sidecar.data_url
        sidecar.on_exit = lambda: fired.append(True)
        wait_for(lambda: fired, "the exit report")
        assert fired == [True]
        assert sidecar.exit_status == 78
        assert front_pid is not None and not alive(front_pid)
        assert refuses(url) and refuses(data_url)
        time.sleep(1.0)
        assert len(deployment.dumps("data")) == 1
        assert sidecar.data_starts == 1

    def test_a_refused_start_fails_the_readiness_wait_with_the_status(self, deployment):
        sidecar = deployment.sidecar(data="78")
        sidecar.spawn()
        with pytest.raises(RuntimeError, match="exited with status 78"):
            sidecar.wait_ready(WAIT_S)

    def test_an_exit_callback_registered_after_the_exit_fires_at_once(self, deployment):
        fired = []
        sidecar = deployment.sidecar(data="78")
        sidecar.spawn()
        wait_for(lambda: sidecar.exit_status is not None and refuses(sidecar.url), "the exit")
        sidecar.on_exit = lambda: fired.append(True)
        wait_for(lambda: fired, "the exit report", 5.0)
        sidecar.stop()
        assert fired == [True]

    def test_front_leaving_ends_the_sidecar(self, deployment):
        fired = []
        sidecar = deployment.sidecar(front="3")
        sidecar.spawn()
        data_pid = sidecar.data_pid
        sidecar.on_exit = lambda: fired.append(True)
        wait_for(lambda: fired, "the exit report")
        assert sidecar.exit_status == 3
        assert data_pid is not None and not alive(data_pid)
        assert refuses(sidecar.data_url)
        assert len(deployment.dumps("front")) == 1

    def test_a_stop_during_the_readiness_wait_ends_it(self, deployment):
        import threading

        sidecar = deployment.sidecar()
        sidecar.spawn()
        deployment.started("data")
        threading.Timer(0.3, sidecar.stop).start()
        with pytest.raises(RuntimeError, match="stopped before it was ready"):
            sidecar.wait_ready(WAIT_S)

    def test_spawning_twice_is_refused(self, deployment):
        sidecar = deployment.sidecar()
        sidecar.spawn()
        with pytest.raises(RuntimeError, match="already running"):
            sidecar.spawn()


class TestPreflight:
    def test_a_terminal_without_a_config_cannot_launch(self, deployment, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG")
        with pytest.raises(RuntimeError, match="OSPREY_CONFIG is not set"):
            deployment.sidecar().preflight()

    def test_a_missing_viewer_is_named(self, deployment, monkeypatch):
        monkeypatch.setattr(sidecar_module.importlib.util, "find_spec", lambda name: object())
        (deployment.payload / "viewer" / "index.html").unlink()
        with pytest.raises(RuntimeError, match="no built viewer under"):
            deployment.sidecar().preflight()

    def test_a_viewer_without_its_ir_map_is_named(self, deployment, monkeypatch):
        """The widget catalog is the viewer's ir-map.json: without it there is no panel builder."""
        monkeypatch.setattr(sidecar_module.importlib.util, "find_spec", lambda name: object())
        (deployment.payload / "viewer" / "ir-map.json").unlink()
        with pytest.raises(RuntimeError, match="no widget catalog: .*ir-map.json does not exist"):
            deployment.sidecar().preflight()

    def test_a_payload_without_a_widgets_directory_passes(self, deployment, monkeypatch):
        """The payload has no widgets/ any more; its absence is not a refusal."""
        monkeypatch.setattr(sidecar_module.importlib.util, "find_spec", lambda name: object())
        assert not (deployment.payload / "widgets").exists()
        deployment.sidecar().preflight()

    def test_a_missing_package_is_named(self, deployment, monkeypatch):
        monkeypatch.setattr(sidecar_module.importlib.util, "find_spec", lambda name: None)
        with pytest.raises(RuntimeError, match="has no canopy installed"):
            deployment.sidecar().preflight()

    def test_a_complete_deployment_passes(self, deployment, monkeypatch):
        monkeypatch.setattr(sidecar_module.importlib.util, "find_spec", lambda name: object())
        deployment.sidecar().preflight()

    def test_the_payload_is_found_beside_the_config(self, deployment):
        sidecar = CanopySidecar(deployment.root, "", None)
        assert sidecar.payload_dir == deployment.payload

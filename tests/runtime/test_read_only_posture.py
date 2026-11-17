"""The read-only posture: a connector under it cannot put, and the arming is the one contract."""

import pytest

from osprey.runtime import raw_put_block
from osprey.runtime.read_only_posture import arm_raw_put_block, enter_sandbox_mode
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.posture_env import POSTURE_ENV_VAR, SANDBOX_MODE, is_readonly_run
from osprey_connectors_tests._write_fakes import (
    RecordingConnector,
    arm_switch,
    writes_enabled_config,
)


@pytest.fixture
def writes_enabled_deployment(monkeypatch, tmp_path):
    """A write tier that allows, the login's switch on, and a readwrite run: only the posture refuses."""
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)
    arm_switch(monkeypatch, tmp_path / "armed-data")
    # The variable is restored at teardown, whatever the posture sets it to.
    monkeypatch.setenv(POSTURE_ENV_VAR, "readwrite")


class TestEnterSandboxMode:
    def test_sets_the_sandbox_value_over_a_readwrite_run(self, monkeypatch):
        monkeypatch.setenv(POSTURE_ENV_VAR, "readwrite")

        enter_sandbox_mode()

        assert is_readonly_run()

    def test_writes_into_a_given_environment_only(self, monkeypatch):
        monkeypatch.delenv(POSTURE_ENV_VAR, raising=False)
        environ: dict[str, str] = {}

        enter_sandbox_mode(environ)

        assert environ == {POSTURE_ENV_VAR: SANDBOX_MODE}
        assert not is_readonly_run()


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment")
class TestConnectorUnderThePosture:
    async def test_connector_puts_before_the_posture(self):
        connector = RecordingConnector()

        result = await connector.write_channel("SR:MAG:QF:01:CURRENT:SP", 1.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert connector.writes == [("SR:MAG:QF:01:CURRENT:SP", 1.0)]

    async def test_connector_cannot_put_under_the_posture(self):
        enter_sandbox_mode()
        connector = RecordingConnector()

        result = await connector.write_channel("SR:MAG:QF:01:CURRENT:SP", 1.0)
        many = await connector.write_multiple_channels([("A:SP", 1.0), ("B:SP", 2.0)])

        assert result.outcome is WriteOutcome.REFUSED
        assert all(r.outcome is WriteOutcome.REFUSED for r in many)
        assert connector.writes == []


class TestArmRawPutBlock:
    """The block is irreversible, so its installer is recorded rather than run."""

    @pytest.fixture
    def installed(self, monkeypatch):
        calls: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            raw_put_block, "install", lambda mode, **contract: calls.append((mode, contract))
        )
        return calls

    def test_arms_the_armed_mode_with_rpc_refused(self, installed):
        arm_raw_put_block()

        [(mode, contract)] = installed
        assert mode == "armed"
        assert contract["refuse_rpc"] is True

    def test_refuses_every_pvaccess_put_and_leaves_no_ca_provider_route(self, installed):
        from osprey.services.python_executor.write_surface import _ARMED_CHECKED

        arm_raw_put_block()

        [(_, contract)] = installed
        assert contract["ca_provider_targets"] == ()
        for dotted, attr in _ARMED_CHECKED:
            assert (dotted, (attr,)) in contract["blocked_targets"]

    def test_an_installer_failure_reaches_the_caller(self, monkeypatch):
        def fail(*_args, **_contract):
            raise RuntimeError("cannot patch")

        monkeypatch.setattr(raw_put_block, "install", fail)

        with pytest.raises(RuntimeError, match="cannot patch"):
            arm_raw_put_block()

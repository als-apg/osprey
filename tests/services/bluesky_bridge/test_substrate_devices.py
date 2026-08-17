"""Unit tests for the canonical EPICS-substrate plan-device derivation.

Covers ``osprey.services.bluesky_bridge.substrate_devices`` -- the single
source shared by ``osprey up`` (``container_lifecycle.
_ensure_bluesky_substrate_env``) and ``tests/e2e/_orm_stack.py`` -- plus, in
``TestEnsureBlueskySubstrateEnv`` below, the ``container_lifecycle`` deploy-path
wiring itself, called directly (Docker-free).
"""

from __future__ import annotations

import json

import pytest

from osprey.services.bluesky_bridge.substrate_devices import (
    READBACKS_ENV,
    SETPOINTS_ENV,
    SUBSTRATE_ENV,
    derive_substrate_env,
    format_readbacks_env,
    format_setpoints_env,
    select_bpms,
    select_correctors,
)

# A synthetic channel_limits.json-shaped dict covering:
#  - two full SR HCM/VCM pyat-coupled corrector SP/RB pairs
#  - one SR corrector SP with NO matching RB (must be excluded)
#  - a non-HCM/VCM SR magnet family (QF; in MAG_FAMILIES but not a corrector)
#  - a BR magnet (sp-echo partition; wrong ring for a corrector)
#  - two SR BPM pyat-coupled X/Y position readbacks
#  - a BPM STATUS field (same family, but classify_partition falls through
#    to static-noisy since the field isn't POSITION)
#  - metadata keys ("_meta", "defaults") that must be ignored entirely
_LIMITS = {
    "SR:MAG:HCM:01:CURRENT:SP": {"min": -10, "max": 10},
    "SR:MAG:HCM:01:CURRENT:RB": {"min": -10, "max": 10},
    "SR:MAG:VCM:02:CURRENT:SP": {"min": -10, "max": 10},
    "SR:MAG:VCM:02:CURRENT:RB": {"min": -10, "max": 10},
    "SR:MAG:HCM:03:CURRENT:SP": {"min": -10, "max": 10},  # no RB counterpart
    "SR:MAG:QF:01:CURRENT:SP": {"min": -10, "max": 10},
    "SR:MAG:QF:01:CURRENT:RB": {"min": -10, "max": 10},
    "BR:MAG:HCM:01:CURRENT:SP": {"min": -10, "max": 10},
    "BR:MAG:HCM:01:CURRENT:RB": {"min": -10, "max": 10},
    "SR:DIAG:BPM:01:POSITION:X": {"min": -5, "max": 5},
    "SR:DIAG:BPM:01:POSITION:Y": {"min": -5, "max": 5},
    "SR:DIAG:BPM:02:POSITION:X": {"min": -5, "max": 5},
    "SR:DIAG:BPM:02:POSITION:Y": {"min": -5, "max": 5},
    "SR:DIAG:BPM:03:STATUS:VALID": {"min": 0, "max": 1},
    "_meta": {"ignored": True},
    "defaults": {"ignored": True},
}


class TestSelectCorrectors:
    def test_full_set_default_returns_all_pyat_coupled_pairs(self) -> None:
        correctors = select_correctors(_LIMITS)
        # Only the two complete SR HCM/VCM SP/RB pairs qualify.
        assert len(correctors) == 2
        pairs = set(correctors.values())
        assert ("SR:MAG:HCM:01:CURRENT:SP", "SR:MAG:HCM:01:CURRENT:RB") in pairs
        assert ("SR:MAG:VCM:02:CURRENT:SP", "SR:MAG:VCM:02:CURRENT:RB") in pairs

    def test_device_name_is_the_setpoint_address(self) -> None:
        """Device name == channel address: an agent that discovered a ``:SP``
        address via channel-finder can name it directly in a plan, with no
        synthetic ``corrector_NN`` namespace in between."""
        correctors = select_correctors(_LIMITS)
        assert set(correctors) == {
            "SR:MAG:HCM:01:CURRENT:SP",
            "SR:MAG:VCM:02:CURRENT:SP",
        }
        assert all(name == sp for name, (sp, _rb) in correctors.items())

    def test_excludes_sp_without_matching_rb(self) -> None:
        correctors = select_correctors(_LIMITS)
        assert not any(sp == "SR:MAG:HCM:03:CURRENT:SP" for sp, _rb in correctors.values())

    def test_excludes_non_hcm_vcm_family(self) -> None:
        correctors = select_correctors(_LIMITS)
        assert not any(sp.startswith("SR:MAG:QF:") for sp, _rb in correctors.values())

    def test_excludes_non_sr_ring(self) -> None:
        correctors = select_correctors(_LIMITS)
        assert not any(sp.startswith("BR:") for sp, _rb in correctors.values())

    def test_count_none_never_raises_regardless_of_availability(self) -> None:
        assert select_correctors({}, count=None) == {}

    def test_count_int_returns_exact_slice(self) -> None:
        correctors = select_correctors(_LIMITS, count=1)
        # Address-keyed on the sliced path too, and the slice takes the
        # lowest-sorting address rather than an arbitrary one.
        assert correctors == {
            "SR:MAG:HCM:01:CURRENT:SP": ("SR:MAG:HCM:01:CURRENT:SP", "SR:MAG:HCM:01:CURRENT:RB")
        }

    def test_count_int_raises_when_insufficient(self) -> None:
        with pytest.raises(AssertionError):
            select_correctors(_LIMITS, count=5)

    def test_ignores_metadata_keys(self) -> None:
        limits = {"_meta": {}, "defaults": {}}
        assert select_correctors(limits, count=None) == {}


class TestSelectBpms:
    def test_full_set_default_returns_all_pyat_coupled_readbacks(self) -> None:
        readbacks = select_bpms(_LIMITS)
        assert len(readbacks) == 4
        addresses = set(readbacks.values())
        assert "SR:DIAG:BPM:01:POSITION:X" in addresses
        assert "SR:DIAG:BPM:01:POSITION:Y" in addresses
        assert "SR:DIAG:BPM:02:POSITION:X" in addresses
        assert "SR:DIAG:BPM:02:POSITION:Y" in addresses

    def test_device_name_is_the_read_address(self) -> None:
        """Same convention as the correctors: the detector's name is the
        address it reads, so a discovered BPM address is directly usable."""
        readbacks = select_bpms(_LIMITS)
        assert all(name == address for name, address in readbacks.items())

    def test_excludes_non_position_field(self) -> None:
        readbacks = select_bpms(_LIMITS)
        assert "SR:DIAG:BPM:03:STATUS:VALID" not in set(readbacks.values())

    def test_count_none_never_raises_regardless_of_availability(self) -> None:
        assert select_bpms({}, count=None) == {}

    def test_count_int_returns_exact_slice(self) -> None:
        readbacks = select_bpms(_LIMITS, count=2)
        # Address-keyed on the sliced path too (see the corrector counterpart).
        assert readbacks == {
            "SR:DIAG:BPM:01:POSITION:X": "SR:DIAG:BPM:01:POSITION:X",
            "SR:DIAG:BPM:01:POSITION:Y": "SR:DIAG:BPM:01:POSITION:Y",
        }

    def test_count_int_raises_when_insufficient(self) -> None:
        with pytest.raises(AssertionError):
            select_bpms(_LIMITS, count=10)


class TestFormatters:
    def test_format_setpoints_env(self) -> None:
        correctors = {
            "SR:MAG:HCM:01:CURRENT:SP": ("SR:MAG:HCM:01:CURRENT:SP", "SR:MAG:HCM:01:CURRENT:RB")
        }
        assert format_setpoints_env(correctors) == (
            "SR:MAG:HCM:01:CURRENT:SP=SR:MAG:HCM:01:CURRENT:SP|SR:MAG:HCM:01:CURRENT:RB"
        )

    def test_format_readbacks_env(self) -> None:
        readbacks = {"SR:DIAG:BPM:01:POSITION:X": "SR:DIAG:BPM:01:POSITION:X"}
        assert (
            format_readbacks_env(readbacks) == "SR:DIAG:BPM:01:POSITION:X=SR:DIAG:BPM:01:POSITION:X"
        )

    def test_format_setpoints_env_joins_multiple_with_commas(self) -> None:
        correctors = {
            "SP1": ("SP1", "RB1"),
            "SP2": ("SP2", "RB2"),
        }
        assert format_setpoints_env(correctors) == "SP1=SP1|RB1,SP2=SP2|RB2"


class TestDeriveSubstrateEnv:
    def test_happy_path_returns_full_env_dict(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "channel_limits.json").write_text(json.dumps(_LIMITS), encoding="utf-8")

        env = derive_substrate_env(tmp_path)

        assert env[SUBSTRATE_ENV] == "1"
        assert env[SETPOINTS_ENV]
        assert env[READBACKS_ENV]
        # Wire format sanity: comma-separated name=value entries.
        assert len(env[SETPOINTS_ENV].split(",")) == 2
        assert len(env[READBACKS_ENV].split(",")) == 4
        for entry in env[SETPOINTS_ENV].split(","):
            name, _, rest = entry.partition("=")
            assert name
            assert "|" in rest

    def test_colon_named_devices_survive_the_bridge_parser(self, tmp_path) -> None:
        """The wire format carries colon addresses AS device names unharmed.

        ``name=SP|RB`` splits on the first ``=`` and on ``|``, neither of which
        is a legal EPICS name character, so an address-named device round-trips
        with its name intact -- and no name collides, so ``_drop_duplicate_names``
        keeps every spec.
        """
        from osprey.services.bluesky_bridge.devices._specs_from_env import specs_from_env

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "channel_limits.json").write_text(json.dumps(_LIMITS), encoding="utf-8")

        motors, detectors = specs_from_env(derive_substrate_env(tmp_path))

        assert {spec.name for spec in motors} == {
            "SR:MAG:HCM:01:CURRENT:SP",
            "SR:MAG:VCM:02:CURRENT:SP",
        }
        assert all(spec.name == spec.setpoint_pv for spec in motors)
        assert all(spec.readback_pv == spec.name[: -len(":SP")] + ":RB" for spec in motors)
        # Nothing dropped as a duplicate: 4 BPM readbacks in, 4 out.
        assert {spec.name for spec in detectors} == {
            "SR:DIAG:BPM:01:POSITION:X",
            "SR:DIAG:BPM:01:POSITION:Y",
            "SR:DIAG:BPM:02:POSITION:X",
            "SR:DIAG:BPM:02:POSITION:Y",
        }
        assert all(spec.name == spec.read_pv for spec in detectors)

    def test_missing_channel_limits_returns_empty_dict(self, tmp_path) -> None:
        assert derive_substrate_env(tmp_path) == {}

    def test_malformed_json_returns_empty_dict(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "channel_limits.json").write_text("{not valid json", encoding="utf-8")

        assert derive_substrate_env(tmp_path) == {}

    def test_no_correctors_returns_empty_dict(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        only_bpms = {k: v for k, v in _LIMITS.items() if "BPM" in k}
        (data_dir / "channel_limits.json").write_text(json.dumps(only_bpms), encoding="utf-8")

        assert derive_substrate_env(tmp_path) == {}

    def test_no_bpms_returns_empty_dict(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        only_correctors = {k: v for k, v in _LIMITS.items() if "MAG" in k}
        (data_dir / "channel_limits.json").write_text(json.dumps(only_correctors), encoding="utf-8")

        assert derive_substrate_env(tmp_path) == {}

    def test_empty_channel_limits_returns_empty_dict(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "channel_limits.json").write_text("{}", encoding="utf-8")

        assert derive_substrate_env(tmp_path) == {}

    def test_non_dict_json_returns_empty_dict(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "channel_limits.json").write_text("[1, 2, 3]", encoding="utf-8")

        assert derive_substrate_env(tmp_path) == {}


class TestEnsureBlueskySubstrateEnv:
    """Deploy-path wiring: ``container_lifecycle._ensure_bluesky_substrate_env``,
    called directly (Docker-free) rather than through the full ``deploy_up``.
    """

    def _write_channel_limits(self, project_dir) -> None:
        data_dir = project_dir / "data"
        data_dir.mkdir()
        (data_dir / "channel_limits.json").write_text(json.dumps(_LIMITS), encoding="utf-8")

    def test_writes_substrate_env_when_va_backed_plan_stack(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_bluesky_substrate_env

        self._write_channel_limits(tmp_path)
        config = {
            "deployed_services": ["bluesky", "virtual_accelerator"],
            "control_system": {"type": "virtual_accelerator"},
        }
        env_path = tmp_path / ".env"

        _ensure_bluesky_substrate_env(config, env_path=env_path)

        from osprey.utils.dotenv import parse_dotenv_file

        env = parse_dotenv_file(env_path)
        assert env[SUBSTRATE_ENV] == "1"
        assert env[SETPOINTS_ENV]
        assert env[READBACKS_ENV]

    def test_already_set_dotenv_values_are_preserved(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_bluesky_substrate_env

        self._write_channel_limits(tmp_path)
        env_path = tmp_path / ".env"
        env_path.write_text(f"{SETPOINTS_ENV}=operator_corrector=OP:SP|OP:RB\n", encoding="utf-8")
        config = {
            "deployed_services": ["bluesky", "virtual_accelerator"],
            "control_system": {"type": "virtual_accelerator"},
        }

        _ensure_bluesky_substrate_env(config, env_path=env_path)

        from osprey.utils.dotenv import parse_dotenv_file

        env = parse_dotenv_file(env_path)
        # Operator-set value untouched...
        assert env[SETPOINTS_ENV] == "operator_corrector=OP:SP|OP:RB"
        # ...but the vars the operator did NOT set are still filled in.
        assert env[SUBSTRATE_ENV] == "1"
        assert env[READBACKS_ENV]

    def test_already_set_process_env_values_are_preserved(self, tmp_path, monkeypatch) -> None:
        from osprey.deployment.container_lifecycle import _ensure_bluesky_substrate_env

        self._write_channel_limits(tmp_path)
        monkeypatch.setenv(SUBSTRATE_ENV, "0")
        config = {
            "deployed_services": ["bluesky", "virtual_accelerator"],
            "control_system": {"type": "virtual_accelerator"},
        }
        env_path = tmp_path / ".env"

        _ensure_bluesky_substrate_env(config, env_path=env_path)

        from osprey.utils.dotenv import parse_dotenv_file

        env = parse_dotenv_file(env_path)
        # A process-env value is never duplicated into .env.
        assert SUBSTRATE_ENV not in env
        # The other, unset vars are still written.
        assert env[SETPOINTS_ENV]
        assert env[READBACKS_ENV]

    def test_mock_control_system_never_arms_the_substrate(self, tmp_path) -> None:
        """A ``control_system.type: mock`` deploy must NOT arm the EPICS
        substrate, even when a VA container is co-deployed alongside bluesky
        (control-assistant always bundles VA). ``control_system.type`` is the
        single source of truth for what the deployment may drive: arming the
        substrate off the mere presence of a VA container would hand the
        queueserver worker real Channel Access devices for a deployment the
        capability surface reports as browse-only.
        See ``container_lifecycle._ensure_bluesky_substrate_env``.
        """
        from osprey.deployment.container_lifecycle import _ensure_bluesky_substrate_env

        self._write_channel_limits(tmp_path)
        config = {
            "deployed_services": ["bluesky", "virtual_accelerator"],
            "control_system": {"type": "mock"},
        }
        env_path = tmp_path / ".env"

        _ensure_bluesky_substrate_env(config, env_path=env_path)

        # No .env written at all -- the substrate stays disarmed, and the worker
        # builds no devices, which is the browse-only signal by design.
        assert not env_path.exists()

    def test_no_write_without_virtual_accelerator_deployed(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_bluesky_substrate_env

        self._write_channel_limits(tmp_path)
        config = {"deployed_services": ["bluesky"]}
        env_path = tmp_path / ".env"

        _ensure_bluesky_substrate_env(config, env_path=env_path)

        assert not env_path.exists()

    def test_no_write_without_bluesky_deployed(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_bluesky_substrate_env

        self._write_channel_limits(tmp_path)
        config = {"deployed_services": ["virtual_accelerator"]}
        env_path = tmp_path / ".env"

        _ensure_bluesky_substrate_env(config, env_path=env_path)

        assert not env_path.exists()

    def test_missing_channel_limits_skips_without_raising(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_bluesky_substrate_env

        config = {
            "deployed_services": ["bluesky", "virtual_accelerator"],
            "control_system": {"type": "virtual_accelerator"},
        }
        env_path = tmp_path / ".env"

        _ensure_bluesky_substrate_env(config, env_path=env_path)  # must not raise

        assert not env_path.exists()

    def test_idempotent_no_duplicate_keys_on_second_run(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_bluesky_substrate_env

        self._write_channel_limits(tmp_path)
        config = {
            "deployed_services": ["bluesky", "virtual_accelerator"],
            "control_system": {"type": "virtual_accelerator"},
        }
        env_path = tmp_path / ".env"

        _ensure_bluesky_substrate_env(config, env_path=env_path)
        _ensure_bluesky_substrate_env(config, env_path=env_path)

        text = env_path.read_text(encoding="utf-8")
        assert text.count(f"{SUBSTRATE_ENV}=") == 1
        assert text.count(f"{SETPOINTS_ENV}=") == 1
        assert text.count(f"{READBACKS_ENV}=") == 1

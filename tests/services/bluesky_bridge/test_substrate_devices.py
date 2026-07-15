"""Unit tests for the canonical EPICS-substrate scan-device derivation.

Covers ``osprey.services.bluesky_bridge.substrate_devices`` -- the single
source shared by ``osprey deploy up`` (``container_lifecycle.
_ensure_scan_substrate_env``) and ``tests/e2e/_orm_stack.py`` -- plus, in
``TestEnsureScanSubstrateEnv`` below, the ``container_lifecycle`` deploy-path
wiring itself (called directly, Docker-free, per task 3.5's test plan).
"""

from __future__ import annotations

import json

import pytest

from osprey.services.bluesky_bridge.substrate_devices import (
    DETECTORS_ENV,
    MOTORS_ENV,
    SUBSTRATE_ENV,
    derive_substrate_env,
    format_detectors_env,
    format_motors_env,
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
        assert len(correctors) == 1

    def test_count_int_raises_when_insufficient(self) -> None:
        with pytest.raises(AssertionError):
            select_correctors(_LIMITS, count=5)

    def test_ignores_metadata_keys(self) -> None:
        limits = {"_meta": {}, "defaults": {}}
        assert select_correctors(limits, count=None) == {}


class TestSelectBpms:
    def test_full_set_default_returns_all_pyat_coupled_readbacks(self) -> None:
        bpms = select_bpms(_LIMITS)
        assert len(bpms) == 4
        addresses = set(bpms.values())
        assert "SR:DIAG:BPM:01:POSITION:X" in addresses
        assert "SR:DIAG:BPM:01:POSITION:Y" in addresses
        assert "SR:DIAG:BPM:02:POSITION:X" in addresses
        assert "SR:DIAG:BPM:02:POSITION:Y" in addresses

    def test_excludes_non_position_field(self) -> None:
        bpms = select_bpms(_LIMITS)
        assert "SR:DIAG:BPM:03:STATUS:VALID" not in set(bpms.values())

    def test_count_none_never_raises_regardless_of_availability(self) -> None:
        assert select_bpms({}, count=None) == {}

    def test_count_int_returns_exact_slice(self) -> None:
        bpms = select_bpms(_LIMITS, count=2)
        assert len(bpms) == 2

    def test_count_int_raises_when_insufficient(self) -> None:
        with pytest.raises(AssertionError):
            select_bpms(_LIMITS, count=10)


class TestFormatters:
    def test_format_motors_env(self) -> None:
        correctors = {"corrector_01": ("SR:MAG:HCM:01:CURRENT:SP", "SR:MAG:HCM:01:CURRENT:RB")}
        assert (
            format_motors_env(correctors)
            == "corrector_01=SR:MAG:HCM:01:CURRENT:SP|SR:MAG:HCM:01:CURRENT:RB"
        )

    def test_format_detectors_env(self) -> None:
        bpms = {"bpm_01": "SR:DIAG:BPM:01:POSITION:X"}
        assert format_detectors_env(bpms) == "bpm_01=SR:DIAG:BPM:01:POSITION:X"

    def test_format_motors_env_joins_multiple_with_commas(self) -> None:
        correctors = {
            "corrector_01": ("SP1", "RB1"),
            "corrector_02": ("SP2", "RB2"),
        }
        assert format_motors_env(correctors) == "corrector_01=SP1|RB1,corrector_02=SP2|RB2"


class TestDeriveSubstrateEnv:
    def test_happy_path_returns_full_env_dict(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "channel_limits.json").write_text(json.dumps(_LIMITS), encoding="utf-8")

        env = derive_substrate_env(tmp_path)

        assert env[SUBSTRATE_ENV] == "1"
        assert env[MOTORS_ENV]
        assert env[DETECTORS_ENV]
        # Wire format sanity: comma-separated name=value entries.
        assert len(env[MOTORS_ENV].split(",")) == 2
        assert len(env[DETECTORS_ENV].split(",")) == 4
        for entry in env[MOTORS_ENV].split(","):
            name, _, rest = entry.partition("=")
            assert name
            assert "|" in rest

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


class TestEnsureScanSubstrateEnv:
    """Deploy-path wiring: ``container_lifecycle._ensure_scan_substrate_env``,
    called directly (Docker-free) rather than through the full ``deploy_up``.
    """

    def _write_channel_limits(self, project_dir) -> None:
        data_dir = project_dir / "data"
        data_dir.mkdir()
        (data_dir / "channel_limits.json").write_text(json.dumps(_LIMITS), encoding="utf-8")

    def test_writes_substrate_env_when_va_backed_scan_stack(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_scan_substrate_env

        self._write_channel_limits(tmp_path)
        config = {"deployed_services": ["bluesky", "virtual_accelerator"]}
        env_path = tmp_path / ".env"

        _ensure_scan_substrate_env(config, env_path=env_path)

        from osprey.utils.dotenv import parse_dotenv_file

        env = parse_dotenv_file(env_path)
        assert env[SUBSTRATE_ENV] == "1"
        assert env[MOTORS_ENV]
        assert env[DETECTORS_ENV]

    def test_already_set_dotenv_values_are_preserved(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_scan_substrate_env

        self._write_channel_limits(tmp_path)
        env_path = tmp_path / ".env"
        env_path.write_text(f"{MOTORS_ENV}=operator_corrector=OP:SP|OP:RB\n", encoding="utf-8")
        config = {"deployed_services": ["bluesky", "virtual_accelerator"]}

        _ensure_scan_substrate_env(config, env_path=env_path)

        from osprey.utils.dotenv import parse_dotenv_file

        env = parse_dotenv_file(env_path)
        # Operator-set value untouched...
        assert env[MOTORS_ENV] == "operator_corrector=OP:SP|OP:RB"
        # ...but the vars the operator did NOT set are still filled in.
        assert env[SUBSTRATE_ENV] == "1"
        assert env[DETECTORS_ENV]

    def test_already_set_process_env_values_are_preserved(self, tmp_path, monkeypatch) -> None:
        from osprey.deployment.container_lifecycle import _ensure_scan_substrate_env

        self._write_channel_limits(tmp_path)
        monkeypatch.setenv(SUBSTRATE_ENV, "0")
        config = {"deployed_services": ["bluesky", "virtual_accelerator"]}
        env_path = tmp_path / ".env"

        _ensure_scan_substrate_env(config, env_path=env_path)

        from osprey.utils.dotenv import parse_dotenv_file

        env = parse_dotenv_file(env_path)
        # A process-env value is never duplicated into .env.
        assert SUBSTRATE_ENV not in env
        # The other, unset vars are still written.
        assert env[MOTORS_ENV]
        assert env[DETECTORS_ENV]

    def test_no_write_without_virtual_accelerator_deployed(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_scan_substrate_env

        self._write_channel_limits(tmp_path)
        config = {"deployed_services": ["bluesky"]}
        env_path = tmp_path / ".env"

        _ensure_scan_substrate_env(config, env_path=env_path)

        assert not env_path.exists()

    def test_no_write_without_bluesky_deployed(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_scan_substrate_env

        self._write_channel_limits(tmp_path)
        config = {"deployed_services": ["virtual_accelerator"]}
        env_path = tmp_path / ".env"

        _ensure_scan_substrate_env(config, env_path=env_path)

        assert not env_path.exists()

    def test_missing_channel_limits_skips_without_raising(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_scan_substrate_env

        config = {"deployed_services": ["bluesky", "virtual_accelerator"]}
        env_path = tmp_path / ".env"

        _ensure_scan_substrate_env(config, env_path=env_path)  # must not raise

        assert not env_path.exists()

    def test_idempotent_no_duplicate_keys_on_second_run(self, tmp_path) -> None:
        from osprey.deployment.container_lifecycle import _ensure_scan_substrate_env

        self._write_channel_limits(tmp_path)
        config = {"deployed_services": ["bluesky", "virtual_accelerator"]}
        env_path = tmp_path / ".env"

        _ensure_scan_substrate_env(config, env_path=env_path)
        _ensure_scan_substrate_env(config, env_path=env_path)

        text = env_path.read_text(encoding="utf-8")
        assert text.count(f"{SUBSTRATE_ENV}=") == 1
        assert text.count(f"{MOTORS_ENV}=") == 1
        assert text.count(f"{DETECTORS_ENV}=") == 1

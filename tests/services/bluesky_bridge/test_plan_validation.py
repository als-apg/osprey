"""Tests for `plan_validation.py` (task 2.1): the three-stage authoring
validator for a session-tier plan-file body — static AST allowlist, narrowed
CA/connector pattern scan, and a mock-RunEngine dry-run.

The dry-run stage actually drives a real bluesky `RunEngine` (in a
subprocess) against real ophyd-async mock devices, so — like every other
bluesky-capable test in this directory — the dry-run tests are guarded by
`pytest.importorskip` rather than failing outright when `bluesky`/
`ophyd_async` aren't installed. The static-allowlist and pattern-scan stages
need neither, but this file keeps the same guard for every test for
consistency with its siblings (`test_exemplar_plans.py`,
`test_runengine_integration.py`).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

bluesky = pytest.importorskip("bluesky")
ophyd_async = pytest.importorskip("ophyd_async")

from osprey.mcp_server.workspace.execution.sandbox_executor import (  # noqa: E402
    _ALLOWED_IMPORTS,
    _ALLOWED_TOP_LEVEL,
    _DANGEROUS_PATTERNS,
    validate_sandbox_code,
)
from osprey.services.bluesky_bridge import plan_validation  # noqa: E402
from osprey.services.bluesky_bridge.plan_validation import (  # noqa: E402
    _CA_ONLY_PATTERNS,
    _EPICS_CA_ENV_NAMES_TO_DROP,
    _EPICS_CA_INERT_ENV,
    _ca_pattern_scan,
    _collect_device_names,
    _static_allowlist_check,
    hash_plan_body,
    validate_plan,
)

# ---------------------------------------------------------------------------
# A tiny, fully-contract-compliant benign plan body: one corrector, one
# detector, harmless (non-control-system) `.put`/`.get` usage that a naive
# pattern scan could mistake for a CA write/read. Reused across the
# accept-path and dry-run tests below.
# ---------------------------------------------------------------------------
BENIGN_PLAN_BODY = textwrap.dedent(
    """\
    import numpy as np
    from bluesky import plan_stubs as bps
    from bluesky import preprocessors as bpp
    from pydantic import BaseModel, Field

    PLAN_METADATA = {
        "name": "tiny_sweep",
        "description": "Sweep one corrector, reading one detector at each point.",
        "category": "accelerator",
        "required_devices": ["correctors", "detectors"],
        "writes": True,
    }


    class PARAMS(BaseModel):
        correctors: list[str] = Field(..., min_length=1)
        detectors: list[str] = Field(..., min_length=1)
        num: int = Field(..., ge=1)


    def build_plan(devices, params):
        # Harmless, non-control-system ".get"/".put" usage that must NOT be
        # mistaken for a CA/connector read or write.
        {}.get("missing")
        arr = np.zeros(params.num)
        np.put(arr, list(range(params.num)), 1.0)

        corrector = devices[params.correctors[0]]
        detector = devices[params.detectors[0]]

        @bpp.stage_decorator([corrector, detector])
        @bpp.run_decorator()
        def _sweep():
            for i in range(params.num):
                yield from bps.mv(corrector, float(i))
                yield from bps.trigger_and_read([corrector, detector])

        return _sweep()
    """
)

BENIGN_SAMPLE_ARGS = {"correctors": ["c1"], "detectors": ["d1"], "num": 3}

# ---------------------------------------------------------------------------
# A raw "author-submitted body" shaped like an actual `PlanSessionWriteRequest
# .body` (task 2.3) -- unlike `BENIGN_PLAN_BODY` above, this has NO embedded
# `PLAN_METADATA` of its own (the real field never does; `write_session_plan`
# in app.py generates and prepends that separately). Used by
# `TestFutureImportPosition` to assemble content exactly the way
# `write_session_plan` does, so those tests exercise the real shape of the
# task 2.12 bug rather than a synthetic approximation of it.
# ---------------------------------------------------------------------------
_SESSION_BODY = textwrap.dedent(
    """\
    from bluesky import plan_stubs as bps
    from bluesky import preprocessors as bpp
    from pydantic import BaseModel, Field


    class PARAMS(BaseModel):
        correctors: list[str] = Field(..., min_length=1)
        detectors: list[str] = Field(..., min_length=1)
        num: int = Field(..., ge=1)


    def build_plan(devices, params):
        corrector = devices[params.correctors[0]]
        detector = devices[params.detectors[0]]

        @bpp.stage_decorator([corrector, detector])
        @bpp.run_decorator()
        def _sweep():
            for i in range(params.num):
                yield from bps.mv(corrector, float(i))
                yield from bps.trigger_and_read([corrector, detector])

        return _sweep()
    """
)


def _assembled_session_content(body: str) -> str:
    """Mirror `write_session_plan`'s (app.py) file assembly exactly: a
    generated `PLAN_METADATA = {...}` assignment prepended ahead of the
    author's own body -- the shape task 2.12's future-import-position check
    exists to guard.
    """
    metadata = {
        "name": "tiny_sweep",
        "description": "",
        "category": "accelerator",
        "required_devices": ["correctors", "detectors"],
        "writes": True,
    }
    return f"PLAN_METADATA = {metadata!r}\n\n{body}"


# =========================================================================
# Regression: the viz sandbox's own constants/behavior are untouched (C10)
# =========================================================================


class TestVizSandboxRegression:
    def test_allowed_top_level_and_imports_unmodified(self):
        """`_ALLOWED_TOP_LEVEL`/`_ALLOWED_IMPORTS` still hold every original
        viz-sandbox entry, and `_ALLOWED_TOP_LEVEL` is still derived from
        `_ALLOWED_IMPORTS` — task 2.1 must never rename or mutate either.
        """
        assert _ALLOWED_TOP_LEVEL == {m.split(".")[0] for m in _ALLOWED_IMPORTS}
        # A representative sample of the original viz whitelist, untouched.
        for name in ("numpy", "pandas", "matplotlib", "plotly", "bokeh", "os", "pathlib"):
            assert name in _ALLOWED_IMPORTS
        # Never widened to admit bluesky or CA-adjacent names by this change.
        assert "bluesky" not in _ALLOWED_TOP_LEVEL
        assert "epics" not in _ALLOWED_TOP_LEVEL

    def test_dangerous_patterns_unmodified(self):
        assert ("epics", "epics module") in _DANGEROUS_PATTERNS
        assert ("write_channel", "write_channel()") in _DANGEROUS_PATTERNS
        assert ("ctypes", "ctypes module") in _DANGEROUS_PATTERNS

    def test_viz_single_arg_call_still_behaves_identically(self):
        """The pre-existing single-positional-arg call (the viz sandbox's own
        caller, `sandbox_executor.py`'s `execute_sandbox_code`) must see the
        exact same behavior after parameterization: same signature, same
        defaults, no keyword required.
        """
        is_safe, violations = validate_sandbox_code(
            "import numpy as np\nimport matplotlib.pyplot as plt\nplt.plot(np.arange(3))"
        )
        assert is_safe
        assert violations == []

        is_safe, violations = validate_sandbox_code("import epics\nepics.caput('PV', 1)")
        assert not is_safe
        assert any("epics" in v for v in violations)

        is_safe, violations = validate_sandbox_code("import bluesky")
        assert not is_safe
        assert any("bluesky" in v for v in violations)


# =========================================================================
# Stage 1: static AST allowlist
# =========================================================================


class TestStaticAllowlistCheck:
    @pytest.mark.parametrize(
        "code",
        [
            "import epics",
            "import epics as e",  # aliasing must not evade the import-name check
            "from epics import caput",
            '__import__("epics")',
            "import ctypes",
            "import os",
            "import importlib",
            "import subprocess",
            "import socket",
            "import aioca",
            "import caproto",
            "import bluesky",  # bare bluesky, no submodule
            "import bluesky.utils",  # a real submodule, but not one of the 3 allowed
            "from bluesky.callbacks import LiveTable",
            "import logging.config",  # dictConfig/fileConfig do instantiation-by-string
            "import logging.handlers",  # e.g. SMTPHandler/SocketHandler
            "from logging import config",  # same submodule, "from X import Y" form
            "from logging import handlers",
            "from logging.config import dictConfig",
        ],
    )
    def test_rejects_disallowed_imports(self, code):
        violations = _static_allowlist_check(code)
        assert violations, f"expected a rejection for: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "from bluesky import plan_stubs as bps",
            "from bluesky.plan_stubs import mv",
            "import bluesky.plan_stubs",
            "from bluesky import plans as bp",
            "from bluesky import preprocessors as bpp",
            "import numpy as np",
            "from scipy import stats",
            "import math",
            "import statistics",
            "import time",
            "import collections",
            "import itertools",
            "import functools",
            "from pydantic import BaseModel, Field",
            "from __future__ import annotations",
            "from typing import Any",
            "import typing",
            "import logging",
            "from logging import getLogger",
        ],
    )
    def test_accepts_allowed_imports(self, code):
        assert _static_allowlist_check(code) == []

    def test_logging_submodule_granularity_bare_accept_config_and_handlers_reject(self):
        """The `logging` top level is allowed (the shipped exemplars need
        `logging.getLogger(...)`), but `logging.config`/`logging.handlers`
        must stay rejected in every import form — mirrors the `bluesky`
        submodule granularity test, inverted (allow the top level, deny
        specific submodules rather than the reverse).
        """
        assert _static_allowlist_check("import logging") == []
        assert _static_allowlist_check("from logging import getLogger") == []
        assert _static_allowlist_check("import logging.config") != []
        assert _static_allowlist_check("import logging.handlers") != []
        assert _static_allowlist_check("from logging import config") != []
        assert _static_allowlist_check("from logging.config import dictConfig") != []

    def test_submodule_granularity_plan_stubs_accept_some_other_reject(self):
        assert _static_allowlist_check("from bluesky import plan_stubs") == []
        assert _static_allowlist_check("from bluesky import some_other") != []

    def test_dunder_import_variant_rejected(self):
        violations = _static_allowlist_check("x = __import__('epics')")
        assert any("__import__" in v for v in violations)

    def test_syntax_error_reported_and_short_circuits(self):
        violations = _static_allowlist_check("def f(:\n  pass")
        assert any("Syntax error" in v for v in violations)


# =========================================================================
# Task 2.12: `from __future__ import ...` cannot survive `write_session_plan`
# (app.py)'s metadata-prepend assembly -- Python requires a future-import to
# be the file's literal first statement (module docstring aside), but
# `write_session_plan` always writes a generated `PLAN_METADATA = {...}`
# assignment ahead of the author's body. `ast.parse` (the syntax gate
# `validate_sandbox_code` runs) does not enforce that positional rule --
# only `compile()`/the import machinery does -- so an unflagged body would
# otherwise sail through stages 1-2 clean and only fail deep in stage 3's
# dry-run subprocess, as a `SyntaxError` naming a temp file and line number
# that point at the generated metadata line, not the real cause.
# =========================================================================

_FUTURE_IMPORT_REJECT_MESSAGE = (
    "session plans cannot use `from __future__` imports because plan "
    "metadata is prepended to the file; omit it — modern type hints "
    "(list[str], dict[str, Any]) work without it on Python 3.9+."
)


class TestFutureImportPosition:
    def test_bare_future_import_at_position_zero_still_accepted(self):
        """A positional check, not a blanket ban: `from __future__ import`
        genuinely at the leading position (as in the shipped `plans_core`
        exemplars, read and validated directly -- never metadata-prepended,
        see `TestShippedExemplarsPassValidation` below) is legal Python and
        must still pass.
        """
        assert _static_allowlist_check("from __future__ import annotations") == []

    def test_future_import_rejected_once_metadata_is_prepended(self):
        content = _assembled_session_content(
            "from __future__ import annotations\n\n" + _SESSION_BODY
        )
        assert _static_allowlist_check(content) == [_FUTURE_IMPORT_REJECT_MESSAGE]

    async def test_validate_plan_rejects_with_clear_message_not_a_syntax_error(self):
        content = _assembled_session_content(
            "from __future__ import annotations\n\n" + _SESSION_BODY
        )
        result = await validate_plan(
            content, plan_name="tiny_sweep", sample_args=BENIGN_SAMPLE_ARGS
        )
        assert result.passed is False
        assert result.reasons == [_FUTURE_IMPORT_REJECT_MESSAGE]
        assert not any("SyntaxError" in r for r in result.reasons)

    async def test_validate_plan_accepts_the_same_body_without_future_import(self):
        content = _assembled_session_content(_SESSION_BODY)
        result = await validate_plan(
            content, plan_name="tiny_sweep", sample_args=BENIGN_SAMPLE_ARGS
        )
        assert result.passed is True, result.reasons


# =========================================================================
# Stage 2: narrowed CA/connector pattern scan
# =========================================================================


class TestCaPatternScan:
    @pytest.mark.parametrize(
        "code",
        [
            "caput('BEAM:CURRENT', 1.0)",
            "epics.caget('BEAM:CURRENT')",
            "write_channel('BEAM:CURRENT', 1.0)",
            "read_channel('BEAM:CURRENT')",
            "device._osprey_connector.write_channel('X', 1.0)",
            "pv = PV('BEAM:CURRENT')",
            "import aioca",
            "import caproto",
        ],
    )
    def test_flags_ca_constructs(self, code):
        assert _ca_pattern_scan(code) != []

    @pytest.mark.parametrize(
        "code",
        [
            "np.put(arr, [0], 1.0)",
            "numpy.put(arr, [0], 1.0)",
            "{}.get('missing')",
            "config.get('key', default)",
            "queue.put(1)",
            "q = queue.Queue()\nq.put(1)",
        ],
    )
    def test_does_not_flag_benign_put_get(self, code):
        assert _ca_pattern_scan(code) == []

    def test_bare_framework_default_patterns_would_have_false_positived(self):
        """Sanity check that this is a real narrowing, not a no-op: the
        framework's own default write patterns (`.put(`) DO match
        `numpy.put(...)` — confirming `_CA_ONLY_PATTERNS` deliberately
        excludes it rather than happening to not match by accident.
        """
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
        )

        default_result = detect_control_system_operations("np.put(arr, [0], 1.0)")
        assert default_result["has_writes"] is True  # the framework default DOES false-positive

        narrowed_result = detect_control_system_operations(
            "np.put(arr, [0], 1.0)", patterns=_CA_ONLY_PATTERNS, pattern_mode="override"
        )
        assert narrowed_result["has_writes"] is False


# =========================================================================
# Content hash helper
# =========================================================================


class TestHashPlanBody:
    def test_stable_and_deterministic(self):
        h1 = hash_plan_body("x = 1\n")
        h2 = hash_plan_body("x = 1\n")
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex digest

    def test_differs_for_different_content(self):
        assert hash_plan_body("x = 1\n") != hash_plan_body("x = 2\n")

    def test_normalizes_line_endings(self):
        assert hash_plan_body("x = 1\r\n") == hash_plan_body("x = 1\n")

    def test_normalizes_trailing_newline_variants_and_bom(self):
        """`hash_plan_body("x")`, `("x\\n")`, `("x\\n\\n\\n")`, and a
        BOM-prefixed `("\\ufeffx\\n")` must all hash IDENTICALLY: this is the
        cross-task key (2.2's store / 2.4's load gate / 2.5's promote gate
        match an in-memory body against an on-disk re-hash), so any of these
        harmless round-trip variations silently diverging would be a real
        mismatch, not a cosmetic one.
        """
        reference = hash_plan_body("x")
        assert hash_plan_body("x\n") == reference
        assert hash_plan_body("x\n\n\n") == reference
        assert hash_plan_body("﻿x\n") == reference

    def test_bom_and_extra_trailing_newlines_alone_still_differ_from_other_content(self):
        assert hash_plan_body("x\n\n\n") != hash_plan_body("y\n\n\n")


# =========================================================================
# Device-name bucketing for the dry-run's mock device factory
# =========================================================================


class TestCollectDeviceNames:
    def test_flat_correctors_and_detectors_fields(self):
        motors, detectors = _collect_device_names(
            {"correctors": ["c1", "c2"], "detectors": ["d1"], "span_a": 1.0, "num": 3}
        )
        assert motors == {"c1", "c2"}
        assert detectors == {"d1"}

    def test_nested_axes_setpoint_field(self):
        """Mirrors `grid_scan`'s PARAMS shape: setpoints nested under
        `axes[].setpoint`, not a flat field named "setpoints"."""
        motors, detectors = _collect_device_names(
            {
                "detectors": ["d1"],
                "axes": [
                    {"setpoint": "m1", "start": 0.0, "stop": 1.0, "num_points": 2},
                    {"setpoint": "m2", "start": 0.0, "stop": 1.0, "num_points": 3},
                ],
            }
        )
        assert motors == {"m1", "m2"}
        assert detectors == {"d1"}

    def test_empty_sample_args(self):
        assert _collect_device_names({}) == (set(), set())


# =========================================================================
# Stage 3: dry-run — environment scrub wiring (subprocess spawn mocked out)
# =========================================================================


class TestDryRunEnvScrub:
    async def test_epics_ca_vars_are_neutralized_in_the_subprocess_env(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Asserts `_dry_run` passes a neutralized env to
        `create_subprocess_exec` — the real subprocess spawn is mocked out so
        this test stays fast and doesn't need a real bluesky dry-run to prove
        the env wiring specifically.

        Deliberately asserts the SET values, not merely "key absent": a CA
        client that sees neither `EPICS_CA_ADDR_LIST` nor an explicit
        `EPICS_CA_AUTO_ADDR_LIST` defaults auto-discovery to YES and
        broadcasts on the local subnet looking for IOCs — so simply deleting
        these keys would have been worse than leaving them alone. The
        assertions below are what actually proves that gap is closed.
        """
        captured_env: dict[str, str] = {}

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return b"", b""

            def kill(self):
                pass

            async def wait(self):
                pass

        async def _fake_create_subprocess_exec(*args, **kwargs):
            captured_env.update(kwargs["env"])
            script_path = Path(args[1])
            result_path = script_path.parent / "result.json"
            result_path.write_text(json.dumps({"success": True}))
            return _FakeProc()

        monkeypatch.setenv("EPICS_CA_ADDR_LIST", "10.0.0.1")
        monkeypatch.setenv("EPICS_CA_NAME_SERVERS", "10.0.0.1:5064")
        monkeypatch.setenv("EPICS_CA_AUTO_ADDR_LIST", "YES")
        monkeypatch.setenv("EPICS_CA_SERVER_PORT", "5064")
        monkeypatch.setattr(
            plan_validation.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
        )

        reasons = await plan_validation._dry_run(
            BENIGN_PLAN_BODY, plan_name="tiny_sweep", sample_args=BENIGN_SAMPLE_ARGS, timeout=5.0
        )

        assert reasons == []
        assert captured_env["EPICS_CA_AUTO_ADDR_LIST"] == "NO"
        assert captured_env["EPICS_CA_ADDR_LIST"] == ""
        assert captured_env["EPICS_CA_NAME_SERVERS"] == ""
        for name in _EPICS_CA_ENV_NAMES_TO_DROP:
            assert name not in captured_env

    def test_inert_env_constants_are_actually_inert(self):
        """`_EPICS_CA_INERT_ENV` is the source of truth the assertions above
        rely on — pin its exact values so a future edit can't quietly
        reintroduce the broadcast-discovery gap this fix closed.
        """
        assert _EPICS_CA_INERT_ENV == {
            "EPICS_CA_ADDR_LIST": "",
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
            "EPICS_CA_NAME_SERVERS": "",
        }


# =========================================================================
# Full pipeline: validate_plan
# =========================================================================


class TestValidateBlueskyPlanRejects:
    async def test_rejects_at_stage_1_for_a_disallowed_import(self):
        result = await validate_plan("import epics\nepics.caput('X', 1)\n")
        assert result.passed is False
        assert any("epics" in r or "Import not allowed" in r for r in result.reasons)
        assert len(result.content_hash) == 64

    async def test_rejects_at_stage_2_for_a_ca_construct_with_no_import(self):
        # `caget(`/`read_channel(` are in `_CA_ONLY_PATTERNS` but NOT in
        # `_DANGEROUS_PATTERNS` (unlike `caput`/`write_channel`, which stage 1
        # already catches via its reused dangerous-pattern scan) — so this
        # body only gets caught once stage 2 actually runs.
        code = "PLAN_METADATA = {}\nvalue = read_channel('BEAM:CURRENT')\n"
        result = await validate_plan(code)
        assert result.passed is False
        assert any("Control-system operation" in r for r in result.reasons)

    async def test_stage_1_short_circuits_before_stage_2(self):
        """A body with BOTH a disallowed import AND a CA construct is
        rejected for the import (stage 1 never reaches stage 2)."""
        code = "import epics\nvalue = read_channel('BEAM:CURRENT')\n"
        result = await validate_plan(code)
        assert result.passed is False
        assert not any("Control-system operation" in r for r in result.reasons)


class TestValidateBlueskyPlanAccepts:
    async def test_benign_plan_passes_all_stages_and_drives_to_completion(self):
        result = await validate_plan(
            BENIGN_PLAN_BODY,
            plan_name="tiny_sweep",
            sample_args=BENIGN_SAMPLE_ARGS,
            dry_run_timeout=30.0,
        )

        assert result.passed is True, result.reasons
        assert result.reasons == []
        assert result.content_hash == hash_plan_body(BENIGN_PLAN_BODY)

    async def test_a_body_that_raises_at_dry_run_time_fails_stage_3_only(self):
        """Imports/patterns are clean, but the plan body itself blows up once
        actually driven — proves stage 3 catches runtime failures stages 1-2
        cannot (they never execute the body)."""
        code = textwrap.dedent(
            """\
            from pydantic import BaseModel

            PLAN_METADATA = {
                "name": "boom",
                "description": "raises at runtime",
                "category": "accelerator",
                "required_devices": [],
                "writes": False,
            }


            class PARAMS(BaseModel):
                pass


            def build_plan(devices, params):
                raise RuntimeError("boom")
                yield  # pragma: no cover - unreachable; makes this a generator
            """
        )
        result = await validate_plan(code, plan_name="boom", sample_args={})
        assert result.passed is False
        assert any("Dry-run failed" in r for r in result.reasons)


# =========================================================================
# Regression: the shipped exemplars (task 1.5) — THE format the
# writing-bluesky-plans skill tells authors to copy — must themselves pass
# validation. Stage 1 rejecting them for reasons unrelated to control-system
# safety (a missing `__future__`/`typing`/`logging` allowlist entry) would
# mean the documented reference format doesn't actually validate.
# =========================================================================

_PLANS_CORE_DIR = (
    Path(__file__).parents[3] / "src" / "osprey" / "services" / "bluesky_bridge" / "plans_core"
)


class TestShippedExemplarsPassValidation:
    def test_orm_source_passes_the_static_and_pattern_stages(self):
        source = (_PLANS_CORE_DIR / "orm.py").read_text(encoding="utf-8")
        assert _static_allowlist_check(source) == []
        assert _ca_pattern_scan(source) == []

    def test_grid_scan_source_passes_the_static_and_pattern_stages(self):
        source = (_PLANS_CORE_DIR / "grid_scan.py").read_text(encoding="utf-8")
        assert _static_allowlist_check(source) == []
        assert _ca_pattern_scan(source) == []

    async def test_orm_source_passes_full_validation_including_dry_run(self):
        source = (_PLANS_CORE_DIR / "orm.py").read_text(encoding="utf-8")
        result = await validate_plan(
            source,
            plan_name="orm",
            sample_args={
                "correctors": ["hcm1", "hcm2"],
                "detectors": ["bpm1", "bpm2"],
                "span_a": 2.0,
                "num": 3,
            },
        )
        assert result.passed is True, result.reasons

    async def test_grid_scan_source_passes_full_validation_including_dry_run(self):
        source = (_PLANS_CORE_DIR / "grid_scan.py").read_text(encoding="utf-8")
        result = await validate_plan(
            source,
            plan_name="grid_scan",
            sample_args={
                "detectors": ["det1"],
                "axes": [
                    {"setpoint": "motor1", "start": 0.0, "stop": 1.0, "num_points": 2},
                    {"setpoint": "motor2", "start": 0.0, "stop": 1.0, "num_points": 3},
                ],
            },
        )
        assert result.passed is True, result.reasons


# =========================================================================
# Documented, accepted residual: obfuscated imports are not a containment
# boundary (see module docstring — stages 1-2 are AST/regex checks, not a
# sandbox; the real backstop is human approval rendering the plan source at
# launch, task 2.6).
# =========================================================================


class TestKnownObfuscationResidual:
    @pytest.mark.xfail(
        reason=(
            "known-uncaught residual: a getattr/string-concatenation-obfuscated "
            "__import__ call evades both the AST import walk and the substring/"
            "regex pattern scan by construction (neither stage's source text nor "
            "AST ever contains a literal 'epics' import or CA-construct name) -- "
            "ACCEPTED, not a bug. See the module docstring's 'not a containment "
            "boundary' note; the real backstop is human approval at launch "
            "(task 2.6), not this validator."
        ),
        strict=True,
    )
    def test_obfuscated_import_evades_the_static_and_pattern_stages(self):
        code = textwrap.dedent(
            """\
            PLAN_METADATA = {}


            def build_plan(devices, params):
                _import_name = "".join(["__", "imp", "ort", "__"])
                _module_name = "".join(["ep", "ics"])
                _reflected = getattr(__builtins__, _import_name)(_module_name)
                return _reflected
            """
        )
        violations = _static_allowlist_check(code) + _ca_pattern_scan(code)
        assert violations != [], (
            "this obfuscated import is now being caught -- if that's a real "
            "improvement, update the module docstring's residual claim and "
            "flip this test's polarity rather than leaving it stale"
        )

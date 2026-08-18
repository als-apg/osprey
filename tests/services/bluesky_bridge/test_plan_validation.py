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

import inspect
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
from osprey.services.bluesky_bridge.plan_fields import (  # noqa: E402
    MOVABLE_ROLE,
    READABLE_ROLE,
    channel_roles,
    scan_metadata,
)
from osprey.services.bluesky_bridge.plan_validation import (  # noqa: E402
    _CA_ONLY_PATTERNS,
    _EPICS_CA_ENV_NAMES_TO_DROP,
    _EPICS_CA_INERT_ENV,
    _ca_pattern_scan,
    _static_allowlist_check,
    hash_plan_body,
    validate_plan,
)

# ---------------------------------------------------------------------------
# A tiny, fully-contract-compliant benign plan body: one movable channel, one
# readable channel, harmless (non-control-system) `.put`/`.get` usage that a
# pattern scan could mistake for a CA write/read. Reused across the
# accept-path and dry-run tests below.
#
# Its `PARAMS` declares both channel roles, which is what the stage-3 dry-run
# reads to decide which mock each channel name gets — an undeclared body would
# get no mocks for the names it passes and fail the dry-run by design. It
# declares that it moves channels, so it also stamps its run with the point
# count that declaration obliges it to (task 3.4's gate).
# ---------------------------------------------------------------------------
BENIGN_PLAN_BODY = textwrap.dedent(
    """\
    import numpy as np
    from bluesky import plan_stubs as bps
    from bluesky import preprocessors as bpp
    from pydantic import BaseModel, Field

    from osprey.services.bluesky_bridge.plan_fields import (
        MovableChannels,
        ReadableChannels,
        scan_metadata,
    )

    PLAN_METADATA = {
        "name": "tiny_sweep",
        "description": "Sweep one corrector, reading one BPM at each point.",
        "writes": True,
    }


    class PARAMS(BaseModel):
        correctors: MovableChannels = Field(..., min_length=1)
        readbacks: ReadableChannels = Field(..., min_length=1)
        num: int = Field(..., ge=1)


    def build_plan(devices, params):
        # Harmless, non-control-system ".get"/".put" usage that must NOT be
        # mistaken for a CA/connector read or write.
        {}.get("missing")
        arr = np.zeros(params.num)
        np.put(arr, list(range(params.num)), 1.0)

        corrector = devices[params.correctors[0]]
        bpm = devices[params.readbacks[0]]

        @bpp.stage_decorator([corrector, bpm])
        @bpp.run_decorator(
            md=scan_metadata(
                movable=params.correctors, readable=params.readbacks, points=params.num
            )
        )
        def _sweep():
            for i in range(params.num):
                yield from bps.mv(corrector, float(i))
                yield from bps.trigger_and_read([corrector, bpm])

        return _sweep()
    """
)

BENIGN_SAMPLE_ARGS = {"correctors": ["c1"], "readbacks": ["d1"], "num": 3}

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

    from osprey.services.bluesky_bridge.plan_fields import (
        MovableChannels,
        ReadableChannels,
        scan_metadata,
    )


    class PARAMS(BaseModel):
        correctors: MovableChannels = Field(..., min_length=1)
        readbacks: ReadableChannels = Field(..., min_length=1)
        num: int = Field(..., ge=1)


    def build_plan(devices, params):
        corrector = devices[params.correctors[0]]
        bpm = devices[params.readbacks[0]]

        @bpp.stage_decorator([corrector, bpm])
        @bpp.run_decorator(
            md=scan_metadata(
                movable=params.correctors, readable=params.readbacks, points=params.num
            )
        )
        def _sweep():
            for i in range(params.num):
                yield from bps.mv(corrector, float(i))
                yield from bps.trigger_and_read([corrector, bpm])

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
            "import osprey",  # bare osprey, no submodule
            "import osprey as o",
            "from osprey.connectors import epics",  # the control-system surface itself
            "from osprey.services.bluesky_bridge.queue_backend import QueueBackend",
            "import osprey.services.bluesky_bridge.queue_backend",
            "from osprey.services.bluesky_bridge.queue import Queue",
            "import osprey.services.bluesky_bridge.queue",
            "from osprey.services.bluesky_bridge.app import app",
            "import osprey.services.bluesky_bridge.app",
            "from osprey.services.python_executor import runner",
            # The parent package of an allowed leaf is NOT itself allowed: this
            # form binds the whole package, from which `figure` is one attribute
            # among all its siblings.
            "from osprey.services.bluesky_bridge import figure",
            "from osprey.services.bluesky_bridge import plan_fields",
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
            # The two inert modules a plan's `render()` needs to exist at all.
            "from osprey.services.bluesky_bridge.figure import Figure, Panel",
            "import osprey.services.bluesky_bridge.figure",
            "from osprey.services.bluesky_bridge.orm_analysis import build_response_matrix",
            # The role-typed field helpers a plan's `PARAMS` model needs to
            # declare `movable`/`readable` channel parameters.
            "from osprey.services.bluesky_bridge.plan_fields import MovableChannels",
            "import osprey.services.bluesky_bridge.plan_fields",
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

    def test_osprey_is_three_leaf_modules_and_nothing_else(self):
        """`osprey` is narrowed exactly as `bluesky` is: the top level is denied
        and three fully-dotted leaves are allowed.

        A plan file may declare a `render()`, and a `Figure` cannot be built
        without importing the module that defines it; a plan that solves
        between steps needs the arithmetic it runs -- but that is the whole of
        the widening. `figure` is pydantic-only, `orm_analysis` and
        `bump_analysis` are numpy-only; the connector, config, queue, and
        executor packages next door are precisely what this allowlist exists to
        keep out of an agent-authored plan body, and admitting bare `osprey`
        would hand a plan every one of them.
        """
        assert _static_allowlist_check("import osprey") != []
        assert _static_allowlist_check("from osprey import connectors") != []
        assert _static_allowlist_check("from osprey.services import bluesky_bridge") != []
        assert _static_allowlist_check("from osprey.services.bluesky_bridge import app") != []
        assert (
            _static_allowlist_check("from osprey.services.bluesky_bridge.figure import Figure")
            == []
        )
        assert (
            _static_allowlist_check(
                "from osprey.services.bluesky_bridge.orm_analysis import build_response_matrix"
            )
            == []
        )

    def test_bump_analysis_passes_stage_one_while_its_siblings_still_do_not(self):
        """`bump_analysis` joins `figure`/`orm_analysis` on the osprey allowlist
        because `orbit_bump_sweep`'s body has to import the arithmetic it runs
        between steps (`fit_probe_response`, `solve_offsets`), and a
        `plans_*/*.py` file is exec'd with no package -- the fully-dotted
        absolute spelling below is the only one that resolves at load.

        It is admitted on the same terms as the other two and no wider terms:
        the module is numpy-only, and widening the allowlist by one leaf must
        not widen it by a package. `queue` next door is the counter-example
        that pins that -- one dotted level away from the module now allowed,
        and still rejected.
        """
        body = textwrap.dedent(
            """\
            from bluesky import plan_stubs as bps
            from pydantic import BaseModel, Field

            from osprey.services.bluesky_bridge.bump_analysis import (
                fit_probe_response,
                solve_offsets,
            )


            class PARAMS(BaseModel):
                correctors: list[str] = Field(..., min_length=1)


            def build_plan(devices, params):
                yield from bps.null()
            """
        )
        assert _static_allowlist_check(body) == []
        assert _static_allowlist_check("import osprey.services.bluesky_bridge.bump_analysis") == []
        assert plan_validation._is_allowed_import("osprey.services.bluesky_bridge.bump_analysis")

        assert (
            _static_allowlist_check("from osprey.services.bluesky_bridge.queue import Queue") != []
        )
        assert not plan_validation._is_allowed_import("osprey.services.bluesky_bridge.queue")

    def test_the_validator_top_level_set_never_admits_osprey_on_its_own(self):
        """`_VALIDATOR_TOP_LEVEL_MODULES` carries `osprey` so
        `validate_sandbox_code`'s coarser top-level check agrees with the walk
        above -- but it is only ever safe paired with the submodule gate, and a
        future edit that drops the gate while keeping the set would admit every
        `osprey` submodule silently. Pin both halves together.
        """
        assert "osprey" in plan_validation._VALIDATOR_TOP_LEVEL_MODULES
        assert "osprey" not in plan_validation._ALLOWED_TOP_LEVEL_MODULES
        assert not plan_validation._is_allowed_import("osprey")
        assert not plan_validation._is_allowed_import("osprey.connectors")
        assert plan_validation._is_allowed_import("osprey.services.bluesky_bridge.figure")
        assert plan_validation._is_allowed_import("osprey.services.bluesky_bridge.plan_fields")

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
        cross-task key (2.2's store / 2.4's load gate / 2.5's launch gate
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
# Task 2.5: which mock the dry-run builds for a channel is READ FROM THE
# PLAN'S OWN DECLARATION, never guessed from a field's name.
#
# These drive the real subprocess rather than unit-testing a bridge-side
# bucketing helper, because there is no longer one to test: the collection
# needs the body's `PARAMS` class and its validated params, and the bridge
# process never loads a plan body. Each body below asserts, from inside
# `build_plan`, exactly which mock devices it was handed — so a PASS is the
# assertion that the bucketing was right, and a FAIL carries the mismatch.
# =========================================================================

_ROLE_MOCK_BODY_HEADER = textwrap.dedent(
    """\
    from bluesky import plan_stubs as bps
    from bluesky import preprocessors as bpp
    from pydantic import BaseModel, Field

    from osprey.services.bluesky_bridge.plan_fields import (
        MovableChannel,
        MovableChannels,
        ReadableChannels,
        scan_metadata,
    )

    PLAN_METADATA = {
        "name": "role_mocks",
        "description": "Asserts the mock devices the dry-run built for it.",
        "writes": True,
    }
    """
)

_ASSERT_MOCK_KINDS = textwrap.dedent(
    """\
    def _assert_mock_kinds(devices, expected):
        kinds = {name: type(device).__name__ for name, device in devices.items()}
        if kinds != expected:
            raise AssertionError(f"mock devices were {kinds}, expected {expected}")
    """
)


def _role_mock_body(params_source: str, build_plan_source: str) -> str:
    """A complete, allowlist-clean plan body around a role-declaring `PARAMS`.

    Both arguments arrive already dedented — they are spliced together, and a
    single `dedent` over the join would key off whichever fragment happens to
    be least indented.
    """
    return (
        _ROLE_MOCK_BODY_HEADER
        + "\n\n"
        + params_source
        + "\n\n"
        + _ASSERT_MOCK_KINDS
        + "\n\n"
        + build_plan_source
    )


# The tail every role-mock body shares: drive one point through a run that
# declares its own size, as a body whose `PLAN_METADATA` says it moves channels
# must (task 3.4). `_DRIVE_ONE_POINT_UNDECLARED` is the same drive without the
# stamp — the gate's negative case, and nothing else's.
_DRIVE_ONE_POINT = """\
    movable = devices[params.correctors[0]]
    readable = devices[params.readbacks[0]]

    @bpp.stage_decorator([movable, readable])
    @bpp.run_decorator(
        md=scan_metadata(movable=params.correctors, readable=params.readbacks, points=1)
    )
    def _sweep():
        yield from bps.mv(movable, 1.0)
        yield from bps.trigger_and_read([movable, readable])

    return _sweep()
"""

_DRIVE_ONE_POINT_UNDECLARED = """\
    movable = devices[params.correctors[0]]
    readable = devices[params.readbacks[0]]

    @bpp.stage_decorator([movable, readable])
    @bpp.run_decorator()
    def _sweep():
        yield from bps.mv(movable, 1.0)
        yield from bps.trigger_and_read([movable, readable])

    return _sweep()
"""


class TestDryRunRoleMocks:
    async def test_movable_gets_a_motor_mock_and_readable_a_detector_mock(self):
        """The two roles bucket into the two mock classes, and a string under a
        field that declared NO role is a plain parameter — never a device."""
        body = _role_mock_body(
            textwrap.dedent(
                """\
                class PARAMS(BaseModel):
                    correctors: MovableChannels = Field(..., min_length=1)
                    readbacks: ReadableChannels = Field(..., min_length=1)
                    label: str = Field(...)
                """
            ),
            textwrap.dedent(
                """\
                def build_plan(devices, params):
                    _assert_mock_kinds(
                        devices, {"c1": "MockSettable", "c2": "MockSettable", "b1": "MockReadable"}
                    )
                """
            )
            + _DRIVE_ONE_POINT,
        )
        result = await validate_plan(
            body,
            plan_name="role_mocks",
            sample_args={
                "correctors": ["c1", "c2"],
                "readbacks": ["b1"],
                # A device-shaped string under a role-less field. Mocking it
                # would be the old name-guessing behavior returning by another
                # route, so its absence from the mapping above IS the assertion.
                "label": "looks_like_a_device",
            },
        )
        assert result.passed is True, result.reasons

    async def test_nested_single_channel_is_collected(self):
        """A `GridAxis.setpoint`-style channel — one name per entry of a nested
        model, not a flat list — is mocked from the same declaration."""
        body = _role_mock_body(
            textwrap.dedent(
                """\
                class Axis(BaseModel):
                    setpoint: MovableChannel
                    start: float


                class PARAMS(BaseModel):
                    axes: list[Axis] = Field(..., min_length=1)
                    readbacks: ReadableChannels = Field(..., min_length=1)
                """
            ),
            textwrap.dedent(
                """\
                def build_plan(devices, params):
                    _assert_mock_kinds(
                        devices, {"m1": "MockSettable", "m2": "MockSettable", "b1": "MockReadable"}
                    )
                    movable = devices[params.axes[0].setpoint]
                    readable = devices[params.readbacks[0]]

                    @bpp.stage_decorator([movable, readable])
                    @bpp.run_decorator(
                        md=scan_metadata(
                            movable=[axis.setpoint for axis in params.axes],
                            readable=params.readbacks,
                            points=1,
                        )
                    )
                    def _sweep():
                        yield from bps.mv(movable, 1.0)
                        yield from bps.trigger_and_read([movable, readable])

                    return _sweep()
                """
            ),
        )
        result = await validate_plan(
            body,
            plan_name="role_mocks",
            sample_args={
                "axes": [{"setpoint": "m1", "start": 0.0}, {"setpoint": "m2", "start": 1.0}],
                "readbacks": ["b1"],
            },
        )
        assert result.passed is True, result.reasons

    async def test_untagged_device_field_fails_naming_the_available_mocks(self):
        """SC-2: a channel the plan never declared gets no mock, and the failure
        says so — naming the sorted mocks that DO exist, the same legibility the
        queue worker's plan wrappers give an unknown device name."""
        body = _role_mock_body(
            textwrap.dedent(
                """\
                class PARAMS(BaseModel):
                    correctors: MovableChannels = Field(..., min_length=1)
                    readbacks: ReadableChannels = Field(..., min_length=1)
                    reference: str = Field(...)
                """
            ),
            textwrap.dedent(
                """\
                def build_plan(devices, params):
                    _reference = devices[params.reference]
                """
            )
            + _DRIVE_ONE_POINT,
        )
        result = await validate_plan(
            body,
            plan_name="role_mocks",
            sample_args={"correctors": ["c1"], "readbacks": ["b1"], "reference": "undeclared1"},
        )
        assert result.passed is False
        assert len(result.reasons) == 1
        reason = result.reasons[0]
        assert "undeclared1" in reason
        assert "available mock devices: ['b1', 'c1']" in reason
        assert "role_mocks" in reason

    async def test_a_plan_declaring_no_roles_gets_only_the_fallback_mocks(self):
        """A body whose `PARAMS` declares nothing gets one mock of each kind and
        nothing else — the device-shaped strings it passes are just strings.

        This is the old name-guessing rule's negative control: `correctors` and
        `detectors` used to be mocked purely because of what they were called.
        """
        body = textwrap.dedent(
            """\
            from bluesky import plan_stubs as bps
            from bluesky import preprocessors as bpp
            from pydantic import BaseModel, Field

            PLAN_METADATA = {
                "name": "undeclared",
                "description": "Declares no channel roles at all.",
                "writes": True,
            }


            class PARAMS(BaseModel):
                correctors: list[str] = Field(..., min_length=1)
                detectors: list[str] = Field(..., min_length=1)


            def build_plan(devices, params):
                movable = devices[params.correctors[0]]
                readable = devices[params.detectors[0]]

                @bpp.stage_decorator([movable, readable])
                @bpp.run_decorator()
                def _sweep():
                    yield from bps.mv(movable, 1.0)
                    yield from bps.trigger_and_read([movable, readable])

                return _sweep()
            """
        )
        result = await validate_plan(
            body,
            plan_name="undeclared",
            sample_args={"correctors": ["c1"], "detectors": ["d1"]},
        )
        assert result.passed is False
        assert "available mock devices: ['rb1', 'sp1']" in result.reasons[0]

    def test_fallback_mock_names_are_the_mock_factory_defaults(self):
        """The fallback names the render seam injects are the ones
        `devices/mock.build_devices` builds by default — the failure message
        above quotes them, so a drift here would make it name devices that
        don't exist.
        """
        from osprey.services.bluesky_bridge.devices import mock as mock_devices

        signature = inspect.signature(mock_devices.build_devices)
        assert (
            tuple(signature.parameters["settable_names"].default)
            == plan_validation._FALLBACK_MOVABLE_MOCKS
        )
        assert (
            tuple(signature.parameters["readable_names"].default)
            == plan_validation._FALLBACK_READABLE_MOCKS
        )


# =========================================================================
# Task 3.4: a plan that says it MOVES channels must leave an operator able to
# see how far along it is — its run has to declare its own point count.
#
# The gate lives on the session tier and only here: a shipped/preset/facility
# plan is reviewed by whoever installs it, while a session plan is written on
# the spot and this validator is the only thing that reads it before it runs.
# A read-only plan is never asked for a count.
# =========================================================================

_GATE_PARAMS = textwrap.dedent(
    """\
    class PARAMS(BaseModel):
        correctors: MovableChannels = Field(..., min_length=1)
        readbacks: ReadableChannels = Field(..., min_length=1)
    """
)

_GATE_SAMPLE_ARGS = {"correctors": ["c1"], "readbacks": ["b1"]}

_READ_ONLY_BODY = textwrap.dedent(
    """\
    from bluesky import plan_stubs as bps
    from bluesky import preprocessors as bpp
    from pydantic import BaseModel, Field

    from osprey.services.bluesky_bridge.plan_fields import ReadableChannels

    PLAN_METADATA = {
        "name": "read_only",
        "description": "Record every declared channel once, moving nothing.",
        "writes": False,
    }


    class PARAMS(BaseModel):
        readbacks: ReadableChannels = Field(..., min_length=1)


    def build_plan(devices, params):
        readable = devices[params.readbacks[0]]

        @bpp.stage_decorator([readable])
        @bpp.run_decorator()
        def _read():
            yield from bps.trigger_and_read([readable])

        return _read()
    """
)


class TestDeclaredPointCountGate:
    async def test_a_moving_plan_that_declares_its_point_count_passes(self):
        body = _role_mock_body(
            _GATE_PARAMS,
            "def build_plan(devices, params):\n" + _DRIVE_ONE_POINT,
        )
        result = await validate_plan(body, plan_name="role_mocks", sample_args=_GATE_SAMPLE_ARGS)
        assert result.passed is True, result.reasons

    async def test_a_moving_plan_whose_run_declares_nothing_is_rejected(self):
        """Same body, same run — only the declaration is missing. The message
        has to hand the author the call that fixes it, not describe a document
        key they never write by hand.
        """
        body = _role_mock_body(
            _GATE_PARAMS,
            "def build_plan(devices, params):\n" + _DRIVE_ONE_POINT_UNDECLARED,
        )
        result = await validate_plan(body, plan_name="role_mocks", sample_args=_GATE_SAMPLE_ARGS)
        assert result.passed is False
        assert len(result.reasons) == 1
        reason = result.reasons[0]
        assert "scan_metadata(" in reason
        assert "point count" in reason
        assert "role_mocks" in reason
        # Nothing crashed — this is a contract rejection, not a runtime failure.
        assert "Dry-run failed" not in reason

    async def test_a_read_only_plan_is_never_asked_for_a_point_count(self):
        """The gate keys off what the plan says it does. A plan that only reads
        has no move for an operator to track, so its unstamped run passes —
        this is the negative control that the gate is not simply "every run
        must be stamped".
        """
        result = await validate_plan(
            _READ_ONLY_BODY, plan_name="read_only", sample_args={"readbacks": ["b1"]}
        )
        assert result.passed is True, result.reasons

    async def test_a_moving_plan_that_opens_no_run_at_all_is_rejected(self):
        """DECISION: a plan declaring that it moves channels while opening no
        run is rejected by this same gate, with the same remedy. It moves a
        device with nothing recording it — no run, no documents, nothing for an
        operator to watch or for a consumer to read afterwards. Treating "no
        run" as "nothing to declare" would let exactly the plan an operator can
        least see through the gate that exists to keep it visible.
        """
        body = _role_mock_body(
            _GATE_PARAMS,
            textwrap.dedent(
                """\
                def build_plan(devices, params):
                    movable = devices[params.correctors[0]]
                    yield from bps.mv(movable, 1.0)
                """
            ),
        )
        result = await validate_plan(body, plan_name="role_mocks", sample_args=_GATE_SAMPLE_ARGS)
        assert result.passed is False
        assert len(result.reasons) == 1
        reason = result.reasons[0]
        assert "opened no run" in reason
        assert "scan_metadata(" in reason

    def test_the_gated_value_is_the_one_the_authoring_helper_stamps(self):
        """The gate reads what `scan_metadata()` writes. Pinned so the two
        cannot drift into a gate that rejects correctly-authored plans.
        """
        stamped = scan_metadata(movable=["c1"], readable=["b1"], points=7)
        assert stamped[plan_validation._DECLARED_POINT_COUNT_KEY] == 7

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("PLAN_METADATA = {'name': 'x', 'description': 'y', 'writes': True}\n", True),
            ("PLAN_METADATA = {'name': 'x', 'description': 'y', 'writes': False}\n", False),
            ("PLAN_METADATA = {}\n", False),
            ("value = 1\n", False),
            # Not a literal the static read can trust — the gate declines
            # rather than guessing, and the load gate still cross-checks the
            # declaration against the plan's role-typed fields.
            ("WRITES = True\nPLAN_METADATA = {'writes': WRITES}\n", False),
        ],
    )
    def test_the_writes_declaration_is_read_statically_off_the_body(self, body, expected):
        assert plan_validation._declares_writes(body) is expected


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
            # Stands in for what the real script writes, declared point count
            # included: `BENIGN_PLAN_BODY` says it moves channels, so a payload
            # without one would be rejected by the point-count gate and this
            # test would be asserting the wrong thing about the env wiring.
            result_path.write_text(json.dumps({"success": True, "declared_points": [3]}))
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


def _declared_channel_fields(model_cls) -> tuple[str, str]:
    """An exemplar's top-level movable and readable field names, read from its
    own role declaration.

    The dry-run builds its mocks from that declaration, so sample args that
    named the fields by a spelling hard-coded here would stop matching the
    moment a shipped plan renames one. A role declared only on a nested model
    (`grid_scan`'s ``axes[].setpoint``) has no top-level field and comes back
    empty — but an exemplar declaring no role at all has nothing to build mocks
    from and could not be dry-run, so that fails here rather than surfacing as
    an inscrutable missing-device failure inside the subprocess.
    """
    roles = channel_roles(model_cls)
    assert roles, (
        f"{model_cls.__module__} declares no channel roles; the dry-run has nothing to "
        "build mock devices from, and the writing-bluesky-plans contract requires them"
    )
    top_level = {role: path for path, role in roles if "." not in path}
    return top_level.get(MOVABLE_ROLE, ""), top_level.get(READABLE_ROLE, "")


class TestShippedExemplarsPassValidation:
    def test_orm_source_passes_the_static_and_pattern_stages(self):
        source = (_PLANS_CORE_DIR / "orm.py").read_text(encoding="utf-8")
        assert _static_allowlist_check(source) == []
        assert _ca_pattern_scan(source) == []

    def test_grid_scan_source_passes_the_static_and_pattern_stages(self):
        source = (_PLANS_CORE_DIR / "grid_scan.py").read_text(encoding="utf-8")
        assert _static_allowlist_check(source) == []
        assert _ca_pattern_scan(source) == []

    def test_orbit_bump_sweep_source_passes_the_static_and_pattern_stages(self):
        source = (_PLANS_CORE_DIR / "orbit_bump_sweep.py").read_text(encoding="utf-8")
        assert _static_allowlist_check(source) == []
        assert _ca_pattern_scan(source) == []

    async def test_orm_source_passes_full_validation_including_dry_run(self):
        """The dry-run mocks an exemplar's channels from its own declaration, so
        the sample args name the exemplar's declared channel fields rather than
        a spelling hard-coded here — that survives the field renames the
        contract cutover makes to the shipped plans.
        """
        from osprey.services.bluesky_bridge.plans_core import orm

        movable_field, readable_field = _declared_channel_fields(orm.PARAMS)
        source = (_PLANS_CORE_DIR / "orm.py").read_text(encoding="utf-8")
        result = await validate_plan(
            source,
            plan_name="orm",
            sample_args={
                movable_field: ["hcm1", "hcm2"],
                readable_field: ["bpm1", "bpm2"],
                "span_a": 2.0,
                "num": 3,
            },
        )
        assert result.passed is True, result.reasons

    async def test_grid_scan_source_passes_full_validation_including_dry_run(self):
        from osprey.services.bluesky_bridge.plans_core import grid_scan

        _movable_field, readable_field = _declared_channel_fields(grid_scan.PARAMS)
        source = (_PLANS_CORE_DIR / "grid_scan.py").read_text(encoding="utf-8")
        result = await validate_plan(
            source,
            plan_name="grid_scan",
            sample_args={
                readable_field: ["det1"],
                "axes": [
                    {"setpoint": "motor1", "start": 0.0, "stop": 1.0, "num_points": 2},
                    {"setpoint": "motor2", "start": 0.0, "stop": 1.0, "num_points": 3},
                ],
            },
        )
        assert result.passed is True, result.reasons

    async def test_orbit_bump_sweep_source_passes_full_validation_including_dry_run(self):
        """The zero-offset dry run: the only bump a physics-free mock can honestly
        pass, and it still walks the plan's whole structure.

        Mock devices have no orbit response — writing a corrector moves no BPM —
        so the probe measures an identically zero response matrix. Asking for a
        real displacement through that would (correctly) abort in `solve_offsets`
        as a degenerate bump, which would say nothing about whether the plan body
        runs. Zero-offset `targets` instead put the run on the demand gate's
        trivially-converged path, where every step's solution is zero and no
        solve is ever attempted — but the profile is still walked in full, so
        this dry run exercises the baseline reads, the probe writes (which run
        before the gate is evaluated), all `2 * num` amplitude steps, the
        terminal working-point verification, and the restore.

        The mock factory builds one device per declared channel, split by the
        roles the plan's `PARAMS` declares (`collect_channels`): the correctors
        are the movables, every BPM name a readable. A mock readable counts up
        on every trigger, so the three baseline reads carry σ = 1.0 — real
        noise the plan's own floor gate measures — which is why the sample
        `tolerance` sits above twice that, and why `best_effort` records the
        counter's drift at each step instead of failing the sweep on it.
        """
        source = (_PLANS_CORE_DIR / "orbit_bump_sweep.py").read_text(encoding="utf-8")
        result = await validate_plan(
            source,
            plan_name="orbit_bump_sweep",
            sample_args={
                "correctors": ["hcm1", "hcm2", "hcm3"],
                "targets": [{"readback": "bpm1", "value": 0.0}],
                "closure_readbacks": ["bpm2", "bpm3"],
                "readbacks": [],
                "num": 2,
                "baseline_reads": 3,
                "probe_amplitude": 0.1,
                "tolerance": 10.0,
                "max_trim_iterations": 1,
                "best_effort": True,
                "settle_s": 0.0,
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

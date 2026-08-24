"""p4p / PVAccess coverage in the framework circumvention patterns.

The framework pattern list is the safety net that catches an agent reaching a
control system directly instead of through ``osprey.runtime``. Until p4p was
added it covered pyepics, PyTango and LabVIEW only, so a PVAccess put was
caught by accident (the generic ``\\.put\\s*\\(``) and a ``ctxt.rpc(...)`` or a
``SharedPV`` server was not caught at all.

These tests pin three things: every new regex fires on code an agent would
plausibly write, none of them fires on ordinary analysis code (notably
``requests.post``), and each one lives in the write or the read list on
purpose - the p4p idioms are operation-agnostic, so the assignment cannot be
left implied.
"""

import re

import pytest

from osprey.services.python_executor.analysis.pattern_detection import (
    detect_control_system_operations,
    get_framework_standard_patterns,
)

# --- the new p4p entries, by list -------------------------------------------

P4P_WRITE_PATTERNS = [
    r"\bp4p\b[\s\S]*?\.put\s*\(",
    r"\bp4p\b[\s\S]*?\.post\s*\(",
    r"\.rpc\s*\(",
    r"\bSharedPV\b",
]

P4P_READ_PATTERNS = [
    r"\bp4p\b[\s\S]*?\.get\s*\(",
    r"\bp4p\b[\s\S]*?\.monitor\s*\(",
    r"\bContext\s*\(",
]


def detect(code: str) -> dict:
    """Detect against the framework patterns only, ignoring any local config."""
    return detect_control_system_operations(
        code,
        patterns=get_framework_standard_patterns(),
        pattern_mode="override",
        control_system_type="epics",
    )


# ============================================================================
# List assignment - which half each new pattern belongs to
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("pattern", P4P_WRITE_PATTERNS)
def test_p4p_write_patterns_are_in_the_write_list(pattern):
    """Put, post, rpc and SharedPV are writes, not reads."""
    patterns = get_framework_standard_patterns()

    assert pattern in patterns["write"]
    assert pattern not in patterns["read"]


@pytest.mark.unit
@pytest.mark.parametrize("pattern", P4P_READ_PATTERNS)
def test_p4p_read_patterns_are_in_the_read_list(pattern):
    """Get, monitor and context creation are reads - creating a Context alone
    puts nothing on the wire, so it must not force an approval prompt."""
    patterns = get_framework_standard_patterns()

    assert pattern in patterns["read"]
    assert pattern not in patterns["write"]


@pytest.mark.unit
def test_no_bare_post_pattern():
    """A bare ``.post(`` would flag every requests.post() in analysis code.

    The existing generic ``\\.put\\s*\\(`` collision is not a precedent to
    extend: the p4p post spelling is only matched when p4p is named too.
    """
    patterns = get_framework_standard_patterns()

    assert r"\.post\s*\(" not in patterns["write"]
    assert r"\.post\s*\(" not in patterns["read"]


@pytest.mark.unit
def test_every_p4p_pattern_compiles():
    """Invalid regexes are swallowed at match time - catch them here instead."""
    for pattern in P4P_WRITE_PATTERNS + P4P_READ_PATTERNS:
        re.compile(pattern)


# ============================================================================
# Writes - realistic p4p code an agent could produce
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "import p4p.client.thread\np4p.client.thread.Context('pva').put('SR:BEND:SP', 1.5)\n",
            id="thread-qualified-put",
        ),
        pytest.param(
            "import p4p.client.asyncio\n"
            "await p4p.client.asyncio.Context('pva').put('SR:BEND:SP', 1.5)\n",
            id="asyncio-qualified-put",
        ),
        pytest.param(
            "import p4p.client.cothread\n"
            "p4p.client.cothread.Context('pva').put('SR:BEND:SP', 1.5)\n",
            id="cothread-qualified-put",
        ),
        pytest.param(
            "import p4p.client.thread\n"
            "ctxt = p4p.client.thread.Context('pva')\n"
            "ctxt.put(['SR:A:SP', 'SR:B:SP'], [1.0, 2.0])\n",
            id="batch-put",
        ),
        pytest.param(
            "from p4p.client.thread import Context\nctxt = Context('pva')\nctxt.put('SR:A:SP', 1)\n",
            id="from-import-put",
        ),
        pytest.param(
            "import p4p.client.thread\n"
            "reply = p4p.client.thread.Context('pva').rpc('SR:CALC', args)\n",
            id="qualified-rpc",
        ),
        pytest.param(
            "from p4p.client.asyncio import Context\n"
            "ctxt = Context('pva')\n"
            "reply = await ctxt.rpc('SR:CALC', args)\n",
            id="aliased-rpc",
        ),
        pytest.param(
            "from p4p.server.thread import SharedPV\n"
            "from p4p.nt import NTScalar\n"
            "pv = SharedPV(nt=NTScalar('d'), initial=0.0)\n",
            id="sharedpv-construction",
        ),
        pytest.param(
            "import p4p.server.thread\npv = p4p.server.thread.SharedPV()\npv.post(42.0)\n",
            id="qualified-post",
        ),
    ],
)
def test_p4p_write_code_is_detected_as_a_write(code):
    result = detect(code)

    assert result["has_writes"] is True, f"no write pattern matched:\n{code}"


@pytest.mark.unit
@pytest.mark.parametrize("pattern", P4P_WRITE_PATTERNS)
def test_each_new_write_pattern_fires_on_some_snippet(pattern):
    """No dead entries: every new write regex earns its place."""
    snippets = [
        "import p4p.client.thread\np4p.client.thread.Context('pva').put('SR:A:SP', 1)\n",
        "import p4p.server.thread\npv = p4p.server.thread.SharedPV()\npv.post(42.0)\n",
        "from p4p.client.thread import Context\nreply = Context('pva').rpc('SR:CALC', args)\n",
        "from p4p.server.thread import SharedPV\npv = SharedPV()\n",
    ]

    assert any(re.search(pattern, code) for code in snippets)


# ============================================================================
# Reads
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "import p4p.client.thread\nval = p4p.client.thread.Context('pva').get('SR:CURRENT')\n",
            id="qualified-get",
        ),
        pytest.param(
            "import p4p.client.asyncio\n"
            "sub = p4p.client.asyncio.Context('pva').monitor('SR:IMAGE', cb)\n",
            id="qualified-monitor",
        ),
        pytest.param(
            "from p4p.client.cothread import Context\n"
            "ctxt = Context('pva')\n"
            "sub = ctxt.monitor('SR:IMAGE', cb)\n",
            id="context-creation-then-monitor",
        ),
    ],
)
def test_p4p_read_code_is_detected_as_a_read(code):
    result = detect(code)

    assert result["has_reads"] is True, f"no read pattern matched:\n{code}"


@pytest.mark.unit
def test_context_creation_alone_is_not_a_write():
    """Opening a client context puts nothing on the wire."""
    code = "from p4p.client.thread import Context\nctxt = Context('pva')\n"

    result = detect(code)

    assert result["has_reads"] is True
    assert result["has_writes"] is False


@pytest.mark.unit
@pytest.mark.parametrize("pattern", P4P_READ_PATTERNS)
def test_each_new_read_pattern_fires_on_some_snippet(pattern):
    snippets = [
        "import p4p.client.thread\nval = p4p.client.thread.Context('pva').get('SR:CURRENT')\n",
        "import p4p.client.asyncio\np4p.client.asyncio.Context('pva').monitor('SR:IMAGE', cb)\n",
        "from p4p.client.thread import Context\nctxt = Context('pva')\n",
    ]

    assert any(re.search(pattern, code) for code in snippets)


# ============================================================================
# No false positives on ordinary analysis code
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "import requests\nrequests.post('https://example.org/api', json=payload)\n",
            id="requests-post",
        ),
        pytest.param(
            "resp = session.post(url, data={'run': 12})\n",
            id="session-post",
        ),
        pytest.param(
            "import numpy as np\nmean = np.mean(samples)\nprint(f'mean={mean:.3f}')\n",
            id="numpy-analysis",
        ),
        pytest.param(
            "import ssl\nctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)\n",
            id="ssl-context",
        ),
    ],
)
def test_ordinary_code_does_not_fire_any_new_p4p_pattern(code):
    """The new entries must be silent on code that has nothing to do with p4p."""
    for pattern in P4P_WRITE_PATTERNS + P4P_READ_PATTERNS:
        assert re.search(pattern, code) is None, f"{pattern!r} matched:\n{code}"


@pytest.mark.unit
def test_post_in_code_that_also_uses_p4p_is_treated_as_a_write():
    """Deliberate and conservative: once p4p is in scope, a post is a write.

    The p4p anchor spans the whole snippet rather than a single line, because
    the realistic server spelling puts the import and the ``pv.post(...)`` call
    on different lines. The cost is that a script which talks PVAccess *and*
    posts to an HTTP endpoint asks for approval it may not need - the safe
    direction to be wrong in.
    """
    code = (
        "import requests\n"
        "from p4p.client.thread import Context\n"
        "val = Context('pva').get('SR:CURRENT')\n"
        "requests.post('https://example.org/api', json={'v': val})\n"
    )

    result = detect(code)

    assert result["has_writes"] is True


@pytest.mark.unit
def test_requests_post_is_not_a_control_system_write():
    """The whole reason there is no bare ``.post(`` entry."""
    code = "import requests\nrequests.post('https://example.org/api', json=payload)\n"

    result = detect(code)

    assert result["has_writes"] is False

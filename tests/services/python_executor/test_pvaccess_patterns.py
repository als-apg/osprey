"""pvaPy (``pvaccess``) coverage in the framework circumvention patterns.

pvaPy speaks both Channel Access and PVAccess, so it is a direct client an
agent can reach for on any EPICS deployment that has it installed. Before
these entries a plain ``Channel.put()`` was caught by accident (the generic
``\\.put\\s*\\(``), while the typed setters, ``asyncPut``, ``parsePut``,
``RpcClient.invoke`` and the whole monitor family went undetected.

As in ``test_p4p_patterns.py``, these tests pin three things: every new regex
fires on code an agent would plausibly write, none fires on ordinary analysis
code, and each lives in the write or the read list on purpose.
"""

import re

import pytest

from osprey.services.python_executor.analysis.pattern_detection import (
    detect_control_system_operations,
    get_framework_standard_patterns,
)

# --- the new pvaccess entries, by list --------------------------------------

PVACCESS_WRITE_PATTERNS = [
    r"\.put(?:Get|Boolean|Byte|Double|Float|Int|Long|Short|String|ScalarArray"
    r"|UByte|UInt|ULong|UShort|AsDoubleArray)\w*\s*\(",
    r"\.asyncPut\s*\(",
    r"\.parsePut\w*\s*\(",
    r"\bRpcClient\s*\(",
    r"\bpvaccess\b[\s\S]*?\.invoke\s*\(",
    r"\b(?:PvaServer|PvaMirrorServer|RpcServer|CaIoc)\b",
]

PVACCESS_READ_PATTERNS = [
    r"\bpvaccess\b[\s\S]*?\.(?:get|asyncGet|getPut|getAsDoubleArray)\s*\(",
    r"\bpvaccess\b[\s\S]*?\.(?:monitor|monitorAsDoubleArray|qMonitor|subscribe|startMonitor)\s*\(",
    r"\bpvaccess\b[\s\S]*?\b(?:Multi)?Channel\s*\(",
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


@pytest.mark.parametrize("pattern", PVACCESS_WRITE_PATTERNS)
def test_pvaccess_write_patterns_are_in_the_write_list(pattern):
    """The put family, rpc and the servers are writes, as in p4p."""
    patterns = get_framework_standard_patterns()

    assert pattern in patterns["write"]
    assert pattern not in patterns["read"]


@pytest.mark.parametrize("pattern", PVACCESS_READ_PATTERNS)
def test_pvaccess_read_patterns_are_in_the_read_list(pattern):
    """Get, the monitor family and channel creation are reads - creating a
    Channel alone puts nothing on the wire, so it must not force approval."""
    patterns = get_framework_standard_patterns()

    assert pattern in patterns["read"]
    assert pattern not in patterns["write"]


def test_the_setter_pattern_names_every_put_the_installed_binding_has():
    """The setters are named rather than matched as any camelCase put, so a
    setter the binding adds would go unseen; the installed one has none."""
    pvaccess = pytest.importorskip("pvaccess", reason="pvaccess is not installed")
    writes = get_framework_standard_patterns()["write"]
    for cls in (pvaccess.Channel, pvaccess.MultiChannel):
        for name in dir(cls):
            if name.startswith(("put", "asyncPut", "parsePut")):
                code = f"x.{name}(1)"
                assert any(re.search(p, code) for p in writes), f"{name} is not detected"


def test_every_pvaccess_pattern_compiles():
    """Invalid regexes are swallowed at match time - catch them here instead."""
    for pattern in PVACCESS_WRITE_PATTERNS + PVACCESS_READ_PATTERNS:
        re.compile(pattern)


# ============================================================================
# Writes - realistic pvaPy code an agent could produce
# ============================================================================


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "import pvaccess\npvaccess.Channel('SR:BEND:SP').put(1.5)\n",
            id="qualified-put",
        ),
        pytest.param(
            "import pvaccess as pva\nch = pva.Channel('SR:BEND:SP', pva.CA)\nch.putDouble(1.5)\n",
            id="aliased-typed-setter",
        ),
        pytest.param(
            "from pvaccess import Channel\nch = Channel('SR:BEND:SP')\nch.putScalarArray([1, 2])\n",
            id="from-import-array-setter",
        ),
        pytest.param(
            "import pvaccess\nch = pvaccess.Channel('SR:BEND:SP')\nch.putGet(1.5)\n",
            id="put-get",
        ),
        pytest.param(
            "import pvaccess\n"
            "ch = pvaccess.Channel('SR:BEND:SP')\n"
            "ch.asyncPut(pvaccess.PvObject({'value': pvaccess.DOUBLE}), cb, err)\n",
            id="async-put",
        ),
        pytest.param(
            "import pvaccess\nch = pvaccess.Channel('SR:BEND:SP')\nch.parsePut(['value=1.5'])\n",
            id="parse-put",
        ),
        pytest.param(
            "import pvaccess\n"
            "mc = pvaccess.MultiChannel(['SR:A:SP', 'SR:B:SP'])\n"
            "mc.putAsDoubleArray([1.0, 2.0])\n",
            id="multichannel-put",
        ),
        pytest.param(
            "import pvaccess\nreply = pvaccess.RpcClient('SR:CALC').invoke(request)\n",
            id="qualified-rpc",
        ),
        pytest.param(
            "from pvaccess import RpcClient as R\nclient = R('SR:CALC')\nclient.invoke(request)\n",
            id="aliased-rpc",
        ),
        pytest.param(
            "import pvaccess\nserver = pvaccess.PvaServer('SR:FAKE', pv)\nserver.update(pv)\n",
            id="pva-server",
        ),
        pytest.param(
            "import pvaccess\nioc = pvaccess.CaIoc()\nioc.putField('SR:REC', 1.0)\n",
            id="ca-ioc",
        ),
    ],
)
def test_pvaccess_write_code_is_detected_as_a_write(code):
    result = detect(code)

    assert result["has_writes"] is True, f"no write pattern matched:\n{code}"


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "import pvaccess as pva\nch = pva.Channel('SR:BEND:SP')\nch.putDouble(1.5)\n",
            id="typed-setter",
        ),
        pytest.param(
            "import pvaccess\nch = pvaccess.Channel('SR:BEND:SP')\nch.asyncPut(pv, cb, err)\n",
            id="async-put",
        ),
        pytest.param(
            "import pvaccess\nch = pvaccess.Channel('SR:BEND:SP')\nch.parsePutGet(['value=1'])\n",
            id="parse-put-get",
        ),
    ],
)
def test_pvaccess_writes_the_generic_put_misses_are_caught(code):
    """The generic ``\\.put\\s*\\(`` is blind to these spellings; the pvaPy
    put-family entries are what catch them."""
    assert re.search(r"\.put\s*\(", code) is None
    assert any(re.search(pattern, code) for pattern in PVACCESS_WRITE_PATTERNS[:3])
    assert detect(code)["has_writes"] is True


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "import importlib\n"
            "importlib.import_module('pva' + 'ccess').Channel('SR:A:SP').putDouble(1.0)\n",
            id="typed-setter",
        ),
        pytest.param(
            "m = __import__('pva' + 'ccess')\nm.Channel('SR:A:SP').asyncPut(pv, cb, err)\n",
            id="async-put",
        ),
        pytest.param(
            "m = __import__('pva' + 'ccess')\nm.Channel('SR:A:SP').parsePutGet(['value=1'])\n",
            id="parse-put-get",
        ),
        pytest.param(
            "m = __import__('pva' + 'ccess')\n"
            "m.MultiChannel(['SR:A', 'SR:B']).putAsDoubleArray([1.0, 2.0])\n",
            id="multichannel-put",
        ),
    ],
)
def test_pvaccess_writes_through_a_built_import_are_caught(code):
    """An import assembled at runtime never writes the token ``pvaccess``, so
    the anchored entry misses it; the unanchored spellings catch it."""
    assert re.search(r"\bpvaccess\b", code) is None
    assert re.search(r"\.put\s*\(", code) is None
    assert detect(code)["has_writes"] is True


@pytest.mark.parametrize("pattern", PVACCESS_WRITE_PATTERNS)
def test_each_new_write_pattern_fires_on_some_snippet(pattern):
    """No dead entries: every new write regex earns its place."""
    snippets = [
        "import pvaccess\npvaccess.Channel('SR:A:SP').putDouble(1)\n",
        "ch.asyncPut(pv, cb, err)\nch.parsePut(['value=1'])\nmc.putAsDoubleArray([1.0])\n",
        "import pvaccess\nreply = pvaccess.RpcClient('SR:CALC').invoke(request)\n",
        "import pvaccess\nioc = pvaccess.CaIoc()\n",
    ]

    assert any(re.search(pattern, code) for code in snippets)


# ============================================================================
# Reads
# ============================================================================


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "import pvaccess\nval = pvaccess.Channel('SR:CURRENT').get()\n",
            id="qualified-get",
        ),
        pytest.param(
            "import pvaccess as pva\nch = pva.Channel('SR:IMAGE')\nch.monitor(cb)\n",
            id="monitor",
        ),
        pytest.param(
            "import pvaccess\n"
            "ch = pvaccess.Channel('SR:IMAGE')\n"
            "ch.subscribe('s1', cb)\n"
            "ch.startMonitor()\n",
            id="subscribe-start-monitor",
        ),
        pytest.param(
            "import pvaccess\nch = pvaccess.Channel('SR:IMAGE')\nch.asyncGet(cb, err)\n",
            id="async-get",
        ),
        pytest.param(
            "import pvaccess\n"
            "mc = pvaccess.MultiChannel(['SR:A', 'SR:B'])\n"
            "vals = mc.getAsDoubleArray()\n",
            id="multichannel-get",
        ),
    ],
)
def test_pvaccess_read_code_is_detected_as_a_read(code):
    result = detect(code)

    assert result["has_reads"] is True, f"no read pattern matched:\n{code}"


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "import pvaccess\nch = pvaccess.Channel('SR:IMAGE')\nch.monitor(cb)\n",
            id="monitor",
        ),
        pytest.param(
            "from pvaccess import Channel\nch = Channel('SR:IMAGE')\nch.subscribe('s', cb)\n",
            id="subscribe",
        ),
        pytest.param(
            "import pvaccess\nmc = pvaccess.MultiChannel(['SR:A', 'SR:B'])\n",
            id="multichannel-creation",
        ),
    ],
)
def test_pvaccess_reads_are_not_writes(code):
    """Opening a channel and monitoring it puts nothing on the wire."""
    result = detect(code)

    assert result["has_reads"] is True
    assert result["has_writes"] is False


@pytest.mark.parametrize("pattern", PVACCESS_READ_PATTERNS)
def test_each_new_read_pattern_fires_on_some_snippet(pattern):
    snippets = [
        "import pvaccess\nval = pvaccess.Channel('SR:CURRENT').asyncGet(cb, err)\n",
        "import pvaccess\npvaccess.Channel('SR:IMAGE').startMonitor()\n",
    ]

    assert any(re.search(pattern, code) for code in snippets)


# ============================================================================
# No false positives on ordinary analysis code
# ============================================================================


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "import queue\nq = queue.Queue()\nq.put(1)\nq.get()\n",
            id="queue-put-get",
        ),
        pytest.param(
            "import asyncio\nch = asyncio.Queue()\nresult = obj.invoke(args)\n",
            id="unrelated-invoke",
        ),
        pytest.param(
            "import numpy as np\nmean = np.mean(samples)\nprint(f'mean={mean:.3f}')\n",
            id="numpy-analysis",
        ),
        pytest.param(
            "from slack_sdk import WebClient\nchannel = client.conversations_open(users=u)\n",
            id="unrelated-channel",
        ),
        pytest.param(
            "import pandas as pd\ndf = pd.read_csv(path)\nout = df.pivot_table(index='t')\n",
            id="pandas-pivot",
        ),
        pytest.param(
            "cache.put_many(items)\noutput = compute(input_data)\nthroughput = n / dt\n",
            id="snake-case-put-and-put-substrings",
        ),
        pytest.param(
            "import cv2\ncv2.putText(frame, 'BPM 3', (10, 30), font, 1.0, (255, 0, 0))\n",
            id="opencv-put-text",
        ),
        pytest.param("device.putChar(b'x')\nnp.putmask(arr, mask, 0)\n", id="other-put-names"),
        pytest.param("reply = chain.invoke({'question': q})\n", id="unrelated-invoke-call"),
    ],
)
def test_ordinary_code_does_not_fire_any_new_pvaccess_pattern(code):
    """The new entries must be silent on code that has nothing to do with pvaPy."""
    for pattern in PVACCESS_WRITE_PATTERNS + PVACCESS_READ_PATTERNS:
        assert re.search(pattern, code) is None, f"{pattern!r} matched:\n{code}"

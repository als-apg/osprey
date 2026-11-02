"""The wheel layer's constraints filter: the frozen set minus the dropped wheels."""

from __future__ import annotations

import subprocess
import sys

import pytest

from osprey.deployment import constraints

FROZEN = [
    "osprey-framework==1.0\n",
    "osprey_connectors==1.0\n",
    "fwdep==1.0\n",
    "Zope.Interface==6.0\n",
    "site-payload @ file:///wheels/site_payload-0.1-py3-none-any.whl\n",
    "\n",
    "# a comment\n",
]


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ("/tmp/ctx/osprey_framework-1.1-py3-none-any.whl", "osprey-framework"),
        ("osprey_connectors-2.0.dev3-py3-none-any.whl", "osprey-connectors"),
        ("/w/Zope.Interface-6.1-cp313-cp313-linux_x86_64.whl", "zope-interface"),
    ],
)
def test_a_wheel_names_its_distribution_normalized(path, name):
    assert constraints.wheel_distribution(path) == name


@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("osprey-framework==1.0\n", "osprey-framework"),
        ("  osprey_framework >= 1.0\n", "osprey-framework"),
        ("pkg[extra]==1.0\n", "pkg"),
        ("pkg; python_version < '3.12'\n", "pkg"),
        ("\n", ""),
    ],
)
def test_a_line_names_its_requirement_normalized(line, name):
    assert constraints.line_distribution(line) == name


def test_only_the_lines_of_the_dropped_wheels_go():
    wheels = [
        "/tmp/ctx/osprey_framework-1.1-py3-none-any.whl",
        "/tmp/ctx/zope_interface-6.1-py3-none-any.whl",
        "/tmp/ctx/site_payload-0.1-py3-none-any.whl",
    ]
    assert constraints.minus_wheels(FROZEN, wheels) == [
        "osprey_connectors==1.0\n",
        "fwdep==1.0\n",
        "\n",
        "# a comment\n",
    ]


def test_no_wheel_keeps_every_line():
    assert constraints.minus_wheels(FROZEN, []) == FROZEN


def test_a_name_is_matched_whole_not_as_a_prefix():
    lines = ["osprey-framework-extras==1.0\n", "osprey-framework==1.0\n"]
    assert constraints.minus_wheels(lines, ["osprey_framework-1.1-py3-none-any.whl"]) == [
        "osprey-framework-extras==1.0\n"
    ]


def test_the_module_writes_the_filtered_file_to_stdout(tmp_path):
    frozen = tmp_path / "constraints.txt"
    frozen.write_text("".join(FROZEN), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "osprey.deployment.constraints",
            str(frozen),
            str(tmp_path / "osprey_framework-1.1-py3-none-any.whl"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "".join(line for line in FROZEN if line != "osprey-framework==1.0\n")


def test_without_arguments_it_prints_usage_and_exits_2(capsys):
    assert constraints.main([]) == 2
    assert "usage:" in capsys.readouterr().err
